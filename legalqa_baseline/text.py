from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable


# Chỉ bỏ các hư từ rất phổ biến. Giữ lại từ khóa pháp lý như "điều", "khoản",
# "nghị định" vì số hiệu đi kèm thường mang tín hiệu truy xuất mạnh.
STOPWORDS = {
    "ai", "bao", "bằng", "bị", "các", "cái", "cho", "có", "của", "cũng",
    "đã", "đang", "để", "đến", "đó", "được", "gì", "hay", "hiện", "hỏi",
    "khi", "không", "là", "làm", "mà", "một", "nào", "này", "như", "những",
    "phải", "ra", "sao", "sẽ", "thế", "thì", "theo", "trên", "trong", "từ",
    "và", "về", "với", "việc", "quy", "định", "thế nào", "như thế nào",
    # Các token pháp lý xuất hiện ở gần như mọi văn bản. Số hiệu và chủ thể
    # cụ thể vẫn được giữ, nên bỏ nhóm này giúp FTS tránh quét posting list lớn.
    "căn", "cứ", "điều", "khoản", "điểm", "pháp", "luật", "nghị", "thông", "tư",
}

TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
ARTICLE_RE = re.compile(
    r"(?im)^[ \t]*(?:ĐIỀU|Điều|điều)[ \t]+\d+[a-zA-ZđĐ]*(?:[ \t]*[.:\-])?.*$"
)
APPENDIX_RE = re.compile(
    r"(?im)^[ \t]*(?:PHỤ[ \t]+LỤC|Phụ[ \t]+lục|MẪU[ \t]+SỐ|Mẫu[ \t]+số).*$"
)


def normalize_text(text: str) -> str:
    """Chuẩn hóa lỗi xuống dòng/Unicode nhưng không đổi từ ngữ pháp lý."""
    text = unicodedata.normalize("NFC", str(text or ""))
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    # Loại các dòng trống lặp lại, vẫn giữ một newline để nhận diện Điều/Phụ lục.
    output: list[str] = []
    for line in lines:
        if line:
            output.append(line)
        elif output and output[-1] != "":
            output.append("")
    return "\n".join(output).strip()


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", str(text or "")).casefold()
    return TOKEN_RE.findall(text)


def query_terms(text: str, max_terms: int = 36) -> list[str]:
    """Rút các token có ích cho FTS5, giữ thứ tự xuất hiện và không lặp."""
    seen: set[str] = set()
    result: list[str] = []
    for token in tokenize(text):
        if token in STOPWORDS or len(token) < 2 or token in seen:
            continue
        seen.add(token)
        result.append(token)
        if len(result) >= max_terms:
            break
    if result:
        return result
    return list(dict.fromkeys(tokenize(text)))[:max_terms]


def _split_on_headings(text: str) -> list[str]:
    all_matches = sorted(
        [*ARTICLE_RE.finditer(text), *APPENDIX_RE.finditer(text)],
        key=lambda match: match.start(),
    )
    if not all_matches:
        return [text]

    matches = []
    seen_starts: set[int] = set()
    for match in all_matches:
        if match.start() not in seen_starts:
            seen_starts.add(match.start())
            matches.append(match)

    pieces: list[str] = []
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if len(preamble.split()) >= 30:
            pieces.append(preamble)
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        piece = text[match.start() : end].strip()
        if piece:
            pieces.append(piece)
    return pieces


def _word_windows(text: str, max_words: int, overlap_words: int) -> Iterable[str]:
    words = text.split()
    if len(words) <= max_words:
        if words:
            yield " ".join(words)
        return

    step = max(1, max_words - overlap_words)
    start = 0
    while start < len(words):
        window = words[start : start + max_words]
        if window:
            yield " ".join(window)
        if start + max_words >= len(words):
            break
        start += step


def chunk_passage(
    passage: str,
    max_words: int = 620,
    overlap_words: int = 100,
    min_words: int = 18,
) -> list[str]:
    """Chia văn bản ưu tiên theo Điều/Phụ lục rồi mới dùng cửa sổ từ."""
    if max_words <= 0 or overlap_words < 0 or overlap_words >= max_words:
        raise ValueError("Cấu hình chunk không hợp lệ")

    text = normalize_text(passage)
    chunks: list[str] = []
    for piece in _split_on_headings(text):
        is_heading_piece = bool(ARTICLE_RE.match(piece) or APPENDIX_RE.match(piece))
        preserve_short_piece = is_heading_piece and len(piece.split()) < min_words
        for chunk in _word_windows(piece, max_words, overlap_words):
            if preserve_short_piece or len(chunk.split()) >= min_words:
                chunks.append(chunk)
    return chunks


def best_excerpt(text: str, question: str, max_words: int = 520) -> str:
    """Lấy cửa sổ trong chunk phủ nhiều từ khóa câu hỏi nhất."""
    words = str(text or "").split()
    if len(words) <= max_words:
        return " ".join(words).strip()

    terms = set(query_terms(question))
    if not terms:
        return " ".join(words[:max_words]).strip()

    window_size = max_words
    step = max(40, window_size // 5)
    best_start = 0
    best_score = float("-inf")
    for start in range(0, max(1, len(words) - window_size + 1), step):
        window = words[start : start + window_size]
        window_tokens = tokenize(" ".join(window))
        unique_overlap = len(terms.intersection(window_tokens))
        repeated_overlap = sum(token in terms for token in window_tokens)
        score = 3.0 * unique_overlap + min(repeated_overlap, 2 * len(terms))
        if score > best_score:
            best_start = start
            best_score = score

    # Kiểm tra cửa sổ cuối vì range có thể không chạm đuôi văn bản.
    last_start = max(0, len(words) - window_size)
    last_tokens = tokenize(" ".join(words[last_start:]))
    last_score = 3.0 * len(terms.intersection(last_tokens)) + min(
        sum(token in terms for token in last_tokens), 2 * len(terms)
    )
    if last_score > best_score:
        best_start = last_start

    return " ".join(words[best_start : best_start + window_size]).strip()
