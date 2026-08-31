from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from typing import Any


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
    # Chuẩn hóa cả các Unicode space như NARROW NO-BREAK SPACE (U+202F),
    # thường xuất hiện giữa "Điều" và số điều trong văn bản PDF/OCR.
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.split("\n")]
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
        if preamble:
            pieces.append(preamble)
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        piece = text[match.start() : end].strip()
        if piece:
            pieces.append(piece)
    return pieces


def _word_windows(
    text: str,
    max_words: int,
    overlap_words: int,
    min_words: int,
) -> Iterable[str]:
    words = text.split()
    if len(words) <= max_words:
        if words:
            yield " ".join(words)
        return

    step = max(1, max_words - overlap_words)
    anchored_start = len(words) - max_words
    last_regular_start = (
        (anchored_start + step - 1) // step
    ) * step
    tail_words = len(words) - last_regular_start

    if tail_words >= min_words:
        for start in range(0, last_regular_start + 1, step):
            yield " ".join(words[start : start + max_words])
        return

    # Rebalance hai cửa sổ cuối khi có thể: cửa sổ cuối đạt min_words,
    # cửa sổ trước được rút ngắn, còn overlap vẫn đúng bằng cấu hình.
    previous_start = last_regular_start - step
    balanced_last_start = len(words) - min_words
    balanced_previous_end = balanced_last_start + overlap_words
    if balanced_previous_end - previous_start >= min_words:
        for start in range(0, previous_start, step):
            yield " ".join(words[start : start + max_words])
        yield " ".join(words[previous_start:balanced_previous_end])
        yield " ".join(words[balanced_last_start:])
        return

    # Với cấu hình như min_words == max_words, không thể vừa giữ mọi token,
    # vừa buộc cả hai chunk đạt min_words. Ưu tiên không mất dữ liệu và giữ
    # overlap chính xác; chỉ chunk cuối được phép ngắn hơn min_words.
    for start in range(0, last_regular_start + 1, step):
        yield " ".join(words[start : start + max_words])


def validate_chunk_parameters(
    max_words: int,
    overlap_words: int,
    min_words: int | None = None,
) -> None:
    values = (max_words, overlap_words)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("Cấu hình chunk phải là số nguyên")
    if max_words <= 0 or overlap_words < 0 or overlap_words >= max_words:
        raise ValueError("Cấu hình chunk không hợp lệ")
    if min_words is not None:
        if isinstance(min_words, bool) or not isinstance(min_words, int):
            raise ValueError("min_words phải là số nguyên")
        if min_words <= 0 or min_words > max_words:
            raise ValueError("min_words phải nằm trong khoảng 1..max_words")


def chunk_passage(
    passage: str,
    max_words: int = 620,
    overlap_words: int = 100,
    min_words: int = 18,
) -> list[str]:
    """Chia văn bản ưu tiên theo Điều/Phụ lục rồi mới dùng cửa sổ từ."""
    validate_chunk_parameters(max_words, overlap_words, min_words)

    text = normalize_text(passage)
    chunks: list[str] = []
    for piece in _split_on_headings(text):
        chunks.extend(_word_windows(piece, max_words, overlap_words, min_words))
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
    return " ".join(words[best_start : best_start + window_size]).strip()


LONG_ANSWER_PATTERNS: tuple[str, ...] = (
    "mẫu",
    "biểu mẫu",
    "phụ lục",
    "liệt kê",
    "bao gồm những",
    "nội dung điều",
    "nội dung khoản",
    "các trường hợp",
    "điều kiện",
    "hồ sơ gồm",
    "quyền và nghĩa vụ",
)


def is_long_answer_question(
    question: str, patterns: Iterable[str] = LONG_ANSWER_PATTERNS
) -> bool:
    """Kiểm tra câu hỏi có thuộc nhóm yêu cầu câu trả lời dài/liệt kê/nguyên văn hay không."""
    if not isinstance(question, str):
        return False
    normalized = unicodedata.normalize("NFC", question.casefold())
    for pattern in patterns:
        pat_norm = unicodedata.normalize("NFC", str(pattern).casefold().strip())
        if pat_norm and pat_norm in normalized:
            return True
    return False


def merge_adjacent_chunks(
    chunks: list[dict[str, Any]],
    max_words: int = 800,
) -> str:
    """Ghép nội dung các chunk (đã sắp xếp theo chunk_no) thành văn bản trích xuất dài."""
    if max_words <= 0:
        raise ValueError("max_words phải lớn hơn 0")
    if not chunks:
        return ""
    sorted_chunks = sorted(chunks, key=lambda c: int(c.get("chunk_no", 0)))
    merged_texts: list[str] = []
    total_words = 0

    for chunk in sorted_chunks:
        text = str(chunk.get("text") or "").strip()
        if not text:
            continue
        words = text.split()
        if not words:
            continue
        # Tránh trùng lặp hoàn toàn nếu 2 chunk liên tiếp giống hệt nhau
        if merged_texts and text == merged_texts[-1]:
            continue
        if total_words + len(words) <= max_words:
            merged_texts.append(text)
            total_words += len(words)
        else:
            remaining = max_words - total_words
            if remaining > 0:
                merged_texts.append(" ".join(words[:remaining]))
                total_words += remaining
            break

    return "\n\n".join(merged_texts).strip()

