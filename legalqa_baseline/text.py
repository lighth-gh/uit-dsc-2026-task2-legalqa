from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from decimal import Decimal, InvalidOperation
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

# Controlled retrieval aliases. Do not convert standalone legal article or
# decree numbers: Arabic/Roman expansion is only safe in the named concept
# "điện 8 / điện VIII".
_POWER_PLAN_NUMBER_RE = re.compile(
    r"(?i)\b(?P<prefix>(?:quy\s+hoạch\s+)?điện\s+)(?P<number>8|viii)\b"
)
_MILLION_AMOUNT_RE = re.compile(
    r"(?i)(?<![\w.,])(?P<amount>\d+(?:[.,]\d+)?)\s*triệu(?:\s+đồng)?\b"
)
_GROUPED_VND_RE = re.compile(
    r"(?i)(?<![\w.,])(?P<amount>\d{1,3}(?:[.]\d{3})+)\s*đồng\b"
)
_PRIORITY_LEGAL_PHRASES = ("mức lương cơ sở",)


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


def _format_grouped_vnd(amount: Decimal) -> str | None:
    dong = amount * Decimal(1_000_000)
    if dong != dong.to_integral_value() or dong <= 0:
        return None
    return f"{int(dong):,}".replace(",", ".") + " đồng"


def _format_million_amount(dong: int) -> str | None:
    if dong <= 0 or dong % 100_000 != 0:
        return None
    millions = Decimal(dong) / Decimal(1_000_000)
    value = format(millions.normalize(), "f").replace(".", ",")
    return f"{value} triệu"


def retrieval_query_aliases(text: str) -> list[str]:
    """Return only high-confidence aliases for named plans and money values."""
    source = unicodedata.normalize("NFC", str(text or ""))
    aliases: list[str] = []

    for match in _POWER_PLAN_NUMBER_RE.finditer(source):
        replacement = "VIII" if match.group("number").casefold() == "8" else "8"
        aliases.append(f"{match.group('prefix')}{replacement}".strip())

    for match in _MILLION_AMOUNT_RE.finditer(source):
        raw_amount = match.group("amount").replace(",", ".")
        try:
            amount = Decimal(raw_amount)
        except InvalidOperation:
            continue
        formatted = _format_grouped_vnd(amount)
        if formatted:
            aliases.append(formatted)

    for match in _GROUPED_VND_RE.finditer(source):
        try:
            dong = int(match.group("amount").replace(".", ""))
        except ValueError:
            continue
        formatted = _format_million_amount(dong)
        if formatted:
            aliases.append(formatted)

    source_folded = source.casefold()
    return [
        alias
        for alias in dict.fromkeys(aliases)
        if alias.casefold() not in source_folded
    ]


def expand_retrieval_query(text: str) -> str:
    """Append controlled aliases while preserving the complete original query."""
    source = unicodedata.normalize("NFC", str(text or "")).strip()
    aliases = retrieval_query_aliases(source)
    return " ".join([source, *aliases]).strip()


def retrieval_priority_phrases(text: str) -> list[str]:
    """Exact phrases that may safely receive lexical priority in FTS retrieval."""
    expanded = expand_retrieval_query(text)
    expanded_folded = expanded.casefold()
    phrases: list[str] = []

    for match in _POWER_PLAN_NUMBER_RE.finditer(expanded):
        phrases.append(match.group(0))
    for phrase in _PRIORITY_LEGAL_PHRASES:
        if phrase in expanded_folded:
            phrases.append(phrase)
    for pattern in (_MILLION_AMOUNT_RE, _GROUPED_VND_RE):
        phrases.extend(match.group(0) for match in pattern.finditer(expanded))

    # FTS5 tokenizes punctuation inside money amounts. Converting each phrase
    # to its token sequence creates a valid exact FTS phrase for both formats.
    normalized_phrases = (" ".join(tokenize(phrase)) for phrase in phrases)
    return [phrase for phrase in dict.fromkeys(normalized_phrases) if phrase]


def query_terms(text: str, max_terms: int = 36) -> list[str]:
    """Rút các token có ích cho FTS5, giữ thứ tự xuất hiện và không lặp."""
    seen: set[str] = set()
    result: list[str] = []
    for token in tokenize(text):
        if token in STOPWORDS or (len(token) < 2 and not token.isdigit()) or token in seen:
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


def truncate_at_text_boundary(text: str, max_words: int) -> str:
    """Limit text at a paragraph/sentence boundary, with a word-safe fallback."""
    if isinstance(max_words, bool) or not isinstance(max_words, int) or max_words <= 0:
        raise ValueError("max_words phải là số nguyên lớn hơn 0")
    source = str(text or "").strip()
    word_matches = list(re.finditer(r"\S+", source))
    if len(word_matches) <= max_words:
        return source

    char_limit = word_matches[max_words - 1].end()
    prefix = source[:char_limit].rstrip()
    boundary_ends: list[int] = []

    # A newline closes a legal-list item/paragraph. Sentence punctuation is
    # also safe, while the word-limited prefix remains the final fallback for
    # OCR text that contains neither kind of boundary.
    boundary_ends.extend(match.start() for match in re.finditer(r"\n+", prefix))
    boundary_ends.extend(
        match.end()
        for match in re.finditer(r"[.!?…](?=(?:[\"'”’\)\]]*)?(?:\s|$))", prefix)
    )
    valid_ends = [end for end in boundary_ends if prefix[:end].strip()]
    if valid_ends:
        return prefix[: max(valid_ends)].strip()
    return prefix


def best_excerpt(text: str, question: str, max_words: int = 520) -> str:
    """Lấy cửa sổ trong chunk phủ nhiều từ khóa câu hỏi nhất."""
    words = str(text or "").split()
    if len(words) <= max_words:
        return " ".join(words).strip()

    terms = set(query_terms(question))
    if not terms:
        return truncate_at_text_boundary(str(text or ""), max_words)

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
    selected = " ".join(words[best_start:]).strip()
    return truncate_at_text_boundary(selected, max_words)


LONG_ANSWER_PATTERNS: tuple[str, ...] = (
    r"\bthủ tục\b",
    r"\btrình tự\b",
    r"\bhồ sơ\b",
    r"\bbao gồm\b",
    r"\bcác trường hợp\b",
    r"\bcác bước\b",
    r"\bđiều kiện\b",
    r"\bquyền và nghĩa vụ\b",
    r"\btrách nhiệm\b",
    r"\bnội dung và phương pháp\b",
    r"\bbiểu mẫu\b",
    # Các cách hỏi dài tương đương thường gặp trong bộ LegalQA.
    r"\bmẫu(?: số)?\b",
    r"\bphụ lục\b",
    r"\bliệt kê\b",
    r"\bnội dung điều\b",
    r"\bnội dung khoản\b",
    r"\bđiều luật\b",
    r"\bnguyên văn\b",
    r"\btoàn văn\b",
    r"\bquy định đầy đủ\b",
)
LONG_PATTERNS = LONG_ANSWER_PATTERNS


def is_long_form_question(
    question: str, patterns: Iterable[str] = LONG_ANSWER_PATTERNS
) -> bool:
    """Kiểm tra câu hỏi có thuộc nhóm yêu cầu câu trả lời dài/liệt kê/nguyên văn hay không."""
    if not isinstance(question, str):
        return False
    normalized = unicodedata.normalize("NFC", question.casefold())
    for pattern in patterns:
        pat_norm = unicodedata.normalize("NFC", str(pattern).casefold().strip())
        if pat_norm and re.search(pat_norm, normalized, flags=re.UNICODE):
            return True
    return False


def is_long_answer_question(
    question: str, patterns: Iterable[str] = LONG_ANSWER_PATTERNS
) -> bool:
    """Alias tương thích cho tên hàm cũ."""
    return is_long_form_question(question, patterns=patterns)


BOILERPLATE_PATTERNS: tuple[str, ...] = (
    r"^\s*dựa trên ngữ cảnh được cung cấp[,:\s]*",
    r"^\s*theo ngữ cảnh được cung cấp[,:\s]*",
    r"^\s*dựa trên thông tin được cung cấp[,:\s]*",
)


def clean_answer(answer: str) -> str:
    """Chỉ bỏ boilerplate ở đầu đáp án, không tóm tắt hoặc cắt nội dung."""
    cleaned = str(answer or "")
    for pattern in BOILERPLATE_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, count=1, flags=re.IGNORECASE)
    return cleaned.strip()


def possibly_cut(text: str) -> bool:
    """Phát hiện đáp án có vẻ dừng giữa câu hoặc giữa một mục liệt kê."""
    answer = str(text or "").strip()
    if not answer or len(answer.split()) < 8:
        return False
    if answer.endswith((".", "!", "?", "…", ")", "]", "}", '"', "”", "’")):
        return False
    if answer.endswith((",", ";", ":", "-", "–", "—", "/")):
        return True
    normalized = " ".join(answer.casefold().split())
    dangling_endings = (
        " và",
        " hoặc",
        " bao gồm",
        " gồm",
        " như sau",
        " theo",
        " tại",
    )
    if normalized.endswith(dangling_endings):
        return True
    if answer.count("(") > answer.count(")") or answer.count("[") > answer.count("]"):
        return True
    return False


def deduplicate_overlaps(texts: Iterable[str]) -> str:
    """Ghép các chunk liên tiếp và bỏ phần word-overlap ở ranh giới chunk."""
    merged = ""
    merged_words: list[str] = []
    for raw_text in texts:
        text = str(raw_text or "").strip()
        words = text.split()
        if not words:
            continue
        if not merged_words:
            merged = text
            merged_words = words
            continue

        max_overlap = min(len(merged_words), len(words))
        overlap = 0
        normalized_left = [word.casefold() for word in merged_words[-max_overlap:]]
        normalized_right = [word.casefold() for word in words[:max_overlap]]
        for size in range(max_overlap, 2, -1):
            if normalized_left[-size:] == normalized_right[:size]:
                overlap = size
                break

        # Vẫn loại chunk ngắn trùng hoàn toàn, nhưng không coi một/hai từ vô
        # tình giống nhau ở ranh giới Điều là overlap vì có thể làm mất ý luật.
        if not overlap and len(words) <= len(merged_words):
            normalized_full = [word.casefold() for word in words]
            if [word.casefold() for word in merged_words[-len(words):]] == normalized_full:
                overlap = len(words)

        suffix = words[overlap:]
        if not suffix:
            continue
        merged = f"{merged}\n\n{' '.join(suffix)}"
        merged_words.extend(suffix)
    return merged.strip()


def build_extractive_answer(chunks: list[dict[str, Any]]) -> str:
    """Tạo raw answer từ các chunk liền kề của cùng văn bản."""
    ordered = sorted(
        chunks,
        key=lambda chunk: int(chunk.get("chunk_no", chunk.get("chunk_index", 0))),
    )
    return deduplicate_overlaps(
        str(chunk.get("text") or "") for chunk in ordered
    )


def merge_adjacent_chunks(
    chunks: list[dict[str, Any]],
    max_words: int = 800,
) -> str:
    """Ghép nội dung các chunk (đã sắp xếp theo chunk_no) thành văn bản trích xuất dài."""
    if max_words <= 0:
        raise ValueError("max_words phải lớn hơn 0")
    if not chunks:
        return ""
    merged = build_extractive_answer(chunks)
    return truncate_at_text_boundary(merged, max_words)
