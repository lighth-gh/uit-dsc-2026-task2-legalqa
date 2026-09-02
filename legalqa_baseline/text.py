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
_DOCUMENT_REFERENCE_RE = re.compile(
    r"(?i)\b(?P<kind>nghị\s+định|quyết\s+định|thông\s+tư|nghị\s+quyết|"
    r"công\s+văn|luật)\s*(?:số\s*)?"
    r"(?P<number>\d+(?:[/.-][\wđĐ-]+)+)"
)
_YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_PRIORITY_LEGAL_PHRASES = ("mức lương cơ sở",)
_MAX_EXACT_PHRASE_TOKENS = 8
_MIN_EXACT_PHRASE_TOKENS = 4

# Các cụm này mô tả đúng vấn đề pháp lý đang được hỏi. Chúng được dùng như
# guardrail sau cross-encoder, không được đưa thẳng vào câu trả lời.
_LEGAL_FOCUS_PATTERNS = (
    re.compile(r"\bthời\s+(?:hiệu|hạn)\s+[^,;?.]{1,80}", re.IGNORECASE),
    re.compile(r"\bnghĩa\s+vụ\s+[^,;?.]{1,64}", re.IGNORECASE),
    re.compile(r"\b(?:đăng\s+ký|cấp|thu\s+hồi|hủy)\s+[^,;?.]{1,72}", re.IGNORECASE),
    re.compile(r"\b(?:giá\s+trị\s+)?bồi\s+thường\s+[^,;?.]{1,80}", re.IGNORECASE),
    re.compile(r"\bhội\s+viên\s+[^,;?.]{1,64}", re.IGNORECASE),
)
_FOCUS_TRAILING_TOKENS = {
    "bao", "lâu", "như", "thế", "nào", "không", "ra", "sao", "gì",
    "được", "xác", "định",
}
_LEGAL_SCOPE_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("vụ án dân sự", ("tố tụng dân sự", "bộ luật tố tụng dân sự")),
    ("tố tụng dân sự", ("tố tụng dân sự", "bộ luật tố tụng dân sự")),
    ("bản án hình sự", ("tố tụng hình sự", "bộ luật tố tụng hình sự")),
    ("vụ án hình sự", ("tố tụng hình sự", "bộ luật tố tụng hình sự")),
    ("tố tụng hình sự", ("tố tụng hình sự", "bộ luật tố tụng hình sự")),
    ("tố tụng hành chính", ("tố tụng hành chính", "luật tố tụng hành chính")),
)
_HEADING_QUERY_IGNORED = STOPWORDS | {
    "bao", "lâu", "như", "thế", "nào", "không", "ra", "sao", "gì",
    "xác", "định", "giá", "trị", "phần",
}


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


def _document_references(text: str) -> set[str]:
    references: set[str] = set()
    for match in _DOCUMENT_REFERENCE_RE.finditer(text):
        kind = " ".join(tokenize(match.group("kind")))
        number = re.sub(r"\s+", "", match.group("number")).casefold()
        references.add(f"{kind} {number}")
    return references


def _money_amounts(text: str) -> set[int]:
    amounts: set[int] = set()
    for match in _MILLION_AMOUNT_RE.finditer(text):
        try:
            amount = Decimal(match.group("amount").replace(",", "."))
        except InvalidOperation:
            continue
        dong = amount * Decimal(1_000_000)
        if dong > 0 and dong == dong.to_integral_value():
            amounts.add(int(dong))
    for match in _GROUPED_VND_RE.finditer(text):
        try:
            amounts.add(int(match.group("amount").replace(".", "")))
        except ValueError:
            continue
    return amounts


def _power_plan_names(text: str) -> set[str]:
    # Both "điện 8" and "Quy hoạch điện VIII" describe the same named plan.
    return {
        "quy hoạch điện viii"
        for _match in _POWER_PLAN_NUMBER_RE.finditer(text)
    }


def _longest_matching_form_name(
    question_tokens: list[str],
    candidate_tokens: list[str],
) -> str | None:
    candidate_text = f" {' '.join(candidate_tokens)} "
    for start, token in enumerate(question_tokens):
        if token == "biểu" and start + 1 < len(question_tokens):
            if question_tokens[start + 1] != "mẫu":
                continue
        elif token != "mẫu":
            continue
        max_size = min(12, len(question_tokens) - start)
        for size in range(max_size, 2, -1):
            phrase = " ".join(question_tokens[start : start + size])
            if f" {phrase} " in candidate_text:
                return phrase
    return None


def _longest_exact_legal_phrase(
    question_tokens: list[str],
    candidate_tokens: list[str],
) -> tuple[str | None, int]:
    candidate_text = f" {' '.join(candidate_tokens)} "
    max_size = min(_MAX_EXACT_PHRASE_TOKENS, len(question_tokens))
    for size in range(max_size, _MIN_EXACT_PHRASE_TOKENS - 1, -1):
        for start in range(0, len(question_tokens) - size + 1):
            window = question_tokens[start : start + size]
            informative = [
                token
                for token in window
                if token not in STOPWORDS and (len(token) >= 2 or token.isdigit())
            ]
            if len(informative) < 3:
                continue
            phrase = " ".join(window)
            if f" {phrase} " in candidate_text:
                return phrase, size
    return None, 0


def legal_question_focus_phrases(text: str) -> list[str]:
    """Extract high-precision legal issue phrases, longest phrase first."""
    source = unicodedata.normalize("NFC", str(text or "")).casefold()
    phrases: list[str] = []
    for pattern in _LEGAL_FOCUS_PATTERNS:
        for match in pattern.finditer(source):
            tokens = tokenize(match.group(0))
            while tokens and tokens[-1] in _FOCUS_TRAILING_TOKENS:
                tokens.pop()
            max_size = min(7, len(tokens))
            # Giữ cả các prefix ngắn: tiêu đề Điều thường chỉ có 3-4 token,
            # còn câu hỏi tiếp tục bằng điều kiện/đối tượng dài hơn.
            # Hai token như "thời hiệu" hoặc "thu hồi" còn quá chung và có
            # thể khớp một thao tác pháp lý khác. Cần ít nhất ba token để bonus
            # hậu reranker được xem là tín hiệu trọng tâm.
            for size in range(max_size, 2, -1):
                phrase = " ".join(tokens[:size])
                if phrase and phrase not in phrases:
                    phrases.append(phrase)
    return phrases


def _matching_question_focus_phrases(question: str, candidate_text: str) -> list[str]:
    candidate = f" {' '.join(tokenize(candidate_text))} "
    matches = [
        phrase
        for phrase in legal_question_focus_phrases(question)
        if f" {phrase} " in candidate
    ]
    # Prefixes của cùng một cụm không được tính lặp nhiều lần.
    kept: list[str] = []
    for phrase in matches:
        if any(f" {phrase} " in f" {existing} " for existing in kept):
            continue
        kept.append(phrase)
    return kept


def _fold_for_lexical_match(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", str(text or "").casefold())
    without_marks = "".join(
        character
        for character in decomposed
        if unicodedata.category(character) != "Mn"
    ).replace("đ", "d")
    return " ".join(TOKEN_RE.findall(without_marks))


def _matching_legal_scopes(question: str, candidate_text: str) -> tuple[list[str], bool]:
    question_normalized = _fold_for_lexical_match(question)
    candidate_normalized = _fold_for_lexical_match(candidate_text)
    requested: list[str] = []
    matched: list[str] = []
    for question_phrase, candidate_aliases in _LEGAL_SCOPE_ALIASES:
        folded_question_phrase = _fold_for_lexical_match(question_phrase)
        if folded_question_phrase not in question_normalized:
            continue
        requested.append(question_phrase)
        if any(
            _fold_for_lexical_match(alias) in candidate_normalized
            for alias in candidate_aliases
        ):
            matched.append(question_phrase)
    return matched, bool(requested)


def _leading_legal_heading(text: str, max_tokens: int = 18) -> str:
    """Return only the leading Điều/Mẫu heading, excluding numbered body items."""
    source = normalize_text(text).replace("\n", " ").strip()
    marker = re.search(
        r"\b(?:điều\s+\d+[a-zđ]*|mẫu(?:\s+số)?\s+[\w./-]+)\b",
        source,
        flags=re.IGNORECASE,
    )
    if marker:
        source = source[marker.start():]
    source = re.sub(
        r"^\s*(?:điều\s+\d+[a-zđ]*|mẫu(?:\s+số)?\s+[\w./-]+)\s*[.:-]?\s*",
        "",
        source,
        count=1,
        flags=re.IGNORECASE,
    )
    body_start = re.search(r"\s+(?:1|I)[.)]\s+", source)
    if body_start:
        source = source[:body_start.start()]
    return " ".join(tokenize(source)[:max_tokens])


def _heading_query_overlap(question: str, candidate_text: str) -> tuple[int, float]:
    heading_terms = {
        token
        for token in tokenize(_leading_legal_heading(candidate_text))
        if token not in _HEADING_QUERY_IGNORED and len(token) >= 2
    }
    question_terms = {
        token
        for token in tokenize(question)
        if token not in _HEADING_QUERY_IGNORED and len(token) >= 2
    }
    if not heading_terms or not question_terms:
        return 0, 0.0
    overlap = len(heading_terms & question_terms)
    return overlap, overlap / len(question_terms)


def legal_retrieval_signal_matches(
    question: str,
    candidate_text: str,
) -> dict[str, Any]:
    """Find exact, high-precision legal signals shared by a query and chunk."""
    question_tokens = tokenize(question)
    candidate_tokens = tokenize(candidate_text)

    document_references = sorted(
        _document_references(question) & _document_references(candidate_text)
    )
    money_amounts = sorted(_money_amounts(question) & _money_amounts(candidate_text))
    years = sorted(set(_YEAR_RE.findall(question)) & set(_YEAR_RE.findall(candidate_text)))
    plan_names = sorted(_power_plan_names(question) & _power_plan_names(candidate_text))
    form_name = _longest_matching_form_name(question_tokens, candidate_tokens)
    long_phrase, long_phrase_tokens = _longest_exact_legal_phrase(
        question_tokens,
        candidate_tokens,
    )
    focus_phrases = _matching_question_focus_phrases(question, candidate_text)
    scope_phrases, scope_requested = _matching_legal_scopes(
        question,
        candidate_text,
    )
    heading_overlap_tokens, heading_query_coverage = _heading_query_overlap(
        question,
        candidate_text,
    )

    return {
        "document_references": document_references,
        "money_amounts_vnd": money_amounts,
        "years": years,
        "plan_names": plan_names,
        "form_names": [form_name] if form_name else [],
        "long_phrase": long_phrase,
        "long_phrase_tokens": long_phrase_tokens,
        "focus_phrases": focus_phrases,
        "scope_phrases": scope_phrases,
        "scope_requested": scope_requested,
        "heading_overlap_tokens": heading_overlap_tokens,
        "heading_query_coverage": round(heading_query_coverage, 6),
    }


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
    # Các dạng hỏi có cấu trúc đã chạm 512 token trong smoke30. Đây là tín
    # hiệu danh sách/quy trình rõ ràng, không phải mọi câu có từ "quy định".
    r"\bđược thực hiện (?:như thế nào|ra sao)\b",
    r"\b(?:các|những)\s+(?:yêu cầu|hình thức|công việc|nội dung|nhiệm vụ|quyền hạn)\b",
    r"\btiêu chuẩn(?:\s+(?:của|đối với|chung|cụ thể))?\b",
    r"\bthời gian hưởng chế độ\b",
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
    r"^\s*dựa\s+(?:trên|vào)\s+(?:các\s+)?ngữ\s+cảnh(?:\s+được\s+cung\s+cấp)?[,;:\s-]*",
    r"^\s*theo\s+(?:các\s+)?ngữ\s+cảnh(?:\s+được\s+cung\s+cấp)?[,;:\s-]*",
    r"^\s*dựa\s+(?:trên|vào)\s+(?:các\s+)?thông\s+tin(?:\s+được\s+cung\s+cấp)?[,;:\s-]*",
)

_REFUSAL_START_MARKERS = (
    "không đủ thông tin trong ngữ cảnh",
    "không có đủ thông tin trong ngữ cảnh",
    "không thể trả lời",
    "tôi không thể trả lời",
    "xin lỗi, tôi không thể",
    "xin lỗi",
    "tôi không có thông tin",
    "không có thông tin",
    "không tìm thấy thông tin",
)
_REFUSAL_EARLY_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\bkhông\s+có\s+thông\s+tin\s+cụ\s+thể\b",
        r"\b(?:các\s+)?ngữ\s+cảnh(?:\s+được\s+cung\s+cấp)?\s+không\s+đề\s+cập\b",
        r"\bkhông\s+thể\s+trả\s+lời\s+chính\s+xác\b",
        r"\bkhông\s+tìm\s+thấy\s+thông\s+tin\b",
    )
)


def clean_answer(answer: str) -> str:
    """Chỉ bỏ boilerplate ở đầu đáp án, không tóm tắt hoặc cắt nội dung."""
    cleaned = str(answer or "")
    for pattern in BOILERPLATE_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, count=1, flags=re.IGNORECASE)
    return cleaned.strip()


def is_refusal_answer(answer: Any, max_sentences: int = 2) -> bool:
    """Detect an LLM refusal in the leading sentences after stripping boilerplate."""
    if not isinstance(answer, str) or not answer.strip():
        return False
    if max_sentences <= 0:
        raise ValueError("max_sentences phải lớn hơn 0")

    cleaned = clean_answer(answer)
    early_sentences = [
        " ".join(sentence.casefold().split()).strip(" .!?:;,-")
        for sentence in re.split(r"(?<=[.!?…])\s+|[\r\n]+", cleaned)
        if sentence.strip()
    ][:max_sentences]
    return any(
        sentence.startswith(_REFUSAL_START_MARKERS)
        or any(pattern.search(sentence) for pattern in _REFUSAL_EARLY_PATTERNS)
        for sentence in early_sentences
    )


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


def _heading_token_start(tokens: list[str], match_start: int) -> int | None:
    """Return a nearby Mẫu/Phụ lục/Điều marker preceding a title match."""
    lower_bound = max(0, match_start - 12)
    for index in range(match_start, lower_bound - 1, -1):
        token = tokens[index]
        if token in {"mẫu", "điều"}:
            return index
        if token == "phụ" and index + 1 < len(tokens) and tokens[index + 1] == "lục":
            return index
    return None


def _matching_legal_heading_start(question: str, text: str) -> int | None:
    """Locate the strongest exact title phrase in a form/article answer."""
    question_tokens = tokenize(question)
    requests_named_heading = (
        "mẫu" in question_tokens
        or "biểu" in question_tokens
        or "phụ" in question_tokens and "lục" in question_tokens
        or "điều" in question_tokens
    )
    if not requests_named_heading:
        return None

    token_matches = list(TOKEN_RE.finditer(text))
    text_tokens = [match.group(0).casefold() for match in token_matches]
    if not text_tokens:
        return None

    minimum_size = 2 if "điều" in question_tokens else 3
    best: tuple[tuple[int, int, int, int], int] | None = None
    maximum_size = min(18, len(question_tokens))
    for size in range(maximum_size, minimum_size - 1, -1):
        for question_start in range(0, len(question_tokens) - size + 1):
            phrase_tokens = question_tokens[question_start : question_start + size]
            informative = [
                token
                for token in phrase_tokens
                if token not in STOPWORDS and (len(token) >= 2 or token.isdigit())
            ]
            is_numbered_article = (
                "điều" in phrase_tokens
                and any(token.isdigit() for token in phrase_tokens)
            )
            if len(informative) < minimum_size and not is_numbered_article:
                continue
            for text_start in range(0, len(text_tokens) - size + 1):
                if text_tokens[text_start : text_start + size] != phrase_tokens:
                    continue
                heading_token_start = _heading_token_start(text_tokens, text_start)
                raw_start = token_matches[text_start].start()
                raw_end = token_matches[text_start + size - 1].end()
                raw_phrase = text[raw_start:raw_end]
                cased_letters = [char for char in raw_phrase if char.isalpha()]
                uppercase_title = bool(cased_letters) and raw_phrase.upper() == raw_phrase
                has_heading_marker = heading_token_start is not None
                if not uppercase_title and not has_heading_marker:
                    continue
                output_token_start = (
                    heading_token_start
                    if heading_token_start is not None
                    else text_start
                )
                # Prefer the longest exact phrase, then a real uppercase title.
                # For equal table-of-contents/title matches, prefer the later
                # occurrence, which is normally the complete form body.
                score = (
                    size,
                    int(uppercase_title),
                    int(has_heading_marker),
                    text_start,
                )
                output_start = token_matches[output_token_start].start()
                if best is None or score > best[0]:
                    best = (score, output_start)
        if best is not None and best[0][0] == size:
            break
    return best[1] if best is not None else None


def _overlap_size(left_words: list[str], right_words: list[str]) -> int:
    maximum = min(len(left_words), len(right_words))
    left = [word.casefold() for word in left_words]
    right = [word.casefold() for word in right_words]
    for size in range(maximum, 2, -1):
        if left[-size:] == right[:size]:
            return size
    return 0


def _build_best_first_answer(
    chunks: list[dict[str, Any]],
    best_chunk_no: int,
) -> str:
    """Fallback order: best chunk, previous chunk, then next chunk."""
    by_number = {
        int(chunk.get("chunk_no", chunk.get("chunk_index", 0))): chunk
        for chunk in chunks
        if str(chunk.get("text") or "").strip()
    }
    best = by_number.get(best_chunk_no)
    if best is None:
        return build_extractive_answer(chunks)

    best_text = str(best.get("text") or "").strip()
    best_words = best_text.split()
    parts = [best_text]

    previous = by_number.get(best_chunk_no - 1)
    if previous is not None:
        previous_words = str(previous.get("text") or "").strip().split()
        overlap = _overlap_size(previous_words, best_words)
        previous_only = previous_words[:-overlap] if overlap else previous_words
        if previous_only:
            parts.append(" ".join(previous_only))

    following = by_number.get(best_chunk_no + 1)
    if following is not None:
        following_words = str(following.get("text") or "").strip().split()
        overlap = _overlap_size(best_words, following_words)
        following_only = following_words[overlap:]
        if following_only:
            parts.append(" ".join(following_only))
    return "\n\n".join(parts).strip()


def build_focused_extractive_answer(
    question: str,
    chunks: list[dict[str, Any]],
    *,
    best_chunk_no: int,
) -> str:
    """Start raw evidence at a matching legal heading, without a char limit."""
    chronological = build_extractive_answer(chunks)
    heading_start = _matching_legal_heading_start(question, chronological)
    if heading_start is not None:
        return chronological[heading_start:].lstrip()
    return _build_best_first_answer(chunks, best_chunk_no)


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
