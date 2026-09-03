from __future__ import annotations

import difflib
import inspect
import math
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from typing import Any, Literal

from .generator import GenerationTokenLimitReached
from .storage import SearchIndex
from .text import (
    LONG_ANSWER_PATTERNS,
    STOPWORDS,
    best_excerpt,
    build_focused_extractive_answer,
    clean_answer,
    expand_retrieval_query,
    is_refusal_answer,
    is_heading_only_answer,
    is_long_form_question,
    is_structured_extractive_question,
    legal_retrieval_signal_matches,
    needs_extended_generation_retry,
    output_artifact_flags,
    possibly_cut,
    query_terms,
    retrieval_priority_phrases,
    retrieval_query_aliases,
    select_relevant_neighbor_chunks,
    tokenize,
)


Mode = Literal["extractive", "knn", "hybrid", "rag", "hybrid_rag"]


@dataclass(frozen=True)
class Prediction:
    answer: str
    route: str
    confidence: float
    evidence: dict[str, Any]


_RETRIEVAL_SCORE_FIELDS = (
    "bm25_score",
    "dense_score",
    "rrf_score",
    "legal_signal_boost",
    "boosted_rrf_score",
    "rerank_score",
    "rerank_guardrail_bonus",
    "rerank_guardrail_bonus_before_protection",
    "final_rerank_score",
)

_LEGAL_SIGNAL_BOOST_WEIGHTS = {
    "document_references": 0.0008,
    "money_amounts_vnd": 0.0008,
    "years": 0.00035,
    "plan_names": 0.0008,
    "form_names": 0.0008,
    "long_phrase": 0.00055,
    "focus_phrases": 0.0012,
    "scope_phrases": 0.0012,
}
_LEGAL_SIGNAL_COMBINATION_BONUS = 0.00035
_MAX_LEGAL_SIGNAL_BOOST = 0.006

def _audit_float(value: Any) -> float | None:
    """Convert model/NumPy scores to finite JSON numbers."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _retrieval_candidate_record(
    candidate: dict[str, Any],
    *,
    rank: int,
    score_field: str,
) -> dict[str, Any]:
    """Keep retrieval metadata useful for diagnosis without copying chunk text."""
    context_id = str(candidate.get("context_id") or "").strip() or None
    document_id = str(candidate.get("document_id") or context_id or "").strip() or None
    raw_chunk_no = candidate.get("chunk_no")
    try:
        chunk_no: int | str | None = int(raw_chunk_no)
    except (TypeError, ValueError, OverflowError):
        chunk_no = str(raw_chunk_no).strip() if raw_chunk_no is not None else None
    title = str(candidate.get("title") or candidate.get("name") or "").strip() or None
    link = str(candidate.get("link") or "").strip() or None
    record: dict[str, Any] = {
        "rank": int(rank),
        "document_id": document_id,
        "context_id": context_id,
        "chunk_no": chunk_no,
        "title": title,
        "link": link,
        "score": _audit_float(candidate.get(score_field)),
        "exact_phrase_matches": int(candidate.get("exact_phrase_matches") or 0),
    }
    for field in _RETRIEVAL_SCORE_FIELDS:
        record[field] = _audit_float(candidate.get(field))
    for field in (
        "rrf_rank_before_boost",
        "rrf_rank_after_boost",
        "rerank_rank_before_guardrail",
        "rerank_rank_after_guardrail",
    ):
        value = candidate.get(field)
        record[field] = (
            int(value)
            if isinstance(value, int) and not isinstance(value, bool)
            else None
        )
    matches = candidate.get("legal_signal_matches")
    record["legal_signal_matches"] = matches if isinstance(matches, dict) else {}
    components = candidate.get("rerank_guardrail_components")
    record["rerank_guardrail_components"] = (
        components if isinstance(components, dict) else {}
    )
    protection = candidate.get("rerank_guardrail_protected_by")
    record["rerank_guardrail_protected_by"] = (
        protection if isinstance(protection, dict) else None
    )
    return record


def _retrieval_stage_record(
    candidates: list[dict[str, Any]],
    *,
    requested_top_k: int,
    score_field: str,
    status: str = "ok",
) -> dict[str, Any]:
    return {
        "status": status,
        "requested_top_k": int(requested_top_k),
        "returned": len(candidates),
        "score_field": score_field,
        "candidates": [
            _retrieval_candidate_record(candidate, rank=rank, score_field=score_field)
            for rank, candidate in enumerate(candidates, start=1)
        ],
    }


def prediction_audit_record(sample_id: str, prediction: Prediction) -> dict[str, Any]:
    """Tạo audit record gọn, ổn định cho từng câu inference."""
    evidence = prediction.evidence
    top_contexts = evidence.get("top_contexts")
    top_context = (
        top_contexts[0]
        if isinstance(top_contexts, list) and top_contexts and isinstance(top_contexts[0], dict)
        else {}
    )
    top_document_id = evidence.get("context_id") or top_context.get("context_id")
    reranker_score = evidence.get("rerank_score")
    if reranker_score is None:
        reranker_score = top_context.get("rerank_score")
    says_no_information = evidence.get("says_no_information")
    if says_no_information is None:
        says_no_information = is_refusal_answer(prediction.answer)
    artifact_flags = output_artifact_flags(prediction.answer)
    return {
        "id": str(sample_id),
        "route": prediction.route,
        "answer_words": len(tokenize(prediction.answer)),
        "context_words": int(evidence.get("context_words") or 0),
        "generated_tokens": evidence.get("generated_tokens"),
        "hit_token_limit": bool(evidence.get("hit_token_limit", False)),
        "initial_hit_token_limit": bool(
            evidence.get("initial_hit_token_limit", False)
        ),
        "generation_attempts": int(evidence.get("generation_attempts") or 0),
        "generation_attempt_seconds": evidence.get("generation_attempt_seconds", []),
        "initial_generation_seconds": evidence.get("initial_generation_seconds"),
        "retry_generation_seconds": evidence.get("retry_generation_seconds"),
        "retry_max_new_tokens": evidence.get("retry_max_new_tokens"),
        "recovery_strategy": evidence.get("recovery_strategy"),
        "routing_decision": evidence.get("routing_decision"),
        "raw_fallback_allowed": bool(evidence.get("raw_fallback_allowed", False)),
        "says_no_information": bool(says_no_information),
        "possibly_cut": bool(evidence.get("possibly_cut", False)),
        "output_artifacts": sorted(artifact_flags),
        "has_markdown": "markdown" in artifact_flags,
        "has_document_slug": "document_slug" in artifact_flags,
        "has_fake_document_number": (
            "fake_document_number_or_page_id" in artifact_flags
        ),
        "top_document_id": top_document_id,
        "reranker_score": reranker_score,
        "raw_reranker_score": evidence.get("raw_reranker_score", reranker_score),
        "fallback_reason": evidence.get("fallback_reason"),
        "reranker_candidates": evidence.get("reranker_candidates"),
        "reranker_max_length": evidence.get("reranker_max_length"),
        "retrieval_trace": evidence.get("retrieval_trace", {}),
        "stage_seconds": evidence.get("stage_seconds", {}),
    }


def _trusted_answer_metadata(evidence: dict[str, Any]) -> list[str]:
    """Collect only retrieval-owned title/link fields used to whitelist citations."""
    values: list[str] = []

    def append_metadata(item: Any) -> None:
        if not isinstance(item, dict):
            return
        for key in ("name", "title", "link"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                values.append(value.strip())

    append_metadata(evidence)
    for key in ("top_contexts", "prompt_contexts"):
        items = evidence.get(key)
        if isinstance(items, list):
            for item in items:
                append_metadata(item)
    return list(dict.fromkeys(values))


def _normalize_similarity_token(token: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFD", token.casefold())
        if unicodedata.category(char) != "Mn"
    )


_SIMILARITY_STOPWORDS = {
    _normalize_similarity_token(term) for term in STOPWORDS
} - {
    _normalize_similarity_token(term)
    for term in ("có", "không", "phải", "được", "bị", "điều", "khoản", "điểm")
}
_LEGAL_SINGLE_TOKENS = {"a", "b", "c", "d", "đ"}
_LEGAL_DISAMBIGUATORS = {
    _normalize_similarity_token(term)
    for term in ("có", "không", "phải", "được", "bị", "điều", "khoản", "điểm")
}
_LEGAL_DISAMBIGUATORS.update(_LEGAL_SINGLE_TOKENS)


def _similarity_terms(text: str) -> list[str]:
    """Tokenize KNN questions without dropping legal polarity/citations."""
    seen: set[str] = set()
    result: list[str] = []
    for raw_token in tokenize(text):
        token = _normalize_similarity_token(raw_token)
        if (
            token in _SIMILARITY_STOPWORDS
            or (len(token) < 2 and not token.isdigit() and token not in _LEGAL_SINGLE_TOKENS)
            or token in seen
        ):
            continue
        seen.add(token)
        result.append(token)
        if len(result) >= 60:
            break
    if result:
        return result
    return list(
        dict.fromkeys(
            _normalize_similarity_token(raw_token)
            for raw_token in tokenize(text)
        )
    )[:60]


def question_similarity(left: str, right: str) -> float:
    left_tokens = _similarity_terms(left)
    right_tokens = _similarity_terms(right)
    left_set, right_set = set(left_tokens), set(right_tokens)
    if not left_set or not right_set:
        return 0.0
    intersection = len(left_set & right_set)
    coverage = intersection / len(left_set)
    jaccard = intersection / len(left_set | right_set)
    sequence = difflib.SequenceMatcher(
        None, " ".join(left_tokens), " ".join(right_tokens), autojunk=False
    ).ratio()
    score = 0.50 * coverage + 0.30 * jaccard + 0.20 * sequence
    left_disambiguators = {
        token
        for token in left_tokens
        if token in _LEGAL_DISAMBIGUATORS or any(char.isdigit() for char in token)
    }
    right_disambiguators = {
        token
        for token in right_tokens
        if token in _LEGAL_DISAMBIGUATORS or any(char.isdigit() for char in token)
    }
    if left_disambiguators != right_disambiguators:
        # A different negation, legal role, article, clause, or numeric class
        # must not pass the high-confidence exact-answer route.
        score *= 0.5
    return score


_KNN_INTENT_PATTERNS: dict[str, tuple[str, ...]] = {
    "penalty": (r"\bmức phạt\b", r"\bxử phạt\b", r"\bchế tài\b"),
    "authority": (r"\bthẩm quyền\b", r"\bcơ quan nào\b", r"\bai\b"),
    "condition": (r"\bđiều kiện\b", r"\btiêu chuẩn\b"),
    "dossier": (r"\bhồ sơ\b", r"\bgiấy tờ\b", r"\btài liệu cần\b"),
    "principle": (r"\bnguyên tắc\b",),
    "procedure": (r"\bthủ tục\b", r"\btrình tự\b", r"\bcác bước\b"),
    "responsibility": (r"\btrách nhiệm\b", r"\bnghĩa vụ\b"),
    "time": (r"\bthời hạn\b", r"\bthời hiệu\b", r"\bbao lâu\b", r"\bkhi nào\b"),
    "yes_no": (r"\b(?:có|được|phải|bị)\b[^?]{0,100}\bkhông\s*[?]?$",),
}

_KNN_GUARD_DIMENSIONS: dict[str, tuple[str, ...]] = {
    "education_level": (
        "sơ cấp",
        "trung cấp",
        "cao đẳng",
        "đại học",
        "thạc sĩ",
        "tiến sĩ",
        "mầm non",
        "tiểu học",
        "trung học cơ sở",
        "thcs",
        "trung học phổ thông",
        "thpt",
    ),
    "subject": (
        "người sử dụng lao động",
        "người lao động",
        "hộ kinh doanh",
        "doanh nghiệp",
        "tổ chức",
        "cá nhân",
        "học sinh",
        "sinh viên",
        "giáo viên",
        "giảng viên",
        "cán bộ",
        "công chức",
        "viên chức",
        "người chưa thành niên",
    ),
    "entity_role": (
        "cơ quan chủ trì soạn thảo",
        "cơ quan thẩm định",
        "cơ quan quản lý nhà nước",
        "ủy ban nhân dân cấp xã",
        "ủy ban nhân dân cấp huyện",
        "ủy ban nhân dân cấp tỉnh",
        "tòa án nhân dân",
        "viện kiểm sát nhân dân",
    ),
    "legal_scope": (
        "bình đẳng giới",
        "bảo hiểm xã hội",
        "bảo hiểm thất nghiệp",
        "hôn nhân và gia đình",
        "tố tụng dân sự",
        "tố tụng hình sự",
        "tố tụng hành chính",
        "dân sự",
        "hình sự",
        "hành chính",
        "đất đai",
        "lao động",
        "giáo dục",
        "thuế",
        "giao thông",
    ),
}


def _question_intents(question: str) -> set[str]:
    normalized = unicodedata.normalize("NFC", str(question or "").casefold())
    return {
        intent
        for intent, patterns in _KNN_INTENT_PATTERNS.items()
        if any(re.search(pattern, normalized) for pattern in patterns)
    }


def _normalized_question_key(question: str) -> str:
    """Normalize case, accents, punctuation and whitespace without reordering words."""
    return " ".join(
        _normalize_similarity_token(token)
        for token in tokenize(question)
    )


def _knn_guard_signature(question: str) -> dict[str, list[str]]:
    normalized = " ".join(tokenize(question))
    padded = f" {normalized} "
    signature: dict[str, list[str]] = {}
    for dimension, phrases in _KNN_GUARD_DIMENSIONS.items():
        matched = [phrase for phrase in phrases if f" {phrase} " in padded]
        signature[dimension] = matched
    return signature


def _knn_guards_match(left: str, right: str) -> tuple[bool, dict[str, Any]]:
    left_intents = _question_intents(left)
    right_intents = _question_intents(right)
    left_signature = _knn_guard_signature(left)
    right_signature = _knn_guard_signature(right)
    mismatches = [
        dimension
        for dimension in _KNN_GUARD_DIMENSIONS
        if set(left_signature[dimension]) != set(right_signature[dimension])
        and (left_signature[dimension] or right_signature[dimension])
    ]
    intents_match = bool(left_intents) and left_intents == right_intents
    return intents_match and not mismatches, {
        "query_intents": sorted(left_intents),
        "candidate_intents": sorted(right_intents),
        "query_signature": left_signature,
        "candidate_signature": right_signature,
        "mismatched_dimensions": mismatches,
    }


def _context_rerank_score(question: str, result: dict[str, Any]) -> float:
    q_list = query_terms(question, max_terms=60)
    q_terms = set(q_list)
    raw_bm25 = result.get("bm25_score")
    bm25 = max(0.0, -float(raw_bm25)) if raw_bm25 is not None else 0.0
    if not q_terms:
        return bm25
    text_tokens = tokenize(f"{result.get('name', '')} {result.get('text', '')}")
    text_set = set(text_tokens)
    coverage = len(q_terms & text_set) / len(q_terms)
    full_q = tokenize(question)
    q_bigrams = set(zip(full_q, full_q[1:]))
    q_trigrams = set(zip(full_q, full_q[1:], full_q[2:]))
    text_bigrams = set(zip(text_tokens, text_tokens[1:]))
    text_trigrams = set(zip(text_tokens, text_tokens[1:], text_tokens[2:]))
    phrase_coverage = (
        len(q_bigrams & text_bigrams) / len(q_bigrams) if q_bigrams else 0.0
    )
    trigram_coverage = (
        len(q_trigrams & text_trigrams) / len(q_trigrams) if q_trigrams else 0.0
    )
    # FTS5 đã tính IDF/toàn bộ tần suất. Chỉ thêm bonus nhỏ cho coverage/cụm từ;
    # nếu đè BM25 bằng số lần lặp, phần đầu một văn bản dài rất dễ thắng sai Điều.
    return bm25 + 2.0 * coverage + 2.0 * phrase_coverage + 6.0 * trigram_coverage


def _rag_confidence(question: str, result: dict[str, Any]) -> float:
    """Điểm chẩn đoán theo scorer mạnh nhất có sẵn, không giả định luôn có BM25."""
    for key in ("rerank_score", "dense_score"):
        value = result.get(key)
        if value is not None:
            try:
                score = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(score):
                continue
            if score >= 0:
                z = math.exp(-score)
                return 1.0 / (1.0 + z)
            z = math.exp(score)
            return z / (1.0 + z)
    if result.get("bm25_score") is not None:
        score = _context_rerank_score(question, result)
        if not math.isfinite(score):
            return 0.0
        return max(0.0, min(1.0, score / 20.0))
    return 0.0


def _has_strong_legal_evidence(candidate: dict[str, Any]) -> bool:
    """Return whether exact legal signals support this reranked candidate."""
    components = candidate.get("rerank_guardrail_components")
    components = components if isinstance(components, dict) else {}
    strong_component = any(
        (_audit_float(components.get(name)) or 0.0) > 0.0
        for name in (
            "exact_form",
            "exact_article",
            "exact_document_reference",
            "exact_document_name",
            "exact_long_phrase",
        )
    )
    exact_focus = _audit_float(components.get("exact_focus")) or 0.0
    try:
        exact_phrase_matches = int(candidate.get("exact_phrase_matches") or 0)
    except (TypeError, ValueError, OverflowError):
        exact_phrase_matches = 0
    matches = candidate.get("legal_signal_matches")
    try:
        long_phrase_tokens = (
            int(matches.get("long_phrase_tokens") or 0)
            if isinstance(matches, dict)
            else 0
        )
    except (TypeError, ValueError, OverflowError):
        long_phrase_tokens = 0
    return bool(
        exact_phrase_matches > 0
        or strong_component
        or exact_focus >= 4.5
        or long_phrase_tokens >= 4
    )


def _safe_generation_failure_extractive(
    question: str,
    candidate: dict[str, Any],
    answer: str,
) -> bool:
    """Allow failure recovery only from strong, directly usable evidence."""
    raw_score = _audit_float(candidate.get("rerank_score"))
    if raw_score is None or raw_score < 2.0:
        return False
    if (
        len(tokenize(answer)) < 20
        or is_heading_only_answer(answer)
        or possibly_cut(answer)
    ):
        return False

    normalized_question = unicodedata.normalize("NFC", str(question).casefold())
    if re.search(
        r"\b(?:phân tích|so sánh|đánh giá|giải thích|tổng hợp|suy luận|"
        r"tại sao|vì sao)\b",
        normalized_question,
    ):
        return False
    if re.search(
        r"\b(?:có|được|phải|bị)\b[^?]{0,140}\b(?:hay\s+)?không\s*[?]?$",
        normalized_question,
    ):
        return False
    return _has_strong_legal_evidence(candidate)


def reciprocal_rank_fusion(
    bm25_results: list[dict[str, Any]],
    dense_results: list[dict[str, Any]],
    rrf_k: int = 60,
    top_k: int = 50,
) -> list[dict[str, Any]]:
    """Hợp nhất kết quả từ BM25 và Dense Search bằng Reciprocal Rank Fusion (RRF)."""
    if rrf_k <= 0:
        raise ValueError("rrf_k phải lớn hơn 0")
    if top_k <= 0:
        raise ValueError("top_k phải lớn hơn 0")

    def ranked_unique(
        results: list[dict[str, Any]],
    ) -> list[tuple[tuple[str, int], dict[str, Any]]]:
        unique: list[tuple[tuple[str, int], dict[str, Any]]] = []
        seen: set[tuple[str, int]] = set()
        for doc in results:
            if not isinstance(doc, dict):
                raise TypeError("RRF candidate phải là dict")
            raw_context_id = doc.get("context_id")
            context_id = "" if raw_context_id is None else str(raw_context_id).strip()
            if not context_id:
                raise ValueError("RRF candidate thiếu context_id")
            try:
                chunk_no = int(doc["chunk_no"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("RRF candidate có chunk_no không hợp lệ") from exc
            key = (context_id, chunk_no)
            if key not in seen:
                seen.add(key)
                unique.append((key, doc))
        return unique

    scores: dict[tuple[str, int], float] = {}
    docs: dict[tuple[str, int], dict[str, Any]] = {}

    for rank, (key, doc) in enumerate(ranked_unique(bm25_results), start=1):
        scores[key] = scores.get(key, 0.0) + (1.0 / (rrf_k + rank))
        docs[key] = dict(doc)

    for rank, (key, doc) in enumerate(ranked_unique(dense_results), start=1):
        scores[key] = scores.get(key, 0.0) + (1.0 / (rrf_k + rank))
        if key not in docs:
            docs[key] = dict(doc)
        else:
            # Giữ nội dung/BM25 từ nhánh lexical nhưng bổ sung score Dense để audit.
            for field in ("dense_score", "dense_rank"):
                if field in doc:
                    docs[key][field] = doc[field]

    sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    fused_results: list[dict[str, Any]] = []
    for key in sorted_keys[:top_k]:
        item = dict(docs[key])
        item["rrf_score"] = float(scores[key])
        fused_results.append(item)
    return fused_results


def _apply_legal_signal_boost(
    question: str,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply a capped exact-match boost without overwriting the raw RRF score."""
    boosted: list[dict[str, Any]] = []
    for rank, candidate in enumerate(candidates, start=1):
        item = dict(candidate)
        title = str(item.get("title") or item.get("name") or "").strip()
        passage = "\n".join(
            part for part in (title, str(item.get("text") or "").strip()) if part
        )
        matches = legal_retrieval_signal_matches(question, passage)
        # Domain scope must come from the document title. A specialized law
        # may mention "tố tụng dân sự" in its body without being the governing
        # Civil Procedure Code for a general civil-procedure question.
        title_scope = legal_retrieval_signal_matches(question, title)
        matches["scope_phrases"] = title_scope["scope_phrases"]
        matches["scope_requested"] = title_scope["scope_requested"]
        matched_categories = [
            field
            for field in _LEGAL_SIGNAL_BOOST_WEIGHTS
            if matches.get(field)
        ]
        boost = sum(_LEGAL_SIGNAL_BOOST_WEIGHTS[field] for field in matched_categories)
        if len(matched_categories) >= 2:
            boost += _LEGAL_SIGNAL_COMBINATION_BONUS
        heading_overlap = int(matches.get("heading_overlap_tokens") or 0)
        heading_coverage = float(matches.get("heading_query_coverage") or 0.0)
        if heading_overlap >= 3 and heading_coverage >= 0.45:
            # A heading that directly names the issue is much safer than a
            # body paragraph that merely repeats generic legal terms. This is
            # enough to keep the right Điều inside the 20-item reranker pool.
            boost += min(0.004, 0.0006 * heading_overlap)
        boost = min(boost, _MAX_LEGAL_SIGNAL_BOOST)

        raw_rrf_score = _audit_float(item.get("rrf_score")) or 0.0
        item["rrf_rank_before_boost"] = rank
        item["legal_signal_matches"] = matches
        item["legal_signal_boost"] = round(boost, 8)
        item["boosted_rrf_score"] = raw_rrf_score + boost
        boosted.append(item)

    boosted.sort(
        key=lambda item: (
            -float(item["boosted_rrf_score"]),
            -float(item.get("rrf_score") or 0.0),
            int(item["rrf_rank_before_boost"]),
        )
    )
    for rank, item in enumerate(boosted, start=1):
        item["rrf_rank_after_boost"] = rank
    return boosted


def _apply_reranker_legal_guardrails(
    question: str,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Blend cross-encoder scores with high-precision legal intent signals.

    Cross-encoders can prefer a semantically nearby Điều with the wrong legal
    operation (for example, ``thời hạn kháng nghị`` instead of ``thời hiệu
    khiếu nại``). Exact form/article/document/long-phrase matches are strong;
    generic heading overlap is only a small tie-breaker. Raw reranker leaders
    backed by exact retrieval evidence are protected from weak heuristic flips
    when raw scores are close or both negative.
    """
    if not candidates:
        return []

    pool_size = max(1, len(candidates))
    adjusted: list[dict[str, Any]] = []
    for rank, candidate in enumerate(candidates, start=1):
        item = dict(candidate)
        title = str(item.get("title") or item.get("name") or "").strip()
        passage = "\n".join(
            part for part in (title, str(item.get("text") or "").strip()) if part
        )
        matches = item.get("legal_signal_matches")
        if not isinstance(matches, dict):
            matches = legal_retrieval_signal_matches(question, passage)
        title_scope = legal_retrieval_signal_matches(question, title)
        matches = dict(matches)
        matches["scope_phrases"] = title_scope["scope_phrases"]
        matches["scope_requested"] = title_scope["scope_requested"]
        # A document name is authoritative only when it matches the metadata
        # title, not when a body paragraph merely mentions another instrument.
        matches["document_names"] = title_scope["document_names"]
        item["legal_signal_matches"] = matches

        focus_phrases = matches.get("focus_phrases")
        longest_focus_tokens = max(
            (len(tokenize(phrase)) for phrase in focus_phrases),
            default=0,
        ) if isinstance(focus_phrases, list) else 0
        focus_bonus = min(6.0, 1.5 * longest_focus_tokens)

        scope_phrases = matches.get("scope_phrases")
        scope_bonus = 5.0 if isinstance(scope_phrases, list) and scope_phrases else 0.0

        heading_overlap = int(matches.get("heading_overlap_tokens") or 0)
        heading_coverage = float(matches.get("heading_query_coverage") or 0.0)
        heading_bonus = (
            min(0.6, 0.12 * heading_overlap)
            if heading_overlap >= 3 and heading_coverage >= 0.45
            else 0.0
        )

        form_names = matches.get("form_names")
        longest_form_tokens = max(
            (len(tokenize(name)) for name in form_names),
            default=0,
        ) if isinstance(form_names, list) else 0
        form_bonus = (
            min(5.0, 1.25 + 0.45 * longest_form_tokens)
            if longest_form_tokens >= 3
            else 0.0
        )

        article_references = matches.get("article_references")
        article_bonus = (
            min(4.0, 3.0 + 0.25 * (len(article_references) - 1))
            if isinstance(article_references, list) and article_references
            else 0.0
        )

        document_references = matches.get("document_references")
        document_reference_bonus = (
            min(4.5, 3.5 + 0.25 * (len(document_references) - 1))
            if isinstance(document_references, list) and document_references
            else 0.0
        )

        document_names = matches.get("document_names")
        longest_document_name_tokens = max(
            (len(tokenize(name)) for name in document_names),
            default=0,
        ) if isinstance(document_names, list) else 0
        document_name_bonus = (
            min(4.0, 0.55 * longest_document_name_tokens)
            if longest_document_name_tokens >= 3
            else 0.0
        )

        long_phrase_tokens = int(matches.get("long_phrase_tokens") or 0)
        long_phrase_bonus = (
            min(4.0, 0.75 + 0.6 * (long_phrase_tokens - 4))
            if long_phrase_tokens >= 4
            else 0.0
        )

        rrf_rank = item.get("rrf_rank_after_boost")
        try:
            rrf_rank_value = max(1, int(rrf_rank))
        except (TypeError, ValueError):
            rrf_rank_value = pool_size
        rrf_prior_bonus = max(
            0.0,
            0.75 * (1.0 - ((rrf_rank_value - 1) / max(1, pool_size - 1))),
        )
        title_tokens = set(tokenize(title))
        authority_tokens_match = (
            {"bộ", "luật", "tố", "tụng"}.issubset(title_tokens)
            or {"bo", "luat", "to", "tung"}.issubset(title_tokens)
        )
        authority_bonus = 0.75 if scope_bonus and authority_tokens_match else 0.0

        raw_score = _audit_float(item.get("rerank_score")) or 0.0
        guardrail_bonus = (
            focus_bonus
            + scope_bonus
            + heading_bonus
            + form_bonus
            + article_bonus
            + document_reference_bonus
            + document_name_bonus
            + long_phrase_bonus
            + rrf_prior_bonus
            + authority_bonus
        )
        exact_strength = (
            focus_bonus
            + scope_bonus
            + form_bonus
            + article_bonus
            + document_reference_bonus
            + document_name_bonus
            + long_phrase_bonus
            + authority_bonus
        )
        try:
            exact_phrase_matches = int(item.get("exact_phrase_matches") or 0)
        except (TypeError, ValueError, OverflowError):
            exact_phrase_matches = 0
        retrieval_exact = bool(
            exact_phrase_matches
            or form_bonus
            or article_bonus
            or document_reference_bonus
            or document_name_bonus
            or long_phrase_tokens >= 6
        )
        item["rerank_rank_before_guardrail"] = rank
        item["rerank_guardrail_bonus"] = round(guardrail_bonus, 6)
        item["final_rerank_score"] = raw_score + guardrail_bonus
        item["rerank_guardrail_components"] = {
            "exact_focus": round(focus_bonus, 6),
            "scope": round(scope_bonus, 6),
            "heading": round(heading_bonus, 6),
            "exact_form": round(form_bonus, 6),
            "exact_article": round(article_bonus, 6),
            "exact_document_reference": round(document_reference_bonus, 6),
            "exact_document_name": round(document_name_bonus, 6),
            "exact_long_phrase": round(long_phrase_bonus, 6),
            "rrf_prior": round(rrf_prior_bonus, 6),
            "authority": round(authority_bonus, 6),
        }
        item["rerank_guardrail_protected_by"] = None
        item["_guardrail_exact_strength"] = exact_strength
        item["_guardrail_retrieval_exact"] = retrieval_exact
        item["_guardrail_raw_score"] = raw_score
        adjusted.append(item)

    # A raw leader with exact lexical/RRF evidence should not lose to a weaker
    # heuristic when the cross-encoder itself is indecisive. Large positive
    # raw-score corrections (such as the exact-focus rescue for ID 34235) are
    # intentionally still allowed.
    for anchor in adjusted:
        if not anchor["_guardrail_retrieval_exact"]:
            continue
        anchor_raw = float(anchor["_guardrail_raw_score"])
        anchor_strength = float(anchor["_guardrail_exact_strength"])
        anchor_rank = int(anchor["rerank_rank_before_guardrail"])
        for challenger in adjusted:
            challenger_rank = int(challenger["rerank_rank_before_guardrail"])
            if challenger_rank <= anchor_rank:
                continue
            challenger_raw = float(challenger["_guardrail_raw_score"])
            raw_gap = anchor_raw - challenger_raw
            raw_is_uncertain = raw_gap <= 1.25 or (
                anchor_raw < 0.0 and challenger_raw < 0.0
            )
            if not raw_is_uncertain:
                continue
            challenger_strength = float(challenger["_guardrail_exact_strength"])
            if challenger_strength > anchor_strength:
                continue
            anchor_final = float(anchor["final_rerank_score"])
            challenger_final = float(challenger["final_rerank_score"])
            if challenger_final < anchor_final:
                continue
            protected_final = anchor_final - 0.000001
            original_bonus = float(challenger["rerank_guardrail_bonus"])
            capped_bonus = max(0.0, protected_final - challenger_raw)
            challenger["rerank_guardrail_bonus_before_protection"] = round(
                original_bonus, 6
            )
            challenger["rerank_guardrail_bonus"] = round(capped_bonus, 6)
            challenger["final_rerank_score"] = challenger_raw + capped_bonus
            challenger["rerank_guardrail_protected_by"] = {
                "context_id": anchor.get("context_id"),
                "chunk_no": anchor.get("chunk_no"),
                "reason": "exact_raw_leader_close_or_both_negative",
                "raw_score_gap": round(raw_gap, 6),
            }

    for item in adjusted:
        item.pop("_guardrail_exact_strength", None)
        item.pop("_guardrail_retrieval_exact", None)
        item.pop("_guardrail_raw_score", None)

    adjusted.sort(
        key=lambda item: (
            -float(item.get("final_rerank_score") or 0.0),
            -float(item.get("rerank_score") or 0.0),
            int(item.get("rerank_rank_before_guardrail") or pool_size),
        )
    )
    for rank, item in enumerate(adjusted, start=1):
        item["rerank_rank_after_guardrail"] = rank
    return adjusted


class LegalQABaseline:
    def __init__(
        self,
        index: SearchIndex,
        top_k: int = 12,
        max_answer_words: int = 520,
        knn_threshold: float = 0.72,
        generator: Any | None = None,
        context_top_k: int = 3,
        dense_index: Any | None = None,
        embedding_model: Any | None = None,
        reranker: Any | None = None,
        bm25_top_k: int = 50,
        dense_top_k: int = 50,
        rrf_k: int = 60,
        rrf_top_k: int = 50,
        reranker_candidate_k: int = 20,
        rerank_top_k: int = 3,
        dense_query_max_length: int = 256,
        reranker_max_length: int = 1024,
        allow_retrieval_fallback: bool = False,
        enable_long_answer_extractive: bool = True,
        max_long_answer_words: int = 640,
        min_llm_answer_tokens: int = 8,
        token_limit_retry_tokens: int = 768,
        guarded_knn_threshold: float = 0.90,
    ):
        positive = {
            "top_k": top_k,
            "max_answer_words": max_answer_words,
            "context_top_k": context_top_k,
            "bm25_top_k": bm25_top_k,
            "dense_top_k": dense_top_k,
            "rrf_k": rrf_k,
            "rrf_top_k": rrf_top_k,
            "reranker_candidate_k": reranker_candidate_k,
            "rerank_top_k": rerank_top_k,
            "dense_query_max_length": dense_query_max_length,
            "reranker_max_length": reranker_max_length,
            "max_long_answer_words": max_long_answer_words,
            "min_llm_answer_tokens": min_llm_answer_tokens,
            "token_limit_retry_tokens": token_limit_retry_tokens,
        }
        invalid = {name: value for name, value in positive.items() if value <= 0}
        if invalid:
            raise ValueError(f"Các tham số pipeline phải lớn hơn 0: {invalid}")
        if not 0.0 <= knn_threshold <= 1.0:
            raise ValueError("knn_threshold phải nằm trong [0, 1]")
        if not 0.90 <= guarded_knn_threshold <= 1.0:
            raise ValueError("guarded_knn_threshold phải nằm trong [0.90, 1]")
        if token_limit_retry_tokens not in (768, 1024):
            raise ValueError("token_limit_retry_tokens phải là 768 hoặc 1024")
        if not (
            context_top_k
            <= rerank_top_k
            <= reranker_candidate_k
            <= rrf_top_k
        ):
            raise ValueError(
                "Cần context_top_k <= rerank_top_k <= "
                "reranker_candidate_k <= rrf_top_k"
            )
        if (dense_index is None) != (embedding_model is None):
            raise ValueError("dense_index và embedding_model phải được truyền cùng nhau")
        self.index = index
        self.top_k = top_k
        self.max_answer_words = max_answer_words
        self.enable_long_answer_extractive = bool(enable_long_answer_extractive)
        self.max_long_answer_words = max_long_answer_words
        self.knn_threshold = knn_threshold
        self.generator = generator
        self.context_top_k = context_top_k
        self.dense_index = dense_index
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.bm25_top_k = bm25_top_k
        self.dense_top_k = dense_top_k
        self.rrf_k = rrf_k
        self.rrf_top_k = rrf_top_k
        self.reranker_candidate_k = reranker_candidate_k
        self.rerank_top_k = rerank_top_k or context_top_k
        self.dense_query_max_length = dense_query_max_length
        self.reranker_max_length = reranker_max_length
        self.allow_retrieval_fallback = allow_retrieval_fallback
        self.min_llm_answer_tokens = min_llm_answer_tokens
        self.token_limit_retry_tokens = token_limit_retry_tokens
        self.guarded_knn_threshold = guarded_knn_threshold

    def _adjacent_chunks(
        self,
        best: dict[str, Any],
        *,
        question: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return only relevant/continuous local chunks around the best hit."""
        context_id = str(best.get("context_id") or "").strip()
        chunk_no = int(best.get("chunk_no", 0))
        adjacent_nos = [
            number
            for number in (chunk_no - 1, chunk_no, chunk_no + 1)
            if number >= 0
        ]
        get_context_chunks = getattr(self.index, "get_context_chunks", None)
        fetched = (
            get_context_chunks(context_id, chunk_nos=adjacent_nos)
            if context_id and callable(get_context_chunks)
            else []
        )

        by_key: dict[tuple[str, int], dict[str, Any]] = {}
        for chunk in [*fetched, best]:
            chunk_context_id = str(chunk.get("context_id") or "").strip()
            try:
                candidate_no = int(chunk.get("chunk_no", 0))
            except (TypeError, ValueError):
                continue
            if (
                chunk_context_id == context_id
                and candidate_no in adjacent_nos
                and str(chunk.get("text") or "").strip()
            ):
                by_key[(chunk_context_id, candidate_no)] = dict(chunk)
        candidates = sorted(by_key.values(), key=lambda chunk: int(chunk["chunk_no"]))
        if not question:
            return candidates
        return select_relevant_neighbor_chunks(
            question,
            candidates,
            best_chunk_no=chunk_no,
        )

    def _merge_raw_chunks(
        self,
        chunks: list[dict[str, Any]],
        *,
        question: str | None = None,
        best: dict[str, Any] | None = None,
    ) -> str:
        """Join adjacent chunks and focus raw output on the requested heading."""
        if question and best is not None:
            return build_focused_extractive_answer(
                question,
                chunks,
                best_chunk_no=int(best.get("chunk_no", 0)),
                max_words=self.max_long_answer_words,
            )
        return build_focused_extractive_answer(
            "",
            chunks,
            best_chunk_no=int(chunks[0].get("chunk_no", 0)) if chunks else 0,
            max_words=self.max_long_answer_words,
        )

    def _call_generator(
        self,
        *,
        context: str,
        question: str,
        max_new_tokens: int | None = None,
    ) -> Any:
        """Call production and lightweight generators with an optional retry budget."""
        if self.generator is None:
            from .generator import ViQwenRAGGenerator
            self.generator = ViQwenRAGGenerator()
        generate = self.generator.generate
        kwargs: dict[str, Any] = {"context": context, "question": question}
        if max_new_tokens is not None:
            try:
                parameters = inspect.signature(generate).parameters
            except (TypeError, ValueError):
                parameters = {}
            accepts_kwargs = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
            if accepts_kwargs or "max_new_tokens" in parameters:
                kwargs["max_new_tokens"] = max_new_tokens
        return generate(**kwargs)

    @staticmethod
    def _finalize_prediction(result: Any) -> Prediction:
        """Enforce the submission answer schema and clean every output route once."""
        fallback_answer = "Không tìm thấy căn cứ đủ tin cậy để trả lời câu hỏi."
        if not isinstance(result, Prediction):
            return Prediction(
                fallback_answer,
                "fallback",
                0.0,
                {
                    "answer_cleaning": {
                        "schema_valid": False,
                        "error": "pipeline_result_is_not_prediction",
                    }
                },
            )

        evidence = result.evidence if isinstance(result.evidence, dict) else {}
        trusted_metadata = _trusted_answer_metadata(evidence)
        try:
            cleaned = clean_answer(
                result.answer,
                trusted_metadata=trusted_metadata,
            )
        except (TypeError, ValueError) as exc:
            return Prediction(
                fallback_answer,
                "fallback",
                0.0,
                {
                    **evidence,
                    "answer_cleaning": {
                        "schema_valid": False,
                        "error": str(exc),
                    },
                },
            )

        route = (
            result.route
            if isinstance(result.route, str) and result.route
            else "fallback"
        )
        confidence = _audit_float(result.confidence)
        return Prediction(
            cleaned,
            route,
            confidence if confidence is not None else 0.0,
            {
                **evidence,
                "answer_cleaning": {
                    "schema_valid": True,
                    "changed": cleaned != result.answer.strip(),
                    "trusted_metadata_fields": len(trusted_metadata),
                },
            },
        )

    def _guarded_knn(
        self,
        question: str,
        *,
        exclude_id: str | None,
    ) -> Prediction | None:
        """Use train answers only for near-duplicates with matching intent/entity guards."""
        neighbors = self.index.search_train(question, top_k=5, exclude_id=exclude_id)
        candidates: list[dict[str, Any]] = []
        query_key = _normalized_question_key(question)
        for neighbor in neighbors:
            if (
                not isinstance(neighbor, dict)
                or not isinstance(neighbor.get("question"), str)
                or not str(neighbor.get("answer") or "").strip()
            ):
                continue
            item = dict(neighbor)
            candidate_question = str(item["question"])
            item["similarity"] = question_similarity(question, candidate_question)
            item["exact_normalized"] = (
                bool(query_key)
                and query_key == _normalized_question_key(candidate_question)
            )
            candidates.append(item)

        candidates.sort(
            key=lambda item: (
                bool(item["exact_normalized"]),
                float(item["similarity"]),
            ),
            reverse=True,
        )
        for candidate in candidates:
            similarity = float(candidate["similarity"])
            exact_normalized = bool(candidate["exact_normalized"])
            candidate_question = str(candidate["question"])
            guards_match, guard_evidence = _knn_guards_match(
                question,
                candidate_question,
            )
            if not exact_normalized and (
                similarity < self.guarded_knn_threshold or not guards_match
            ):
                continue
            match_type = "exact_normalized" if exact_normalized else "near_duplicate"
            return Prediction(
                str(candidate["answer"]),
                "knn_exact" if exact_normalized else "knn_guarded",
                similarity,
                {
                    "sample_id": candidate.get("sample_id"),
                    "question": candidate_question,
                    "bm25_score": candidate.get("bm25_score"),
                    "guarded_knn_threshold": self.guarded_knn_threshold,
                    "knn_match_type": match_type,
                    "guard_checks": guard_evidence,
                },
            )
        return None

    def _requery_candidates(
        self,
        question: str,
        *,
        excluded_keys: set[tuple[str, int]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Run one controlled lexical re-query and retain only confident reranker hits."""
        requery = " ".join(query_terms(question, max_terms=24)).strip()
        trace: dict[str, Any] = {
            "query": requery,
            "status": "skipped",
            "returned": 0,
        }
        if not requery:
            return [], trace
        try:
            candidates = self.index.search_contexts(requery, top_k=self.bm25_top_k)
            candidates = [
                dict(candidate)
                for candidate in candidates
                if (
                    str(candidate.get("context_id") or "").strip(),
                    int(candidate.get("chunk_no", 0)),
                ) not in excluded_keys
            ][: self.reranker_candidate_k]
            if self.reranker is not None and candidates:
                candidates = self.reranker.rerank(
                    question,
                    candidates,
                    top_k=len(candidates),
                    max_length=self.reranker_max_length,
                )
            candidates = _apply_reranker_legal_guardrails(question, candidates)
            confident = [
                candidate
                for candidate in candidates
                if (_audit_float(candidate.get("rerank_score")) or float("-inf")) >= 2.0
            ]
            trace.update(
                {
                    "status": "ok" if confident else "no_confident_candidate",
                    "returned": len(confident),
                    "candidates": [
                        _retrieval_candidate_record(
                            candidate,
                            rank=rank,
                            score_field="final_rerank_score",
                        )
                        for rank, candidate in enumerate(confident[:3], start=1)
                    ],
                }
            )
            return confident, trace
        except Exception as exc:
            trace.update({"status": "error", "error": str(exc)})
            return [], trace

    def _build_generation_context(
        self,
        question: str,
        best: dict[str, Any],
        ranked_chunks: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, str]:
        adjacent_chunks = self._adjacent_chunks(best, question=question)
        prompt_chunks = self._prioritize_prompt_chunks(
            [best, *adjacent_chunks],
            ranked_chunks,
        )
        context_blocks: list[str] = []
        for index, chunk in enumerate(prompt_chunks, start=1):
            name = str(chunk.get("name") or "").strip()
            text = str(chunk.get("text") or "").strip()
            context_blocks.append(
                f"[{index}] Văn bản: {name}\n{text}" if name else f"[{index}] {text}"
            )
        joined_context = "\n\n".join(context_blocks)
        raw_answer = self._merge_raw_chunks(
            adjacent_chunks,
            question=question,
            best=best,
        )
        return adjacent_chunks, prompt_chunks, joined_context, raw_answer

    def _invalid_generation_reason(self, answer: Any) -> str | None:
        """Classify unusable LLM output that must fall back to raw evidence."""
        if not isinstance(answer, str) or not answer.strip():
            return "empty"

        if is_refusal_answer(answer):
            return "refusal"
        if len(tokenize(answer)) < self.min_llm_answer_tokens:
            return "too_short"
        if possibly_cut(answer):
            return "possibly_cut"
        return None

    @staticmethod
    def _prioritize_prompt_chunks(
        adjacent_chunks: list[dict[str, Any]],
        reranked_chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Put contiguous evidence first, followed by other reranker results."""
        ordered: list[dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()
        for chunk in [*adjacent_chunks, *reranked_chunks]:
            key = (
                str(chunk.get("context_id") or "").strip(),
                int(chunk.get("chunk_no", 0)),
            )
            if key in seen or not str(chunk.get("text") or "").strip():
                continue
            seen.add(key)
            ordered.append(chunk)
        return ordered

    def _extractive(self, question: str) -> Prediction | None:
        contexts = self.index.search_contexts(question, top_k=self.top_k)
        if not contexts:
            return None
        ranked = sorted(
            contexts,
            key=lambda item: _context_rerank_score(question, item),
            reverse=True,
        )
        best = ranked[0]
        if self.enable_long_answer_extractive and is_long_form_question(question):
            adjacent_chunks = self._adjacent_chunks(best, question=question)
            if adjacent_chunks:
                answer = self._merge_raw_chunks(
                    adjacent_chunks,
                    question=question,
                    best=best,
                )
                evidence = {
                    "context_id": best["context_id"],
                    "chunk_no": best["chunk_no"],
                    "name": best["name"],
                    "link": best["link"],
                    "bm25_score": best["bm25_score"],
                    "merged_chunk_nos": [c["chunk_no"] for c in adjacent_chunks],
                    "context_words": len(tokenize(answer)),
                    "generated_tokens": 0,
                    "hit_token_limit": False,
                    "says_no_information": False,
                    "possibly_cut": False,
                }
                score = _context_rerank_score(question, best)
                confidence = (
                    max(0.0, min(1.0, score / 20.0))
                    if math.isfinite(score)
                    else 0.0
                )
                return Prediction(answer, "extractive_long", confidence, evidence)

        answer = best_excerpt(
            best["text"], question=question, max_words=self.max_answer_words
        )
        evidence = {
            "context_id": best["context_id"],
            "chunk_no": best["chunk_no"],
            "name": best["name"],
            "link": best["link"],
            "bm25_score": best["bm25_score"],
        }
        score = _context_rerank_score(question, best)
        confidence = (
            max(0.0, min(1.0, score / 20.0))
            if math.isfinite(score)
            else 0.0
        )
        return Prediction(answer, "extractive", confidence, evidence)

    def retrieve_only(self, question: str) -> dict[str, Any]:
        """Run the exact RAG retrieval path without initializing or calling the LLM."""
        result = self._rag(question, retrieval_only=True)
        if result is None:
            return {
                "question": question,
                "status": "empty",
                "retrieval_trace": {},
                "stage_seconds": {},
                "top_contexts": [],
            }
        return {
            "question": question,
            "status": "ok",
            **result.evidence,
        }

    def _rag(
        self,
        question: str,
        *,
        retrieval_only: bool = False,
        exclude_id: str | None = None,
    ) -> Prediction | None:
        rag_started = time.perf_counter()
        stage_seconds: dict[str, float] = {}
        expanded_retrieval_query = expand_retrieval_query(question)
        query_aliases = retrieval_query_aliases(question)
        priority_phrases = retrieval_priority_phrases(question)

        # 1. Truy xuất BM25 Top-50
        stage_started = time.perf_counter()
        bm25_candidates = self.index.search_contexts(question, top_k=self.bm25_top_k)
        stage_seconds["bm25"] = round(time.perf_counter() - stage_started, 4)

        # 2. Truy xuất Dense FAISS Top-50 (nếu có index và embedding model)
        stage_started = time.perf_counter()
        dense_candidates: list[dict[str, Any]] = []
        dense_status = "disabled"
        if self.dense_index is not None and self.embedding_model is not None:
            try:
                normalize = (
                    getattr(self.dense_index, "normalization", "l2") == "l2"
                    or getattr(self.dense_index, "similarity", "cosine") == "cosine"
                )
                encode = self.embedding_model.encode
                try:
                    parameters = inspect.signature(encode).parameters
                except (TypeError, ValueError):
                    parameters = {}
                accepts_kwargs = any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in parameters.values()
                )
                encode_kwargs: dict[str, Any] = {}
                if accepts_kwargs or "max_length" in parameters:
                    encode_kwargs["max_length"] = self.dense_query_max_length
                if accepts_kwargs or "normalize_embeddings" in parameters:
                    encode_kwargs["normalize_embeddings"] = normalize
                encoded_query = encode([expanded_retrieval_query], **encode_kwargs)
                q_vec = encoded_query[0]
                dense_candidates = self.dense_index.search(q_vec, top_k=self.dense_top_k)
                dense_status = "ok" if dense_candidates else "empty"
            except Exception as exc:
                if not self.allow_retrieval_fallback:
                    raise RuntimeError(f"Dense search thất bại: {exc}") from exc
                print(f"[pipeline] Lỗi dense search ({exc}), fallback BM25.", file=sys.stderr)
                dense_status = "fallback_bm25"
        stage_seconds["dense"] = round(time.perf_counter() - stage_started, 4)

        # 3. Hợp nhất RRF (Reciprocal Rank Fusion k=60 -> Top-50) hoặc fallback Heuristic BM25
        stage_started = time.perf_counter()
        fusion_status = "rrf"
        if dense_candidates:
            try:
                fused_candidates = reciprocal_rank_fusion(
                    bm25_candidates, dense_candidates, rrf_k=self.rrf_k, top_k=self.rrf_top_k
                )
            except Exception as exc:
                if not self.allow_retrieval_fallback:
                    raise RuntimeError(f"RRF thất bại: {exc}") from exc
                print(f"[pipeline] Lỗi RRF ({exc}), fallback BM25.", file=sys.stderr)
                fusion_status = "fallback_bm25"
                fused_candidates = sorted(
                    bm25_candidates,
                    key=lambda item: _context_rerank_score(question, item),
                    reverse=True,
                )[: self.rrf_top_k]
        else:
            # Fallback thuần BM25
            fusion_status = "bm25_only"
            fused_candidates = sorted(
                bm25_candidates,
                key=lambda item: _context_rerank_score(question, item),
                reverse=True,
            )[: self.rrf_top_k]
        if fusion_status == "rrf":
            fused_candidates = _apply_legal_signal_boost(question, fused_candidates)
        stage_seconds["fusion"] = round(time.perf_counter() - stage_started, 4)

        if not fused_candidates:
            return None

        # 4. Tái xếp hạng bằng Vietnamese_Reranker (Top-3 chunks)
        stage_started = time.perf_counter()
        rerank_candidates = fused_candidates[: self.reranker_candidate_k]
        reranked_pool: list[dict[str, Any]] = []
        reranker_status = "disabled"
        if self.reranker is not None:
            try:
                # The cross-encoder already scores the complete candidate pool.
                # Return all scores so audit can show where a relevant document
                # was lost, then take Top-K for the answer below.
                reranked_pool = self.reranker.rerank(
                    question,
                    rerank_candidates,
                    top_k=len(rerank_candidates),
                    max_length=self.reranker_max_length,
                )
                if not isinstance(reranked_pool, list) or not reranked_pool:
                    raise ValueError("Reranker trả về danh sách rỗng/không hợp lệ")
                if len(reranked_pool) != len(rerank_candidates):
                    raise ValueError(
                        "Reranker không trả đủ điểm cho candidate pool: "
                        f"expected={len(rerank_candidates)}, got={len(reranked_pool)}"
                    )
                if any(
                    not isinstance(chunk, dict)
                    or not str(chunk.get("context_id") or "").strip()
                    or "chunk_no" not in chunk
                    for chunk in reranked_pool
                ):
                    raise ValueError("Reranker trả về candidate không hợp lệ")
                reranker_status = "ok"
            except Exception as exc:
                if not self.allow_retrieval_fallback:
                    raise RuntimeError(f"Reranker thất bại: {exc}") from exc
                print(f"[pipeline] Lỗi reranker ({exc}), fallback RRF order.", file=sys.stderr)
                reranked_pool = [dict(candidate) for candidate in rerank_candidates]
                reranker_status = "fallback_rrf"
        else:
            reranked_pool = [dict(candidate) for candidate in rerank_candidates]
        reranked_pool = _apply_reranker_legal_guardrails(question, reranked_pool)
        stage_seconds["reranker"] = round(time.perf_counter() - stage_started, 4)

        reranker_top_chunks = reranked_pool[: self.rerank_top_k]
        retrieval_trace = {
            "query": {
                "original": question,
                "expanded": expanded_retrieval_query,
                "aliases": query_aliases,
                "priority_phrases": priority_phrases,
            },
            "bm25": _retrieval_stage_record(
                bm25_candidates,
                requested_top_k=self.bm25_top_k,
                score_field="bm25_score",
                status="ok" if bm25_candidates else "empty",
            ),
            "dense": _retrieval_stage_record(
                dense_candidates,
                requested_top_k=self.dense_top_k,
                score_field="dense_score",
                status=dense_status,
            ),
            "rrf": _retrieval_stage_record(
                fused_candidates,
                requested_top_k=self.rrf_top_k,
                score_field=(
                    "boosted_rrf_score" if fusion_status == "rrf" else "rrf_score"
                ),
                status=fusion_status,
            ),
            "reranker_pool": _retrieval_stage_record(
                reranked_pool,
                requested_top_k=self.reranker_candidate_k,
                score_field="final_rerank_score",
                status=reranker_status,
            ),
            "reranker_top": _retrieval_stage_record(
                reranker_top_chunks,
                requested_top_k=self.rerank_top_k,
                score_field="final_rerank_score",
                status=reranker_status,
            ),
        }

        # Reranker có thể trả nhiều candidate để audit; prompt chỉ nhận context_top_k.
        top_chunks = reranker_top_chunks[: self.context_top_k]
        if not top_chunks:
            return None

        if retrieval_only:
            diagnostic_stages = {
                "top50": (
                    fused_candidates,
                    "boosted_rrf_score" if fusion_status == "rrf" else "bm25_score",
                ),
                "top20": (reranked_pool, "final_rerank_score"),
                "top3": (reranker_top_chunks, "final_rerank_score"),
            }
            diagnostic_candidates: dict[str, list[dict[str, Any]]] = {}
            for stage_name, (candidates, score_field) in diagnostic_stages.items():
                records: list[dict[str, Any]] = []
                for rank, candidate in enumerate(candidates, start=1):
                    record = _retrieval_candidate_record(
                        candidate,
                        rank=rank,
                        score_field=score_field,
                    )
                    text = " ".join(str(candidate.get("text") or "").split())
                    record["text_preview"] = text[:600]
                    records.append(record)
                diagnostic_candidates[stage_name] = records
            total_seconds = round(time.perf_counter() - rag_started, 4)
            return Prediction(
                answer="",
                route="retrieval_only",
                confidence=0.0,
                evidence={
                    "retrieval_trace": retrieval_trace,
                    "diagnostic_candidates": diagnostic_candidates,
                    "top_contexts": diagnostic_candidates["top3"],
                    "stage_seconds": {
                        **stage_seconds,
                        "generation": 0.0,
                        "total": total_seconds,
                    },
                },
            )

        best = top_chunks[0]
        raw_reranker_score = _audit_float(best.get("rerank_score"))
        low_confidence_retrieval = (
            raw_reranker_score is None or raw_reranker_score < 2.0
        )
        requery_candidates: list[dict[str, Any]] = []
        if low_confidence_retrieval:
            guarded_knn = self._guarded_knn(question, exclude_id=exclude_id)
            if guarded_knn is not None:
                return Prediction(
                    guarded_knn.answer,
                    "knn_guarded_low_confidence",
                    guarded_knn.confidence,
                    {
                        **guarded_knn.evidence,
                        "retrieval_trace": retrieval_trace,
                        "raw_reranker_score": raw_reranker_score,
                        "recovery_strategy": "guarded_knn",
                        "routing_decision": "low_confidence_guarded_knn",
                        "generated_tokens": 0,
                        "generation_attempts": 0,
                        "hit_token_limit": False,
                        "says_no_information": False,
                        "possibly_cut": False,
                        "stage_seconds": {
                            **stage_seconds,
                            "generation": 0.0,
                            "total": round(time.perf_counter() - rag_started, 4),
                        },
                    },
                )
            excluded_keys = {
                (str(chunk.get("context_id") or ""), int(chunk.get("chunk_no", 0)))
                for chunk in top_chunks
            }
            requery_candidates, requery_trace = self._requery_candidates(
                question,
                excluded_keys=excluded_keys,
            )
            retrieval_trace["recovery"] = {
                "trigger": "raw_reranker_score_below_2",
                **requery_trace,
            }
            if requery_candidates:
                recovered_best = requery_candidates[0]
                unique_chunks = [recovered_best, *top_chunks]
                seen_keys: set[tuple[str, int]] = set()
                top_chunks = []
                for chunk in unique_chunks:
                    key = (
                        str(chunk.get("context_id") or ""),
                        int(chunk.get("chunk_no", 0)),
                    )
                    if key not in seen_keys:
                        seen_keys.add(key)
                        top_chunks.append(chunk)
                    if len(top_chunks) >= self.context_top_k:
                        break
                best = top_chunks[0]
                raw_reranker_score = _audit_float(best.get("rerank_score"))
                low_confidence_retrieval = False

        adjacent_chunks = self._adjacent_chunks(best, question=question)

        if (
            self.enable_long_answer_extractive
            and is_structured_extractive_question(question)
        ):
            if adjacent_chunks:
                merged_answer = self._merge_raw_chunks(
                    adjacent_chunks,
                    question=question,
                    best=best,
                )
                direct_extractive_allowed = (
                    raw_reranker_score is not None and raw_reranker_score >= 2.0
                )
                extractive_is_usable = bool(
                    merged_answer
                    and not is_heading_only_answer(merged_answer)
                    and not possibly_cut(merged_answer)
                )
                if direct_extractive_allowed and extractive_is_usable:
                    evidence = {
                        "num_contexts": len(top_chunks),
                        "reranker_candidates": min(
                            len(fused_candidates), self.reranker_candidate_k
                        ),
                        "reranker_max_length": self.reranker_max_length,
                        "retrieval_trace": retrieval_trace,
                        "top_contexts": [
                            {
                                "context_id": c["context_id"],
                                "chunk_no": c["chunk_no"],
                                "name": c.get("name"),
                                "bm25_score": c.get("bm25_score"),
                                "dense_score": c.get("dense_score"),
                                "rrf_score": c.get("rrf_score"),
                                "legal_signal_boost": c.get("legal_signal_boost"),
                                "boosted_rrf_score": c.get("boosted_rrf_score"),
                                "rerank_score": c.get("rerank_score"),
                                "rerank_guardrail_bonus": c.get("rerank_guardrail_bonus"),
                                "final_rerank_score": c.get("final_rerank_score"),
                            }
                            for c in top_chunks
                        ],
                        "merged_chunk_nos": [c["chunk_no"] for c in adjacent_chunks],
                        "context_words": len(tokenize(merged_answer)),
                        "raw_reranker_score": raw_reranker_score,
                        "routing_decision": "focused_extractive_good_retrieval",
                        "raw_fallback_allowed": False,
                        "generation_attempts": 0,
                        "generated_tokens": 0,
                        "hit_token_limit": False,
                        "says_no_information": False,
                        "possibly_cut": False,
                        "stage_seconds": {
                            **stage_seconds,
                            "generation": 0.0,
                            "total": round(time.perf_counter() - rag_started, 4),
                        },
                    }
                    confidence = _rag_confidence(question, best)
                    return Prediction(merged_answer, "extractive_long", confidence, evidence)

        adjacent_chunks, prompt_chunks, joined_context, raw_context_answer = (
            self._build_generation_context(question, best, top_chunks)
        )
        generation_trusted_metadata = [
            chunk.get(key)
            for chunk in prompt_chunks
            for key in ("name", "title", "link")
            if isinstance(chunk.get(key), str) and str(chunk.get(key)).strip()
        ]
        refusal_anchor = {
            "best": best,
            "top_chunks": list(top_chunks),
            "adjacent_chunks": list(adjacent_chunks),
            "prompt_chunks": list(prompt_chunks),
            "joined_context": joined_context,
            "raw_context_answer": raw_context_answer,
            "trusted_metadata": list(generation_trusted_metadata),
        }
        generation_started = time.perf_counter()
        generation_attempts = 0
        generation_attempt_seconds: list[float] = []
        initial_hit_token_limit = False
        retry_budget: int | None = None
        recovery_strategy: str | None = None
        route: str | None = None

        def generate_once(
            context: str,
            *,
            max_new_tokens: int | None = None,
        ) -> tuple[str, str | None, dict[str, Any]]:
            nonlocal generation_attempts
            generation_attempts += 1
            attempt_started = time.perf_counter()
            try:
                raw_answer = self._call_generator(
                    context=context,
                    question=question,
                    max_new_tokens=max_new_tokens,
                )
            except GenerationTokenLimitReached as exc:
                stats = getattr(self.generator, "last_generation_stats", {})
                if not isinstance(stats, dict):
                    stats = {}
                generation_attempt_seconds.append(
                    round(time.perf_counter() - attempt_started, 4)
                )
                return "", "token_limit", {
                    **stats,
                    "generated_tokens": stats.get("generated_tokens", exc.generated_tokens),
                    "max_new_tokens": stats.get("max_new_tokens", exc.max_new_tokens),
                    "hit_token_limit": True,
                }
            generation_attempt_seconds.append(
                round(time.perf_counter() - attempt_started, 4)
            )
            generation_stats = getattr(self.generator, "last_generation_stats", {})
            if not isinstance(generation_stats, dict):
                generation_stats = {}
            if raw_answer is None:
                return "", "empty", generation_stats
            if not isinstance(raw_answer, str):
                return "", "invalid_schema", generation_stats
            try:
                cleaned = clean_answer(
                    raw_answer,
                    trusted_metadata=generation_trusted_metadata,
                )
            except ValueError:
                return "", "empty", generation_stats
            reason = (
                "token_limit"
                if generation_stats.get("hit_token_limit")
                else self._invalid_generation_reason(cleaned)
            )
            return cleaned, reason, generation_stats

        answer, invalid_reason, generation_stats = generate_once(joined_context)
        initial_budget = int(
            generation_stats.get("max_new_tokens")
            or getattr(self.generator, "max_new_tokens", 512)
            or 512
        )

        if invalid_reason == "token_limit":
            initial_hit_token_limit = True
            if needs_extended_generation_retry(question):
                if initial_budget < self.token_limit_retry_tokens:
                    retry_budget = self.token_limit_retry_tokens
                elif initial_budget < 1024:
                    retry_budget = 1024
                if retry_budget is not None:
                    answer, invalid_reason, generation_stats = generate_once(
                        joined_context,
                        max_new_tokens=retry_budget,
                    )
                    if invalid_reason is None:
                        route = f"generated_retry_{retry_budget}"
                        recovery_strategy = "token_limit_retry"

        refusal_recovery_requested = (
            invalid_reason == "refusal" and generation_attempts == 1
        )
        if refusal_recovery_requested:
            guarded_knn = self._guarded_knn(question, exclude_id=exclude_id)
            if guarded_knn is not None:
                answer = guarded_knn.answer
                invalid_reason = None
                route = "knn_guarded_refusal"
                recovery_strategy = "guarded_knn"

        if refusal_recovery_requested and invalid_reason == "refusal":
            anchor_best = refusal_anchor["best"]
            anchor_answer = str(refusal_anchor["raw_context_answer"] or "").strip()
            focused_context_usable = bool(
                _has_strong_legal_evidence(anchor_best)
                and len(tokenize(anchor_answer)) >= 20
                and not is_heading_only_answer(anchor_answer)
                and not possibly_cut(anchor_answer)
            )
            if focused_context_usable:
                best = anchor_best
                top_chunks = list(refusal_anchor["top_chunks"])
                adjacent_chunks = list(refusal_anchor["adjacent_chunks"])
                prompt_chunks = list(refusal_anchor["prompt_chunks"])
                raw_context_answer = anchor_answer
                generation_trusted_metadata = list(
                    refusal_anchor["trusted_metadata"]
                )
                anchor_name = str(anchor_best.get("name") or "").strip()
                joined_context = (
                    f"[1] Văn bản: {anchor_name}\n{anchor_answer}"
                    if anchor_name
                    else f"[1] {anchor_answer}"
                )
                answer, invalid_reason, generation_stats = generate_once(joined_context)
                recovery_strategy = "focused_context"
                if invalid_reason is None:
                    route = "generated_refusal_recovery"

        # Focused context is only the first refusal retry. If it also refuses,
        # continue with a different candidate/re-query; this is essential for
        # yes/no questions where direct extractive fallback is intentionally blocked.
        if refusal_recovery_requested and invalid_reason == "refusal":
            current_key = (
                str(best.get("context_id") or ""),
                int(best.get("chunk_no", 0)),
            )
            alternative_pool = [
                candidate
                for candidate in [*requery_candidates, *reranked_pool]
                if (
                    str(candidate.get("context_id") or ""),
                    int(candidate.get("chunk_no", 0)),
                ) != current_key
                and (_audit_float(candidate.get("rerank_score")) or float("-inf"))
                >= 2.0
            ]
            if not alternative_pool and not requery_candidates:
                recovered, requery_trace = self._requery_candidates(
                    question,
                    excluded_keys={current_key},
                )
                retrieval_trace["recovery"] = {
                    "trigger": "generator_refusal",
                    **requery_trace,
                }
                alternative_pool = recovered
            if alternative_pool:
                alternative_best = alternative_pool[0]
                alternative_chunks = [alternative_best, *top_chunks]
                (
                    adjacent_chunks,
                    prompt_chunks,
                    joined_context,
                    raw_context_answer,
                ) = self._build_generation_context(
                    question,
                    alternative_best,
                    alternative_chunks,
                )
                generation_trusted_metadata = [
                    chunk.get(key)
                    for chunk in prompt_chunks
                    for key in ("name", "title", "link")
                    if isinstance(chunk.get(key), str)
                    and str(chunk.get(key)).strip()
                ]
                answer, invalid_reason, generation_stats = generate_once(joined_context)
                best = alternative_best
                top_chunks = alternative_chunks[: self.context_top_k]
                raw_reranker_score = _audit_float(best.get("rerank_score"))
                recovery_strategy = "alternate_candidate"
                if invalid_reason is None:
                    route = "generated_refusal_recovery"

        focused_extractive_used = False
        failure_before_extractive = invalid_reason
        if (
            invalid_reason is not None
            and (refusal_recovery_requested or initial_hit_token_limit)
        ):
            anchor_best = refusal_anchor["best"]
            anchor_answer = str(refusal_anchor["raw_context_answer"] or "").strip()
            if _safe_generation_failure_extractive(
                question,
                anchor_best,
                anchor_answer,
            ):
                best = anchor_best
                top_chunks = list(refusal_anchor["top_chunks"])
                adjacent_chunks = list(refusal_anchor["adjacent_chunks"])
                prompt_chunks = list(refusal_anchor["prompt_chunks"])
                joined_context = str(refusal_anchor["joined_context"])
                raw_context_answer = anchor_answer
                generation_trusted_metadata = list(
                    refusal_anchor["trusted_metadata"]
                )
                answer = anchor_answer
                invalid_reason = None
                route = "extractive_fallback"
                recovery_strategy = (
                    "token_limit_focused_extractive"
                    if failure_before_extractive == "token_limit"
                    else "refusal_focused_extractive"
                )
                raw_reranker_score = _audit_float(best.get("rerank_score"))
                focused_extractive_used = True

        raw_fallback_allowed = bool(
            focused_extractive_used
            or (
                raw_reranker_score is not None
                and raw_reranker_score >= 2.0
                and is_structured_extractive_question(question)
                and raw_context_answer
                and not is_heading_only_answer(raw_context_answer)
                and not possibly_cut(raw_context_answer)
            )
        )
        generation_timing = {
            "generation_attempt_seconds": generation_attempt_seconds,
            "initial_generation_seconds": (
                generation_attempt_seconds[0]
                if generation_attempt_seconds
                else None
            ),
            "retry_generation_seconds": (
                round(sum(generation_attempt_seconds[1:]), 4)
                if len(generation_attempt_seconds) > 1
                else 0.0
            ),
        }
        if invalid_reason is None:
            if route is None:
                route = f"generated_{initial_budget}"
            generation_evidence = {
                "generated_tokens": generation_stats.get("generated_tokens"),
                "max_new_tokens": generation_stats.get("max_new_tokens", initial_budget),
                "hit_token_limit": False,
                "initial_hit_token_limit": initial_hit_token_limit,
                "generation_attempts": generation_attempts,
                **generation_timing,
                "retry_max_new_tokens": retry_budget,
                "recovery_strategy": recovery_strategy,
                "raw_reranker_score": raw_reranker_score,
                "raw_fallback_allowed": raw_fallback_allowed,
                "routing_decision": (
                    "guarded_knn_after_refusal"
                    if route == "knn_guarded_refusal"
                    else (
                        "guarded_focused_extractive_after_generation_failure"
                        if focused_extractive_used
                        else "generator_success"
                    )
                ),
                "says_no_information": False,
                "possibly_cut": False,
            }
        elif raw_fallback_allowed:
            answer = raw_context_answer
            route = "extractive_fallback"
            generation_evidence = {
                "fallback_reason": invalid_reason,
                "generated_tokens": generation_stats.get("generated_tokens"),
                "max_new_tokens": generation_stats.get("max_new_tokens", initial_budget),
                "hit_token_limit": invalid_reason == "token_limit",
                "initial_hit_token_limit": initial_hit_token_limit,
                "generation_attempts": generation_attempts,
                **generation_timing,
                "retry_max_new_tokens": retry_budget,
                "recovery_strategy": recovery_strategy,
                "raw_reranker_score": raw_reranker_score,
                "raw_fallback_allowed": True,
                "routing_decision": "guarded_focused_extractive_fallback",
                "says_no_information": False,
                "possibly_cut": False,
            }
        else:
            if invalid_reason != "refusal" or not answer:
                answer = "Không tìm thấy căn cứ đủ tin cậy để trả lời câu hỏi."
            route = "recovery_exhausted"
            generation_evidence = {
                "fallback_reason": invalid_reason,
                "generated_tokens": generation_stats.get("generated_tokens"),
                "max_new_tokens": generation_stats.get("max_new_tokens", initial_budget),
                "hit_token_limit": invalid_reason == "token_limit",
                "initial_hit_token_limit": initial_hit_token_limit,
                "generation_attempts": generation_attempts,
                **generation_timing,
                "retry_max_new_tokens": retry_budget,
                "recovery_strategy": recovery_strategy or "none_available",
                "raw_reranker_score": raw_reranker_score,
                "raw_fallback_allowed": False,
                "routing_decision": "recovery_exhausted_without_raw_dump",
                "says_no_information": True,
                "possibly_cut": invalid_reason == "possibly_cut",
            }
        stage_seconds["generation"] = round(
            time.perf_counter() - generation_started,
            4,
        )
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("Generator và raw context đều trả về answer rỗng")
        answer = answer.strip()
        evidence = {
            "num_contexts": len(top_chunks),
            "reranker_candidates": min(
                len(fused_candidates), self.reranker_candidate_k
            ),
            "reranker_max_length": self.reranker_max_length,
            "retrieval_trace": retrieval_trace,
            "top_contexts": [
                {
                    "context_id": c["context_id"],
                    "chunk_no": c["chunk_no"],
                    "name": c.get("name"),
                    "bm25_score": c.get("bm25_score"),
                    "dense_score": c.get("dense_score"),
                    "rrf_score": c.get("rrf_score"),
                    "legal_signal_boost": c.get("legal_signal_boost"),
                    "boosted_rrf_score": c.get("boosted_rrf_score"),
                    "rerank_score": c.get("rerank_score"),
                    "rerank_guardrail_bonus": c.get("rerank_guardrail_bonus"),
                    "final_rerank_score": c.get("final_rerank_score"),
                }
                for c in top_chunks
            ],
            "adjacent_chunk_nos": [c["chunk_no"] for c in adjacent_chunks],
            "prompt_contexts": [
                {
                    "context_id": c["context_id"],
                    "chunk_no": c["chunk_no"],
                    "name": c.get("name"),
                }
                for c in prompt_chunks
            ],
            "context_words": len(tokenize(joined_context)),
            "stage_seconds": {
                **stage_seconds,
                "total": round(time.perf_counter() - rag_started, 4),
            },
            **generation_evidence,
        }
        best = top_chunks[0]
        confidence = _rag_confidence(question, best)
        return Prediction(answer, route, confidence, evidence)

    def _knn(self, question: str, exclude_id: str | None = None) -> Prediction | None:
        neighbors = self.index.search_train(question, top_k=5, exclude_id=exclude_id)
        neighbors = [
            neighbor
            for neighbor in neighbors
            if isinstance(neighbor, dict)
            and isinstance(neighbor.get("question"), str)
            and str(neighbor.get("answer") or "").strip()
        ]
        if not neighbors:
            return None
        for neighbor in neighbors:
            neighbor["similarity"] = question_similarity(question, neighbor["question"])
        best = max(neighbors, key=lambda item: item["similarity"])
        return Prediction(
            str(best["answer"]),
            "knn",
            float(best["similarity"]),
            {
                "sample_id": best["sample_id"],
                "question": best["question"],
                "bm25_score": best["bm25_score"],
            },
        )

    def predict_one(
        self,
        question: str,
        mode: Mode = "hybrid_rag",
        exclude_id: str | None = None,
    ) -> Prediction:
        if mode not in ("extractive", "knn", "hybrid", "rag", "hybrid_rag"):
            raise ValueError(f"Mode không hỗ trợ: {mode}")
        if not isinstance(question, str):
            raise TypeError("question phải là chuỗi")
        if not question.strip():
            return self._finalize_prediction(
                Prediction(
                    "Không tìm thấy căn cứ phù hợp trong kho văn bản được cung cấp.",
                    "fallback",
                    0.0,
                    {},
                )
            )
        if mode == "extractive":
            result = self._extractive(question)
        elif mode == "knn":
            result = self._knn(question, exclude_id=exclude_id)
        elif mode == "hybrid":
            knn = self._knn(question, exclude_id=exclude_id)
            if knn is not None and knn.confidence >= self.knn_threshold:
                result = knn
            else:
                result = self._extractive(question)
        elif mode == "rag":
            result = self._rag(question, exclude_id=exclude_id)
        elif mode == "hybrid_rag":
            knn = self._guarded_knn(question, exclude_id=exclude_id)
            if knn is not None:
                result = knn
            else:
                result = self._rag(question, exclude_id=exclude_id)
        else:
            raise ValueError(f"Mode không hỗ trợ: {mode}")

        if result is None:
            result = Prediction(
                "Không tìm thấy căn cứ phù hợp trong kho văn bản được cung cấp.",
                "fallback",
                0.0,
                {},
            )
        return self._finalize_prediction(result)
