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
    build_extractive_answer,
    clean_answer,
    expand_retrieval_query,
    is_long_form_question,
    legal_retrieval_signal_matches,
    possibly_cut,
    query_terms,
    retrieval_priority_phrases,
    retrieval_query_aliases,
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
)

_LEGAL_SIGNAL_BOOST_WEIGHTS = {
    "document_references": 0.0008,
    "money_amounts_vnd": 0.0008,
    "years": 0.00035,
    "plan_names": 0.0008,
    "form_names": 0.0008,
    "long_phrase": 0.00055,
}
_LEGAL_SIGNAL_COMBINATION_BONUS = 0.00035
_MAX_LEGAL_SIGNAL_BOOST = 0.0025

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
    for field in ("rrf_rank_before_boost", "rrf_rank_after_boost"):
        value = candidate.get(field)
        record[field] = (
            int(value)
            if isinstance(value, int) and not isinstance(value, bool)
            else None
        )
    matches = candidate.get("legal_signal_matches")
    record["legal_signal_matches"] = matches if isinstance(matches, dict) else {}
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
    answer_lower = prediction.answer.casefold()
    says_no_information = evidence.get("says_no_information")
    if says_no_information is None:
        says_no_information = (
            "không đủ thông tin" in answer_lower
            or "không có thông tin" in answer_lower
        )
    return {
        "id": str(sample_id),
        "route": prediction.route,
        "answer_words": len(tokenize(prediction.answer)),
        "context_words": int(evidence.get("context_words") or 0),
        "generated_tokens": evidence.get("generated_tokens"),
        "hit_token_limit": bool(evidence.get("hit_token_limit", False)),
        "says_no_information": bool(says_no_information),
        "possibly_cut": bool(evidence.get("possibly_cut", False)),
        "top_document_id": top_document_id,
        "reranker_score": reranker_score,
        "reranker_candidates": evidence.get("reranker_candidates"),
        "reranker_max_length": evidence.get("reranker_max_length"),
        "retrieval_trace": evidence.get("retrieval_trace", {}),
        "stage_seconds": evidence.get("stage_seconds", {}),
    }


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
        matched_categories = [
            field
            for field in _LEGAL_SIGNAL_BOOST_WEIGHTS
            if matches.get(field)
        ]
        boost = sum(_LEGAL_SIGNAL_BOOST_WEIGHTS[field] for field in matched_categories)
        if len(matched_categories) >= 2:
            boost += _LEGAL_SIGNAL_COMBINATION_BONUS
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
        max_long_answer_words: int = 800,
        min_llm_answer_tokens: int = 8,
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
            "min_llm_answer_tokens": min_llm_answer_tokens,
        }
        invalid = {name: value for name, value in positive.items() if value <= 0}
        if invalid:
            raise ValueError(f"Các tham số pipeline phải lớn hơn 0: {invalid}")
        if not 0.0 <= knn_threshold <= 1.0:
            raise ValueError("knn_threshold phải nằm trong [0, 1]")
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

    def _adjacent_chunks(self, best: dict[str, Any]) -> list[dict[str, Any]]:
        """Return previous/current/next chunks from the best chunk's document."""
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
        return sorted(by_key.values(), key=lambda chunk: int(chunk["chunk_no"]))

    @staticmethod
    def _merge_raw_chunks(chunks: list[dict[str, Any]]) -> str:
        """Join complete adjacent chunks and remove their configured overlap."""
        return build_extractive_answer(chunks)

    def _invalid_generation_reason(self, answer: Any) -> str | None:
        """Classify unusable LLM output that must fall back to raw evidence."""
        if not isinstance(answer, str) or not answer.strip():
            return "empty"

        # Run this cleanup here as well as in the generation path so direct
        # callers cannot hide a refusal behind "Dựa trên (các) ngữ cảnh...".
        cleaned = clean_answer(answer)
        early_sentences = [
            " ".join(sentence.casefold().split()).strip(" .!?:;,-")
            for sentence in re.split(r"(?<=[.!?…])\s+|[\r\n]+", cleaned)
            if sentence.strip()
        ][:2]
        if any(
            sentence.startswith(_REFUSAL_START_MARKERS)
            or any(pattern.search(sentence) for pattern in _REFUSAL_EARLY_PATTERNS)
            for sentence in early_sentences
        ):
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
            adjacent_chunks = self._adjacent_chunks(best)
            if adjacent_chunks:
                answer = self._merge_raw_chunks(adjacent_chunks)
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

    def _rag(self, question: str) -> Prediction | None:
        if self.generator is None:
            from .generator import ViQwenRAGGenerator
            self.generator = ViQwenRAGGenerator()

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
                score_field="rerank_score",
                status=reranker_status,
            ),
            "reranker_top": _retrieval_stage_record(
                reranker_top_chunks,
                requested_top_k=self.rerank_top_k,
                score_field="rerank_score",
                status=reranker_status,
            ),
        }

        # Reranker có thể trả nhiều candidate để audit; prompt chỉ nhận context_top_k.
        top_chunks = reranker_top_chunks[: self.context_top_k]
        if not top_chunks:
            return None

        best = top_chunks[0]
        adjacent_chunks = self._adjacent_chunks(best)

        if self.enable_long_answer_extractive and is_long_form_question(question):
            if adjacent_chunks:
                merged_answer = self._merge_raw_chunks(adjacent_chunks)
                if merged_answer:
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
                            }
                            for c in top_chunks
                        ],
                        "merged_chunk_nos": [c["chunk_no"] for c in adjacent_chunks],
                        "context_words": len(tokenize(merged_answer)),
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

        # Preserve the legal document's local continuity first. Other reranker
        # results come later and are retained only while the generator's prompt
        # budget still has room (the generator truncates context from the tail).
        prompt_chunks = self._prioritize_prompt_chunks(adjacent_chunks, top_chunks)
        context_blocks = []
        for idx, chunk in enumerate(prompt_chunks, start=1):
            name = str(chunk.get("name") or "").strip()
            text = str(chunk.get("text") or "").strip()
            if name:
                context_blocks.append(f"[{idx}] Văn bản: {name}\n{text}")
            else:
                context_blocks.append(f"[{idx}] {text}")
        joined_context = "\n\n".join(context_blocks)

        raw_context_answer = self._merge_raw_chunks(adjacent_chunks)
        generator_kwargs: dict[str, Any] = {
            "context": joined_context,
            "question": question,
        }
        generation_started = time.perf_counter()
        try:
            raw_generated_answer = self.generator.generate(**generator_kwargs)
            answer = clean_answer(raw_generated_answer)
            generation_stats = getattr(self.generator, "last_generation_stats", {})
            if not isinstance(generation_stats, dict):
                generation_stats = {}
            invalid_reason = (
                "token_limit"
                if generation_stats.get("hit_token_limit")
                else self._invalid_generation_reason(answer)
            )
            if invalid_reason is not None:
                if not raw_context_answer:
                    raise RuntimeError(
                        "Generator output is unusable but the context fallback is empty"
                    )
                answer = raw_context_answer
                route = "extractive_fallback"
                generation_evidence = {
                    "fallback_reason": invalid_reason,
                    "generated_tokens": generation_stats.get("generated_tokens"),
                    "max_new_tokens": generation_stats.get("max_new_tokens", 512),
                    "hit_token_limit": bool(generation_stats.get("hit_token_limit", False)),
                    "says_no_information": invalid_reason == "refusal",
                    "possibly_cut": invalid_reason in ("possibly_cut", "token_limit"),
                }
            else:
                route = "generated_512"
                generation_evidence = {
                    "generated_tokens": generation_stats.get("generated_tokens"),
                    "max_new_tokens": generation_stats.get("max_new_tokens", 512),
                    "hit_token_limit": bool(generation_stats.get("hit_token_limit", False)),
                    "says_no_information": False,
                    "possibly_cut": False,
                }
        except GenerationTokenLimitReached as exc:
            if not raw_context_answer:
                raise RuntimeError(
                    "Generator reached its token limit but the context fallback is empty"
                ) from exc
            answer = raw_context_answer
            route = "extractive_fallback"
            generation_evidence = {
                "hit_token_limit": True,
                "generated_tokens": exc.generated_tokens,
                "max_new_tokens": exc.max_new_tokens,
                "fallback_reason": "token_limit",
                "says_no_information": False,
                "possibly_cut": True,
            }
        finally:
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
        mode: Mode = "hybrid",
        exclude_id: str | None = None,
    ) -> Prediction:
        if mode not in ("extractive", "knn", "hybrid", "rag", "hybrid_rag"):
            raise ValueError(f"Mode không hỗ trợ: {mode}")
        if not isinstance(question, str):
            raise TypeError("question phải là chuỗi")
        if not question.strip():
            return Prediction(
                "Không tìm thấy căn cứ phù hợp trong kho văn bản được cung cấp.",
                "fallback",
                0.0,
                {},
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
            result = self._rag(question)
        elif mode == "hybrid_rag":
            knn = self._knn(question, exclude_id=exclude_id)
            if knn is not None and knn.confidence >= self.knn_threshold:
                result = knn
            else:
                result = self._rag(question)
        else:
            raise ValueError(f"Mode không hỗ trợ: {mode}")

        if result is None:
            return Prediction(
                "Không tìm thấy căn cứ phù hợp trong kho văn bản được cung cấp.",
                "fallback",
                0.0,
                {},
            )
        return result
