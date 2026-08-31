from __future__ import annotations

import difflib
import inspect
import math
import sys
import unicodedata
from dataclasses import dataclass
from typing import Any, Literal

from .storage import SearchIndex
from .text import STOPWORDS, best_excerpt, query_terms, tokenize


Mode = Literal["extractive", "knn", "hybrid", "rag", "hybrid_rag"]


@dataclass(frozen=True)
class Prediction:
    answer: str
    route: str
    confidence: float
    evidence: dict[str, Any]


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
        rerank_top_k: int = 3,
        dense_query_max_length: int = 256,
        reranker_max_length: int = 2304,
        allow_retrieval_fallback: bool = False,
    ):
        positive = {
            "top_k": top_k,
            "max_answer_words": max_answer_words,
            "context_top_k": context_top_k,
            "bm25_top_k": bm25_top_k,
            "dense_top_k": dense_top_k,
            "rrf_k": rrf_k,
            "rrf_top_k": rrf_top_k,
            "rerank_top_k": rerank_top_k,
            "dense_query_max_length": dense_query_max_length,
            "reranker_max_length": reranker_max_length,
        }
        invalid = {name: value for name, value in positive.items() if value <= 0}
        if invalid:
            raise ValueError(f"Các tham số pipeline phải lớn hơn 0: {invalid}")
        if not 0.0 <= knn_threshold <= 1.0:
            raise ValueError("knn_threshold phải nằm trong [0, 1]")
        if context_top_k > rerank_top_k or rerank_top_k > rrf_top_k:
            raise ValueError("Cần context_top_k <= rerank_top_k <= rrf_top_k")
        if (dense_index is None) != (embedding_model is None):
            raise ValueError("dense_index và embedding_model phải được truyền cùng nhau")
        self.index = index
        self.top_k = top_k
        self.max_answer_words = max_answer_words
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
        self.rerank_top_k = rerank_top_k or context_top_k
        self.dense_query_max_length = dense_query_max_length
        self.reranker_max_length = reranker_max_length
        self.allow_retrieval_fallback = allow_retrieval_fallback

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

        # 1. Truy xuất BM25 Top-50
        bm25_candidates = self.index.search_contexts(question, top_k=self.bm25_top_k)

        # 2. Truy xuất Dense FAISS Top-50 (nếu có index và embedding model)
        dense_candidates: list[dict[str, Any]] = []
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
                encoded_query = encode([question], **encode_kwargs)
                q_vec = encoded_query[0]
                dense_candidates = self.dense_index.search(q_vec, top_k=self.dense_top_k)
            except Exception as exc:
                if not self.allow_retrieval_fallback:
                    raise RuntimeError(f"Dense search thất bại: {exc}") from exc
                print(f"[pipeline] Lỗi dense search ({exc}), fallback BM25.", file=sys.stderr)

        # 3. Hợp nhất RRF (Reciprocal Rank Fusion k=60 -> Top-50) hoặc fallback Heuristic BM25
        if dense_candidates:
            try:
                fused_candidates = reciprocal_rank_fusion(
                    bm25_candidates, dense_candidates, rrf_k=self.rrf_k, top_k=self.rrf_top_k
                )
            except Exception as exc:
                if not self.allow_retrieval_fallback:
                    raise RuntimeError(f"RRF thất bại: {exc}") from exc
                print(f"[pipeline] Lỗi RRF ({exc}), fallback BM25.", file=sys.stderr)
                fused_candidates = sorted(
                    bm25_candidates,
                    key=lambda item: _context_rerank_score(question, item),
                    reverse=True,
                )[: self.rrf_top_k]
        else:
            # Fallback thuần BM25
            fused_candidates = sorted(
                bm25_candidates,
                key=lambda item: _context_rerank_score(question, item),
                reverse=True,
            )[: self.rrf_top_k]

        if not fused_candidates:
            return None

        # 4. Tái xếp hạng bằng Vietnamese_Reranker (Top-3 chunks)
        if self.reranker is not None:
            try:
                reranked_chunks = self.reranker.rerank(
                    question,
                    fused_candidates,
                    top_k=self.rerank_top_k,
                    max_length=self.reranker_max_length,
                )
                if not isinstance(reranked_chunks, list) or not reranked_chunks:
                    raise ValueError("Reranker trả về danh sách rỗng/không hợp lệ")
                if any(
                    not isinstance(chunk, dict)
                    or not str(chunk.get("context_id") or "").strip()
                    or "chunk_no" not in chunk
                    for chunk in reranked_chunks
                ):
                    raise ValueError("Reranker trả về candidate không hợp lệ")
                top_chunks = reranked_chunks[: self.rerank_top_k]
            except Exception as exc:
                if not self.allow_retrieval_fallback:
                    raise RuntimeError(f"Reranker thất bại: {exc}") from exc
                print(f"[pipeline] Lỗi reranker ({exc}), fallback RRF order.", file=sys.stderr)
                top_chunks = fused_candidates[: self.rerank_top_k]
        else:
            top_chunks = fused_candidates[: self.rerank_top_k]

        # Reranker có thể trả nhiều candidate để audit; prompt chỉ nhận context_top_k.
        top_chunks = top_chunks[: self.context_top_k]
        if not top_chunks:
            return None

        context_blocks = []
        for idx, chunk in enumerate(top_chunks, start=1):
            name = str(chunk.get("name") or "").strip()
            text = str(chunk.get("text") or "").strip()
            if name:
                context_blocks.append(f"[{idx}] Văn bản: {name}\n{text}")
            else:
                context_blocks.append(f"[{idx}] {text}")
        joined_context = "\n\n".join(context_blocks)

        answer = self.generator.generate(context=joined_context, question=question)
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("Generator trả về answer rỗng/không hợp lệ")
        answer = answer.strip()
        evidence = {
            "num_contexts": len(top_chunks),
            "top_contexts": [
                {
                    "context_id": c["context_id"],
                    "chunk_no": c["chunk_no"],
                    "name": c.get("name"),
                    "bm25_score": c.get("bm25_score"),
                    "dense_score": c.get("dense_score"),
                    "rrf_score": c.get("rrf_score"),
                    "rerank_score": c.get("rerank_score"),
                }
                for c in top_chunks
            ],
        }
        best = top_chunks[0]
        confidence = _rag_confidence(question, best)
        return Prediction(answer, "rag", confidence, evidence)

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
