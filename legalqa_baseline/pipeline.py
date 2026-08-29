from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Any, Literal

from .storage import SearchIndex
from .text import best_excerpt, query_terms, tokenize


Mode = Literal["extractive", "knn", "hybrid", "rag", "hybrid_rag"]


@dataclass(frozen=True)
class Prediction:
    answer: str
    route: str
    confidence: float
    evidence: dict[str, Any]


def question_similarity(left: str, right: str) -> float:
    left_tokens = query_terms(left, max_terms=60)
    right_tokens = query_terms(right, max_terms=60)
    left_set, right_set = set(left_tokens), set(right_tokens)
    if not left_set or not right_set:
        return 0.0
    intersection = len(left_set & right_set)
    coverage = intersection / len(left_set)
    jaccard = intersection / len(left_set | right_set)
    sequence = difflib.SequenceMatcher(
        None, " ".join(left_tokens), " ".join(right_tokens), autojunk=False
    ).ratio()
    return 0.50 * coverage + 0.30 * jaccard + 0.20 * sequence


def _context_rerank_score(question: str, result: dict[str, Any]) -> float:
    q_list = query_terms(question, max_terms=60)
    q_terms = set(q_list)
    bm25 = max(0.0, -float(result["bm25_score"]))
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


class LegalQABaseline:
    def __init__(
        self,
        index: SearchIndex,
        top_k: int = 12,
        max_answer_words: int = 520,
        knn_threshold: float = 0.72,
        generator: Any | None = None,
        context_top_k: int = 3,
    ):
        self.index = index
        self.top_k = top_k
        self.max_answer_words = max_answer_words
        self.knn_threshold = knn_threshold
        self.generator = generator
        self.context_top_k = context_top_k

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
        confidence = min(1.0, _context_rerank_score(question, best) / 20.0)
        return Prediction(answer, "extractive", confidence, evidence)

    def _rag(self, question: str) -> Prediction | None:
        if self.generator is None:
            from .generator import ViQwenRAGGenerator
            self.generator = ViQwenRAGGenerator()

        contexts = self.index.search_contexts(question, top_k=self.top_k)
        if not contexts:
            return None

        ranked = sorted(
            contexts,
            key=lambda item: _context_rerank_score(question, item),
            reverse=True,
        )
        top_chunks = ranked[: self.context_top_k]

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
        evidence = {
            "num_contexts": len(top_chunks),
            "top_contexts": [
                {
                    "context_id": c["context_id"],
                    "chunk_no": c["chunk_no"],
                    "name": c.get("name"),
                    "bm25_score": c["bm25_score"],
                }
                for c in top_chunks
            ],
        }
        best = top_chunks[0]
        confidence = min(1.0, _context_rerank_score(question, best) / 20.0)
        return Prediction(answer, "rag", confidence, evidence)

    def _knn(self, question: str, exclude_id: str | None = None) -> Prediction | None:
        neighbors = self.index.search_train(question, top_k=5, exclude_id=exclude_id)
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
        if mode == "extractive":
            result = self._extractive(question)
        elif mode == "knn":
            result = self._knn(question, exclude_id=exclude_id)
        elif mode == "hybrid":
            knn = self._knn(question, exclude_id=exclude_id)
            if knn is not None and knn.confidence >= self.knn_threshold:
                result = knn
            else:
                result = self._extractive(question) or knn
        elif mode == "rag":
            result = self._rag(question)
        elif mode == "hybrid_rag":
            knn = self._knn(question, exclude_id=exclude_id)
            if knn is not None and knn.confidence >= self.knn_threshold:
                result = knn
            else:
                result = self._rag(question) or knn
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
