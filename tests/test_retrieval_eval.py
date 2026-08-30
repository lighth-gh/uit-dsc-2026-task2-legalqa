from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from legalqa_baseline.retrieval_eval import (
    answer_evidence_score,
    build_pseudo_gold,
    compute_stage_ranking_metrics,
    evaluate_retrieval,
)


GOLD_TEXT = "Đáp án quy định mức phạt là năm triệu đồng đối với hành vi này"


def chunk(context_id: str, text: str) -> dict[str, Any]:
    return {
        "context_id": context_id,
        "chunk_no": 0,
        "name": f"Document {context_id}",
        "text": text,
        "bm25_score": -10.0,
    }


class MockIndex:
    def search_contexts(self, query: str, top_k: int = 50) -> list[dict[str, Any]]:
        if query == GOLD_TEXT:
            return [
                chunk("gold", GOLD_TEXT),
                chunk("noise", "Nội dung không liên quan đến câu trả lời."),
            ][:top_k]
        return [
            chunk("noise", "Nội dung đứng đầu nhưng không phải chứng cứ."),
            chunk("gold", GOLD_TEXT),
            chunk("other", "Một quy định pháp luật khác."),
        ][:top_k]


class MockEmbedding:
    def encode(self, texts: list[str], **_: Any) -> np.ndarray:
        return np.ones((len(texts), 2), dtype=np.float32)


class MockDenseIndex:
    similarity = "dot_product"

    def search(self, query_vector: Any, top_k: int = 50) -> list[dict[str, Any]]:
        return [
            {**chunk("gold", GOLD_TEXT), "dense_score": 1.0},
            {**chunk("other", "Nội dung khác"), "dense_score": 0.5},
            {**chunk("noise", "Nhiễu"), "dense_score": 0.1},
        ][:top_k]


class MockReranker:
    def rerank(
        self,
        question: str,
        candidates: list[dict[str, Any]],
        top_k: int,
        max_length: int,
    ) -> list[dict[str, Any]]:
        return sorted(
            candidates,
            key=lambda item: item["context_id"] == "gold",
            reverse=True,
        )[:top_k]


class RetrievalEvaluationTests(unittest.TestCase):
    def test_answer_evidence_and_pseudo_gold(self) -> None:
        self.assertAlmostEqual(answer_evidence_score(GOLD_TEXT, GOLD_TEXT), 1.0)
        gold, audit = build_pseudo_gold(MockIndex(), GOLD_TEXT)
        self.assertEqual(gold, {("gold", 0)})
        self.assertEqual(audit[0]["context_id"], "gold")

    def test_compute_stage_ranking_metrics(self) -> None:
        results = [
            chunk("doc1", "a"),
            chunk("gold1", "b"),
            chunk("gold2", "c"),
        ]
        gold = {("gold1", 0), ("gold2", 0)}
        metrics = compute_stage_ranking_metrics(results, gold, ks=(1, 2, 3))
        self.assertEqual(metrics["recall@1"], 0.0)
        self.assertEqual(metrics["recall@2"], 1.0)
        self.assertEqual(metrics["recall@3"], 1.0)
        self.assertAlmostEqual(metrics["mrr"], 0.5)
        self.assertAlmostEqual(metrics["mrr@1"], 0.0)
        self.assertAlmostEqual(metrics["mrr@2"], 0.5)
        self.assertAlmostEqual(metrics["gold_recall@2"], 0.5)
        self.assertAlmostEqual(metrics["gold_recall@3"], 1.0)
        self.assertGreater(metrics["ndcg@3"], 0.0)
        self.assertGreater(metrics["map@3"], 0.0)

    def test_recall_at_1_3_5_for_each_stage(self) -> None:
        report = evaluate_retrieval(
            {"sample": {"question": "Mức phạt là bao nhiêu?", "answer": GOLD_TEXT}},
            MockIndex(),
            dense_index=MockDenseIndex(),
            embedding_model=MockEmbedding(),
            reranker=MockReranker(),
            limit=1,
        )
        self.assertEqual(report["samples_evaluated"], 1)
        self.assertEqual(report["metrics"]["bm25"]["recall@1"], 0.0)
        self.assertEqual(report["metrics"]["bm25"]["recall@3"], 1.0)
        self.assertEqual(report["metrics"]["bm25"]["recall@5"], 1.0)
        self.assertEqual(report["metrics"]["bm25"]["mrr@3"], 0.5)
        self.assertEqual(report["metrics"]["dense"]["recall@1"], 1.0)
        self.assertEqual(report["metrics"]["dense"]["mrr@1"], 1.0)
        self.assertEqual(report["metrics"]["rrf"]["recall@1"], 1.0)
        self.assertEqual(report["metrics"]["reranker"]["recall@1"], 1.0)


if __name__ == "__main__":
    unittest.main()
