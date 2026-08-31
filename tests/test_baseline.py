from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from legalqa_baseline.metrics import answer_token_f1, meteor_exact, rouge_l_f1
from legalqa_baseline.pipeline import (
    LegalQABaseline,
    question_similarity,
    reciprocal_rank_fusion,
)
from legalqa_baseline.storage import build_index, load_qa, write_predictions
from legalqa_baseline.text import best_excerpt, chunk_passage, query_terms


class TextTests(unittest.TestCase):
    def test_article_chunking(self) -> None:
        text = "Mở đầu văn bản.\nĐiều 1. Quy định chung " + "nội dung " * 700
        chunks = chunk_passage(text, max_words=120, overlap_words=20)
        self.assertGreater(len(chunks), 5)
        self.assertTrue(chunks[0].startswith("Điều 1"))

    def test_query_terms(self) -> None:
        terms = query_terms("Mức phạt theo khoản 3 Điều 17 là bao nhiêu?")
        self.assertIn("phạt", terms)
        self.assertIn("17", terms)

    def test_excerpt_limit(self) -> None:
        text = ("khác " * 300) + ("kiểm dịch động vật " * 100)
        excerpt = best_excerpt(text, "kiểm dịch động vật", max_words=100)
        self.assertLessEqual(len(excerpt.split()), 100)
        self.assertIn("kiểm dịch", excerpt)


class MetricTests(unittest.TestCase):
    def test_identical_scores(self) -> None:
        text = "a b c d"
        self.assertAlmostEqual(rouge_l_f1(text, text), 1.0)
        self.assertGreater(meteor_exact(text, text), 0.99)
        self.assertAlmostEqual(answer_token_f1(text, text), 1.0)

    def test_question_similarity(self) -> None:
        same = question_similarity("Mức xử phạt là bao nhiêu?", "Mức xử phạt là bao nhiêu?")
        other = question_similarity("Mức xử phạt là bao nhiêu?", "Thủ tục cấp hộ chiếu")
        self.assertGreater(same, other)

    def test_question_similarity_keeps_legal_disambiguators(self) -> None:
        self.assertLess(
            question_similarity(
                "\u0110i\u1ec1u 5 x\u1eed ph\u1ea1t xe m\u00e1y",
                "\u0110i\u1ec1u 6 x\u1eed ph\u1ea1t xe m\u00e1y",
            ),
            0.72,
        )
        self.assertLess(
            question_similarity(
                "Ngh\u0129a v\u1ee5 ng\u01b0\u1eddi \u0111\u01b0\u1ee3c thi h\u00e0nh \u00e1n",
                "Ngh\u0129a v\u1ee5 ng\u01b0\u1eddi ph\u1ea3i thi h\u00e0nh \u00e1n",
            ),
            0.72,
        )


class PipelineRoutingTests(unittest.TestCase):
    class EmptyIndex:
        def search_contexts(self, question: str, top_k: int = 12) -> list[dict[str, object]]:
            return []

        def search_train(
            self,
            question: str,
            top_k: int = 5,
            exclude_id: str | None = None,
        ) -> list[dict[str, object]]:
            return [{
                "sample_id": "wrong",
                "question": "thu tuc cap ho chieu",
                "answer": "wrong answer",
                "bm25_score": -1.0,
            }]

    def test_hybrid_does_not_return_knn_below_threshold(self) -> None:
        pipeline = LegalQABaseline(index=self.EmptyIndex())  # type: ignore[arg-type]
        prediction = pipeline.predict_one("muc phat giao thong", mode="hybrid")
        self.assertEqual(prediction.route, "fallback")

    def test_blank_question_returns_fallback_without_retrieval(self) -> None:
        class TrackingIndex(self.EmptyIndex):
            def search_contexts(self, question: str, top_k: int = 12) -> list[dict[str, object]]:
                raise AssertionError("blank question must not query contexts")

            def search_train(
                self,
                question: str,
                top_k: int = 5,
                exclude_id: str | None = None,
            ) -> list[dict[str, object]]:
                raise AssertionError("blank question must not query train")

        prediction = LegalQABaseline(index=TrackingIndex()).predict_one(" \t", mode="rag")  # type: ignore[arg-type]
        self.assertEqual(prediction.route, "fallback")

    def test_partial_dense_configuration_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cùng nhau"):
            LegalQABaseline(index=self.EmptyIndex(), dense_index=object())  # type: ignore[arg-type]

    def test_blank_knn_answers_are_skipped(self) -> None:
        class BlankAnswerIndex(self.EmptyIndex):
            def search_train(
                self,
                question: str,
                top_k: int = 5,
                exclude_id: str | None = None,
            ) -> list[dict[str, object]]:
                return [{
                    "sample_id": "blank",
                    "question": question,
                    "answer": "",
                    "bm25_score": -1.0,
                }]

        prediction = LegalQABaseline(index=BlankAnswerIndex()).predict_one("same", mode="hybrid")  # type: ignore[arg-type]
        self.assertEqual(prediction.route, "fallback")

    def test_rrf_validates_and_deduplicates_candidates(self) -> None:
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([], [], rrf_k=0)
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([], [], top_k=0)
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([{"text": "missing identity"}], [])

        fused = reciprocal_rank_fusion(
            [
                {"context_id": "doc", "chunk_no": 0, "text": "first"},
                {"context_id": "doc", "chunk_no": 0, "text": "duplicate"},
            ],
            [],
        )
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]["text"], "first")


class IoTests(unittest.TestCase):
    def test_schema_and_submission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.json"
            source.write_text(
                json.dumps({"1": {"question": "Câu hỏi?", "answer": None}}, ensure_ascii=False),
                encoding="utf-8",
            )
            self.assertEqual(load_qa(source)["1"]["question"], "Câu hỏi?")
            output = root / "prediction.json"
            write_predictions(output, {"1": "Câu trả lời"})
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["1"]["answer"], "Câu trả lời")


    def test_build_index_rejects_missing_train_answers_before_creating_db(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contexts = root / "contexts"
            contexts.mkdir()
            (contexts / "context_1.json").write_text(
                json.dumps({"id": "ctx", "name": "Doc", "link": "", "passage": ""}),
                encoding="utf-8",
            )
            train = root / "train.json"
            train.write_text(
                json.dumps({"1": {"question": "Question", "answer": None}}),
                encoding="utf-8",
            )
            database = root / "index.sqlite"
            with self.assertRaisesRegex(ValueError, "thiếu answer"):
                build_index(contexts, train, database)
            self.assertFalse(database.exists())


if __name__ == "__main__":
    unittest.main()
