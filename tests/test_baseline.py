from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from legalqa_baseline.metrics import meteor_exact, rouge_l_f1
from legalqa_baseline.pipeline import question_similarity
from legalqa_baseline.storage import load_qa, write_predictions
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

    def test_question_similarity(self) -> None:
        same = question_similarity("Mức xử phạt là bao nhiêu?", "Mức xử phạt là bao nhiêu?")
        other = question_similarity("Mức xử phạt là bao nhiêu?", "Thủ tục cấp hộ chiếu")
        self.assertGreater(same, other)


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


if __name__ == "__main__":
    unittest.main()
