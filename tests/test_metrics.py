from __future__ import annotations

import unittest

from legalqa_baseline.metrics import (
    aggregate_scores,
    answer_token_f1,
    bleu_score,
    exact_match,
    meteor_exact,
    rouge_l_f1,
    token_precision_recall_f1,
)


class MetricsTests(unittest.TestCase):
    def test_exact_match(self) -> None:
        self.assertEqual(exact_match("Điều 1. Phạm vi áp dụng", "Điều 1. Phạm vi áp dụng"), 1.0)
        self.assertEqual(exact_match("điều 1", "Điều 1"), 1.0)
        self.assertEqual(exact_match("Điều 1", "Điều 2"), 0.0)

    def test_token_precision_recall_f1(self) -> None:
        pred = "khoản 1 điều 5 luật giao thông"
        ref = "khoản 1 điều 5 luật doanh nghiệp"
        p, r, f1 = token_precision_recall_f1(pred, ref)
        # 5 out of 7 match: khoản, 1, điều, 5, luật
        self.assertAlmostEqual(p, 5 / 7)
        self.assertAlmostEqual(r, 5 / 7)
        self.assertAlmostEqual(f1, 5 / 7)
        self.assertAlmostEqual(answer_token_f1(pred, ref), 5 / 7)

    def test_rouge_l_f1(self) -> None:
        pred = "quy định về mức xử phạt vi phạm"
        ref = "quy định về các mức xử phạt vi phạm hành chính"
        score = rouge_l_f1(pred, ref)
        self.assertGreater(score, 0.5)
        self.assertLessEqual(score, 1.0)
        self.assertEqual(rouge_l_f1("", ref), 0.0)
        self.assertEqual(rouge_l_f1(pred, ""), 0.0)

    def test_meteor_exact(self) -> None:
        pred = "mức phạt là năm triệu đồng"
        ref = "mức phạt là năm triệu đồng"
        self.assertGreater(meteor_exact(pred, ref), 0.99)
        self.assertEqual(meteor_exact("", ref), 0.0)
        self.assertGreater(meteor_exact("phạt năm triệu đồng", ref), 0.0)

    def test_bleu_score(self) -> None:
        text = "chính phủ ban hành nghị định về xử phạt vi phạm hành chính"
        self.assertAlmostEqual(bleu_score(text, text, max_n=4), 1.0)
        self.assertAlmostEqual(bleu_score(text, text, max_n=1), 1.0)

        partial = "chính phủ ban hành nghị định"
        b1 = bleu_score(partial, text, max_n=1, weights=(1.0,))
        self.assertGreater(b1, 0.0)
        self.assertLess(b1, 1.0)
        self.assertEqual(bleu_score("", text), 0.0)

    def test_aggregate_scores(self) -> None:
        preds = [
            "mức phạt là năm triệu đồng",
            "theo điều 15 nghị định 100",
        ]
        refs = [
            "mức phạt là năm triệu đồng",
            "theo quy định tại điều 15 nghị định 100",
        ]
        scores = aggregate_scores(preds, refs)
        self.assertIn("meteor_exact_approx", scores)
        self.assertIn("rougeL", scores)
        self.assertIn("answer_token_f1", scores)
        self.assertIn("answer_token_precision", scores)
        self.assertIn("answer_token_recall", scores)
        self.assertIn("exact_match", scores)
        self.assertIn("bleu_1", scores)
        self.assertIn("bleu_2", scores)
        self.assertIn("bleu_4", scores)
        self.assertIn("avg_prediction_words", scores)
        self.assertIn("length_ratio", scores)
        self.assertEqual(scores["samples"], 2)
        self.assertGreater(scores["meteor_exact_approx"], 0.7)

    def test_aggregate_scores_validation_error(self) -> None:
        with self.assertRaises(ValueError):
            aggregate_scores(["a"], ["a", "b"])
        with self.assertRaises(ValueError):
            aggregate_scores([], [])


if __name__ == "__main__":
    unittest.main()
