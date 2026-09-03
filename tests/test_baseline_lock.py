from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from legalqa_baseline.baseline_lock import (
    REGRESSION_IDS,
    aggregate_retrieval_items,
    load_locked_split,
    load_regression_samples,
    make_baseline_manifest,
    score_answer,
    score_retrieval_trace,
    write_locked_manifest,
)


class _Index:
    def search_contexts(self, query: str, top_k: int = 100):
        return [
            {
                "context_id": "gold",
                "chunk_no": 0,
                "name": "Gold",
                "text": "đây là câu trả lời pháp luật đủ dài để tạo nhãn giả",
            }
        ][:top_k]


class BaselineLockTests(unittest.TestCase):
    def test_manifest_freezes_nested_100_and_300_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.json"
            public_path = root / "public.json"
            train = {
                str(i): {"question": f"q{i}", "answer": f"a{i}"}
                for i in range(350)
            }
            public = {
                sample_id: {"question": f"public {sample_id}", "answer": None}
                for sample_id in REGRESSION_IDS
            }
            train_path.write_text(json.dumps(train), encoding="utf-8")
            public_path.write_text(json.dumps(public), encoding="utf-8")

            manifest = make_baseline_manifest(
                train,
                train_path,
                public=public,
                public_path=public_path,
            )
            self.assertEqual(len(manifest["splits"]["validation_100"]), 100)
            self.assertEqual(len(manifest["splits"]["validation_300"]), 300)
            self.assertEqual(
                manifest["splits"]["validation_300"][:100],
                manifest["splits"]["validation_100"],
            )
            self.assertEqual(manifest["regression_ids"], list(REGRESSION_IDS))

            lock_path = root / "lock.json"
            write_locked_manifest(lock_path, manifest)
            write_locked_manifest(lock_path, manifest)
            ids, loaded = load_locked_split(
                lock_path, "validation_100", train, train_path
            )
            self.assertEqual(ids, manifest["splits"]["validation_100"])
            regression = load_regression_samples(loaded, public, public_path)
            self.assertEqual(list(regression), list(REGRESSION_IDS))

    def test_manifest_refuses_changed_data_and_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.json"
            train = {
                str(i): {"question": f"q{i}", "answer": f"a{i}"}
                for i in range(300)
            }
            train_path.write_text(json.dumps(train), encoding="utf-8")
            manifest = make_baseline_manifest(train, train_path)
            lock_path = root / "lock.json"
            write_locked_manifest(lock_path, manifest)
            with self.assertRaises(FileExistsError):
                write_locked_manifest(lock_path, {**manifest, "seed": 1})

            train_path.write_text(json.dumps({**train, "extra": {}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_locked_split(lock_path, "validation_100", train, train_path)

    def test_per_question_answer_and_retrieval_metrics(self) -> None:
        reference = "đây là câu trả lời pháp luật đủ dài để tạo nhãn giả"
        answer = score_answer(reference, reference)
        self.assertEqual(answer["rougeL"], 1.0)
        self.assertEqual(answer["prediction_words"], answer["reference_words"])

        trace = {
            "bm25": {
                "candidates": [
                    {"context_id": "gold", "chunk_no": 0, "rank": 1}
                ]
            }
        }
        retrieval = score_retrieval_trace(_Index(), reference, trace)
        self.assertTrue(retrieval["pseudo_gold_available"])
        self.assertEqual(retrieval["stages"]["bm25"]["recall@1"], 1.0)
        aggregate = aggregate_retrieval_items([{"retrieval": retrieval}])
        self.assertEqual(aggregate["metrics"]["bm25"]["recall@1"], 1.0)


if __name__ == "__main__":
    unittest.main()
