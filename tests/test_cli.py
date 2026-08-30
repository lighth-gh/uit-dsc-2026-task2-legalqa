from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from legalqa_baseline.cli import (
    _load_checkpoint,
    _write_checkpoint,
    _write_prediction_progress,
    command_build,
    command_build_dense_index,
    make_parser,
)


class CliTests(unittest.TestCase):
    def test_rag_defaults_match_hybrid_retrieval_contract(self) -> None:
        args = make_parser().parse_args(
            [
                "predict",
                "--input",
                "input.json",
                "--db",
                "index.sqlite",
                "--output",
                "submission.json",
                "--mode",
                "rag",
            ]
        )
        self.assertEqual(args.rrf_top_k, 50)
        self.assertEqual(args.rerank_top_k, 3)
        self.assertEqual(args.context_top_k, 3)
        self.assertEqual(args.temperature, 0.0)
        self.assertEqual(args.checkpoint_interval, 1)
        self.assertFalse(args.allow_retrieval_fallback)

    def test_checkpoint_filters_ids_and_keeps_route_counts_consistent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prediction.checkpoint.json"
            _write_checkpoint(
                path,
                {"1": "Đáp án 1", "extra": "Không thuộc input"},
                {"rag": 2},
            )
            predictions, routes = _load_checkpoint(path, {"1"})
            self.assertEqual(predictions, {"1": "Đáp án 1"})
            self.assertEqual(routes, {"resumed_unknown": 1})

    def test_prediction_progress_keeps_checkpoint_and_partial_submission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "submission.checkpoint.json"
            output = Path(directory) / "submission.json"
            _write_prediction_progress(
                checkpoint,
                output,
                {"1": "answer one", "2": "answer two"},
                {"rag": 2},
            )

            checkpoint_payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            output_payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(checkpoint_payload["schema_version"], 1)
            self.assertEqual(checkpoint_payload["routes"], {"rag": 2})
            self.assertEqual(output_payload["1"], {"answer": "answer one"})
            self.assertEqual(output_payload["2"], {"answer": "answer two"})

    def test_build_commands_forward_only_supported_parameters(self) -> None:
        parser = make_parser()
        bm25_args = parser.parse_args(
            [
                "build-index",
                "--contexts",
                "contexts.zip",
                "--train",
                "train.json",
                "--db",
                "index.sqlite",
            ]
        )
        with patch("legalqa_baseline.cli.build_index", return_value={}) as mocked_bm25:
            self.assertEqual(command_build(bm25_args), 0)
        self.assertNotIn("embedding_max_length", mocked_bm25.call_args.kwargs)

        dense_args = parser.parse_args(
            [
                "build-dense-index",
                "--contexts",
                "contexts.zip",
                "--dense-index",
                "dense",
                "--embedding-max-length",
                "1024",
                "--resume",
                "--checkpoint-chunks",
                "16",
            ]
        )
        with patch("legalqa_baseline.dense.build_dense_index", return_value={}) as mocked_dense:
            self.assertEqual(command_build_dense_index(dense_args), 0)
        self.assertEqual(mocked_dense.call_args.kwargs["embedding_max_length"], 1024)
        self.assertTrue(mocked_dense.call_args.kwargs["resume"])
        self.assertEqual(mocked_dense.call_args.kwargs["checkpoint_chunks"], 16)

    def test_dense_batch_default_is_t4_safe(self) -> None:
        args = make_parser().parse_args(
            [
                "build-dense-index",
                "--contexts",
                "contexts.zip",
                "--dense-index",
                "dense",
            ]
        )
        self.assertEqual(args.batch_size, 8)


if __name__ == "__main__":
    unittest.main()
