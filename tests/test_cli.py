from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from legalqa_baseline.cli import (
    _append_audit_record,
    _load_checkpoint,
    _pipeline,
    _write_checkpoint,
    _write_prediction_progress,
    command_build,
    command_build_dense_index,
    command_evaluate_retrieval,
    command_score,
    command_validate,
    main,
    make_parser,
)
from legalqa_baseline.pipeline import Prediction


class CliTests(unittest.TestCase):
    def test_predict_defaults_to_guarded_hybrid_rag(self) -> None:
        args = make_parser().parse_args(
            [
                "predict",
                "--input",
                "input.json",
                "--db",
                "index.sqlite",
                "--output",
                "submission.json",
            ]
        )

        self.assertEqual(args.mode, "hybrid_rag")
        self.assertEqual(args.guarded_knn_threshold, 0.90)

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
        self.assertEqual(args.reranker_candidate_k, 20)
        self.assertEqual(args.rerank_top_k, 3)
        self.assertEqual(args.reranker_max_length, 1024)
        self.assertEqual(args.context_top_k, 3)
        self.assertEqual(args.temperature, 0.0)
        self.assertEqual(args.max_input_tokens, 7168)
        self.assertEqual(args.max_new_tokens, 512)
        self.assertEqual(args.repetition_penalty, 1.05)
        self.assertEqual(args.min_llm_answer_tokens, 8)
        self.assertEqual(args.token_limit_retry_tokens, 768)
        self.assertEqual(args.guarded_knn_threshold, 0.90)
        self.assertEqual(args.checkpoint_interval, 1)
        self.assertIsNone(args.audit_output)
        self.assertIsNone(args.embedding_revision)
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

    def test_audit_record_is_appended_as_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audit_path = Path(directory) / "submission.audit.jsonl"
            _append_audit_record(audit_path, {"id": "1", "route": "generated_512"})
            _append_audit_record(audit_path, {"id": "2", "route": "extractive_long"})

            records = [
                json.loads(line)
                for line in audit_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([record["id"] for record in records], ["1", "2"])

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
                "--embedding-revision",
                "abc123",
                "--resume",
                "--checkpoint-chunks",
                "16",
            ]
        )
        with patch("legalqa_baseline.dense.build_dense_index", return_value={}) as mocked_dense:
            self.assertEqual(command_build_dense_index(dense_args), 0)
        self.assertEqual(mocked_dense.call_args.kwargs["embedding_max_length"], 1024)
        self.assertEqual(mocked_dense.call_args.kwargs["embedding_model_revision"], "abc123")
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

    def test_dense_fallback_resets_both_dense_index_and_embedding_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dense_path = Path(directory) / "dense"
            dense_path.touch()
            meta_path = Path(directory) / "dense.meta.json"
            meta_path.touch()

            parser = make_parser()
            # 1. With --allow-retrieval-fallback: if validate_against_bm25 fails, both should be None
            args_with_fallback = parser.parse_args(
                [
                    "predict",
                    "--input", "in.json",
                    "--db", "db.sqlite",
                    "--output", "out.json",
                    "--mode", "rag",
                    "--dense-index", str(dense_path),
                    "--allow-retrieval-fallback",
                ]
            )
            mock_index = MagicMock()
            mock_dense = MagicMock()
            mock_dense.validate_against_bm25.side_effect = ValueError("BM25 metadata mismatch")

            with patch("legalqa_baseline.generator.ViQwenRAGGenerator"), \
                 patch("legalqa_baseline.dense.DenseVectorIndex.load", return_value=mock_dense):
                pipeline = _pipeline(args_with_fallback, mock_index, need_generator=True)
                self.assertIsNone(pipeline.dense_index)
                self.assertIsNone(pipeline.embedding_model)

            # 2. Without --allow-retrieval-fallback: should raise ValueError
            args_no_fallback = parser.parse_args(
                [
                    "predict",
                    "--input", "in.json",
                    "--db", "db.sqlite",
                    "--output", "out.json",
                    "--mode", "rag",
                    "--dense-index", str(dense_path),
                ]
            )
            with patch("legalqa_baseline.generator.ViQwenRAGGenerator"), \
                 patch("legalqa_baseline.dense.DenseVectorIndex.load", return_value=mock_dense):
                with self.assertRaises(ValueError):
                    _pipeline(args_no_fallback, mock_index, need_generator=True)

    def test_reranker_init_failure_honors_allow_retrieval_fallback(self) -> None:
        parser = make_parser()
        mock_index = MagicMock()

        args_no_fallback = parser.parse_args(
            [
                "predict",
                "--input", "in.json",
                "--db", "db.sqlite",
                "--output", "out.json",
                "--mode", "rag",
                "--reranker-model", "AITeamVN/Vietnamese_Reranker",
            ]
        )
        with patch("legalqa_baseline.generator.ViQwenRAGGenerator"), \
             patch("legalqa_baseline.reranker.VietnameseReranker", side_effect=RuntimeError("Reranker OOM")):
            with self.assertRaises(RuntimeError):
                _pipeline(args_no_fallback, mock_index, need_generator=True)

        args_with_fallback = parser.parse_args(
            [
                "predict",
                "--input", "in.json",
                "--db", "db.sqlite",
                "--output", "out.json",
                "--mode", "rag",
                "--reranker-model", "AITeamVN/Vietnamese_Reranker",
                "--allow-retrieval-fallback",
            ]
        )
        with patch("legalqa_baseline.generator.ViQwenRAGGenerator"), \
             patch("legalqa_baseline.reranker.VietnameseReranker", side_effect=RuntimeError("Reranker OOM")):
            pipeline = _pipeline(args_with_fallback, mock_index, need_generator=True)
            self.assertIsNone(pipeline.reranker)

    def test_main_catches_sqlite_and_os_errors_cleanly(self) -> None:
        with patch("legalqa_baseline.cli.command_predict", side_effect=sqlite3.DatabaseError("Corrupt DB")):
            code = main(["predict", "--input", "in.json", "--db", "db.sqlite", "--output", "out.json"])
            self.assertEqual(code, 2)

        with patch("legalqa_baseline.cli.command_predict", side_effect=OSError("Disk full")):
            code = main(["predict", "--input", "in.json", "--db", "db.sqlite", "--output", "out.json"])
            self.assertEqual(code, 2)

    def test_score_validates_prediction_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ref_path = Path(directory) / "ref.json"
            pred_path = Path(directory) / "pred.json"

            ref_path.write_text(json.dumps({"1": {"question": "q", "answer": "a"}}), encoding="utf-8")

            # 1. Top-level is a list
            pred_path.write_text(json.dumps([{"answer": "a"}]), encoding="utf-8")
            args = make_parser().parse_args(["score", "--reference", str(ref_path), "--prediction", str(pred_path)])
            with self.assertRaises(ValueError):
                command_score(args)

            # 2. Missing answer key
            pred_path.write_text(json.dumps({"1": {}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                command_score(args)

            # 3. answer is null
            pred_path.write_text(json.dumps({"1": {"answer": None}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                command_score(args)

            # 4. Valid schema
            pred_path.write_text(json.dumps({"1": {"answer": "a"}}), encoding="utf-8")
            self.assertEqual(command_score(args), 0)

    def test_validate_rejects_empty_modes_and_invalid_limit(self) -> None:
        parser = make_parser()
        empty_modes_args = parser.parse_args(
            ["validate", "--train", "train.json", "--db", "db.sqlite", "--output", "out.json", "--modes", ""]
        )
        with self.assertRaises(ValueError):
            command_validate(empty_modes_args)

        zero_limit_args = parser.parse_args(
            ["validate", "--train", "train.json", "--db", "db.sqlite", "--output", "out.json", "--limit", "0"]
        )
        with self.assertRaises(ValueError):
            command_validate(zero_limit_args)

    def test_locked_rag_validation_requires_dense_without_fallback(self) -> None:
        parser = make_parser()
        args = parser.parse_args(
            [
                "validate",
                "--train", "train.json",
                "--db", "db.sqlite",
                "--output", "out.json",
                "--modes", "rag",
                "--split-manifest", "lock.json",
            ]
        )
        fake_manifest = {"seed": 2026}
        with patch("legalqa_baseline.cli.load_qa", return_value={"1": {"question": "q", "answer": "a"}}), \
             patch("legalqa_baseline.cli.load_locked_split", return_value=(["1"], fake_manifest)), \
             patch("legalqa_baseline.cli.file_sha256", return_value="hash"):
            with self.assertRaises(ValueError):
                command_validate(args)

        args.dense_index = "dense"
        args.allow_retrieval_fallback = True
        with patch("legalqa_baseline.cli.load_qa", return_value={"1": {"question": "q", "answer": "a"}}), \
             patch("legalqa_baseline.cli.load_locked_split", return_value=(["1"], fake_manifest)), \
             patch("legalqa_baseline.cli.file_sha256", return_value="hash"):
            with self.assertRaises(ValueError):
                command_validate(args)

    def test_locked_validation_writes_per_question_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "report.json"
            args = make_parser().parse_args(
                [
                    "validate",
                    "--train", "train.json",
                    "--db", "db.sqlite",
                    "--output", str(output),
                    "--modes", "rag",
                    "--split-manifest", "lock.json",
                    "--dense-index", "dense",
                ]
            )
            data = {"1": {"question": "câu hỏi", "answer": "câu trả lời"}}
            manifest = {"seed": 2026}
            result = Prediction(
                answer="câu trả lời",
                route="generated_512",
                confidence=1.0,
                evidence={"context_words": 10, "generated_tokens": 3, "retrieval_trace": {}},
            )
            index = MagicMock()
            index.metadata.return_value = {"corpus_sha256": "corpus"}
            search_context = MagicMock()
            search_context.__enter__.return_value = index
            search_context.__exit__.return_value = False
            pipeline = MagicMock()
            pipeline.predict_one.return_value = result
            pipeline.dense_index.manifest = {"schema_version": 5}

            with patch("legalqa_baseline.cli.load_qa", return_value=data), \
                 patch("legalqa_baseline.cli.load_locked_split", return_value=(["1"], manifest)), \
                 patch("legalqa_baseline.cli.file_sha256", return_value="hash"), \
                 patch("legalqa_baseline.cli.SearchIndex", return_value=search_context), \
                 patch("legalqa_baseline.cli._pipeline", return_value=pipeline):
                self.assertEqual(command_validate(args), 0)

            payload = json.loads(output.read_text(encoding="utf-8"))
            item = payload["results"]["rag"]["items"][0]
            self.assertEqual(item["id"], "1")
            self.assertEqual(item["route"], "generated_512")
            self.assertIn("meteor_exact_approx", item["metrics"])
            self.assertIn("rougeL", item["metrics"])
            self.assertEqual(item["length"]["prediction_words"], 3)
            self.assertIn("retrieval", item)

    def test_evaluate_retrieval_rejects_invalid_numeric_parameters(self) -> None:
        parser = make_parser()

        # Invalid ks
        args = parser.parse_args(
            ["evaluate-retrieval", "--train", "train.json", "--db", "db.sqlite", "--output", "out.json", "--ks", "0,5"]
        )
        with patch("legalqa_baseline.cli.load_qa", return_value={}):
            with self.assertRaises(ValueError):
                command_evaluate_retrieval(args)

        # Invalid gold_candidate_k
        args = parser.parse_args(
            ["evaluate-retrieval", "--train", "train.json", "--db", "db.sqlite", "--output", "out.json", "--gold-candidate-k", "0"]
        )
        with patch("legalqa_baseline.cli.load_qa", return_value={}):
            with self.assertRaises(ValueError):
                command_evaluate_retrieval(args)

        # Invalid gold_max_chunks
        args = parser.parse_args(
            ["evaluate-retrieval", "--train", "train.json", "--db", "db.sqlite", "--output", "out.json", "--gold-max-chunks", "-1"]
        )
        with patch("legalqa_baseline.cli.load_qa", return_value={}):
            with self.assertRaises(ValueError):
                command_evaluate_retrieval(args)

        # Invalid gold_min_score (> 1.0)
        args = parser.parse_args(
            ["evaluate-retrieval", "--train", "train.json", "--db", "db.sqlite", "--output", "out.json", "--gold-min-score", "1.5"]
        )
        with patch("legalqa_baseline.cli.load_qa", return_value={}):
            with self.assertRaises(ValueError):
                command_evaluate_retrieval(args)


if __name__ == "__main__":
    unittest.main()
