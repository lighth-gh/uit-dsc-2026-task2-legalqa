from __future__ import annotations

import json
import unittest
from pathlib import Path

from legalqa_baseline.dense import DENSE_SCHEMA_VERSION
from legalqa_baseline.storage import CORPUS_HASH_VERSION, SCHEMA_VERSION


class NotebookCacheContractTests(unittest.TestCase):
    @staticmethod
    def _notebook_code(name: str) -> str:
        notebook_path = Path(__file__).resolve().parents[1] / name
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        return "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "code"
        )

    def test_cache_gate_matches_current_index_contract(self) -> None:
        code = self._notebook_code("uit-dsc-2026-task2-legalqa.ipynb")

        self.assertIn(f'REQUIRED_BM25_SCHEMA = "{SCHEMA_VERSION}"', code)
        self.assertIn(f"REQUIRED_DENSE_SCHEMA = {DENSE_SCHEMA_VERSION}", code)
        self.assertIn(
            f'REQUIRED_CORPUS_HASH_VERSION = "{CORPUS_HASH_VERSION}"',
            code,
        )
        self.assertIn(
            'metadata.get("corpus_hash_version") '
            "== REQUIRED_CORPUS_HASH_VERSION",
            code,
        )
        self.assertIn(
            'manifest.get("corpus_hash_version") '
            "== REQUIRED_CORPUS_HASH_VERSION",
            code,
        )
        self.assertIn('(\"\", \"-wal\", \"-shm\", \"-journal\")', code)

    def test_full_prediction_keeps_smoke30_speed_contract(self) -> None:
        full_code = self._notebook_code("uit-dsc-2026-task2-legalqa.ipynb")
        smoke_code = self._notebook_code("legalqa-pipeline-smoke30-ready.ipynb")

        for expected in (
            "RRF_TOP_K = 50",
            "RERANKER_MAX_LENGTH = 1024",
            '"--reranker-candidate-k", str(RERANKER_CANDIDATE_K)',
            '"--reranker-max-length", str(RERANKER_MAX_LENGTH)',
            'SUBMISSION_PATH = WORK_DIR / "submission.json"',
            'BM25 local: {DB_PATH}',
            "ENFORCE_SMOKE30_SPEED_PROFILE = True",
        ):
            self.assertIn(expected, full_code)

        self.assertIn("RERANKER_CANDIDATE_K = 20", full_code)
        self.assertIn("MAX_INPUT_TOKENS = 7168", full_code)
        self.assertIn("MAX_NEW_TOKENS = 512", full_code)
        self.assertIn("RERANKER_CANDIDATE_K, RERANK_TOP_K = 20, 3", smoke_code)
        self.assertIn("DENSE_QUERY_MAX_LENGTH, RERANKER_MAX_LENGTH = 256, 1024", smoke_code)
        self.assertIn("MAX_NEW_TOKENS, MAX_INPUT_TOKENS = 512, 7168", smoke_code)

    def test_smoke_notebooks_use_shared_refusal_detector(self) -> None:
        for notebook_name in (
            "legalqa-generation-smoke-test.ipynb",
            "legalqa-pipeline-smoke30-ready.ipynb",
        ):
            with self.subTest(notebook=notebook_name):
                code = self._notebook_code(notebook_name)
                self.assertIn("is_refusal_answer", code)
                self.assertNotIn("no_info_re =", code)
                self.assertNotIn(
                    '"says_no_information": bool("không đủ thông tin"',
                    code,
                )

    def test_smoke30_enforces_two_round_release_gate(self) -> None:
        code = self._notebook_code("legalqa-pipeline-smoke30-ready.ipynb")

        for expected in (
            '"diagnose-retrieval"',
            "RETRIEVAL_MEDIAN_MAX_SECONDS = 2.0",
            "SMOKE_MEDIAN_MAX_SECONDS = 15.0",
            "RETRIEVAL_EXPECTED_DOCUMENT_IDS",
            '("top50", "top20", "top3")',
            "MANUAL_REVIEW_APPROVED_IDS",
            'full_gate["full_1000_unlocked"]',
        ):
            self.assertIn(expected, code)
        self.assertLess(code.index('"diagnose-retrieval"'), code.index('"predict"'))
        self.assertIn("if not retrieval_speed_gate_pass:", code)
        self.assertIn('"document_gate_status": "pending"', code)
        self.assertNotIn('if not retrieval_gate["pass"]:\n    raise RuntimeError', code)


if __name__ == "__main__":
    unittest.main()
