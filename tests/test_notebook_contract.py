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


if __name__ == "__main__":
    unittest.main()
