from __future__ import annotations

import json
import unittest
from pathlib import Path

from legalqa_baseline.dense import DENSE_SCHEMA_VERSION
from legalqa_baseline.storage import CORPUS_HASH_VERSION, SCHEMA_VERSION


class NotebookCacheContractTests(unittest.TestCase):
    def test_cache_gate_matches_current_index_contract(self) -> None:
        notebook_path = (
            Path(__file__).resolve().parents[1]
            / "uit-dsc-2026-task2-legalqa.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "code"
        )

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


if __name__ == "__main__":
    unittest.main()
