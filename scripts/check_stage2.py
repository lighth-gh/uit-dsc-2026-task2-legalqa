"""Stage 2 preflight: input/import, artifact provenance, retrieval and generation."""
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tests'))

TEST_MODULES = (
    'test_core', 'test_stages', 'test_main02_inputs', 'test_main02_private_import',
    'test_bm25_cache', 'test_phrase_precise', 'test_phrase_readers', 'test_generation_multigpu',
)


def main():
    print('Stage 2 preflight modules:', ', '.join(TEST_MODULES), flush=True)
    suite = unittest.defaultTestLoader.loadTestsFromNames(TEST_MODULES)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
