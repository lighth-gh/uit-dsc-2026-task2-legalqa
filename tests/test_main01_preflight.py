import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ('legalqa_main_01_qlora_train.ipynb', 'legalqa-main-01-qlora-train.ipynb')


def runner_source(name):
    notebook = json.loads((ROOT / name).read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        for node in ast.parse(''.join(cell['source'])).body:
            if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == 'STAGE1_TEST_RUNNER'
                    for target in node.targets):
                return ast.literal_eval(node.value)
    raise AssertionError(f'Missing preflight runner: {name}')


class Main01PreflightTests(unittest.TestCase):
    def test_spawn_import_does_not_discover_or_run_tests(self):
        for name in NOTEBOOKS:
            with self.subTest(notebook=name), patch.object(
                    unittest.defaultTestLoader, 'discover',
                    side_effect=AssertionError('Child must not rediscover tests')), \
                    patch.object(sys, 'path', list(sys.path)):
                exec(compile(runner_source(name), name, 'exec'), {'__name__': '__mp_main__'})

    def test_notebook_alias_uses_same_runner(self):
        self.assertEqual(runner_source(NOTEBOOKS[0]), runner_source(NOTEBOOKS[1]))

    def test_runner_real_spawn_and_failure_exit_status(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            tests = root / 'tests'
            tests.mkdir()
            # Keep the existing Stage 4 skip contract while running the actual
            # multi-process regression that failed in Kaggle's preflight.
            (tests / 'test_repair.py').write_text(
                'import unittest\n'
                'class RepairTests(unittest.TestCase):\n'
                '    def test_notebook_cells_compile_and_bundle_matches_sources(self):\n'
                '        self.fail("Stage 4-only test must be excluded")\n', encoding='utf-8')
            (tests / 'test_spawn.py').write_text(
                'from test_generation_multigpu import MultiGpuGenerationTests\n', encoding='utf-8')
            runner = root / 'legalqa_stage1_tests.py'
            runner.write_text(runner_source(NOTEBOOKS[0]), encoding='utf-8')
            env = {**os.environ, 'PYTHONIOENCODING': 'utf-8',
                   'PYTHONPATH': os.pathsep.join([str(ROOT), str(ROOT / 'tests'),
                                                 os.environ.get('PYTHONPATH', '')])}

            def run():
                return subprocess.run([sys.executable, '-B', str(runner)], cwd=root,
                                      env=env, capture_output=True, text=True,
                                      encoding='utf-8', timeout=60)

            result = run()
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(result.stdout.count('Stage 1 preflight skips'), 1)
            self.assertEqual(result.stderr.count('test_real_spawn_two_processes_global_odd_cap ('), 1)
            (tests / 'test_failure.py').write_text(
                'import unittest\n'
                'class FailureTests(unittest.TestCase):\n'
                '    def test_failure(self):\n'
                '        self.fail("expected preflight failure")\n', encoding='utf-8')
            result = run()
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn('expected preflight failure', result.stderr)


if __name__ == '__main__':
    unittest.main()
