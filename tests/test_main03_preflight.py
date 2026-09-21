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


def preflight_cell():
    notebook = json.loads((ROOT / 'legalqa_main_03_generate_submit.ipynb').read_text(encoding='utf-8'))
    return next(''.join(cell['source']) for cell in notebook['cells']
                if cell['cell_type'] == 'code' and 'STAGE3_TEST_RUNNER =' in ''.join(cell['source']))


def runner_source():
    for node in ast.parse(preflight_cell()).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == 'STAGE3_TEST_RUNNER'
                for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError('Missing Stage 3 runner')


class Main03PreflightTests(unittest.TestCase):
    def test_spawn_import_does_not_load_or_run_tests(self):
        with patch.object(unittest.defaultTestLoader, 'loadTestsFromNames',
                          side_effect=AssertionError('Child must not load tests')), \
                patch.object(unittest.TextTestRunner, 'run',
                             side_effect=AssertionError('Child must not run tests')):
            exec(compile(runner_source(), '<stage3>', 'exec'), {'__name__': '__mp_main__'})

    def test_cell_writes_runner_outside_pinned_checkout(self):
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder)
            code = work / 'pinned_code'
            code.mkdir()
            calls = []
            exec(preflight_cell(), {
                'WORK': work, 'CODE': code, 'STAGE': 3, 'sys': sys,
                'bounded_process': lambda *args, **kwargs: calls.append((args, kwargs)),
            })
            runner = work / 'legalqa_stage3_tests.py'
            self.assertEqual(runner.read_text(encoding='utf-8'), runner_source())
            self.assertEqual(list(code.iterdir()), [])
            self.assertEqual(calls[-1], (([sys.executable, '-B', runner],),
                                        {'cwd': code, 'seconds': 300}))

    def test_real_spawn_excludes_unrelated_tests_and_propagates_failures(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            tests = root / 'tests'
            tests.mkdir()
            # Reproduce both blockers from the log; neither belongs to Stage 3.
            for name in ('test_main01_preflight', 'test_repair'):
                (tests / f'{name}.py').write_text(
                    'raise AssertionError("Unrelated notebook/bundle test imported")\n', encoding='utf-8')
            for name in ('test_core', 'test_stages'):
                (tests / f'{name}.py').write_text(
                    'import unittest\nclass Checks(unittest.TestCase):\n'
                    '    def test_ok(self): pass\n', encoding='utf-8')
            runner = root / 'legalqa_stage3_tests.py'
            runner.write_text(runner_source(), encoding='utf-8')
            env = {**os.environ, 'PYTHONIOENCODING': 'utf-8',
                   'PYTHONPATH': os.pathsep.join([str(ROOT), str(ROOT / 'tests')])}

            def run():
                return subprocess.run([sys.executable, '-B', str(runner)], cwd=root,
                                      env=env, capture_output=True, text=True,
                                      encoding='utf-8', timeout=60)

            result = run()
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(result.stdout.count('Stage 3 preflight modules:'), 1)
            self.assertEqual(result.stderr.count('test_real_spawn_two_processes_global_odd_cap ('), 1)
            (tests / 'test_core.py').write_text(
                'import unittest\nclass Checks(unittest.TestCase):\n'
                '    def test_failure(self): self.fail("expected generation failure")\n', encoding='utf-8')
            result = run()
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn('expected generation failure', result.stderr)
            (tests / 'test_core.py').write_text('import missing_stage3_dependency\n', encoding='utf-8')
            result = run()
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn('missing_stage3_dependency', result.stderr)


if __name__ == '__main__':
    unittest.main()
