import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock
from zipfile import ZipFile

from legalqa.adaptive_inputs import resolve_adaptive_input


def archive(path, member):
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, 'w') as z:
        z.writestr(member, '{}')
    return path


class AdaptiveDiscoveryTests(unittest.TestCase):
    def test_renamed_archives_found_by_contents_not_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage2 = archive(root / 'datasets/user/output/diagnostics (5).ZIP', 'stage2_manifest.json')
            baseline = archive(root / 'datasets/user/baseline/submission (2).zip', 'submission.json')
            archive(root / 'public.zip', 'stage3_manifest.json')
            (root / 'broken.zip').write_bytes(b'not a zip')
            self.assertEqual(resolve_adaptive_input(root, 'stage2'), stage2)
            self.assertEqual(resolve_adaptive_input(root, 'submission'), baseline)

    def test_unpacked_nested_input_and_explicit_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage2 = root / 'mounted/nested/stage2'
            stage2.mkdir(parents=True)
            manifest = stage2 / 'stage2_manifest.json'
            manifest.write_text('{}')
            baseline = root / 'mounted/submission.json'
            baseline.write_text('{}')
            self.assertEqual(resolve_adaptive_input(root, 'stage2'), stage2)
            self.assertEqual(resolve_adaptive_input(root, 'stage2', manifest), stage2)
            self.assertEqual(resolve_adaptive_input(root, 'stage2', root / 'mounted'), stage2)
            self.assertEqual(resolve_adaptive_input(root, 'submission'), baseline)

    def test_missing_stage2_reports_add_input_and_inventory_not_stage3_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            public = archive(root / 'old-public-stage3.zip', 'stage3_manifest.json')
            with self.assertRaises(FileNotFoundError) as caught:
                resolve_adaptive_input(root, 'stage2')
            message = str(caught.exception)
            for text in ('STAGE2_DIAGNOSTICS', 'Add Input', '(5).zip', 'Downloads', str(public)):
                self.assertIn(text, message)
            with self.assertRaisesRegex(ValueError, 'STAGE2_DIAGNOSTICS'):
                resolve_adaptive_input(root, 'stage2', public)

    def test_ambiguous_archives_require_explicit_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = archive(root / 'one.zip', 'stage2_manifest.json')
            second = archive(root / 'two.zip', 'stage2_manifest.json')
            with self.assertRaisesRegex(ValueError, 'STAGE2_DIAGNOSTICS') as caught:
                resolve_adaptive_input(root, 'stage2')
            self.assertIn(str(first), str(caught.exception))
            self.assertIn(str(second), str(caught.exception))
            self.assertEqual(resolve_adaptive_input(root, 'stage2', second), second)

    def test_zip_preferred_over_unpacked_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = archive(root / 'renamed.zip', 'stage2_manifest.json')
            (root / 'stage2_manifest.json').write_text('{}')
            self.assertEqual(resolve_adaptive_input(root, 'stage2'), path)

    def test_explicit_missing_path_and_missing_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, 'BASELINE_SUBMISSION'):
                resolve_adaptive_input(root, 'submission', root / 'missing.zip')
            with self.assertRaisesRegex(FileNotFoundError, '0.5713'):
                resolve_adaptive_input(root, 'submission')

    def test_notebook_missing_input_stops_before_pip(self):
        notebook = json.loads((Path(__file__).parents[1] / 'legalqa_main_04_repair_submit.ipynb').read_text(encoding='utf-8'))
        cells = {c['id']: ''.join(c['source']) for c in notebook['cells']}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / 'input'
            inputs.mkdir()
            command = Mock()
            namespace = {'WORK': root / 'working', 'INPUT': inputs, 'MODE': 'adaptive_dev',
                         'STAGE2_DIAGNOSTICS': None, 'BASELINE_SUBMISSION': None,
                         'sys': sys, 'run_bounded': command}
            exec(cells['s4bundle'], namespace)
            old_path = sys.path[:]
            try:
                with self.assertRaisesRegex(FileNotFoundError, 'STAGE2_DIAGNOSTICS'):
                    exec(cells['s4setup'], namespace)
            finally:
                sys.path[:] = old_path
            command.assert_not_called()


if __name__ == '__main__':
    unittest.main()
