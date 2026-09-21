import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Main02InputTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.input = Path(temporary.name)
        self.notebook = json.loads(
            (ROOT / 'legalqa_main_02_select_retrieve.ipynb').read_text(encoding='utf-8')
        )
        config = ''.join(self.notebook['cells'][1]['source'])
        self.discovery = config[config.index('if UPSTREAM_OUTPUT is None:'):]

    def artifact(self, relative):
        root = self.input / relative / 'legalqa_main_stage1_v8'
        root.mkdir(parents=True)
        (root / 'stage1_manifest.json').write_text('{}', encoding='utf-8')
        return root

    def discover(self, override=None):
        namespace = dict(Path=Path, INPUT=self.input, UPSTREAM_OUTPUT=override)
        exec(self.discovery, namespace)
        return namespace['UPSTREAM_OUTPUT']

    def test_display_name_and_mount_slug_can_differ(self):
        for layout in ('legalqa_main_01', 'legalqa-main-01', 'legalqamain01',
                       'datasets/lighth/renamed-export/versions/3/output'):
            with self.subTest(layout=layout):
                expected = self.artifact(layout)
                self.assertEqual(self.discover(), expected)
                (expected / 'stage1_manifest.json').unlink()

    def test_notebook_and_model_mounts_are_not_dataset_sources(self):
        for namespace in ('notebooks', 'kernels', 'models', 'competitions'):
            self.artifact(f'{namespace}/lighth/legalqa-main-01')
        with self.assertRaises(RuntimeError):
            self.discover()
        expected = self.artifact('datasets/lighth/renamed-dataset')
        self.assertEqual(self.discover(), expected)

    def test_missing_artifact_reports_searched_mount(self):
        mount = self.input / 'actual-dataset-slug'
        mount.mkdir()
        with self.assertRaises(RuntimeError) as caught:
            self.discover()
        self.assertIn(mount.as_posix(), str(caught.exception))
        self.assertIn('stage1_manifest.json', str(caught.exception))

    def test_multiple_sources_require_explicit_override(self):
        first = self.artifact('datasets/owner/first')
        second = self.artifact('datasets/owner/second')
        with self.assertRaises(RuntimeError) as caught:
            self.discover()
        self.assertIn(first.as_posix(), str(caught.exception))
        self.assertIn(second.as_posix(), str(caught.exception))
        self.assertEqual(self.discover(second), second)

    def test_explicit_override_requires_manifest(self):
        with self.assertRaises(FileNotFoundError):
            self.discover(self.input / 'missing')

    def test_notebook_cells_compile(self):
        for index, cell in enumerate(self.notebook['cells']):
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), f'main02:cell{index}', 'exec')


if __name__ == '__main__':
    unittest.main()
