import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from legalqa.io import config, digest, file_hash, read_json, source_hash, write_json
from legalqa.retrieval_import import CACHE_NAME, LEGACY_CODE, compatible_code, import_training_retrieval


class RetrievalImportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root/'source'
        self.output = self.root/'new'/CACHE_NAME
        self.c, self.lock = config(), {'models':{'fixture':'revision'}}
        self.questions = {'1':{'question':'question one'}, '2':{'question':'question two'}}
        self.payload = {'identity':{'questions_hash':digest(self.questions), 'models':self.lock,
            'index_hash':'index', 'retrieval':self.c['retrieval'], 'mode':'lexical',
            'mode_config':self.c['training']['lexical_pool_k'], 'code':LEGACY_CODE},
            'records':{key:{'question':row['question'], 'contexts':[{'parent_id':key,'text':'original context'}],
                'stages':{'bm25':[3,2,1]}, 'seconds':{'bm25':2.7}} for key,row in self.questions.items()}}
        self.publish()

    def publish(self):
        write_json(self.source/CACHE_NAME, self.payload)
        path = self.source/CACHE_NAME
        write_json(self.source/'stage1_manifest.json', {'schema':2, 'stage':1, 'status':'failed',
            'source_hash':self.payload['identity']['code'], 'code_commit':'old-commit',
            'files':{CACHE_NAME:{'size':path.stat().st_size, 'sha256':file_hash(path)}}})

    def run_import(self, source=None):
        return import_training_retrieval(source or self.source, self.output, self.questions,
                                         self.c, self.lock, 'index')

    def test_failed_training_snapshot_preserves_complete_cache_and_provenance(self):
        before = file_hash(self.source/CACHE_NAME)
        note = self.run_import()
        imported = read_json(self.output)
        self.assertEqual(imported['records'], self.payload['records'])
        self.assertEqual(imported['identity']['code'], source_hash())
        self.assertEqual(note['original_identity'], self.payload['identity'])
        self.assertEqual(note['records'], 2)
        self.assertEqual(file_hash(self.source/CACHE_NAME), before)
        self.assertEqual(list(self.output.parent.iterdir()), [self.output])
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            self.run_import()

    def test_zip_input_and_direct_cache_path(self):
        archive = self.root/'diagnostics.zip'
        with ZipFile(archive, 'w') as z:
            for name in (CACHE_NAME, 'stage1_manifest.json'):
                z.write(self.source/name, name)
        self.run_import(archive)
        self.assertEqual(read_json(self.output)['records'], self.payload['records'])
        self.output = self.root/'other'/CACHE_NAME
        self.run_import(self.source/CACHE_NAME)

    def test_rejects_changed_settings_models_index_mode_questions_and_unknown_code(self):
        original = copy.deepcopy(self.payload)
        changes = [('models', {}), ('index_hash','other'), ('mode','full'), ('mode_config',5),
                   ('retrieval',{}), ('questions_hash','different'), ('code','unknown')]
        for key, value in changes:
            with self.subTest(key=key):
                self.payload = copy.deepcopy(original)
                self.payload['identity'][key] = value
                self.publish()
                with self.assertRaises(ValueError):
                    self.run_import()
                self.assertFalse(self.output.exists())

    def test_rejects_incomplete_ids_altered_question_or_modified_artifact(self):
        original = copy.deepcopy(self.payload)
        self.payload['records'].pop('2')
        self.publish()
        with self.assertRaisesRegex(ValueError, 'exactly'):
            self.run_import()
        self.payload = copy.deepcopy(original)
        self.payload['records']['1']['question'] = 'different question'
        self.publish()
        with self.assertRaisesRegex(ValueError, 'text differs'):
            self.run_import()
        (self.source/CACHE_NAME).write_text('{}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'checksum/size'):
            self.run_import()

    def test_legacy_cache_rejected_if_retrieval_implementation_changes(self):
        self.assertTrue(compatible_code(LEGACY_CODE))
        with patch.dict('legalqa.retrieval_import.LEGACY_DEPENDENCIES', {'retrieval.py':'different'}):
            with self.assertRaisesRegex(ValueError, 'compatibility'):
                self.run_import()

    def test_notebook_import_skips_discovery_of_old_resume_commit(self):
        from legalqa.io import ROOT
        nb = read_json(ROOT/'legalqa_main_01_qlora_train.ipynb')
        setup = next(''.join(cell['source']) for cell in nb['cells'] if cell.get('id') == 's1c3')
        prefix = setup[:setup.index('pins = []')]
        ns = dict(Path=Path, WORK_HOURS=9, SESSION_STARTED=0, INPUT=self.root, STAGE=1,
                  RETRIEVAL_INPUT=self.source, PREVIOUS_OUTPUT=None, UPSTREAM_OUTPUT=None, LEGACY_INPUT_ROOT=None)
        exec(prefix, ns)
        self.assertIsNone(ns['PREVIOUS_OUTPUT'])
        ns['RETRIEVAL_INPUT'] = None
        exec(prefix, ns)
        self.assertEqual(ns['PREVIOUS_OUTPUT'], self.source)

    def test_stage_imports_cache_then_launches_fit_without_retrieval(self):
        from legalqa.stages import Stage
        stage = Stage.__new__(Stage)
        stage.root, stage.data = self.output.parent, self.output.parent/'data'
        stage.o, stage.c = {'retrieval_input':str(self.source)}, self.c
        stage.lock, stage.index_hash, stage.identity, stage.legacy = self.lock, 'index', {}, None
        write_json(stage.data/'split_manifest.json', {})
        write_json(stage.data/'train.json', {k:{**q,'answer':'reference answer'} for k,q in self.questions.items()})
        with patch.object(stage, 'command', return_value=True) as command, \
             patch('legalqa.stages.should_pause', return_value=False):
            stage.train()
        self.assertEqual(command.call_count, 1)
        self.assertEqual(command.call_args.args[0], 'fit')
        self.assertEqual(read_json(self.output)['records'], self.payload['records'])
        self.assertFalse((stage.root/'sft').exists())


if __name__ == '__main__':
    unittest.main()
