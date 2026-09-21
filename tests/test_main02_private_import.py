import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from legalqa.data import prepare
from legalqa.generation import adapter_identity
from legalqa.io import config, digest, file_hash, load_questions, read_json, write_json
from legalqa.stages import Stage, finalize, verify_snapshot, verify_training
from legalqa.training import select_training_questions


class Main02PrivateImportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.version3 = self.root/'version3'
        self.dataset = self.root/'dataset'
        self.c = config()
        lock = {'models': {}}
        for role, model in self.c['models'].items():
            path = self.version3/'models'/role/'config.json'
            write_json(path, {'fixture': role})
            lock['models'][role] = {'model_id': model, 'revision': 'fixture', 'config_sha256': file_hash(path)}
        write_json(self.version3/'models/models.lock.json', lock)
        write_json(self.version3/'index/index_manifest.json', {
            'chunks': 407107, 'documents': 8507, 'identity': {'embedding': lock['models']['embedding']}})
        write_json(self.dataset/'train.json', {
            str(i): {'question': f'question {i}', 'answer': f'answer {i}'} for i in range(20)})
        write_json(self.dataset/'public-official.json', {'public': {'question': 'old public question'}})
        write_json(self.dataset/'private-official.json', {'private': {'question': 'new private question'}})
        self.source = self.stage('source', stage=1, commit='a'*40, code='old-code')
        prepare(self.dataset/'train.json', self.dataset/'public-official.json', self.source.data, self.c['seed'])
        qa = select_training_questions(load_questions(self.source.data/'train.json', answers=True), self.c)
        write_json(self.source.root/'sft/training_manifest.json', {
            'config': self.c, 'models': self.source.lock, 'code': 'old-code',
            'qa_hash': digest(qa), 'qa_ids': list(qa),
            'retrieval': {'index_hash': self.source.index_hash, 'mode': self.c['training']['retrieval_mode']}})
        for epoch in (1, 2):
            path = self.source.root/f'sft/epoch-{epoch:02d}'
            write_json(path/'adapter_config.json', {'r': 16})
            (path/'adapter_model.safetensors').write_bytes(f'weights-{epoch}'.encode())
            write_json(path/'trainer_state.json', {'epoch': epoch, 'global_step': epoch*10})
            write_json(path/'epoch_complete.json', {'step': epoch*10, 'adapter': adapter_identity(path)})
        write_json(self.source.root/'public.retrieval.json', {'old': 'must not import'})
        write_json(self.source.root/'selection.json', {'old': 'must not import'})
        write_json(self.source.root/'dev100.retrieval.json', {'old': 'must recompute with current code'})
        self.source.progress(complete=True, phase='training_complete')
        finalize(self.source.root, 'ok')

    def stage(self, name, stage=2, commit='b'*40, code='new-code', **extra):
        options = dict(stage=stage, root=str(self.root/name/'run'), version3=str(self.version3),
                       dataset=str(self.dataset), **extra)
        def copy_link(link, target, target_is_directory=False):
            shutil.copytree(target, link)
        with patch.object(Path, 'symlink_to', copy_link), \
             patch('legalqa.stages.subprocess.check_output', return_value=commit), \
             patch('legalqa.stages.source_hash', return_value=code):
            return Stage(options)

    def import_private(self, name='private', **extra):
        return self.stage(name, upstream=str(self.source.root), import_stage1_private=True, **extra)

    def test_import_preserves_training_and_dev_replaces_only_test_and_resumes(self):
        original = read_json(self.source.root/'stage1_manifest.json')
        second = self.import_private()
        for relative in ('data/train.json', 'data/dev100.questions.json', 'data/dev100.references.json',
                         'sft/training_manifest.json', 'sft/epoch-01/adapter_model.safetensors',
                         'sft/epoch-02/adapter_model.safetensors'):
            self.assertEqual(file_hash(second.root/relative), file_hash(self.source.root/relative))
        for relative in ('public.retrieval.json', 'selection.json', 'dev100.retrieval.json'):
            self.assertFalse((second.root/relative).exists())
        self.assertEqual(load_questions(second.data/'test.questions.json'),
                         load_questions(self.dataset/'private-official.json'))
        split = read_json(self.source.data/'split_manifest.json')
        self.assertEqual(read_json(second.data/'split_manifest.json'), {
            **split, 'test_sha256': file_hash(self.dataset/'private-official.json')})
        self.assertEqual(second.identity['code_commit'], 'b'*40)
        self.assertEqual(second.identity['training_origin_code'], 'old-code')
        self.assertEqual(second.identity['stage1_private_import']['code_commit'], 'a'*40)
        verify_training(second.root/'sft', self.c, second.data, second.lock, second.index_hash, 'old-code')
        self.assertEqual(verify_snapshot(self.source.root, 1), original)
        second.progress(complete=False, phase='dev_retrieval')
        finalize(second.root)
        for name, extra in (('resume_alone', {}), ('resume_upstream', {
                'upstream': str(self.source.root), 'import_stage1_private': True})):
            resumed = self.stage(name, previous=str(second.root), **extra)
            self.assertEqual(resumed.identity, second.identity)

    def test_cross_commit_restore_without_import_still_fails(self):
        with self.assertRaisesRegex(ValueError, 'provenance mismatch: code_commit'):
            self.stage('strict', upstream=str(self.source.root))

    def test_import_rejects_partial_training(self):
        finalize(self.source.root, 'paused')
        with self.assertRaisesRegex(ValueError, 'partial'):
            self.import_private()

    def test_import_rejects_changed_training_dataset(self):
        write_json(self.dataset/'train.json', {'different': {'question': 'q', 'answer': 'a'}})
        with self.assertRaisesRegex(ValueError, 'training dataset differs'):
            self.import_private()

    def test_import_rejects_incompatible_config_models_and_index(self):
        session = read_json(self.source.root/'session.json')
        for key in ('config_hash', 'models', 'index_hash'):
            with self.subTest(key=key):
                write_json(self.source.root/'session.json', {**session, key: 'changed'})
                finalize(self.source.root, 'ok')
                with self.assertRaisesRegex(ValueError, f'provenance mismatch: {key}'):
                    self.import_private(name=key)

    def test_import_rejects_tampered_adapter(self):
        (self.source.root/'sft/epoch-01/adapter_model.safetensors').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'missing/changed'):
            self.import_private()

    def test_import_rejects_changed_training_origin(self):
        path = self.source.root/'sft/training_manifest.json'
        write_json(path, {**read_json(path), 'code': 'unrelated-code'})
        finalize(self.source.root, 'ok')
        with self.assertRaisesRegex(ValueError, 'Training config/model/code differs'):
            self.import_private()

    def test_resume_rejects_changed_private_questions(self):
        second = self.import_private()
        finalize(second.root)
        write_json(self.dataset/'private-official.json', {'other': {'question': 'changed question'}})
        with self.assertRaisesRegex(ValueError, 'Snapshot test questions differ'):
            self.stage('resume', previous=str(second.root))


if __name__ == '__main__':
    unittest.main()
