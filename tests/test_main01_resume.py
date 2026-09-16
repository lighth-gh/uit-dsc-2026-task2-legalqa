import ast
import json
import shutil
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from zipfile import ZipFile

from legalqa.io import ROOT, config, digest, file_hash, read_json, source_hash, write_json
from legalqa.memory_guard import available_ram_mb
from legalqa.stages import Stage, RESUME_FILES, finalize
from legalqa.training import select_training_questions


class Main01ResumeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.input = self.root/'input'
        self.input.mkdir()
        notebook = read_json(ROOT/'legalqa_main_01_qlora_train.ipynb')
        self.setup = ''.join(notebook['cells'][3]['source'])
        self.version3 = self.input/'renamed'/'nested'/'version3'
        self.c, self.lock = config(), {'models':{}}
        for role, model in self.c['models'].items():
            path = self.version3/'models'/role/'config.json'
            write_json(path, {'fixture':role})
            (path.parent/'model.safetensors').write_bytes(b'weights')
            self.lock['models'][role] = {'model_id':model, 'revision':'fixture', 'config_sha256':file_hash(path)}
        write_json(self.version3/'models/models.lock.json', self.lock)
        self.index = {'chunks':407107, 'documents':8507, 'identity':{'embedding':self.lock['models']['embedding']}}
        write_json(self.version3/'index/index_manifest.json', self.index)
        (self.version3/'index/corpus.sqlite').touch()

    def discover(self, **extra):
        ns = dict(Path=Path, json=json, WORK_HOURS=9, SESSION_STARTED=0, INPUT=self.input,
                  STAGE=1, INPUT_MODE='auto', PREVIOUS_OUTPUT=None, UPSTREAM_OUTPUT=None,
                  LEGACY_INPUT_ROOT=None, RETRIEVAL_INPUT=None, VERSION3_ROOT=None,
                  DATASET_ROOT=None, REPO_REVISION=None)
        ns.update(extra)
        exec(self.setup[:self.setup.index('def available_ram_mb')], ns)
        return ns

    def dataset(self):
        data = self.input/'new-dataset-name'/'inner'
        write_json(data/'train.json', {'train':'fixture'})
        write_json(data/'public-official.json', {'public':'fixture'})
        return data

    def test_discovery_new_run_and_missing_or_ambiguous_files(self):
        with self.assertRaisesRegex(FileNotFoundError, 'dataset BTC'):
            self.discover()
        dataset = self.dataset()
        found = self.discover()
        self.assertEqual(found['DATASET_ROOT'], dataset)
        self.assertEqual(found['VERSION3_ROOT'], self.version3)
        self.assertEqual(found['PIN'], 'main')
        shutil.copytree(dataset, self.input/'second')
        with self.assertRaisesRegex(RuntimeError, 'dataset BTC'):
            self.discover()
        self.assertEqual(self.discover(DATASET_ROOT=dataset.parent)['DATASET_ROOT'], dataset)

    def test_discovery_resume_without_original_dataset_pins_commit(self):
        previous = self.input/'notebook-output'/'nested'/'run'
        write_json(previous/'stage1_manifest.json', {'schema':2, 'code_commit':'a'*40})
        write_json(previous/'data/split_manifest.json', {})
        for extra in ({}, {'PREVIOUS_OUTPUT':previous.parent}, {'INPUT_MODE':'resume'}):
            ns = self.discover(**extra)
            self.assertEqual(ns['PREVIOUS_OUTPUT'], previous)
            self.assertEqual(ns['PIN'], 'a'*40)
            self.assertIsNone(ns['DATASET_ROOT'])
        with self.assertRaisesRegex(ValueError, 'commit'):
            self.discover(REPO_REVISION='main')
        other = self.input/'different-output'
        write_json(other/'stage1_manifest.json', {'schema':2, 'code_commit':'b'*40})
        with self.assertRaisesRegex(RuntimeError, 'Stage 1'):
            self.discover()

    def test_zip_cache_discovery_and_explicit_fresh_ignores_old_output(self):
        self.dataset()
        with ZipFile(self.input/'arbitrary-name.zip', 'w') as z:
            z.writestr('stage1_manifest.json', '{}')
            z.writestr('train.sft.lexical.retrieval.json', '{}')
        ns = self.discover()
        self.assertEqual(ns['RETRIEVAL_INPUT'], self.input/'arbitrary-name.zip')
        self.assertIsNone(ns['PREVIOUS_OUTPUT'])
        self.assertEqual(ns['PIN'], 'main')
        self.assertIsNone(self.discover(INPUT_MODE='fresh')['RETRIEVAL_INPUT'])
        with self.assertRaises(FileNotFoundError):
            self.discover(INPUT_MODE='resume')

    def test_full_output_restores_cache_and_latest_complete_ddp_checkpoint(self):
        def link_fixture(link, target, target_is_directory=False):
            shutil.copytree(target, link)
        def options(name, **kw):
            return {'stage':1, 'root':str(self.root/name/'run'), 'version3':str(self.version3), **kw}
        with patch.object(Path, 'symlink_to', link_fixture):
            first = Stage(options('first'))
            qa = {'1':{'question':'one', 'answer':'first answer'}, '2':{'question':'two', 'answer':'second answer'}}
            write_json(first.data/'train.json', qa)
            write_json(first.data/'dev100.questions.json', {'dev':{'question':'independent'}})
            write_json(first.data/'split_manifest.json', {'seed':2026})
            selected = select_training_questions(qa, self.c)
            identity = {'index_hash':digest(self.index), 'mode':'lexical', 'code':source_hash(),
                        'models':self.lock, 'retrieval':self.c['retrieval'],
                        'mode_config':self.c['training']['lexical_pool_k'],
                        'questions_hash':digest({k:{'question':v['question']} for k,v in selected.items()})}
            write_json(first.root/'sft/training_manifest.json', {'config':self.c, 'models':self.lock,
                'code':source_hash(), 'qa_hash':digest(selected), 'qa_ids':list(selected), 'retrieval':identity})
            cache = first.root/'train.sft.lexical.retrieval.json'
            write_json(cache, {'identity':identity, 'records':{k:{'question':v['question'],
                'contexts':[{'parent_id':k,'text':'Original BM25 context'}]} for k,v in selected.items()}})
            for step in (10, 20):
                checkpoint = first.root/f'sft/checkpoint-{step}'
                checkpoint.mkdir()
                for name in RESUME_FILES | {'rng_state_0.pth', 'rng_state_1.pth'}:
                    (checkpoint/name).write_bytes(b'fixture')
                write_json(checkpoint/'trainer_state.json', {'global_step':step, 'epoch':.25})
            write_json(first.root/'sft/checkpoint-30/trainer_state.json', {'global_step':30, 'epoch':.5})
            first.progress(complete=False, phase='training_paused')
            finalize(first.root, 'ok')
            source_hash_before = file_hash(cache)
            second = Stage(options('second', previous=str(first.root), dataset=None))
            with patch.object(second, 'command', return_value=True) as command, \
                 patch('legalqa.stages.should_pause', return_value=False):
                second.train()
            command.assert_called_once()
            args = command.call_args.args
            self.assertEqual(args[0], 'fit')
            self.assertEqual(args[args.index('--resume')+1], second.root/'sft/checkpoint-20')
            self.assertEqual(file_hash(second.root/cache.name), source_hash_before)
            self.assertEqual(file_hash(cache), source_hash_before)
            self.assertFalse((second.root/'sft/checkpoint-30').exists())
            from legalqa.training_cache import training_retrieval
            rows, restored_identity = training_retrieval(second.root/cache.name, selected, self.c, second.models)
            self.assertEqual(set(dict(rows)), set(selected))
            self.assertEqual(restored_identity, identity)

    def test_ram_guard_uses_container_headroom(self):
        proc = self.root/'meminfo'
        proc.write_text('MemTotal: 64000000 kB\nMemAvailable: 32000000 kB\n')
        group = self.root/'cgroup'
        group.mkdir()
        (group/'memory.max').write_text(str(8*1024**3))
        (group/'memory.current').write_text(str(7*1024**3))
        self.assertEqual(available_ram_mb(proc, group), 1024)
        (group/'memory.max').write_text('max')
        self.assertEqual(available_ram_mb(proc, group), 32000000/1024)

    def test_supervisor_stops_group_on_deadline_or_low_ram_then_can_export(self):
        declarations = [n for n in ast.parse(self.setup).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        for low_ram in (False, True):
            with self.subTest(low_ram=low_ram):
                clock_values = iter([95, 95, 95, 101])
                process = Mock(pid=42)
                process.wait.side_effect = [subprocess.TimeoutExpired('worker', 2), 0, 0] if not low_ram else [0, 0]
                kill = Mock()
                ns = dict(Path=Path, WORK_END=100, MIN_FREE_RAM_MB=3072,
                    time=types.SimpleNamespace(monotonic=lambda:next(clock_values)),
                    os=types.SimpleNamespace(killpg=kill), signal=types.SimpleNamespace(SIGTERM=15, SIGKILL=9),
                    subprocess=types.SimpleNamespace(Popen=Mock(return_value=process),
                        TimeoutExpired=subprocess.TimeoutExpired, CalledProcessError=subprocess.CalledProcessError))
                exec(compile(ast.Module(body=declarations, type_ignores=[]), '<main01>', 'exec'), ns)
                ns['available_ram_mb'] = lambda:1024 if low_ram else 10000
                with self.assertRaises(ns['BudgetPause']):
                    ns['bounded_process'](['worker'], seconds=600)
                self.assertEqual([call.args for call in kill.call_args_list], [(42,15), (42,9)])
                self.assertTrue(ns['subprocess'].Popen.call_args.kwargs['start_new_session'])


if __name__ == '__main__':
    unittest.main()
