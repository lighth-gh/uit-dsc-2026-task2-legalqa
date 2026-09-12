import ast
import json
import os
import shutil
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from legalqa.io import ROOT, Journal, config, digest, file_hash, read_json, source_hash, write_json
from legalqa.generation import adapter_identity, generate
from legalqa.retrieval import read_retrieval
from legalqa.retrieval import retrieve
from legalqa.runtime import should_pause
from legalqa.stages import (Stage, finalize, verify_snapshot, copy_artifacts,
                            validate_selection, valid_resume, harvest_epochs, epochs, verify_training,
                            generation_ready, copy_file)


class StageTests(unittest.TestCase):
    def notebook(self, stage):
        names = {1:'legalqa_main_01_qlora_train.ipynb',2:'legalqa_main_02_select_retrieve.ipynb',
                 3:'legalqa_main_03_generate_submit.ipynb'}
        return read_json(ROOT/names[stage])

    def test_notebook_supervisor_kills_group_and_honors_remaining_budget(self):
        for stage in (1,2,3):
            nb = self.notebook(stage)
            code = next(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code'
                        and 'def bounded_process' in ''.join(c['source']))
            declarations = [n for n in ast.parse(code).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
            process = Mock(pid=42)
            process.wait.side_effect = [subprocess.TimeoutExpired('worker', 5), 0, 0]
            popen = Mock(return_value=process)
            kill = Mock()
            namespace = {'time':types.SimpleNamespace(monotonic=lambda:95),'WORK_END':100,
                         'os':types.SimpleNamespace(killpg=kill),
                         'signal':types.SimpleNamespace(SIGTERM=15,SIGKILL=9),
                         'subprocess':types.SimpleNamespace(Popen=popen,TimeoutExpired=subprocess.TimeoutExpired,
                                                          CalledProcessError=subprocess.CalledProcessError)}
            exec(compile(ast.Module(body=declarations,type_ignores=[]),'<supervisor>','exec'),namespace)
            with self.assertRaises(namespace['BudgetPause']):
                namespace['bounded_process'](['worker'],seconds=600)
            self.assertEqual(process.wait.call_args_list[0].kwargs['timeout'],5)
            self.assertEqual([c.args for c in kill.call_args_list],[(42,15),(42,9)])
            self.assertTrue(popen.call_args.kwargs['start_new_session'])
            namespace['WORK_END']=90
            with self.assertRaises(namespace['BudgetPause']):
                namespace['bounded_process'](['never-start'])
            self.assertEqual(popen.call_count,1)

    def test_notebooks_derive_pin_from_input_not_moving_main(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            write_json(root/'stage1_manifest.json',{'schema':2,'code_commit':'a'*40})
            nb=self.notebook(2)
            code=next(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code'
                      and 'def bounded_process' in ''.join(c['source']))
            block=code[:code.index('class BudgetPause')]
            ns={'WORK_HOURS':9,'SESSION_STARTED':0,'STAGE':2,'INPUT':root,
                'PREVIOUS_OUTPUT':None,'UPSTREAM_OUTPUT':root,'LEGACY_INPUT_ROOT':None,
                'REPO_REVISION':None,'Path':Path,'json':json}
            exec(block,ns)
            self.assertEqual(ns['PIN'],'a'*40)
            ns['WORK_HOURS']=12
            with self.assertRaises(ValueError):exec(block,ns)

    def test_three_notebooks_share_supervisor_setup_and_runner(self):
        codes=[[''.join(c['source']) for c in self.notebook(i)['cells'] if c['cell_type']=='code']
               for i in (1,2,3)]
        for position in range(1,len(codes[0])):
            self.assertEqual(codes[0][position],codes[1][position])
            self.assertEqual(codes[0][position],codes[2][position])
        for stage,cells in enumerate(codes,1):
            self.assertIn('WORK_HOURS = 9.0',cells[0])
            self.assertIn('EXPORT_SECONDS = 600',cells[0])
            self.assertIn('MAX_NEW_QUESTIONS = 200',cells[0])
            self.assertIn(f'STAGE = {stage}',cells[0])

    def test_cooperative_budget_and_item_cap(self):
        with patch.dict(os.environ,{'LEGALQA_DEADLINE':'1000','LEGALQA_MAX_ITEMS':'200'}), patch('legalqa.runtime.time.time',return_value=500):
            self.assertFalse(should_pause(199))
            self.assertTrue(should_pause(200))
        with patch.dict(os.environ,{'LEGALQA_DEADLINE':'1000','LEGALQA_MAX_ITEMS':'0'}), patch('legalqa.runtime.time.time',return_value=821):
            self.assertTrue(should_pause())

    def test_stage_inputs_resume_and_advance_with_portable_provenance(self):
        # Run the real Stage initializer/restore with tiny model files. On Windows
        # replace directory symlinks with fixture copies (no elevated privileges).
        from legalqa.io import file_hash
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);version3=root/'version3';c=config();lock={'models':{}}
            for role,model in c['models'].items():
                path=version3/'models'/role/'config.json'
                write_json(path,{'fixture':role})
                lock['models'][role]={'model_id':model,'revision':'fixture','config_sha256':file_hash(path)}
            write_json(version3/'models/models.lock.json',lock)
            write_json(version3/'index/index_manifest.json',{'chunks':407107,'documents':8507,
                         'identity':{'embedding':lock['models']['embedding']}})
            def create_link(link,target,target_is_directory=False):
                shutil.copytree(target,link)
            def options(stage,name,**extra):
                return {'stage':stage,'root':str(root/name/'run'),'version3':str(version3),**extra}
            with patch.object(Path,'symlink_to',create_link):
                first=Stage(options(1,'first'))
                write_json(first.data/'split_manifest.json',{'seed':42})
                write_json(first.root/'sft/fixture.json',{'train':'retained'})
                first.progress(complete=False,phase='training_paused')
                finalize(first.root,'ok')
                with self.assertRaisesRegex(ValueError,'partial'):
                    Stage(options(2,'too_early',upstream=str(first.root)))
                first.progress(complete=True,phase='training_complete')
                finalize(first.root,'ok')
                second=Stage(options(2,'second',upstream=str(first.root)))
                self.assertTrue((second.root/'sft/fixture.json').is_file())
                self.assertEqual(read_json(second.data/'split_manifest.json'),{'seed':42})
                second.progress(complete=False,phase='public_retrieval')
                finalize(second.root,'ok')
                resumed=Stage(options(2,'resumed',previous=str(second.root)))
                self.assertEqual(resumed.identity,second.identity)
                resumed.progress(complete=True,phase='public_ready')
                finalize(resumed.root,'ok')
                third=Stage(options(3,'third',upstream=str(resumed.root)))
                self.assertFalse((third.root/'sft').exists())
                self.assertTrue((third.data/'split_manifest.json').exists())

    def test_generation_bundle_requires_audit_and_manifest_after_interruption(self):
        with tempfile.TemporaryDirectory() as folder:
            pred=Path(folder)/'submission.json';qa={'1':{'answer':'one'}}
            write_json(pred,qa)
            self.assertFalse(generation_ready(pred))
            write_json(pred.with_suffix('.manifest.json'),{'prediction_hash':digest(qa)})
            self.assertFalse(generation_ready(pred))
            write_json(pred.with_suffix('.audit.json'),{'1':{}})
            self.assertTrue(generation_ready(pred))
            write_json(pred.with_suffix('.audit.json'),{})
            with self.assertRaisesRegex(ValueError,'audit'):generation_ready(pred)

    def test_interrupted_data_prepare_can_resume_and_rejects_changed_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);dataset=root/'dataset';run=root/'run'
            original={str(i):{'question':f'question {i}','answer':f'answer {i}'} for i in range(20)}
            write_json(dataset/'train.json',original)
            write_json(dataset/'public-official.json',{'public':{'question':'test'}})
            write_json(run/'config.json',config());write_json(run/'models.lock.json',{})
            stage=Stage.__new__(Stage);stage.root=run;stage.data=run/'data';stage.c=config()
            stage.o={'dataset':str(dataset)};stage.identity={'stage':1,'code_commit':'a'*40,'source_hash':'code'}
            stage.legacy=None;stage.index=root/'index';stage.command=Mock(return_value=False)
            with patch('legalqa.data.prepare',side_effect=InterruptedError('prepare interrupted')):
                with self.assertRaises(InterruptedError):stage.train()
            finalize(run);verify_snapshot(run,1) # Partial stage 1 need not have a split yet.
            changed=json.loads(json.dumps(original));changed['0']['answer']='changed answer'
            write_json(dataset/'train.json',changed)
            with self.assertRaisesRegex(ValueError,'Dataset changed'):stage.train()
            write_json(dataset/'train.json',original)
            stage.train()
            self.assertTrue((stage.data/'split_manifest.json').exists())
            self.assertFalse(read_json(run/'progress.json')['complete'])

    def test_retrieval_journal_resumes_without_repeating_completed_queries(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);qa=root/'qa.json';write_json(qa,{str(i):{'question':str(i)} for i in range(3)})
            lock={'models':{'embedding':{}}};engine=Mock()
            engine.manifest={'identity':{'embedding':{}}}
            engine.retrieve_one_lexical.side_effect=lambda question:{'question':question,'contexts':[]}
            with patch('legalqa.retrieval.Retriever',return_value=engine), \
                 patch('legalqa.retrieval.model_lock',return_value=lock), \
                 patch.dict(os.environ,{'LEGALQA_MAX_ITEMS':'2','LEGALQA_DEADLINE':'0'}):
                cache=root/'first/cache.json'
                self.assertEqual(retrieve(config(),qa,'models','index',cache,'cpu','lexical')['status'],'paused')
                self.assertFalse(cache.exists())
                resumed=root/'resumed/cache.json';resumed.parent.mkdir()
                for suffix in ('.checkpoint.jsonl','.checkpoint.jsonl.meta.json'):
                    copy_file(cache.with_suffix(suffix),resumed.with_suffix(suffix))
                self.assertEqual(retrieve(config(),qa,'models','index',resumed,'cpu','lexical')['records'],3)
                self.assertEqual(engine.retrieve_one_lexical.call_count,3)

    def test_stage3_packages_only_complete_prediction_bundle(self):
        from zipfile import ZipFile
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);adapter=root/'selected_adapter'
            write_json(adapter/'adapter_config.json',{'r':16});(adapter/'adapter_model.safetensors').write_bytes(b'weights')
            write_json(root/'selection.json',{'meteor':.5,'prediction_manifest':{'identity':{'adapter':adapter_identity(adapter)}}})
            write_json(root/'data/test.questions.json',{'1':{'question':'one'},'2':{'question':'two'}})
            write_json(root/'public.retrieval.json',{})
            stage=Stage.__new__(Stage);stage.root=root;stage.data=root/'data';stage.c=config()
            stage.o={'max_new_questions':200};stage.command=Mock(return_value=True)
            pred=root/'submissions/public/submission.json';partial={'1':{'answer':'one'}}
            write_json(pred.with_suffix('.partial.json'),partial)
            stage.generate_submit()
            self.assertFalse(pred.with_suffix('.zip').exists())
            self.assertEqual(stage.command.call_args.kwargs['cap'],200)
            answers={**partial,'2':{'answer':'two'}}
            write_json(pred,answers);write_json(pred.with_suffix('.audit.json'),{'1':{},'2':{}})
            write_json(pred.with_suffix('.manifest.json'),{'prediction_hash':digest(answers)})
            stage.generate_submit()
            self.assertTrue(read_json(root/'progress.json')['complete'])
            with ZipFile(pred.with_suffix('.zip')) as archive:
                self.assertEqual(archive.namelist(),[config()['submission_filename']])
                self.assertEqual(json.loads(archive.read(archive.namelist()[0])),answers)

    def test_interrupted_copy_preserves_destination_and_can_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source';destination=root/'destination'
            source.write_bytes(b'complete new contents');destination.write_bytes(b'old contents')
            def interrupted(src,dst):
                Path(dst).write_bytes(b'partial')
                raise InterruptedError('simulated interruption during file copy')
            with patch('legalqa.io.shutil.copy2',side_effect=interrupted):
                with self.assertRaises(InterruptedError):copy_file(source,destination)
            self.assertEqual(destination.read_bytes(),b'old contents')
            copy_file(source,destination)
            self.assertEqual(destination.read_bytes(),source.read_bytes())

    def test_restore_commit_marker_is_written_only_after_all_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source';dest=root/'dest'
            session={'stage':1,'code_commit':'a'*40,'source_hash':'code'}
            write_json(source/'session.json',session);write_json(source/'config.json',config())
            write_json(source/'models.lock.json',{});write_json(source/'data/split_manifest.json',{})
            write_json(source/'zz_last.json',{'last':'artifact'})
            finalize(source)
            stage=Stage.__new__(Stage);stage.root=dest;stage.previous=source;stage.upstream=None
            stage.number=1;stage.identity=session.copy()
            def interrupted(src,dst):
                if Path(src).name=='zz_last.json':raise InterruptedError('copy paused')
                return copy_file(src,dst)
            with patch('legalqa.stages.copy_file',side_effect=interrupted):
                with self.assertRaises(InterruptedError):stage.restore()
            self.assertFalse((dest/'session.json').exists())
            stage.restore()
            self.assertEqual(read_json(dest/'session.json'),session)
            self.assertTrue((dest/'zz_last.json').exists())

    def test_legacy_import_interruption_resumes_without_retraining(self):
        from legalqa.training import select_training_questions
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);old=root/'old';run=root/'run';c=config()
            qa={'1':{'question':'train','answer':'answer'}}
            write_json(run/'data/train.json',qa)
            write_json(run/'data/dev100.questions.json',{'2':{'question':'dev'}})
            write_json(run/'data/split_manifest.json',{'split':'fixture'})
            write_json(old/'data_public/split_manifest.json',{'split':'fixture'})
            write_json(old/'config.json',c);write_json(run/'config.json',c)
            write_json(run/'models.lock.json',{})
            selected=select_training_questions(qa,c)
            manifest={'config':c,'models':{},'code':'original-code','qa_hash':digest(selected),
                      'qa_ids':list(selected),'retrieval':{'index_hash':'index','mode':'lexical'}}
            write_json(old/'sft/training_manifest.json',manifest)
            write_json(old/'sft/training_result.json',{'epochs':2})
            for name in ('checkpoint-96','checkpoint-192','adapter_last'):
                path=old/'sft'/name
                write_json(path/'adapter_config.json',{'r':16})
                (path/'adapter_model.safetensors').write_bytes(name.encode())
                if name.startswith('checkpoint'):
                    step=int(name.split('-')[-1])
                    write_json(path/'trainer_state.json',{'global_step':step,'epoch':step/96})
            def stage_at(path):
                stage=Stage.__new__(Stage);stage.root=path;stage.data=path/'data'
                stage.legacy=None;stage.c=c;stage.lock={};stage.index_hash='index';stage.code='new-code'
                stage.identity={'stage':1,'code_commit':'a'*40,'source_hash':'new-code'}
                return stage
            stage=stage_at(run);stage.legacy=old
            def interrupted(src,dst):
                if Path(src).name=='adapter_model.safetensors':raise InterruptedError('legacy copy paused')
                return copy_file(src,dst)
            with patch('legalqa.stages.copy_file',side_effect=interrupted):
                with self.assertRaises((InterruptedError,shutil.Error)):stage.import_legacy()
            self.assertFalse((run/'sft').exists())
            self.assertIn('pending_legacy',read_json(run/'session.json'))
            snap=finalize(run)
            self.assertFalse(any(p.startswith('legacy_import.tmp/') for p in snap['files']))
            resumed=root/'resumed';copy_artifacts(run,resumed,verify_snapshot(run,1))
            stage=stage_at(resumed);stage.identity=read_json(resumed/'session.json')
            stage.train() # Discovers pending import; no command()/GPU training required.
            self.assertTrue(read_json(resumed/'progress.json')['complete'])
            self.assertEqual(stage.identity['training_origin_code'],'original-code')
            self.assertNotIn('pending_legacy',stage.identity)
            self.assertEqual(len(epochs(resumed/'sft')),2)

    def test_training_callback_saves_once_and_keeps_completed_epoch(self):
        tree=ast.parse((ROOT/'legalqa/training.py').read_text(encoding='utf-8'))
        callback=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='BudgetCallback')
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder);checkpoint=output/'checkpoint-96'
            for name in ('adapter_config.json','trainer_state.json'):
                write_json(checkpoint/name,{'epoch':1.0,'global_step':96})
            (checkpoint/'adapter_model.safetensors').write_bytes(b'fixture')
            ns={'TrainerCallback':object,'should_pause':lambda:True,'output':output,
                'copy_file':copy_file,'file_hash':file_hash,'write_json':write_json}
            exec(compile(ast.Module(body=[callback],type_ignores=[]),'<callback>','exec'),ns)
            cb=ns['BudgetCallback']();state=types.SimpleNamespace(epoch=.5,global_step=50)
            control=types.SimpleNamespace(should_save=False,should_training_stop=False)
            cb.on_step_end(None,state,control)
            self.assertTrue(control.should_save and control.should_training_stop)
            cb.on_save(None,state,control)
            cb.on_epoch_end(None,state,control)
            self.assertFalse(control.should_save) # Do not overwrite the same checkpoint twice.
            self.assertFalse((output/'epoch-01').exists())
            state.epoch=1.0;state.global_step=96
            cb.on_epoch_end(None,state,control)
            self.assertTrue(control.should_save)
            cb.on_save(None,state,control)
            self.assertEqual(read_json(output/'epoch-01/epoch_complete.json')['step'],96)

    def test_external_training_resume_reads_checkpoint_parent_manifest(self):
        tree=ast.parse((ROOT/'legalqa/training.py').read_text(encoding='utf-8'))
        branch=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=='resume')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);manifest={'identity':'training'}
            write_json(root/'input/sft/training_manifest.json',manifest)
            ns={'resume':root/'input/sft/checkpoint-10','output':root/'working/new-sft',
                'manifest':manifest,'Path':Path,'read_json':read_json}
            exec(compile(ast.Module(body=[branch],type_ignores=[]),'<resume>','exec'),ns)
            ns['manifest']={'identity':'changed'}
            with self.assertRaisesRegex(ValueError,'fingerprint'):
                exec(compile(ast.Module(body=[branch],type_ignores=[]),'<resume>','exec'),ns)

    def test_cache_rejects_code_mode_index_and_lexical_config_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'cache.json'
            c=config(); lock={'models':'fixture'}; qa={'q':{'question':'fixture'}}
            good={'identity':{'code':source_hash(),'models':lock,'retrieval':c['retrieval'],
                             'mode':'full','index_hash':'index'},
                  'records':{'q':{'question':'fixture','contexts':[]}}}
            with patch('legalqa.retrieval.model_lock',return_value=lock):
                write_json(path,good)
                read_retrieval(path,qa,c,'models',expected_mode='full',expected_index_hash='index')
                for key,value in [('code','old-code'),('mode','lexical'),('index_hash','other-index')]:
                    wrong=json.loads(json.dumps(good));wrong['identity'][key]=value;write_json(path,wrong)
                    with self.assertRaises(ValueError):
                        read_retrieval(path,qa,c,'models',expected_mode='full',expected_index_hash='index')
                wrong['identity'].update(code=source_hash(),mode='lexical',mode_config=-1)
                write_json(path,wrong)
                with self.assertRaises(ValueError):read_retrieval(path,qa,c,'models',expected_mode='lexical')

    def test_generation_pause_copy_journal_resume_no_duplicate_ids(self):
        # Exercise the real generation loop in extractive mode without GPU dependencies.
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);qa=root/'qa.json';cache=root/'cache.json'
            questions={str(i):{'question':f'question {i}'} for i in range(3)}
            write_json(qa,questions);write_json(cache,{})
            records={k:{'contexts':[]} for k in questions}
            tokenizer=types.SimpleNamespace(from_pretrained=lambda *a,**k:object())
            deps={'torch':types.SimpleNamespace(),'transformers':types.SimpleNamespace(AutoTokenizer=tokenizer,set_seed=lambda _:None)}
            with patch.dict('sys.modules',deps), patch('legalqa.generation.read_retrieval',return_value=(records,{})), \
                 patch('legalqa.generation.model_lock',return_value={}), \
                 patch('legalqa.generation.pack_prompt',return_value=([1],[])), \
                 patch('legalqa.generation.extractive_fallback',return_value='safe answer') as fallback, \
                 patch.dict(os.environ,{'LEGALQA_MAX_ITEMS':'2','LEGALQA_DEADLINE':'0'}):
                first=root/'first'/'submission.json'
                result=generate(config(),qa,cache,'models',first,'cpu',mode='extractive')
                self.assertEqual(result['status'],'paused');self.assertFalse(first.exists())
                self.assertEqual(len(read_json(first.with_suffix('.partial.json'))),2)
                second=root/'second'/'submission.json';second.parent.mkdir()
                for suffix in ('.checkpoint.jsonl','.checkpoint.jsonl.meta.json'):
                    shutil.copy2(first.with_suffix(suffix),second.with_suffix(suffix))
                result=generate(config(),qa,cache,'models',second,'cpu',mode='extractive')
                self.assertEqual(result['samples'],3)
                self.assertEqual(set(read_json(second)),set(questions))
                self.assertEqual(fallback.call_count,3)

    def test_entity_guard_has_priority_over_truncation(self):
        tree=ast.parse((ROOT/'legalqa/generation.py').read_text(encoding='utf-8'))
        branch=next(n for n in ast.walk(tree) if isinstance(n,ast.If)
                    and ast.unparse(n.test)=="entity_conflict['conflict']")
        ns={'entity_conflict':{'conflict':True},'hit_limit':True,'invalid':False,
            'flags':{'refusal':False},'questions':{'q':{'question':'question'}},'key':'q',
            'packed':[],'refusal_support':{},'extractive_fallback':lambda *a:'safe evidence'}
        exec(compile(ast.Module(body=[branch],type_ignores=[]),'<guard>','exec'),ns)
        self.assertEqual(ns['route'],'entity_guard_fallback')
        self.assertEqual(ns['answer'],'safe evidence')

    def test_snapshot_partial_is_portable_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'source';destination=Path(folder)/'dest'
            write_json(root/'session.json',{'stage':3,'code_commit':'a'*40,'source_hash':'code'})
            write_json(root/'config.json',config());write_json(root/'models.lock.json',{})
            write_json(root/'data/split_manifest.json',{'seed':2026})
            write_json(root/'progress.json',{'complete':False,'phase':'generate'})
            Journal(root/'submission.checkpoint.jsonl',{'run':'identity'}).append('1',{'prediction':{'answer':'one'}})
            snap=finalize(root,'ok')
            self.assertEqual(snap['status'],'paused')
            verify_snapshot(root,3)
            copy_artifacts(root,destination,snap)
            self.assertEqual(len(Journal(destination/'submission.checkpoint.jsonl',{'run':'identity'}).records),1)
            write_json(root/'config.json',{'tampered':True})
            with self.assertRaisesRegex(ValueError,'missing/changed'):verify_snapshot(root,3)

    def test_timeout_cannot_publish_complete_from_old_progress(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            write_json(root/'session.json',{'stage':2,'code_commit':'abc','source_hash':'code'})
            write_json(root/'progress.json',{'complete':True})
            self.assertEqual(finalize(root,'paused')['status'],'paused')

    def test_interrupted_diagnostics_zip_does_not_destroy_previous_zip_or_snapshot(self):
        from zipfile import ZipFile
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            write_json(root/'session.json',{'stage':1,'code_commit':'a'*40,'source_hash':'code'})
            write_json(root/'config.json',config());write_json(root/'models.lock.json',{})
            finalize(root)
            zip_path=root/'legalqa_main_stage1_v8_diagnostics.zip';previous_hash=file_hash(zip_path)
            write_json(root/'progress.json',{'complete':False,'phase':'training_paused'})
            original_write=ZipFile.write
            def interrupted(archive,filename,*args,**kwargs):
                if Path(filename).name=='config.json':raise InterruptedError('zip interrupted')
                return original_write(archive,filename,*args,**kwargs)
            with patch.object(ZipFile,'write',interrupted):
                with self.assertRaises(InterruptedError):finalize(root)
            self.assertEqual(file_hash(zip_path),previous_hash)
            verify_snapshot(root,1) # The portable snapshot was published before compression.
            finalize(root)
            with ZipFile(zip_path) as archive:
                self.assertIsNone(archive.testzip())
                self.assertIn('progress.json',archive.namelist())

    def test_selection_rejects_another_adapter_same_architecture(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);adapter=root/'selected_adapter'
            write_json(adapter/'adapter_config.json',{'r':16})
            (adapter/'adapter_model.safetensors').write_bytes(b'first weights')
            write_json(root/'selection.json',{'prediction_manifest':{'identity':{'adapter':adapter_identity(adapter)}}})
            validate_selection(root)
            (adapter/'adapter_model.safetensors').write_bytes(b'different weights')
            with self.assertRaisesRegex(ValueError,'adapter bytes'):validate_selection(root)

    def test_epoch_adapter_survives_rotation_and_partial_epoch_not_selected(self):
        with tempfile.TemporaryDirectory() as folder:
            sft=Path(folder)
            for step,epoch in [(96,1.0),(110,1.14)]:
                ck=sft/f'checkpoint-{step}'
                for name in ('adapter_model.safetensors','optimizer.pt','scheduler.pt','scaler.pt','rng_state.pth'):
                    ck.mkdir(exist_ok=True);(ck/name).write_bytes(b'fixture')
                write_json(ck/'adapter_config.json',{'r':16})
                write_json(ck/'trainer_state.json',{'global_step':step,'epoch':epoch})
                self.assertTrue(valid_resume(ck))
            harvest_epochs(sft)
            shutil.rmtree(sft/'checkpoint-96')
            self.assertEqual([p.name for p in epochs(sft)],['epoch-01'])

    def test_training_reuse_rejects_changed_config_and_split(self):
        from legalqa.training import select_training_questions
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);data=root/'data';sft=root/'sft';c=config();lock={}
            qa={'1':{'question':'train question','answer':'train answer'}}
            write_json(data/'train.json',qa)
            write_json(data/'dev100.questions.json',{'2':{'question':'dev question'}})
            qa=select_training_questions(qa,c)
            original={'config':c,'models':lock,'code':'code','qa_hash':digest(qa),'qa_ids':list(qa),
                      'retrieval':{'index_hash':'index','mode':'lexical'}}
            write_json(sft/'training_manifest.json',original)
            verify_training(sft,c,data,lock,'index','code')
            changed=json.loads(json.dumps(c));changed['generation']['max_new_tokens']+=1
            with self.assertRaisesRegex(ValueError,'config/model/code'):
                verify_training(sft,changed,data,lock,'index','code')
            write_json(data/'dev100.questions.json',{'1':{'question':'train question'}})
            with self.assertRaisesRegex(ValueError,'leakage'):
                verify_training(sft,c,data,lock,'index','code')


if __name__=='__main__':
    unittest.main()
