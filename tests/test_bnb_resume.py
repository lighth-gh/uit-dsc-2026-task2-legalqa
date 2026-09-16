import ast
import importlib.util
import types
import unittest
from pathlib import Path

import numpy as np

from legalqa.io import ROOT, read_json

spec = importlib.util.spec_from_file_location('bnb_resume_compat', ROOT/'scripts/bnb_resume_compat.py')
compat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compat)


class FakeDevice:
    def __init__(self, name):
        self.name, self.type = name, name.split(':')[0]
    def __str__(self):
        return self.name


class FakeTensor:
    def __init__(self, data, device, *, paged=False):
        self.data, self.device = data, device
        if paged:
            self.is_paged, self.page_deviceid = True, 0
    def detach(self):
        return FakeTensor(self.data, self.device)
    def to(self, *, device, copy):
        if not copy:
            raise AssertionError('Restored state must have fresh storage')
        return FakeTensor(self.data.copy(), device)
    def numel(self):
        return self.data.size
    def element_size(self):
        return self.data.itemsize


class BnbResumeTests(unittest.TestCase):
    def tensor(self, data, device, paged=False):
        return FakeTensor(np.asarray(data, dtype=np.uint8), FakeDevice(device), paged=paged)

    def test_restored_moments_keep_bytes_and_steps_but_use_each_rank_device(self):
        for rank in (0,1):
            with self.subTest(rank=rank):
                parameter = self.tensor([1,2], f'cuda:{rank}')
                first = self.tensor([255,127,0,1], 'cuda:0', paged=True)
                second = self.tensor([0,22,44,66], 'cpu', paged=True)
                qmap = self.tensor([0,1,2], f'cuda:{rank}')
                state = {'state1':first,'state2':second,'qmap1':qmap,'step':1218}
                optimizer = types.SimpleNamespace(is_paged=True, param_groups=[{'params':[parameter], 'lr':.00001}],
                                                  state={parameter:state})
                note = compat.restore_cuda_storage(optimizer, lambda v:isinstance(v, FakeTensor))
                for key, before in (('state1',first),('state2',second)):
                    after = state[key]
                    np.testing.assert_array_equal(after.data, before.data)
                    self.assertIsNot(after.data, before.data)
                    self.assertFalse(after.is_paged)
                    self.assertFalse(hasattr(after,'page_deviceid'))
                    self.assertEqual(str(after.device), f'cuda:{rank}')
                    self.assertTrue(before.is_paged)
                self.assertIs(state['qmap1'],qmap)
                self.assertEqual(state['step'],1218)
                self.assertEqual(optimizer.param_groups[0]['lr'],.00001)
                self.assertEqual(note, {'tensors':2,'bytes':8,'devices':[f'cuda:{rank}']})

    def test_nonpaged_states_and_fresh_optimizer_unchanged(self):
        parameter = self.tensor([1], 'cuda:1')
        value = self.tensor([2], 'cuda:1')
        for paged, state in ((False, {parameter:{'state1':value}}), (True,{}), (True,{parameter:{'state1':value}})):
            optimizer = types.SimpleNamespace(is_paged=paged,param_groups=[{'params':[parameter]}],state=state)
            self.assertEqual(compat.restore_cuda_storage(optimizer, lambda v:isinstance(v,FakeTensor))['tensors'],0)
            if state:
                self.assertIs(state[parameter]['state1'],value)

    def test_wrapper_repairs_after_load_once_and_preserves_loader_validation(self):
        parameter = self.tensor([1], 'cuda:1')
        events = []
        class Optimizer:
            is_paged = True
            param_groups = [{'params':[parameter]}]
            def load_state_dict(self, checkpoint, move_to_device=True):
                events.append(('load',move_to_device))
                if checkpoint is None:
                    raise ValueError('checkpoint does not match parameter groups')
                self.state = checkpoint
                return 'original result'
        compat.patch_optimizer_class(Optimizer, lambda v:isinstance(v,FakeTensor), lambda note:events.append(('report',note)))
        first_wrapper = Optimizer.load_state_dict
        compat.patch_optimizer_class(Optimizer, lambda v:False, lambda note:None)
        self.assertIs(Optimizer.load_state_dict, first_wrapper)
        optimizer = Optimizer()
        result = optimizer.load_state_dict({parameter:{'state1':self.tensor([17], 'cpu', paged=True),'step':1218}}, move_to_device=False)
        self.assertEqual(result, 'original result')
        self.assertEqual(events[0], ('load',False))
        self.assertEqual(events[1][0],'report')
        self.assertEqual(str(optimizer.state[parameter]['state1'].device),'cuda:1')
        with self.assertRaisesRegex(ValueError,'parameter groups'):
            optimizer.load_state_dict(None)
        self.assertEqual(len(events),3)  # no repair/report after loader error

    def test_notebook_embeds_reviewed_hook_and_gpu_smoke_before_stage_run(self):
        notebook = read_json(ROOT/'legalqa_main_01_qlora_train.ipynb')
        setup = ''.join(notebook['cells'][5]['source'])
        literals = {n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(setup).body
                    if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)
                    and n.targets[0].id in {'BNB_RESUME_PATCH','BNB_RESUME_SMOKE'}}
        self.assertEqual(literals['BNB_RESUME_PATCH'], (ROOT/'scripts/bnb_resume_compat.py').read_text(encoding='utf-8'))
        self.assertEqual(literals['BNB_RESUME_SMOKE'], (ROOT/'scripts/check_bnb_resume.py').read_text(encoding='utf-8'))
        self.assertIn("'--nproc_per_node=2'",setup)
        self.assertIn("'sitecustomize.py'",setup)
        runner = ''.join(notebook['cells'][7]['source'])
        self.assertIn('**RUNTIME_ENV',runner)
        self.assertIn("RUN_ROOT/'runtime_compat'",runner)


if __name__ == '__main__':
    unittest.main()
