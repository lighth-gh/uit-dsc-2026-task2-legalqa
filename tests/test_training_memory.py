import importlib.util
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from legalqa.training_memory import compact_samples, enable_fused_loss, padded_batch


class TrainingMemoryTests(unittest.TestCase):
    def test_compaction_preserves_targets_masks_order_and_padding(self):
        rows = [dict(input_ids=[1, 151935, 3], labels=[-100, 151935, 3], attention_mask=[1, 1, 1]),
                dict(input_ids=[4, 5], labels=[-100, 5], attention_mask=[1, 1])]
        expected = padded_batch(rows, 42)
        self.assertIs(compact_samples(rows), rows)
        actual = padded_batch(rows, 42)
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])
            self.assertEqual(actual[key].dtype, np.int64)
        self.assertEqual(rows[0]['input_ids'].dtype, np.int32)
        self.assertEqual(rows[0]['attention_mask'].dtype, np.uint8)
        self.assertEqual(sum(v.nbytes for row in rows for v in row.values()), 5 * 9)
        self.assertEqual(actual['labels'][1, -1], -100)

    def test_patch_is_instance_local_and_preserves_loss_kwargs(self):
        class Model:
            config = types.SimpleNamespace(model_type='qwen2')
            lm_head = types.SimpleNamespace(bias=None, weight=types.SimpleNamespace(requires_grad=False))
            def forward(self, **kwargs):
                return 'original'
        def fused(self, **kwargs):
            return self, kwargs
        module = types.SimpleNamespace(lce_forward=fused)
        first, second = Model(), Model()
        with patch.dict(sys.modules, {'liger_kernel.transformers.model.qwen2':module}):
            enable_fused_loss(first)
        owner, kwargs = first.forward(labels='targets', num_items_in_batch=1234)
        self.assertIs(owner, first)
        self.assertEqual(kwargs, {'labels':'targets', 'num_items_in_batch':1234})
        self.assertEqual(second.forward(), 'original')

    def test_rejects_unsupported_head_before_training(self):
        for kind, bias, trainable in [('other', None, False), ('qwen2', 1, False), ('qwen2', None, True)]:
            model = types.SimpleNamespace(config=types.SimpleNamespace(model_type=kind),
                lm_head=types.SimpleNamespace(bias=bias, weight=types.SimpleNamespace(requires_grad=trainable)))
            with self.assertRaises(ValueError):
                enable_fused_loss(model)


@unittest.skipUnless(sys.platform == 'linux' and importlib.util.find_spec('torch') is not None,
                     'requires Linux CUDA and installed training requirements')
class FusedLossCudaTests(unittest.TestCase):
    def test_fp16_frozen_head_loss_gradients_and_global_token_denominator(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('CUDA unavailable')
        from transformers import Qwen2Config, Qwen2ForCausalLM
        torch.manual_seed(2026)
        model = Qwen2ForCausalLM(Qwen2Config(vocab_size=151936, hidden_size=64,
            intermediate_size=128, num_hidden_layers=1, num_attention_heads=4,
            num_key_value_heads=2, use_cache=False, attention_dropout=0.0)).cuda().train()
        model.lm_head.weight.requires_grad_(False)
        original = model.forward
        ids = torch.randint(0, 151936, (2, 23), device='cuda')
        labels = ids.clone()
        labels[:, :11] = -100
        labels[1, 20:] = -100
        for denominator in (None, torch.tensor(83, device='cuda')):
            results = []
            for fused in (False, True):
                model.zero_grad(set_to_none=True)
                if fused:
                    enable_fused_loss(model)
                else:
                    model.forward = original
                with torch.autocast('cuda', dtype=torch.float16):
                    out = model(input_ids=ids, labels=labels, num_items_in_batch=denominator)
                (out.loss * 128).backward()
                results.append((out.loss.detach(), {n:p.grad.detach().clone()/128
                    for n,p in model.named_parameters() if p.grad is not None}))
                if fused:
                    self.assertIsNone(out.logits)
            torch.testing.assert_close(results[1][0], results[0][0], rtol=2e-4, atol=2e-4)
            self.assertEqual(results[0][1].keys(), results[1][1].keys())
            for name in results[0][1]:
                torch.testing.assert_close(results[1][1][name], results[0][1][name], rtol=.03, atol=2e-4)


if __name__ == '__main__':
    unittest.main()
