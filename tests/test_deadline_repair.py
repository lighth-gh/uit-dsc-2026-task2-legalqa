import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from zipfile import ZipFile
import json

from legalqa.deadline_repair import run_queue
from legalqa.io import config
from test_adaptive import data_fixture, generation, ANSWER
from test_core import TinyTokenizer


class DeadlineRepairTests(unittest.TestCase):
    def run_case(self, folder, generate, deadline=1000, clock=lambda: 0, two=True):
        data = data_fixture()
        if two:
            for name in ('predictions', 'questions', 'records', 'audit'):
                data[name]['b'] = copy.deepcopy(data[name]['a'])
        queue = {k: {'group': 'truncated'} for k in data['questions']}
        result = run_queue(data, queue, config(), TinyTokenizer(), generate, folder,
                           {'context_limit': 8192}, deadline, clock)
        with ZipFile(Path(folder) / 'submission.zip') as z:
            self.assertEqual(z.namelist(), ['submission.json'])
            self.assertIsNone(z.testzip())
            pred = json.loads(z.read('submission.json'))
        self.assertEqual(set(pred), set(data['predictions']))
        return result, pred, data

    def test_single_attempt_accept_reject_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            generate = Mock(side_effect=[generation(), generation(hit=True)])
            result, pred, data = self.run_case(folder, generate)
            self.assertEqual(result['changed'], 1)
            self.assertEqual(result['attempted'], 2)
            self.assertEqual(pred['a']['answer'], ANSWER)
            self.assertEqual(pred['b'], data['predictions']['b'])
            self.assertEqual(generate.call_count, 2)
            self.assertEqual(generate.call_args.args[0]['generation']['max_new_tokens'], 3072)
            again = Mock(side_effect=AssertionError('must not regenerate'))
            resumed, after, _ = self.run_case(folder, again)
            again.assert_not_called()
            self.assertEqual(after, pred)
            self.assertEqual(resumed['changed'], 1)

    def test_deadline_before_generation_exports_baseline(self):
        with tempfile.TemporaryDirectory() as folder:
            generate = Mock()
            result, pred, data = self.run_case(folder, generate, deadline=60)
            generate.assert_not_called()
            self.assertEqual(result['status'], 'deadline')
            self.assertEqual(pred, data['predictions'])

    def test_error_preserves_completed_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            result, pred, data = self.run_case(folder, Mock(side_effect=[generation(), RuntimeError('OOM')]))
            self.assertEqual(result['status'], 'stopped_error')
            self.assertEqual(result['changed'], 1)
            self.assertEqual(pred['a']['answer'], ANSWER)
            self.assertEqual(pred['b'], data['predictions']['b'])

    def test_time_stopped_without_eos_keeps_baseline(self):
        with tempfile.TemporaryDirectory() as folder:
            result, pred, data = self.run_case(folder, Mock(return_value=generation(ended_with_eos=False)), two=False)
            self.assertEqual(result['changed'], 0)
            self.assertEqual(pred, data['predictions'])


if __name__ == '__main__':
    unittest.main()
