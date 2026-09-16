import io
import json
import tempfile
import tracemalloc
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from legalqa.io import config, digest, source_hash, write_json
from legalqa.training import training_examples
from legalqa.training_cache import stream_identity, stream_records, training_retrieval


class TrainingCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name)/'retrieval.json'
        self.c, self.lock = config(), {'fixture':'model'}
        self.qa = {'2':{'question':'second', 'answer':'CD'}, '1':{'question':'first', 'answer':'AB'}}
        self.records = {key:{'question':row['question'], 'contexts':[{'text':'context', 'parent_id':key}]}
                        for key, row in reversed(list(self.qa.items()))}
        self.identity = {'questions_hash':digest({key:{'question':row['question']} for key,row in self.qa.items()}),
            'code':source_hash(), 'models':self.lock, 'retrieval':self.c['retrieval'], 'mode':'lexical',
            'mode_config':self.c['training']['lexical_pool_k'], 'index_hash':'fixture-index'}

    def read(self):
        with patch('legalqa.training_cache.model_lock', return_value=self.lock), \
             patch.dict('os.environ', {'LEGALQA_INDEX_HASH':'fixture-index'}):
            return training_retrieval(self.path, self.qa, self.c, 'models')

    def publish(self):
        write_json(self.path, {'records':self.records, 'identity':self.identity})

    def test_stream_preserves_samples_targets_and_question_order(self):
        class Tokenizer:
            eos_token_id = 1
            def __call__(self, text, **kwargs):
                return {'input_ids':[ord(c) for c in text]}
        self.publish()
        rows, identity = self.read()
        self.assertEqual(identity, self.identity)
        with patch('legalqa.training.pack_prompt', return_value=([11,12], ['context'])):
            original, report1 = training_examples(self.qa, self.records, Tokenizer(), self.c)
            streamed, report2 = training_examples(self.qa, rows, Tokenizer(), self.c)
        self.assertEqual(report1, report2)
        self.assertEqual([r['id'] for r in report2['lengths']], ['2','1'])
        np.testing.assert_array_equal(streamed[0]['labels'], [-100,-100,67,68,1])
        for before, after in zip(original, streamed):
            for key in before:
                np.testing.assert_array_equal(before[key], after[key])
        self.assertEqual(streamed[0]['input_ids'].dtype, np.int32)
        self.assertEqual(streamed[0]['attention_mask'].dtype, np.uint8)

    def test_stream_rejects_wrong_identity_ids_and_question_text(self):
        for key in self.identity:
            with self.subTest(key=key):
                old = self.identity[key]
                self.identity[key] = 'different'
                self.publish()
                with self.assertRaisesRegex(ValueError, key):
                    self.read()
                self.identity[key] = old
        self.records['1']['question'] = 'changed'
        self.publish()
        with self.assertRaisesRegex(ValueError, 'question differs'):
            list(self.read()[0])
        del self.records['1']
        self.publish()
        with self.assertRaisesRegex(ValueError, 'Missing'):
            list(self.read()[0])

    def test_duplicate_keys_and_truncated_json_rejected_without_full_load(self):
        for body in (b'{"identity":{},"identity":{},"records":{}}',
                     b'{"identity":{},"records":{"1":{},"1":{}}}',
                     b'{"identity":{},"records":{"1":{"contexts":[],"contexts":[]}}}'):
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                stream_identity(io.BytesIO(body))
        with self.assertRaises(ValueError):
            list(stream_records(io.BytesIO(b'{"records":{"1":{"contexts":[')))
        self.assertEqual(stream_identity(io.BytesIO(b'\xef\xbb\xbf{"identity":{},"records":{}}')), {})

    def test_large_context_cache_peak_memory_is_bounded(self):
        # Write 16 MiB without materializing a full payload in this test either.
        with self.path.open('w', encoding='utf-8') as out:
            out.write('{"identity":{"fixture":true},"records":{')
            for i in range(256):
                if i:
                    out.write(',')
                out.write(json.dumps(str(i))+':'+json.dumps({'contexts':[{'text':'x'*65536}]}))
            out.write('}}')
        tracemalloc.start()
        try:
            with self.path.open('rb') as stream:
                self.assertEqual(stream_identity(stream), {'fixture':True})
            with self.path.open('rb') as stream:
                self.assertEqual(sum(1 for _ in stream_records(stream)), 256)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertLess(peak, 4*1024**2)
        print(f'Streaming cache check: file={self.path.stat().st_size} bytes, Python peak={peak} bytes')


if __name__ == '__main__':
    unittest.main()
