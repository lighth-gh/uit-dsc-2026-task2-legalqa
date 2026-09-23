import contextlib
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from zipfile import ZipFile

from legalqa.adaptive import (SessionBudget, attempt_config, candidate_queue, check_candidate,
                              evaluate_dev, focused_contexts, next_attempt, run_queue, screen, token_ladder)
from legalqa.adaptive_inputs import load_adaptive_inputs
from legalqa.io import config, digest, read_json, write_json
from test_core import TinyTokenizer


QUESTION = 'Hồ sơ đăng ký giấy phép gồm những tài liệu nào?'
ANSWER = ('Hồ sơ đăng ký giấy phép gồm đơn đề nghị cấp giấy phép và bản sao giấy chứng nhận. '
          'Người nộp hồ sơ phải cung cấp đầy đủ các tài liệu theo quy định trong trích đoạn.')
CONTEXT = {'parent_id': 'doc:0', 'doc_id': 'doc', 'text': ANSWER, 'heading': 'Hồ sơ đăng ký giấy phép'}


def generation(text=ANSWER, hit=False, route=None, **extra):
    audit = {'route': route or ('generated_truncated' if hit else 'generated'),
             'raw_answer': text, 'hit_token_limit': hit, 'ended_with_eos': not hit,
             'flags_before_fallback': {}, 'entity_conflict': {'conflict': False},
             'output_tokens': 2048 if hit else 80, 'seconds': 2.0}
    audit.update(extra)
    return {'prediction': {'answer': text}, 'audit': audit}


def data_fixture():
    return {'predictions': {'a': {'answer': 'Hồ sơ bao gồm:'}},
            'questions': {'a': {'question': QUESTION}},
            'records': {'a': {'question': QUESTION, 'contexts': [CONTEXT]}},
            'audit': {'a': {'route': 'generated_truncated', 'hit_token_limit': True}}}


def stage2_fixture(root):
    """Small, fully self-consistent diagnostics including manifest/journal provenance."""
    c = config()
    models = {'models': {'generator': {'revision': 'fixture'}}}
    q = {'a': {'question': QUESTION}}
    pred = {'a': {'answer': ANSWER}}
    audit = {'a': {**generation()['audit'], 'context_parent_ids': ['doc:0']}}
    records = {'a': {'question': QUESTION, 'contexts': [CONTEXT]}}
    ri = {'questions_hash': digest(q), 'models': models, 'code': 'fixture',
          'index_hash': 'index', 'retrieval': c['retrieval']}
    files = {}

    def put(name, value):
        files[name] = json.dumps(value, ensure_ascii=False).encode()

    def journal(name, values):
        files[name] = ''.join(json.dumps({'id': k, 'value': v}, ensure_ascii=False) + '\n'
                              for k, v in values.items()).encode()

    put('config.json', c)
    put('session.json', {'config_hash': digest(c), 'source_hash': 'fixture', 'models': models, 'index_hash': 'index'})
    put('models.lock.json', models)
    for prefix, question_file in [('dev100', 'dev100'), ('public', 'test')]:
        put(f'data/{question_file}.questions.json', q)
        put(f'{prefix}.retrieval.json', {'identity': ri, 'records': records})
        put(f'{prefix}.retrieval.checkpoint.jsonl.meta.json', ri)
        journal(f'{prefix}.retrieval.checkpoint.jsonl', records)
    put('selected_adapter/adapter_config.json', {'r': 16})
    adapter = {'adapter_config.json': hashlib.sha256(files['selected_adapter/adapter_config.json']).hexdigest(),
               'adapter_model.safetensors': 'weight_hash'}
    ident = {'config': c, 'models': models, 'code': 'fixture', 'adapter': adapter,
             'questions_hash': digest(q), 'retrieval': ri,
             'retrieval_file_hash': hashlib.sha256(files['dev100.retrieval.json']).hexdigest()}
    pm = {'identity': ident, 'prediction_hash': digest(pred)}
    refs = {'a': ANSWER}
    selection = {'label': 'epoch-01', 'prediction_manifest': pm, 'reference_hash': digest(refs),
                 'meteor': .5, 'rougeL': .5}
    put('selection.json', selection)
    put('data/dev100.references.json', refs)
    stem = 'dev100.epoch-01'
    for suffix, value in [('.json', pred), ('.audit.json', audit), ('.manifest.json', pm),
                          ('.checkpoint.jsonl.meta.json', ident),
                          ('.metrics.json', {'reference_hash': digest(refs), 'prediction_hash': digest(pred),
                                            'meteor': .5, 'rougeL': .5})]:
        put(stem + suffix, value)
    journal(stem + '.checkpoint.jsonl', {'a': {'prediction': pred['a'], 'audit': audit['a']}})
    manifest = {'stage': 2, 'schema': 2, 'quality_version': 'v8', 'status': 'complete',
                'progress': {'complete': True}, 'source_hash': 'fixture',
                'files': {n: {'size': len(b), 'sha256': hashlib.sha256(b).hexdigest()} for n, b in files.items()}}
    manifest['files']['selected_adapter/adapter_model.safetensors'] = {'size': 10, 'sha256': 'weight_hash'}
    put('stage2_manifest.json', manifest)
    archive = root / 'stage2.zip'
    with ZipFile(archive, 'w') as z:
        for n, b in files.items():
            z.writestr(n, b)
    submission = root / 'submission.zip'
    with ZipFile(submission, 'w') as z:
        z.writestr('submission.json', json.dumps(pred))
    return archive, submission, files


class AdaptiveInputsTests(unittest.TestCase):
    def test_stage2_submission_without_private_audit_and_unpacked_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive, sub, files = stage2_fixture(root)
            result = load_adaptive_inputs(archive, sub)
            self.assertEqual(set(result['private']['predictions']), {'a'})
            self.assertIsNone(result['private']['audit'])
            unpacked = root / 'unpacked'
            for name, value in files.items():
                path = unpacked / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(value)
            self.assertEqual(load_adaptive_inputs(unpacked, sub), result)

    def test_wrong_private_ids_and_provenance_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            archive, sub, _ = stage2_fixture(Path(folder))
            other = {'public': {'questions': {'public_id': {'question': QUESTION}}}}
            with patch('legalqa.adaptive_inputs.load_diagnostics', return_value=other):
                with self.assertRaisesRegex(ValueError, 'IDs/questions differ'):
                    load_adaptive_inputs(archive, sub, 'public-stage3.zip')
            bundle = load_adaptive_inputs(archive, sub)
            other = {'public': {'questions': bundle['private']['questions'], 'records': {}},
                     'config': bundle['config']}
            with patch('legalqa.adaptive_inputs.load_diagnostics', return_value=other):
                with self.assertRaisesRegex(ValueError, 'provenance differs'):
                    load_adaptive_inputs(archive, sub, 'wrong-stage3.zip')

    def test_tamper_duplicate_json_and_schema_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive, sub, files = stage2_fixture(root)
            files['config.json'] = b'{}'
            with ZipFile(archive, 'w') as z:
                for name, payload in files.items():
                    z.writestr(name, payload)
            with self.assertRaisesRegex(ValueError, 'hash/size mismatch'):
                load_adaptive_inputs(archive, sub)
            archive, sub, _ = stage2_fixture(root)
            with ZipFile(sub, 'w') as z:
                z.writestr('submission.json', '{"a":{"answer":"x"},"a":{"answer":"y"}}')
            with self.assertRaisesRegex(ValueError, 'Duplicate JSON key'):
                load_adaptive_inputs(archive, sub)
            with ZipFile(sub, 'w') as z:
                z.writestr('submission.json', '{"wrong":{"answer":"x"}}')
            with self.assertRaisesRegex(ValueError, 'ID mismatch'):
                load_adaptive_inputs(archive, sub)


class AdaptivePolicyTests(unittest.TestCase):
    def test_detection_distinguishes_history_and_heuristic_without_selecting_loop_only(self):
        data = data_fixture()
        data['predictions']['b'] = {'answer': 'Câu trả lời đã hoàn chỉnh.'}
        data['questions']['b'] = {'question': QUESTION}
        data['records']['b'] = data['records']['a']
        data['audit']['b'] = {'route': 'source_fallback', 'hit_token_limit': False}
        queue = candidate_queue(data, config(), TinyTokenizer(), 'audit')
        self.assertEqual(queue['b']['group'], 'fallback')
        self.assertIn('audit_token_limit', queue['a']['reasons'])
        queue = candidate_queue(data, config(), TinyTokenizer(), 'heuristic')
        self.assertEqual(set(queue), {'a'})
        self.assertNotIn('audit_token_limit', queue['a']['reasons'])
        data['predictions']['a']['answer'] = ' '.join(['dài'] * 1383) + '.'
        queue = candidate_queue(data, config(), TinyTokenizer(), 'heuristic')
        self.assertIn('suspected_truncated_near_limit', queue['a']['reasons'])

    def test_exact_fallback_reconstruction(self):
        data = data_fixture()
        data['predictions']['a']['answer'] = ANSWER
        queue = candidate_queue(data, config(), TinyTokenizer(), 'heuristic')
        self.assertEqual(queue['a']['group'], 'fallback')
        self.assertIn('suspected_fallback_exact_reconstruction', queue['a']['reasons'])

    def test_acceptance_allows_no_anchor_but_rejects_bad_evidence_and_generations(self):
        support = {'coverage': .7, 'longest_phrase': 4, 'strong': False, 'anchor_match': False,
                   'marker_ok': True, 'age_value_present': True}
        with patch('legalqa.adaptive.localized_evidence_support', return_value=support):
            self.assertTrue(check_candidate(generation(), QUESTION, [CONTEXT])['accepted'])
            for row in [generation(hit=True), generation(route='source_fallback'),
                        generation(ended_with_eos=False), generation('Hồ sơ bao gồm:'),
                        generation(flags_before_fallback={'unsupported_document_numbers': ['12/2019/tt-btc']}),
                        generation(entity_conflict={'conflict': True}),
                        generation('```invalid```')]:
                self.assertFalse(check_candidate(row, QUESTION, [CONTEXT])['accepted'])
            support['marker_ok'] = False
            self.assertFalse(check_candidate(generation(), QUESTION, [CONTEXT])['accepted'])
            support['marker_ok'], support['age_value_present'] = True, False
            self.assertFalse(check_candidate(generation(), QUESTION, [CONTEXT])['accepted'])
        bad = dict(CONTEXT, text='Nội dung không liên quan.', heading='')
        self.assertFalse(check_candidate(generation(), QUESTION, [bad])['accepted'])

    def test_retry_budget_fault_focus_and_stop(self):
        previous = []
        self.assertEqual(next_attempt(previous, [2048, 3072, 4096]), (2048, False))
        for cap, following in [(2048, 3072), (3072, 4096), (4096, None)]:
            previous.append({'budget': cap, 'focused': False, 'generation': generation(hit=True),
                             'check': {'accepted': False, 'faulty': False}})
            self.assertEqual(next_attempt(previous, [2048, 3072, 4096]),
                             (following, False) if following else None)
        previous = [dict(previous[0], check={'accepted': False, 'faulty': True})]
        self.assertEqual(next_attempt(previous, [2048, 3072, 4096]), (2048, True))
        previous.append(dict(previous[0], focused=True))
        self.assertIsNone(next_attempt(previous, [2048, 3072, 4096]))
        previous[-1]['check'] = {'accepted': True, 'faulty': False}
        self.assertIsNone(next_attempt(previous, [2048, 3072, 4096]))
        self.assertEqual(token_ladder(8192), [2048, 3072, 4096])
        self.assertEqual(token_ladder(7000), [2048, 2904])
        with self.assertRaisesRegex(ValueError, 'Model window'):
            token_ladder(6000)

    def test_focus_keeps_best_and_one_same_document_in_original_order(self):
        contexts = [dict(CONTEXT, doc_id='other'), dict(CONTEXT, parent_id='doc:1'),
                    dict(CONTEXT, parent_id='doc:2'), dict(CONTEXT, parent_id='doc:3')]
        with patch('legalqa.adaptive.localized_evidence_support', return_value={'context_index': 2}):
            self.assertEqual([c['parent_id'] for c in focused_contexts(QUESTION, contexts)], ['doc:1', 'doc:2'])
        cfg = attempt_config(config(), 'fallback', 4096, True)
        self.assertEqual(cfg['generation']['max_input_tokens'] + cfg['generation']['max_new_tokens'], 8192)
        self.assertTrue(cfg['generation']['complete_legal_units'])

    def test_screen_requires_changes_no_meteor_loss_and_no_damage_increase(self):
        before = {'a': {'answer': 'Ban đầu.'}}
        after = {'a': {'answer': ANSWER}}
        metric = {'meteor': .5, 'rougeL': .5}
        self.assertTrue(screen(before, after, metric, metric)['passes'])
        self.assertFalse(screen(before, after, metric, {**metric, 'meteor': .499999})['passes'])
        self.assertFalse(screen(before, before, metric, metric)['passes'])
        self.assertFalse(screen(before, {'a': {'answer': 'Bao gồm:'}}, metric, metric)['passes'])


class AdaptiveJournalTests(unittest.TestCase):
    def test_resume_between_attempts_and_identity_guard(self):
        data, tokenizer = data_fixture(), TinyTokenizer()
        queue = candidate_queue(data, config(), tokenizer, 'audit')
        identity = {'context_limit': 8192}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = Mock(return_value=generation(hit=True))
            budget = SessionBudget(50, pause=Mock(side_effect=[False, True]))
            self.assertIsNone(run_queue(data, queue, config(), tokenizer, first, root, identity,
                                        [2048, 3072, 4096], budget, 'private'))
            self.assertEqual(first.call_count, 1)
            second = Mock(return_value=generation())
            result = run_queue(data, queue, config(), tokenizer, second, root, identity,
                               [2048, 3072, 4096], SessionBudget(50, lambda: False), 'private')
            self.assertEqual(second.call_args.args[0]['generation']['max_new_tokens'], 3072)
            self.assertEqual(result['a']['answer'], ANSWER)
            self.assertEqual(read_json(root / 'outcomes.json')['a']['attempts'], 2)
            with self.assertRaisesRegex(ValueError, 'identity differs'):
                run_queue(data, queue, config(), tokenizer, second, root, {**identity, 'code': 'new'},
                          [2048, 3072, 4096], SessionBudget(50), 'private')

    def test_gpu_failure_retains_committed_attempt_and_baseline(self):
        data, tokenizer = data_fixture(), TinyTokenizer()
        original = copy.deepcopy(data)
        queue = candidate_queue(data, config(), tokenizer, 'audit')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fail = Mock(side_effect=[generation(hit=True), RuntimeError('CUDA out of memory')])
            with self.assertRaisesRegex(RuntimeError, 'CUDA'):
                run_queue(data, queue, config(), tokenizer, fail, root, {'context_limit': 8192},
                          [2048, 3072, 4096], SessionBudget(50, lambda: False), 'private')
            self.assertEqual(data, original)
            self.assertEqual(len((root / 'attempts.jsonl').read_text(encoding='utf-8').splitlines()), 1)
            self.assertEqual(read_json(root / 'status.json')['status'], 'failed')

    def test_last_limit_keeps_baseline_and_unselected_ids_unchanged(self):
        data, tokenizer = data_fixture(), TinyTokenizer()
        data['predictions']['safe'] = {'answer': ANSWER}
        data['questions']['safe'] = {'question': QUESTION}
        queue = {'a': {'group': 'truncated', 'reasons': ['audit_token_limit']}}
        with tempfile.TemporaryDirectory() as folder:
            result = run_queue(data, queue, config(), tokenizer, Mock(return_value=generation(hit=True)),
                               folder, {'context_limit': 8192}, [2048, 3072, 4096],
                               SessionBudget(50, lambda: False), 'private')
            self.assertEqual(result, data['predictions'])

    def test_item_limit_shared_across_detection_methods(self):
        budget = SessionBudget(1, lambda: False)
        self.assertTrue(budget.allow(('dev', 'a')))
        self.assertTrue(budget.allow(('dev', 'a')))
        self.assertFalse(budget.allow(('dev', 'b')))


class AdaptiveScoringTests(unittest.TestCase):
    def test_scores_full_dev_and_gates_groups_independently(self):
        data = data_fixture()
        data['predictions']['b'] = {'answer': 'Câu thứ hai.'}
        data['original'] = copy.deepcopy(data['predictions'])
        data['references'] = {'a': 'Gold a', 'b': 'Gold b'}
        data['stored_metrics'] = {'meteor': .5, 'rougeL': .5}
        candidate = {'a': {'answer': ANSWER}, 'b': {'answer': 'Thay b.'}}
        queue = {'a': {'group': 'fallback'}, 'b': {'group': 'truncated'}}
        seen = []

        def scorer(pred, refs, output):
            seen.append(set(read_json(pred)))
            scores = {'original': .5, 'baseline': .5, 'fallback': .51, 'truncated': .49, 'combined': .51}
            write_json(output, {'meteor': scores[pred.stem], 'rougeL': .5})

        with tempfile.TemporaryDirectory() as folder:
            result = evaluate_dev(data, candidate, queue, folder, scorer)
            self.assertEqual(result['enabled_groups'], ['fallback'])
            self.assertTrue(all(keys == {'a', 'b'} for keys in seen))
            self.assertEqual(read_json(Path(folder) / 'selected.json')['b'], data['predictions']['b'])

    def test_failed_combined_screen_restores_selected_baseline(self):
        data = data_fixture()
        data.update(original=copy.deepcopy(data['predictions']), references={'a': ANSWER},
                    stored_metrics={'meteor': .5, 'rougeL': .5})
        def scorer(pred, refs, report):
            score = .49 if pred.stem == 'combined' else .5
            write_json(report, {'meteor': score, 'rougeL': .5})
        with tempfile.TemporaryDirectory() as folder:
            result = evaluate_dev(data, {'a': {'answer': ANSWER}}, {'a': {'group': 'truncated'}}, folder, scorer)
            self.assertEqual(result['enabled_groups'], [])
            self.assertEqual(read_json(Path(folder) / 'selected.json'), data['predictions'])
            self.assertEqual(result['selected']['meteor'], .5)


class EffectiveGenerationLimitTests(unittest.TestCase):
    def test_override_and_eos_at_exact_boundary(self):
        from legalqa.generation import _generate_one
        class Sequence:
            def __init__(self, tokens): self.tokens = tokens
            def __getitem__(self, key): return SimpleNamespace(tolist=lambda: self.tokens)
        fake_torch = SimpleNamespace(tensor=lambda *a, **k: object(), ones_like=lambda v: object(),
                                     inference_mode=contextlib.nullcontext)
        model = SimpleNamespace(generate=Mock())
        tokenizer = SimpleNamespace(eos_token_id=2, pad_token_id=2, decode=lambda *a, **k: ANSWER)
        with patch.dict('sys.modules', {'torch': fake_torch}), \
             patch('legalqa.generation.pack_prompt', return_value=([1], [CONTEXT])):
            args = ({'generation': {'max_new_tokens': 2}}, 'a', {'a': {'question': QUESTION}},
                    {'a': {'contexts': [CONTEXT]}}, model, tokenizer, 'cuda:0', 'generate')
            for tokens, limit, hit, eos in [([9, 9, 9, 9], 4, True, False),
                                            ([9, 9, 9, 2], 4, False, True),
                                            ([9, 9, 2], 4, False, True),
                                            ([9, 9, 9], 4, False, False),
                                            ([], 4, False, False)]:
                model.generate.return_value = Sequence(tokens)
                value = _generate_one(*args, generation_overrides={'max_new_tokens': limit})
                self.assertEqual(value['audit']['hit_token_limit'], hit)
                self.assertEqual(value['audit']['ended_with_eos'], eos)
                self.assertEqual(value['audit']['effective_max_new_tokens'], limit)


class AdaptiveWorkflowTests(unittest.TestCase):
    def test_smoke_resume_dev_gate_and_private_zip(self):
        from legalqa.adaptive import run
        data = data_fixture()
        for field in ('predictions', 'questions', 'records', 'audit'):
            data[field] = {str(i): copy.deepcopy(data[field]['a']) for i in range(6)}
        data['original'] = copy.deepcopy(data['predictions'])
        data['references'] = {k: ANSWER for k in data['predictions']}
        data['stored_metrics'] = {'meteor': .5, 'rougeL': .5}
        private = copy.deepcopy(data)
        private['audit'] = None
        bundle = {'dev': data, 'private': private, 'config': config()}
        tok = TinyTokenizer()
        tok.model_max_length = 8192
        tok.eos_token = '</s>'
        transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tok),
                                       set_seed=lambda seed: None)
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))

        def evaluate(pred, refs, report):
            values = read_json(pred)
            self.assertEqual(len(values), 6)
            score = .51 if any(v['answer'] == ANSWER for v in values.values()) else .5
            write_json(report, {'meteor': score, 'rougeL': score})

        with tempfile.TemporaryDirectory() as folder, \
             patch.dict('sys.modules', {'torch': torch, 'transformers': transformers}), \
             patch('legalqa.adaptive.load_adaptive_inputs', return_value=bundle), \
             patch('legalqa.adaptive.runtime_identity', return_value={'context_limit': 8192}), \
             patch('legalqa.models.load_generator', return_value=(object(), tok)), \
             patch('legalqa.generation._generate_one', return_value=generation()) as gen, \
             patch('legalqa.metrics.evaluate', side_effect=evaluate):
            args = ('s2', 'submission', 'models', 'adapter', folder)
            with self.assertRaisesRegex(ValueError, 'Finish both'):
                run(*args, mode='adaptive_private')
            first = run(*args, mode='adaptive_dev')
            self.assertEqual(first['status'], 'paused')
            self.assertEqual(gen.call_count, 5)
            self.assertTrue((Path(folder) / 'dev.smoke.json').is_file())
            second = run(*args, mode='adaptive_dev')
            self.assertEqual(second['status'], 'complete')
            self.assertEqual(set(second['summary']), {'audit', 'heuristic'})
            for decision in second['summary'].values():
                self.assertEqual(decision['enabled_groups'], ['truncated'])
            self.assertEqual(run(*args, mode='adaptive_private')['status'], 'paused')
            self.assertFalse((Path(folder) / 'submission_adaptive.zip').exists())
            self.assertEqual(run(*args, mode='adaptive_private')['status'], 'complete')
            with ZipFile(Path(folder) / 'submission_adaptive.zip') as z:
                self.assertEqual(z.namelist(), ['submission.json'])
                exported = json.loads(z.read('submission.json'))
                self.assertEqual(set(exported), set(private['predictions']))
                self.assertTrue(all(v == {'answer': ANSWER} for v in exported.values()))
            self.assertTrue(all(v['answer'] == 'Hồ sơ bao gồm:' for v in private['predictions'].values()))
            previous = gen.call_count
            run(*args, mode='adaptive_private')
            self.assertEqual(gen.call_count, previous)


if __name__ == '__main__':
    unittest.main()
