import ast
import copy
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from legalqa.generation import adapter_identity
from legalqa.io import config, digest, file_hash, read_json, source_hash, write_json
from scripts import dev100_repetition as experiment


ROOT = Path(__file__).resolve().parents[1]


class Dev100RepetitionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.upstream = self.root / 'main2'
        self.models = self.root / 'models'
        self.output = self.root / 'experiment'
        self.base = config()
        self.questions = {str(i): {'question': f'question {i}'} for i in range(100)}
        self.truth = {str(i): f'reference {i}' for i in range(100)}
        lock = {'models': {}}
        for role, model_id in self.base['models'].items():
            path = self.models / role / 'config.json'
            write_json(path, {'role': role})
            lock['models'][role] = {'model_id': model_id, 'revision': 'fixture',
                                    'config_sha256': file_hash(path)}
        write_json(self.models / 'models.lock.json', lock)
        write_json(self.upstream / 'models.lock.json', lock)
        write_json(self.upstream / 'config.json', self.base)
        write_json(self.upstream / 'data/dev100.questions.json', self.questions)
        write_json(self.upstream / 'data/dev100.references.json', self.truth)
        write_json(self.upstream / 'data/split_manifest.json', {'fixture': True})
        write_json(self.upstream / 'selected_adapter/adapter_config.json', {'r': 16})
        (self.upstream / 'selected_adapter/adapter_model.safetensors').write_bytes(b'fixture adapter')
        adapter = adapter_identity(self.upstream / 'selected_adapter')
        write_json(self.upstream / 'selection.json', {
            'label': 'epoch-02', 'prediction_manifest': {'identity': {'adapter': adapter}}})
        session = {'code_commit': 'a' * 40, 'source_hash': source_hash(),
                   'config_hash': digest(self.base), 'models': lock, 'index_hash': 'index-fixture'}
        write_json(self.upstream / 'session.json', session)
        write_json(self.upstream / 'dev100.retrieval.json', {
            'identity': {'models': lock, 'retrieval': self.base['retrieval'],
                         'code': source_hash(), 'mode': 'full', 'index_hash': 'index-fixture'},
            'records': {key: {**question, 'contexts': []} for key, question in self.questions.items()}})
        self.refresh_snapshot()

    def refresh_snapshot(self, status='complete'):
        files = {p.relative_to(self.upstream).as_posix(): {'size': p.stat().st_size, 'sha256': file_hash(p)}
                 for p in self.upstream.rglob('*') if p.is_file() and p.name != 'stage2_manifest.json'}
        write_json(self.upstream / 'stage2_manifest.json', {
            'schema': 2, 'stage': 2, 'status': status, 'code_commit': 'a' * 40,
            'source_hash': source_hash(), 'files': files})

    def prepare(self):
        return experiment.prepare(self.upstream, self.models, self.output)

    def results(self):
        identity = self.prepare()
        repeated = 'This is a sufficiently long repeated sentence with many words.'
        for penalty, loops in zip(experiment.PENALTIES, (3, 1, 2)):
            directory = self.output / experiment.label(penalty)
            predictions = {key: {'answer': f'final answer {key}'} for key in self.truth}
            audit = {key: {'raw_answer': '\n'.join([repeated] * 5) if int(key) < loops else f'answer {key}',
                           'hit_token_limit': int(key) < loops, 'route': 'generated',
                           'output_tokens': 150, 'seconds': 10} for key in self.truth}
            manifest = {'prediction_hash': digest(predictions), 'identity': {
                'config': experiment.variant_config(identity['config'], penalty),
                'code': identity['source_hash'], 'models': identity['models'],
                'adapter': identity['adapter'], 'questions_hash': identity['questions_hash'],
                'retrieval_file_hash': identity['retrieval_file_hash'], 'mode': 'generate'}}
            report = {'samples': 100, 'label': experiment.label(penalty), 'meteor': .6, 'rougeL': .5,
                      'reference_hash': identity['reference_hash'], 'prediction_hash': digest(predictions),
                      'metric_identity': {'fixture': True}, 'prediction_manifest': manifest,
                      'per_question': {key: {'meteor': .6, 'rougeL': .5} for key in self.truth}}
            write_json(directory / 'predictions.json', predictions)
            write_json(directory / 'predictions.audit.json', audit)
            write_json(directory / 'predictions.manifest.json', manifest)
            write_json(directory / 'metrics.json', report)
        return identity

    def test_prepare_reuses_exact_main2_and_changes_only_penalty(self):
        before = {p: file_hash(p) for p in self.upstream.rglob('*') if p.is_file()}
        self.prepare()
        for penalty in experiment.PENALTIES:
            variant = read_json(self.output / experiment.label(penalty) / 'config.json')
            expected = copy.deepcopy(self.base)
            expected['generation']['repetition_penalty'] = penalty
            self.assertEqual(variant, expected)
        self.assertEqual(read_json(self.output / 'data/dev100.questions.json'), self.questions)
        self.assertEqual(file_hash(self.output / 'dev100.retrieval.json'), file_hash(self.upstream / 'dev100.retrieval.json'))
        self.assertEqual(adapter_identity(self.output / 'selected_adapter'), adapter_identity(self.upstream / 'selected_adapter'))
        self.assertEqual(before, {p: file_hash(p) for p in before})

    def test_prepare_rejects_partial_tampered_and_wrong_retrieval(self):
        self.refresh_snapshot('paused')
        with self.assertRaisesRegex(ValueError, 'complete'):
            self.prepare()
        self.refresh_snapshot()
        write_json(self.upstream / 'data/dev100.questions.json', {'bad': {'question': 'different'}})
        with self.assertRaisesRegex(ValueError, 'missing/changed'):
            self.prepare()
        write_json(self.upstream / 'data/dev100.questions.json', self.questions)
        cache = read_json(self.upstream / 'dev100.retrieval.json')
        cache['identity']['index_hash'] = 'wrong'
        write_json(self.upstream / 'dev100.retrieval.json', cache)
        self.refresh_snapshot()
        with self.assertRaisesRegex(ValueError, 'different index'):
            self.prepare()

    def test_resume_copies_checkpoint_and_rejects_changed_identity(self):
        self.prepare()
        checkpoint = self.output / 'rp_1.00/predictions.checkpoint.jsonl'
        checkpoint.write_text('{"id":"0","value":{}}\n', encoding='utf-8')
        resumed = self.root / 'resumed'
        experiment.prepare(self.upstream, self.models, resumed, self.output)
        self.assertEqual((resumed / checkpoint.relative_to(self.output)).read_bytes(), checkpoint.read_bytes())
        identity = read_json(self.output / 'experiment.identity.json')
        identity['adapter'] = {'wrong': True}
        write_json(self.output / 'experiment.identity.json', identity)
        with self.assertRaisesRegex(ValueError, 'Resume identity differs'):
            experiment.prepare(self.upstream, self.models, self.root / 'bad-resume', self.output)

    def test_report_raw_loops_not_hidden_by_clean_final_and_exports(self):
        self.results()
        rows, selected = experiment.summarize(self.output)
        self.assertEqual([row['raw_loop_count'] for row in rows], [3, 1, 2])
        self.assertEqual([row['final_loop_count'] for row in rows], [0, 0, 0])
        self.assertEqual(selected['recommended_penalty'], 1.03)
        self.assertTrue(selected['improvement_found'])
        self.assertEqual(len((self.output / 'per_question.csv').read_text(encoding='utf-8-sig').splitlines()), 301)
        self.assertTrue((self.output / 'rp_1.03/paired_comparison.json').is_file())
        self.assertTrue((self.output / 'comparison.csv').is_file())

    def test_no_promotion_when_either_metric_drops_or_loops_do_not_improve(self):
        rows = [{'repetition_penalty': penalty, 'meteor': .6, 'rougeL': .5,
                 'raw_loop_count': 3 if penalty == 1.0 else 1, 'final_loop_count': 0,
                 'raw_heading_only_count': 0, 'final_heading_only_count': 0,
                 'raw_repetition_mean': .1, 'final_repetition_mean': 0}
                for penalty in experiment.PENALTIES]
        for metric in ('meteor', 'rougeL'):
            bad = copy.deepcopy(rows)
            for row in bad[1:]:
                row[metric] -= .001
            self.assertEqual(experiment.choose(bad)['recommended_penalty'], 1.0)
        for row in rows:
            row['raw_loop_count'] = 0
        self.assertFalse(experiment.choose(rows)['improvement_found'])

    def test_measurement_catches_inline_loop_numeric_run_and_heading_only(self):
        # Minimal reproductions of Main 3 IDs 62249 and 73447, with no gold answers.
        inline = 'Nội dung: ' + ', '.join(['cho thuê mua lại'] * 12)
        numeric = 'Các Điều ' + ', '.join(str(i) for i in range(10, 322))
        heading = ('Căn cứ theo khoản 1 Điều 10 Nghị định 88/2022/NĐ-CP '
                   'quy định về thời hiệu xử phạt như sau:\nThời hiệu xử phạt\n1.')
        for text, reason in [(inline, 'inline_phrase_loop'), (numeric, 'long_numeric_run'),
                             (heading, 'heading_only'), ('## Hồ sơ\na)', 'heading_only')]:
            with self.subTest(reason=reason):
                self.assertIn(reason, experiment.measurement_reasons(text))
        self.assertIn('inline_phrase_loop', experiment.measurement_reasons(
            ', '.join(['CHO THUÊ MUA LẠI', 'cho thuê mua lại'] * 4)))
        # Once substantive content exists, a dangling last item is a different issue.
        self.assertNotIn('heading_only', experiment.measurement_reasons(
            'Hồ sơ:\n1. Đơn đề nghị\n2.'))

    def test_measurement_does_not_flag_short_answers_or_normal_legal_lists(self):
        cases = ['Có.', 'Không phải đăng ký.', '02 năm', 'Hai năm',
                 'Căn cứ quy định như sau:\nHai năm',
                 'Điều 1. Thời hiệu là 02 năm.',
                 'Hồ sơ:\n1. Đơn đề nghị\n2. Bản sao giấy chứng nhận',
                 'Các Điều 1, 2, 3, 4, 5 và 6 của Nghị định 01/2021/NĐ-CP.',
                 'Ngày 01/02/2023, mức phạt 1.000.000 đồng.',
                 ', '.join(['cho thuê mua lại'] * 3),
                 '\n'.join(f'{i}. Nội dung điều kiện thứ {i}' for i in range(1, 25)),
                 'Hồ sơ gồm đơn đề nghị', '']
        for text in cases:
            with self.subTest(text=text):
                reasons = experiment.measurement_reasons(text)
                self.assertFalse(set(reasons) & {'inline_phrase_loop', 'long_numeric_run', 'heading_only'})

    def test_selection_rejects_fewer_loops_but_more_heading_only_answers(self):
        rows = [{'repetition_penalty': p, 'meteor': .6, 'rougeL': .5,
                 'raw_loop_count': 3 if p == 1.0 else 1, 'final_loop_count': 0,
                 'raw_heading_only_count': 0, 'final_heading_only_count': 0,
                 'raw_repetition_mean': 0, 'final_repetition_mean': 0}
                for p in experiment.PENALTIES]
        for key in ('raw_heading_only_count', 'final_heading_only_count'):
            bad = copy.deepcopy(rows)
            for row in bad[1:]:
                row[key] = 1
            self.assertFalse(experiment.choose(bad)['improvement_found'])

    def test_report_exports_new_flags_and_blocks_degenerate_winner(self):
        self.results()
        for penalty in experiment.PENALTIES:
            directory = self.output / experiment.label(penalty)
            audit = read_json(directory / 'predictions.audit.json')
            audit['0']['raw_answer'] = ', '.join(['cho thuê mua lại'] * 8)
            audit['1']['raw_answer'] = ', '.join(str(i) for i in range(1, 40))
            write_json(directory / 'predictions.audit.json', audit)
            if penalty != 1.0:
                pred = read_json(directory / 'predictions.json')
                pred['0']['answer'] = 'Thời hiệu xử phạt\n1.'
                write_json(directory / 'predictions.json', pred)
                manifest = read_json(directory / 'predictions.manifest.json')
                manifest['prediction_hash'] = digest(pred)
                write_json(directory / 'predictions.manifest.json', manifest)
                metrics = read_json(directory / 'metrics.json')
                metrics['prediction_hash'] = digest(pred)
                metrics['prediction_manifest'] = manifest
                write_json(directory / 'metrics.json', metrics)
        with redirect_stdout(io.StringIO()):
            rows, selected = experiment.summarize(self.output)
        self.assertEqual([r['raw_inline_phrase_loop_count'] for r in rows], [1, 1, 1])
        self.assertEqual([r['raw_long_numeric_run_count'] for r in rows], [1, 1, 1])
        self.assertEqual([r['final_heading_only_count'] for r in rows], [0, 1, 1])
        self.assertEqual(selected['recommended_penalty'], 1.0)
        self.assertEqual(read_json(self.output / 'comparison.json')['measurement_policy']['version'], 2)
        import csv
        with (self.output / 'per_question.csv').open(encoding='utf-8-sig', newline='') as stream:
            details = list(csv.DictReader(stream))
        flagged = next(r for r in details if r['id'] == '0' and r['repetition_penalty'] == '1.03')
        self.assertEqual(flagged['final_quality_reasons'], 'heading_only')
        self.assertEqual(flagged['final_loop_reasons'], '')

    def test_report_rejects_incomplete_and_mismatched_metrics(self):
        self.results()
        metrics_path = self.output / 'rp_1.03/metrics.json'
        metrics = read_json(metrics_path)
        write_json(metrics_path, {**metrics, 'reference_hash': 'wrong'})
        with self.assertRaisesRegex(ValueError, 'Metrics do not match'):
            experiment.summarize(self.output)
        write_json(metrics_path, metrics)
        (self.output / 'rp_1.05/predictions.audit.json').unlink()
        with self.assertRaisesRegex(ValueError, 'Incomplete variant'):
            experiment.summarize(self.output)
        self.assertFalse((self.output / 'comparison.json').exists())

    def test_notebook_compiles_and_embeds_exact_helper(self):
        notebook = read_json(ROOT / 'legalqa_dev100_pipeline.ipynb')
        helper = None
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                for node in ast.parse(''.join(cell['source'])).body:
                    if isinstance(node, ast.Assign) and any(
                            isinstance(t, ast.Name) and t.id == 'EXPERIMENT_HELPER' for t in node.targets):
                        helper = ast.literal_eval(node.value)
        self.assertEqual(helper, (ROOT / 'scripts/dev100_repetition.py').read_text(encoding='utf-8'))

    def test_notebook_runs_three_variants_and_exports_on_pause_or_failure(self):
        notebook = read_json(ROOT / 'legalqa_dev100_pipeline.ipynb')
        source = next(''.join(c['source']) for c in notebook['cells']
                      if c['cell_type'] == 'code' and 'def run_variant' in ''.join(c['source']))
        for outcome in ('complete', 'paused', 'failed'):
            with self.subTest(outcome=outcome):
                output = self.root / outcome
                output.mkdir()
                commands = []

                def bounded(command, **kwargs):
                    commands.append(command)
                    if 'generate' in command:
                        self.assertIn('--multi-gpu', command)
                        self.assertEqual(command[command.index('--adapter') + 1], output / 'selected_adapter')
                        if outcome == 'failed':
                            raise RuntimeError('fixture worker failed')
                        if outcome == 'paused':
                            return
                        path = command[command.index('--output') + 1]
                        for suffix in ('.json', '.audit.json', '.manifest.json'):
                            write_json(path.with_suffix(suffix), {})

                namespace = {'RUN_ROOT': output, 'PENALTIES': experiment.PENALTIES,
                             'MAX_NEW_QUESTIONS_PER_VARIANT': 0, 'WORK_END': 1e20,
                             'CODE': ROOT, 'MODELS': self.models, 'HELPER': self.root / 'helper.py',
                             'PIN': 'a' * 40, 'bounded_process': bounded, 'BudgetPause': TimeoutError,
                             'os': os, 'sys': __import__('sys'), 'time': __import__('time'), 'json': json}
                if outcome == 'failed':
                    with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'fixture worker failed'):
                        exec(source, namespace)
                else:
                    with redirect_stdout(io.StringIO()):
                        exec(source, namespace)
                self.assertEqual(read_json(output / 'experiment.status.json')['status'], outcome)
                self.assertEqual(sum('generate' in c for c in commands), 3 if outcome == 'complete' else 1)
                self.assertEqual(sum('evaluate' in c for c in commands), 3 if outcome == 'complete' else 0)
                self.assertEqual(sum('summarize' in c for c in commands), int(outcome == 'complete'))
                with ZipFile(output / 'dev100_repetition_diagnostics.zip') as archive:
                    self.assertIn('experiment.status.json', archive.namelist())
                    if outcome == 'complete':
                        self.assertIn('rp_1.00/predictions.json', archive.namelist())
                        self.assertIn('rp_1.05/predictions.json', archive.namelist())


if __name__ == '__main__':
    unittest.main()
