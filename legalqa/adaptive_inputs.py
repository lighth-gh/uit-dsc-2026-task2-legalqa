"""Verified Stage 2 + external submission inputs for adaptive Main 04 repair.

Archives are data only. A mismatched Stage 3 is never used to infer private routes.
"""
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from .io import _unique_pairs, digest, file_hash, validate_predictions
from .repair import _require, load_diagnostics, repair_predictions, POLICY


class Snapshot:
    """Read a manifest-backed ZIP or an unpacked Kaggle diagnostics directory."""

    def __init__(self, path, stage=2):
        self.path = Path(path)
        self.archive = None
        try:
            if self.path.is_dir():
                names = [p.relative_to(self.path).as_posix() for p in self.path.rglob('*') if p.is_file()]
            else:
                self.archive = ZipFile(self.path)
                names = self.archive.namelist()
                _require(self.archive.testzip() is None, 'ZIP CRC failed')
            _require(len(names) == len(set(names)), 'Duplicate archive member')
            for name in names:
                part = PurePosixPath(name)
                _require(not part.is_absolute() and '..' not in part.parts
                         and ':' not in name and '\\' not in name, 'Invalid diagnostic path')
            self.names = set(names)
            manifest_name = f'stage{stage}_manifest.json'
            self.manifest = self.read(manifest_name)
            m = self.manifest
            _require(m.get('stage') == stage and m.get('schema') == 2
                     and m.get('quality_version') == 'v8', 'Expected Stage 2 V8 snapshot')
            _require(m.get('status') == 'complete' and m.get('progress', {}).get('complete'),
                     'Stage 2 is not complete')
            omitted = set()
            for name, info in m['files'].items():
                if name not in self.names:
                    _require(bool(re.fullmatch(r'(selected_adapter|sft/epoch-\d+)/adapter_model\.safetensors', name)),
                             f'Missing manifest file: {name}')
                    omitted.add(name)
                    continue
                data = self.bytes(name)
                _require(len(data) == info['size'] and hashlib.sha256(data).hexdigest() == info['sha256'],
                         f'Manifest hash/size mismatch: {name}')
            _require(self.names == (set(m['files']) - omitted) | {manifest_name},
                     'Unmanifested diagnostic files')
            self.identity = digest(m)
        except BaseException:
            self.close()
            raise

    def bytes(self, name):
        _require(name in self.names, f'Missing diagnostic: {name}')
        if self.archive:
            return self.archive.read(name)
        target = (self.path / name).resolve()
        _require(target.is_relative_to(self.path.resolve()), 'Unsafe diagnostic path')
        return target.read_bytes()

    def read(self, name):
        return json.loads(self.bytes(name), object_pairs_hook=_unique_pairs)

    def journal(self, name):
        rows = {}
        for line in self.bytes(name).splitlines(keepends=True):
            _require(line.endswith(b'\n'), 'Incomplete journal line')
            row = json.loads(line, object_pairs_hook=_unique_pairs)
            _require(row['id'] not in rows, 'Duplicate journal ID')
            rows[row['id']] = row['value']
        return rows

    def close(self):
        if self.archive:
            self.archive.close()


def read_submission(path):
    path = Path(path)
    if path.is_dir():
        path = path / 'submission.json'
    if path.suffix.lower() == '.zip':
        with ZipFile(path) as z:
            _require(z.namelist() == ['submission.json'], 'Submission ZIP must contain only submission.json')
            _require(z.testzip() is None, 'Submission ZIP CRC failed')
            raw = z.read('submission.json')
    else:
        raw = path.read_bytes()
    return json.loads(raw.decode('utf-8-sig'), object_pairs_hook=_unique_pairs)


def load_adaptive_inputs(stage2, submission, private_diagnostics=None):
    s = Snapshot(stage2)
    try:
        config, session = s.read('config.json'), s.read('session.json')
        _require(digest(config) == session['config_hash'], 'Session/config mismatch')
        _require(session['source_hash'] == s.manifest['source_hash'], 'Session/code mismatch')
        models = s.read('models.lock.json')
        _require(models == session['models'], 'Session/model mismatch')
        selection = s.read('selection.json')
        label = selection['label']
        _require(bool(re.fullmatch(r'epoch-\d+', label)), 'Invalid adapter selection')
        pm = s.read(f'dev100.{label}.manifest.json')
        expected = pm['identity']
        _require(selection['prediction_manifest'] == pm, 'Selection/dev manifest mismatch')
        _require(expected['config'] == config and expected['models'] == models
                 and expected['code'] == session['source_hash'], 'Dev provenance mismatch')
        for name, sha in expected['adapter'].items():
            _require(s.manifest['files'][f'selected_adapter/{name}']['sha256'] == sha,
                     'Selected adapter hash mismatch')
        bundle = {'config': config, 'expected': expected}
        for split, prefix in [('dev', 'dev100'), ('private', 'public')]:
            q = s.read(f'data/{"dev100" if split == "dev" else "test"}.questions.json')
            _require(q and all(isinstance(v.get('question'), str) and v['question'].strip()
                               for v in q.values()), 'Invalid questions')
            retrieval = s.read(f'{prefix}.retrieval.json')
            ri, records = retrieval['identity'], retrieval['records']
            _require(ri['questions_hash'] == digest(q) and ri['models'] == models
                     and ri['code'] == session['source_hash']
                     and ri['retrieval'] == config['retrieval']
                     and ri['index_hash'] == session['index_hash'], 'Retrieval provenance mismatch')
            _require(set(q) == set(records), 'Retrieval IDs differ')
            _require(s.read(f'{prefix}.retrieval.checkpoint.jsonl.meta.json') == ri,
                     'Retrieval journal identity mismatch')
            _require(s.journal(f'{prefix}.retrieval.checkpoint.jsonl') == records,
                     'Retrieval journal content mismatch')
            for key, record in records.items():
                _require(record['question'] == q[key]['question'], 'Retrieval/question mismatch')
                _require(record['contexts'] and all(c['text'].strip() for c in record['contexts']),
                         'Empty evidence')
            bundle[split] = {'questions': q, 'records': records, 'audit': None}
            if split == 'dev':
                _require(expected['retrieval'] == ri and expected['questions_hash'] == digest(q)
                         and expected['retrieval_file_hash'] == hashlib.sha256(
                             s.bytes(f'{prefix}.retrieval.json')).hexdigest(), 'Dev retrieval mismatch')
        dev = bundle['dev']
        stem = f'dev100.{label}'
        pred, audit = s.read(stem + '.json'), s.read(stem + '.audit.json')
        validate_predictions(pred, dev['questions'])
        _require(set(audit) == set(pred) and digest(pred) == pm['prediction_hash'], 'Dev predictions mismatch')
        _require(s.read(stem + '.checkpoint.jsonl.meta.json') == expected, 'Dev journal identity mismatch')
        _require(s.journal(stem + '.checkpoint.jsonl') == {
            k: {'prediction': pred[k], 'audit': audit[k]} for k in pred}, 'Dev journal content mismatch')
        for k, row in audit.items():
            _require(set(row['context_parent_ids']) <= {c['parent_id'] for c in dev['records'][k]['contexts']},
                     'Unknown dev context parent')
        refs = s.read('data/dev100.references.json')
        refs = {k: v['answer'] if isinstance(v, dict) else v for k, v in refs.items()}
        _require(set(refs) == set(pred) and all(isinstance(v, str) and v.strip() for v in refs.values()),
                 'Invalid dev references')
        metrics = s.read(stem + '.metrics.json')
        _require(digest(refs) == metrics['reference_hash'] == selection['reference_hash']
                 and metrics['prediction_hash'] == pm['prediction_hash'], 'Dev metric hashes mismatch')
        _require(all(abs(metrics[k] - selection[k]) <= 1e-12 for k in ('meteor', 'rougeL')),
                 'Selection metrics mismatch')
        dev.update(predictions=repair_predictions(pred, audit, POLICY)[0], original=pred,
                   audit=audit, references=refs, stored_metrics=metrics)
        private = bundle['private']
        private['predictions'] = read_submission(submission)
        validate_predictions(private['predictions'], private['questions'])
        source = {'stage2': s.identity, 'baseline': digest(private['predictions']),
                  'private_audit': None}
        if private_diagnostics:
            # Keep the established complete-Stage-3 verifier unchanged.
            other = load_diagnostics(private_diagnostics)
            _require(other['public']['questions'] == private['questions'], 'Private audit IDs/questions differ')
            _require(other['config'] == config and other['public']['records'] == private['records'],
                     'Private audit config/retrieval provenance differs')
            ident = other['public']['prediction_manifest']['identity']
            _require(ident['adapter'] == expected['adapter'] and ident['models'] == models,
                     'Private audit model/adapter provenance differs')
            private.update(audit=other['public']['audit'], original=other['public']['predictions'])
            source['private_audit'] = file_hash(private_diagnostics)
        bundle['source'] = source
        return bundle
    finally:
        s.close()
