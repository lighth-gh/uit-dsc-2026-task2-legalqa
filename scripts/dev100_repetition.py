"""Helpers embedded in dev100 notebook; execute against the Main 2 pinned checkout."""
import argparse
import copy
import csv
import io
from pathlib import Path
import sys

# The notebook writes this script outside CODE and starts it with cwd=CODE.
sys.path.insert(0, str(Path.cwd()))

from legalqa.generation import adapter_identity
from legalqa.io import (config, copy_file, digest, file_hash, load_questions,
                       read_json, source_hash, validate_predictions, write_json)
from legalqa.metrics import compare_reports, references
from legalqa.models import model_lock
from legalqa.repair import repetition_ratio
from legalqa.repair_v2 import loop_reasons
from legalqa.retrieval import read_retrieval
from legalqa.stages import generation_ready, validate_selection, verify_snapshot


PENALTIES = (1.0, 1.03, 1.05)


def label(penalty):
    return f'rp_{penalty:.2f}'


def variant_config(base, penalty):
    if penalty not in PENALTIES:
        raise ValueError('Unsupported repetition penalty')
    result = copy.deepcopy(base)
    result['generation']['repetition_penalty'] = penalty
    return result


def copy_checked(source, target):
    target = Path(target)
    if target.exists():
        if file_hash(source) != file_hash(target):
            raise ValueError(f'Conflicting artifact; choose a new output: {target}')
    else:
        copy_file(source, target)


def prepare(upstream, models, output, previous=None):
    upstream, models, output = map(Path, (upstream, models, output))
    snapshot = verify_snapshot(upstream, 2)
    if snapshot['status'] != 'complete':
        raise ValueError('Main 2 must be complete')
    required = {'config.json', 'session.json', 'models.lock.json', 'selection.json',
                'data/dev100.questions.json', 'data/dev100.references.json',
                'data/split_manifest.json', 'dev100.retrieval.json',
                'selected_adapter/adapter_config.json',
                'selected_adapter/adapter_model.safetensors'}
    if not required.issubset(snapshot['files']):
        raise ValueError(f'Main 2 lacks required artifacts: {required - set(snapshot["files"])}')
    base = read_json(upstream / 'config.json')
    session = read_json(upstream / 'session.json')
    if source_hash() != snapshot['source_hash'] or config() != base:
        raise ValueError('Use the exact Main 2 pinned code/config')
    if session['config_hash'] != digest(base):
        raise ValueError('Main 2 config hash mismatch')
    lock = model_lock(base, models)
    if lock != read_json(upstream / 'models.lock.json') or lock != session['models']:
        raise ValueError('Version 3 model revisions differ from Main 2')
    questions = load_questions(upstream / 'data/dev100.questions.json')
    truth = references(upstream / 'data/dev100.references.json')
    if len(questions) != 100 or set(questions) != set(truth):
        raise ValueError('Expected the same 100 dev questions and references from Main 2')
    selection = validate_selection(upstream)
    read_retrieval(upstream / 'dev100.retrieval.json', questions, base, models,
                   expected_mode='full', expected_index_hash=session['index_hash'])
    identity = {
        'schema': 1, 'code_commit': snapshot['code_commit'], 'source_hash': source_hash(),
        'helper_hash': digest(Path(__file__).read_text(encoding='utf-8')),
        'upstream_manifest_hash': digest(snapshot), 'config': base, 'models': lock,
        'adapter': adapter_identity(upstream / 'selected_adapter'),
        'questions_hash': digest(questions), 'reference_hash': digest(truth),
        'retrieval_file_hash': file_hash(upstream / 'dev100.retrieval.json'),
        'penalties': list(PENALTIES), 'expected_questions': 100,
    }
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / 'experiment.identity.json'
    if identity_path.exists() and read_json(identity_path) != identity:
        raise ValueError('Existing experiment identity differs; use a new output')
    if previous is not None:
        previous = Path(previous)
        if previous.resolve() == output.resolve():
            raise ValueError('Previous output must be a separate input directory')
        if read_json(previous / 'experiment.identity.json') != identity:
            raise ValueError('Resume identity differs; attach the same Main 2 and notebook version')
        # Restore only experiment files, excluding generated archives and temporary writes.
        for path in sorted(previous.rglob('*')):
            if path.is_file() and not path.is_symlink() and path.suffix in {'.json', '.jsonl', '.csv', '.txt'}:
                if not path.name.endswith('.tmp'):
                    copy_checked(path, output / path.relative_to(previous))
    for relative in sorted(required - {'session.json'}):
        copy_checked(upstream / relative, output / relative)
    write_json(identity_path, identity)
    for penalty in PENALTIES:
        path = output / label(penalty) / 'config.json'
        candidate = variant_config(base, penalty)
        if path.exists() and read_json(path) != candidate:
            raise ValueError(f'Variant config changed: {path}')
        write_json(path, candidate)
    print('Reuse Main 2 dev100, retrieval and adapter:', selection['label'], flush=True)
    print('Evaluation objective:', base['evaluation'], flush=True)
    print('Retrieval settings (unchanged):', base['retrieval'], flush=True)
    print('Generation settings (only repetition_penalty varies):', base['generation'], flush=True)
    return identity


def choose(rows):
    baseline = next(row for row in rows if row['repetition_penalty'] == 1.0)
    for row in rows:
        row['delta_meteor'] = row['meteor'] - baseline['meteor']
        row['delta_rougeL'] = row['rougeL'] - baseline['rougeL']
        row['quality_preserved'] = row['delta_meteor'] >= -1e-12 and row['delta_rougeL'] >= -1e-12
        row['passes_screen'] = (
            row['repetition_penalty'] != 1.0 and row['quality_preserved']
            and row['raw_loop_count'] < baseline['raw_loop_count']
            and row['final_loop_count'] <= baseline['final_loop_count']
            and row['raw_repetition_mean'] <= baseline['raw_repetition_mean'] + 1e-12
            and row['final_repetition_mean'] <= baseline['final_repetition_mean'] + 1e-12)
    candidates = [row for row in rows if row['passes_screen']]
    best = min(candidates, key=lambda row: (row['raw_loop_count'], row['final_loop_count'],
                                           -row['meteor'], -row['rougeL'], row['repetition_penalty'])) if candidates else baseline
    return {'recommended_penalty': best['repetition_penalty'],
            'improvement_found': bool(candidates),
            'rule': 'Both METEOR and ROUGE-L >= baseline; fewer raw loops; no increase in final loops or mean repetition. '
                    'Rank by raw loops, final loops, METEOR, ROUGE-L, then lower penalty.',
            'note': 'Dev100 was used for adapter selection; this is tuning, not independent validation. '
                    'No Main 3 configuration is changed automatically.'}


def csv_report(path, rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    Path(path).write_text(stream.getvalue(), encoding='utf-8-sig', newline='')


def summarize(output):
    output = Path(output)
    identity = read_json(output / 'experiment.identity.json')
    truth = references(output / 'data/dev100.references.json')
    if len(truth) != 100 or digest(truth) != identity['reference_hash']:
        raise ValueError('Comparison requires the original 100 references')
    rows, details, metric_identity = [], [], None
    for penalty in PENALTIES:
        directory = output / label(penalty)
        prediction = directory / 'predictions.json'
        if not generation_ready(prediction):
            raise ValueError(f'Incomplete variant: {label(penalty)}; resume before comparing')
        pred = read_json(prediction)
        validate_predictions(pred, truth)
        report = read_json(directory / 'metrics.json')
        audit = read_json(prediction.with_suffix('.audit.json'))
        manifest = read_json(prediction.with_suffix('.manifest.json'))
        expected_config = variant_config(identity['config'], penalty)
        generated = manifest['identity']
        expected = {'config': expected_config, 'code': identity['source_hash'],
                    'adapter': identity['adapter'], 'models': identity['models'],
                    'questions_hash': identity['questions_hash'],
                    'retrieval_file_hash': identity['retrieval_file_hash'], 'mode': 'generate'}
        if any(generated.get(key) != value for key, value in expected.items()):
            raise ValueError(f'Generation provenance mismatch: {label(penalty)}')
        if (report['samples'] != 100 or set(report['per_question']) != set(truth)
                or report['reference_hash'] != identity['reference_hash']
                or report['prediction_hash'] != digest(pred)
                or report['prediction_manifest'] != manifest):
            raise ValueError(f'Metrics do not match this variant: {label(penalty)}')
        if metric_identity is not None and report['metric_identity'] != metric_identity:
            raise ValueError('Metric implementations differ')
        metric_identity = report['metric_identity']
        local = []
        for key in truth:
            item = audit[key]
            raw, final = item['raw_answer'], pred[key]['answer']
            row = {'id': key, 'repetition_penalty': penalty,
                   'meteor': report['per_question'][key]['meteor'],
                   'rougeL': report['per_question'][key]['rougeL'],
                   'raw_loop_reasons': '|'.join(loop_reasons(raw)),
                   'final_loop_reasons': '|'.join(loop_reasons(final)),
                   'raw_repetition': repetition_ratio(raw), 'final_repetition': repetition_ratio(final),
                   'hit_token_limit': bool(item['hit_token_limit']), 'route': item['route'],
                   'output_tokens': item['output_tokens'], 'generation_seconds': item['seconds']}
            local.append(row)
        details.extend(local)
        rows.append({'repetition_penalty': penalty, 'samples': 100,
                     'meteor': report['meteor'], 'rougeL': report['rougeL'],
                     'raw_loop_count': sum(bool(r['raw_loop_reasons']) for r in local),
                     'final_loop_count': sum(bool(r['final_loop_reasons']) for r in local),
                     'raw_repetition_mean': sum(r['raw_repetition'] for r in local) / 100,
                     'final_repetition_mean': sum(r['final_repetition'] for r in local) / 100,
                     'token_limit_count': sum(r['hit_token_limit'] for r in local),
                     'fallback_count': sum('fallback' in r['route'] for r in local),
                     'mean_output_tokens': sum(r['output_tokens'] for r in local) / 100,
                     'mean_question_seconds': sum(r['generation_seconds'] for r in local) / 100})
    selection = choose(rows)
    for penalty in PENALTIES[1:]:
        compare_reports(output / label(1.0) / 'metrics.json',
                        output / label(penalty) / 'metrics.json',
                        output / label(penalty) / 'paired_comparison.json',
                        seed=identity['config']['seed'])
    write_json(output / 'comparison.json', {'rows': rows, 'selection': selection,
        'loop_definition': 'legalqa.repair_v2.loop_reasons, measured on raw and final answers',
        'repetition_definition': 'duplicate fraction of lines >=35 characters; heuristic, not legal correctness'})
    csv_report(output / 'comparison.csv', rows)
    csv_report(output / 'per_question.csv', details)
    print('DEV100 REPETITION COMPARISON COMPLETE', flush=True)
    for row in rows:
        print(row, flush=True)
    print(selection, flush=True)
    return rows, selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'summarize'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--upstream', type=Path)
    parser.add_argument('--models', type=Path)
    parser.add_argument('--previous', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        if not args.upstream or not args.models:
            parser.error('prepare requires --upstream and --models')
        prepare(args.upstream, args.models, args.output, args.previous)
    else:
        summarize(args.output)


if __name__ == '__main__':
    main()
