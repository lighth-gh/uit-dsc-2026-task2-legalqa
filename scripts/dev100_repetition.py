"""Helpers embedded in dev100 notebook; execute against the Main 2 pinned checkout."""
import argparse
import copy
import csv
import io
from pathlib import Path
import re
import sys
import unicodedata

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
MEASUREMENT_POLICY = {
    'version': 2, 'inline_min_phrase_words': 3, 'inline_max_phrase_words': 24,
    'inline_min_repeats': 4, 'inline_min_total_words': 24,
    'numeric_min_run': 20, 'heading_max_words': 16,
}


def inline_phrase_loop(text):
    """Repeated adjacent word blocks within one line; punctuation/case are ignored."""
    policy = MEASUREMENT_POLICY
    for line in text.splitlines():
        words = re.findall(r'[^\W_]+', line.casefold())
        for width in range(policy['inline_min_phrase_words'], policy['inline_max_phrase_words'] + 1):
            needed = max(policy['inline_min_repeats'],
                         (policy['inline_min_total_words'] + width - 1) // width)
            for start in range(len(words) - width * needed + 1):
                block = words[start:start + width]
                if not any(any(c.isalpha() for c in word) for word in block):
                    continue
                if all(words[start + n * width:start + (n + 1) * width] == block
                       for n in range(1, needed)):
                    return True
    return False


def long_numeric_run(text):
    """Consecutive integers separated only by whitespace/commas/semicolons."""
    previous, end, count = None, 0, 0
    for match in re.finditer(r'(?<![\w./-])\d{1,6}(?![\w./-])', text):
        number = int(match.group())
        contiguous = previous is not None and re.fullmatch(r'[\s,;]+', text[end:match.start()])
        count = count + 1 if contiguous and number == previous + 1 else 1
        if count >= MEASUREMENT_POLICY['numeric_min_run']:
            return True
        previous, end = number, match.end()
    return False


def heading_only(text):
    """Conservative structural check; a short substantive answer is not a heading."""
    lines = [line.strip().strip('#>*_` ').strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    scaffold, headings = False, []
    for line in lines:
        if re.fullmatch(r'(?:\d+(?:\.\d+)*|[a-zđ]|[ivxlcdm]+)[.)]', line, re.I):
            scaffold = True
            continue
        if re.match(r'^(?:căn cứ|theo)\b', line, re.I) and re.search(
                r'(?:như sau|sau đây)\s*[:.]?$', line, re.I):
            scaffold = True
            continue
        section = re.match(r'^(?:điều|khoản|chương|mục|phần)\s+(?:\d+|[ivxlcdm]+)[.:]?\s*', line, re.I)
        if section:
            scaffold = True
            line = line[section.end():].strip()
        elif re.match(r'^(?:\d+|[a-zđ])[.)]\s+\S', line, re.I):
            return False  # A list item with a body, e.g. "1. Đơn đề nghị".
        if not line:
            continue
        if (len(line.split()) > MEASUREMENT_POLICY['heading_max_words']
                or re.search(r'[.;!?]', line)
                or re.search(r'\b(?:là|phải|được|không|có|gồm|cần|chịu)\b', line, re.I)):
            return False
        headings.append((line, bool(section)))
    if not headings:
        return scaffold
    # Require a recognizable title, never just an introduction or low word count.
    return all(not re.search(r'\d', line) and (
        is_section or line.endswith(':') or re.match(
            r'^(?:thời hiệu|thời hạn|điều kiện|hồ sơ|thủ tục|mức phạt|trách nhiệm|'
            r'quyền hạn|căn cứ pháp lý|nội dung|hình thức thi|kiểm định|nguyên tắc)\b', line, re.I))
        for line, is_section in headings)


def measurement_reasons(text):
    text = unicodedata.normalize('NFC', text)
    reasons = set(loop_reasons(text))
    if inline_phrase_loop(text):
        reasons.add('inline_phrase_loop')
    if long_numeric_run(text):
        reasons.add('long_numeric_run')
    if heading_only(text):
        reasons.add('heading_only')
    return sorted(reasons)


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
            and row['raw_heading_only_count'] <= baseline['raw_heading_only_count']
            and row['final_heading_only_count'] <= baseline['final_heading_only_count']
            and row['raw_repetition_mean'] <= baseline['raw_repetition_mean'] + 1e-12
            and row['final_repetition_mean'] <= baseline['final_repetition_mean'] + 1e-12)
    candidates = [row for row in rows if row['passes_screen']]
    best = min(candidates, key=lambda row: (row['raw_loop_count'], row['final_loop_count'],
                                           -row['meteor'], -row['rougeL'], row['repetition_penalty'])) if candidates else baseline
    return {'recommended_penalty': best['repetition_penalty'],
            'improvement_found': bool(candidates),
            'rule': 'Both METEOR and ROUGE-L >= baseline; fewer raw loops; no increase in final loops, '
                    'raw/final heading-only answers or mean repetition. '
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
            raw_reasons, final_reasons = measurement_reasons(raw), measurement_reasons(final)
            row = {'id': key, 'repetition_penalty': penalty,
                   'meteor': report['per_question'][key]['meteor'],
                   'rougeL': report['per_question'][key]['rougeL'],
                   'raw_quality_reasons': '|'.join(raw_reasons),
                   'final_quality_reasons': '|'.join(final_reasons),
                   'raw_loop_reasons': '|'.join(r for r in raw_reasons if r != 'heading_only'),
                   'final_loop_reasons': '|'.join(r for r in final_reasons if r != 'heading_only'),
                   **{f'{side}_{reason}': reason in reasons
                      for side, reasons in [('raw', raw_reasons), ('final', final_reasons)]
                      for reason in ('inline_phrase_loop', 'long_numeric_run', 'heading_only')},
                   'raw_repetition': repetition_ratio(raw), 'final_repetition': repetition_ratio(final),
                   'hit_token_limit': bool(item['hit_token_limit']), 'route': item['route'],
                   'output_tokens': item['output_tokens'], 'generation_seconds': item['seconds']}
            local.append(row)
        details.extend(local)
        rows.append({'repetition_penalty': penalty, 'samples': 100,
                     'meteor': report['meteor'], 'rougeL': report['rougeL'],
                     'raw_loop_count': sum(bool(r['raw_loop_reasons']) for r in local),
                     'final_loop_count': sum(bool(r['final_loop_reasons']) for r in local),
                     **{f'{side}_{reason}_count': sum(r[f'{side}_{reason}'] for r in local)
                        for side in ('raw', 'final')
                        for reason in ('inline_phrase_loop', 'long_numeric_run', 'heading_only')},
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
        'measurement_policy': MEASUREMENT_POLICY,
        'loop_definition': 'Existing loop_reasons plus inline phrase loops and long numeric runs; raw and final. '
                           'Heading-only is counted separately and gates selection.',
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
