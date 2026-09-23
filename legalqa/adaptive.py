"""Selective fallback/truncation regeneration; no gold enters candidate selection."""
import argparse
import copy
from collections import Counter
from pathlib import Path

from .adaptive_inputs import load_adaptive_inputs
from .io import Journal, digest, file_hash, read_json, source_hash, validate_predictions, write_json
from .prompts import (answer_flags, citation_context_conflict, extractive_fallback,
                      localized_evidence_support, pack_prompt, refusal_evidence_support)
from .repair import POLICY as CPU_POLICY, _incomplete, _package, _require, deduplicate_answer, repetition_ratio
from .repair_v2 import loop_reasons
from .runtime import should_pause


POLICY = {'version': 1, 'budgets': [2048, 3072, 4096], 'input_tokens': 4096,
          'total_tokens': 8192, 'max_attempts': 4, 'near_limit_ratio': .90,
          'coverage': .65, 'phrase_words': 3, 'baseline_private_score': .5713}
SUFFIX = {
    'truncated': 'Đây là lượt viết lại câu trả lời có thể bị cắt ngắn. Trả lời đủ các ý được hỏi, '
                 'điều kiện, ngoại lệ, mức tiền và thời hạn có căn cứ; không chép phần không liên quan. '
                 'Mỗi ý chỉ trình bày một lần, giữ nguyên thuật ngữ pháp lý.',
    'fallback': 'Đây là lượt viết lại câu trả lời từ nguồn. Trả lời trực tiếp dựa vào trích đoạn; '
                'chỉ ghi số hiệu văn bản hiện diện trong trích đoạn, không bổ sung số hiệu từ trí nhớ. '
                'Giữ đủ các ý có căn cứ, điều kiện và ngoại lệ; mỗi ý chỉ trình bày một lần.',
}


def candidate_queue(data, config, tokenizer, detection):
    _require(detection in {'audit', 'heuristic'}, 'Unknown detection mode')
    _require(detection != 'audit' or data.get('audit') is not None, 'Audit is unavailable')
    queue = {}
    for key, value in data['predictions'].items():
        text, reasons = value['answer'], []
        tokens = len(tokenizer(text, add_special_tokens=False)['input_ids'])
        fallback = False
        if detection == 'audit':
            row = data['audit'][key]
            fallback = 'fallback' in row['route']
            if fallback:
                reasons.append('audit_fallback')
            if row.get('hit_token_limit') or row['route'] == 'generated_truncated':
                reasons.append('audit_token_limit')
        else:
            if tokens >= POLICY['near_limit_ratio'] * config['generation']['max_new_tokens']:
                reasons.append('suspected_truncated_near_limit')
            _, packed = pack_prompt(data['questions'][key]['question'], data['records'][key]['contexts'],
                                    tokenizer, config)
            support = refusal_evidence_support(data['questions'][key]['question'], packed)
            rebuilt = extractive_fallback(data['questions'][key]['question'], packed, support)
            # Compare also to the known conservative CPU cleanup, never to gold.
            fallback = text in (rebuilt, deduplicate_answer(rebuilt, CPU_POLICY)[0])
            if fallback:
                reasons.append('suspected_fallback_exact_reconstruction')
        if _incomplete(text):
            reasons.append('unfinished_lead_in')
        if reasons:
            queue[key] = {'group': 'fallback' if fallback else 'truncated', 'reasons': reasons,
                          'detection': detection, 'baseline_tokens': tokens,
                          'baseline_differs_from_original': (
                              data.get('original', {}).get(key) != value if data.get('original') else None)}
    return dict(sorted(queue.items()))


def focused_contexts(question, contexts):
    support = localized_evidence_support(question, contexts)
    best = support.get('context_index')
    if best is None:
        return contexts[:1]
    chosen = {best}
    document = contexts[best].get('doc_id')
    if document:
        other = next((i for i, c in enumerate(contexts)
                      if i != best and c.get('doc_id') == document), None)
        if other is not None:
            chosen.add(other)
    return [c for i, c in enumerate(contexts) if i in chosen]


def attempt_config(base, group, tokens, focused=False):
    config = copy.deepcopy(base)
    config['generation'].update(max_input_tokens=POLICY['input_tokens'], max_new_tokens=tokens,
                                repetition_penalty=1.0, no_repeat_ngram_size=0,
                                contexts_k=4, same_document_as_top=False,
                                complete_legal_units=focused, system_suffix=SUFFIX[group])
    if focused:
        config['generation']['system_suffix'] += (
            ' Chỉ dùng các trích đoạn đã chọn, không trộn thực thể hoặc tự tạo mục lặp; '
            'nếu thiếu số hiệu thì không đoán số hiệu.')
    return config


def check_candidate(value, question, packed):
    """Recheck the generated text against precisely the evidence shown to the model."""
    row = value['audit']
    text, edits = deduplicate_answer(value['prediction']['answer'], CPU_POLICY)
    flags = answer_flags(text, '\n'.join(c['text'] for c in packed))
    support = localized_evidence_support(question, packed)
    conflict = citation_context_conflict(question, text, packed)
    reasons = []
    if row['hit_token_limit']:
        reasons.append('hit_token_limit')
    if row['route'] != 'generated':
        reasons.append('route:' + row['route'])
    if not row.get('ended_with_eos', False):
        reasons.append('no_natural_stop')
    if _incomplete(text):
        reasons.append('unfinished_lead_in')
    if loop_reasons(text):
        reasons.append('loop')
    if len(text) < .2 * len(value['prediction']['answer']) and len(text.split()) < 40:
        reasons.append('dedup_too_short')
    for name in ('empty', 'artifact', 'refusal', 'unsupported_document_numbers'):
        if flags.get(name) or row.get('flags_before_fallback', {}).get(name):
            reasons.append('guard:' + name)
    if conflict['conflict'] or row.get('entity_conflict', {}).get('conflict'):
        reasons.append('entity_conflict')
    if (support['coverage'] < POLICY['coverage'] or support['longest_phrase'] < POLICY['phrase_words']
            or not support.get('marker_ok', True) or not support.get('age_value_present', True)):
        reasons.append('insufficient_evidence')
    # Retry routing uses raw loops even if CPU dedup concealed them.
    faulty = (bool(loop_reasons(row.get('raw_answer', text)))
              or any(r.startswith('guard:') for r in reasons)
              or any(r in reasons for r in ('loop', 'entity_conflict', 'insufficient_evidence'))
              or 'fallback' in row['route'])
    return {'prediction': {'answer': text}, 'accepted': not reasons, 'rejected': reasons,
            'faulty': faulty, 'support': support, 'dedup_changes': edits}


def next_attempt(previous, ladder):
    """Return (budget, focus) or stop. There is only one focused repair per ID."""
    if not previous:
        return ladder[0], False
    last = previous[-1]
    if last['check']['accepted'] or len(previous) >= POLICY['max_attempts']:
        return None
    if last['check']['faulty']:
        return None if any(p['focused'] for p in previous) else (last['budget'], True)
    if last['generation']['audit']['hit_token_limit']:
        larger = next((n for n in ladder if n > last['budget']), None)
        if larger:
            return larger, last['focused']
    return None


def token_ladder(model_limit):
    room = min(POLICY['total_tokens'], model_limit) - POLICY['input_tokens']
    _require(room >= POLICY['budgets'][0], 'Model window cannot fit 4096 input + 2048 output tokens')
    return sorted({min(n, room) for n in POLICY['budgets']})


class SessionBudget:
    def __init__(self, maximum, pause=should_pause):
        _require(maximum > 0, 'max_items must be positive')
        self.maximum, self.pause, self.touched = maximum, pause, set()

    def allow(self, key):
        if self.pause() or (key not in self.touched and len(self.touched) >= self.maximum):
            return False
        self.touched.add(key)
        return True


def run_queue(data, queue, config, tokenizer, generate, root, identity, ladder, budget, split):
    root = Path(root)
    journal = Journal(root / 'attempts.jsonl', {**identity, 'queue': queue, 'split': split})
    merged = copy.deepcopy(data['predictions'])
    outcomes = {}
    valid_keys = {f'{k}:{n}' for k in queue for n in range(1, POLICY['max_attempts'] + 1)}
    _require(set(journal.records) <= valid_keys, 'Unknown attempt journal key')
    for key, item in queue.items():
        previous = []
        while True:
            schedule = next_attempt(previous, ladder)
            if schedule is None:
                _require(not any(f'{key}:{n}' in journal.records
                                 for n in range(len(previous) + 1, POLICY['max_attempts'] + 1)),
                         'Unexpected attempts after terminal decision')
                break
            tokens, focus = schedule
            attempt_id = f'{key}:{len(previous) + 1}'
            cfg = attempt_config(config, item['group'], tokens, focus)
            question = data['questions'][key]['question']
            record = copy.deepcopy(data['records'][key])
            if focus:
                record['contexts'] = focused_contexts(question, record['contexts'])
            prompt_ids, packed = pack_prompt(question, record['contexts'], tokenizer, cfg)
            _require(len(prompt_ids) <= POLICY['input_tokens']
                     and len(prompt_ids) + tokens <= identity['context_limit'], 'Prompt/output exceeds context limit')
            if attempt_id in journal.records:
                saved = journal.records[attempt_id]
                checked = check_candidate(saved['generation'], question, packed)
                _require(saved['budget'] == tokens and saved['focused'] == focus
                         and saved['prompt_hash'] == digest(prompt_ids) and saved['check'] == checked,
                         'Attempt journal content differs')
            else:
                if not budget.allow((split, key)):
                    write_json(root / 'status.json', {'status': 'paused', 'finished': len(outcomes), 'total': len(queue)})
                    return None
                print(f'adaptive {split} {key} attempt={len(previous)+1} max_new_tokens={tokens} '
                      f'focused={focus} contexts={len(packed)}', flush=True)
                try:
                    value = generate(cfg, key, {key: data['questions'][key]}, {key: record})
                except Exception as exc:
                    write_json(root / 'status.json', {'status': 'failed', 'id': key,
                                                      'attempt': len(previous) + 1, 'error': str(exc)})
                    raise
                checked = check_candidate(value, question, packed)
                saved = {'budget': tokens, 'focused': focus, 'prompt_hash': digest(prompt_ids),
                         'context_parent_ids': [c['parent_id'] for c in packed],
                         'generation': value, 'check': checked}
                journal.append(attempt_id, saved)
            previous.append(saved)
        final = previous[-1]['check']
        if final['accepted']:
            merged[key] = final['prediction']
        outcomes[key] = {'group': item['group'], 'reasons': item['reasons'], 'attempts': len(previous),
                         'accepted': final['accepted'], 'changed': merged[key] != data['predictions'][key],
                         'rejected': final['rejected'],
                         'output_tokens': sum(p['generation']['audit']['output_tokens'] for p in previous),
                         'seconds': sum(p['generation']['audit']['seconds'] for p in previous)}
    validate_predictions(merged, data['questions'])
    write_json(root / 'candidate.json', merged)
    write_json(root / 'outcomes.json', outcomes)
    write_json(root / 'unresolved.json', {k: v for k, v in outcomes.items() if not v['accepted']})
    summary = {'status': 'complete', 'candidates': len(queue),
               'accepted': sum(v['accepted'] for v in outcomes.values()),
               'changed': sum(v['changed'] for v in outcomes.values()),
               'output_tokens': sum(v['output_tokens'] for v in outcomes.values()),
               'seconds': sum(v['seconds'] for v in outcomes.values()),
               'rejections': dict(Counter(r for v in outcomes.values() for r in v['rejected']))}
    write_json(root / 'status.json', summary)
    return merged


def damage_counts(predictions):
    return {'heavy_loop': sum(repetition_ratio(v['answer']) >= .5 for v in predictions.values()),
            'unfinished': sum(_incomplete(v['answer']) for v in predictions.values())}


def screen(before, after, baseline_metrics, metrics):
    old, new = damage_counts(before), damage_counts(after)
    changed = sum(before[k] != after[k] for k in before)
    return {'passes': bool(changed and metrics['meteor'] >= baseline_metrics['meteor']
                           and all(new[k] <= old[k] for k in old)),
            'changed': changed, 'delta_meteor': metrics['meteor'] - baseline_metrics['meteor'],
            'delta_rougeL': metrics['rougeL'] - baseline_metrics['rougeL'],
            'before': old, 'after': new}


def evaluate_dev(data, candidate, queue, root, evaluate):
    """Screen whole policies, including disabled groups; never select by per-ID score."""
    root = Path(root)
    before = data['predictions']
    refs = root / 'references.json'
    write_json(refs, data['references'])

    def score(name, predictions):
        path = root / (name + '.json')
        report = root / (name + '.metrics.json')
        write_json(path, predictions)
        evaluate(path, refs, report)
        return read_json(report)

    original = score('original', data['original'])
    _require(all(abs(original[k] - data['stored_metrics'][k]) <= 1e-10 for k in ('meteor', 'rougeL')),
             'Stored dev metrics were not reproduced')
    base = score('baseline', before)
    groups, enabled = {}, []
    for group in ('fallback', 'truncated'):
        merged = copy.deepcopy(before)
        for key, item in queue.items():
            if item['group'] == group:
                merged[key] = candidate[key]
        metric = score(group, merged)
        groups[group] = {**screen(before, merged, base, metric),
                         'meteor': metric['meteor'], 'rougeL': metric['rougeL']}
        if groups[group]['passes']:
            enabled.append(group)
    merged = copy.deepcopy(before)
    for key, item in queue.items():
        if item['group'] in enabled:
            merged[key] = candidate[key]
    metric = score('combined', merged)
    combined = screen(before, merged, base, metric)
    if not combined['passes']:
        enabled = []
        merged, selected_metric = before, base
    else:
        selected_metric = metric
    write_json(root / 'selected.json', merged)
    decision = {'status': 'complete', 'groups': groups, 'combined': combined, 'enabled_groups': enabled,
                'baseline': {k: base[k] for k in ('meteor', 'rougeL')},
                'selected': {k: selected_metric[k] for k in ('meteor', 'rougeL')}}
    write_json(root / 'decision.json', decision)
    return decision


def runtime_identity(bundle, models, adapter):
    from .generation import adapter_identity
    from .models import model_lock
    expected = bundle['expected']
    _require(adapter_identity(adapter) == expected['adapter'], 'Selected adapter hash mismatch')
    _require(model_lock(bundle['config'], models) == expected['models'], 'Model lock differs from Stage 2')
    generator = Path(models) / 'generator'
    files = {p.relative_to(generator).as_posix(): file_hash(p) for p in sorted(generator.rglob('*')) if p.is_file()}
    _require(any(n.endswith('.safetensors') for n in files), 'Missing generator weights')
    limit = min(POLICY['total_tokens'], int(read_json(generator / 'config.json')['max_position_embeddings']))
    return {'source': bundle['source'], 'policy': POLICY, 'code': source_hash(),
            'adapter': expected['adapter'], 'models': expected['models'],
            'generator_files': files, 'context_limit': limit}


def run(stage2, submission, models, adapter, output, mode, *, private_diagnostics=None,
        max_items=50, device='cuda:0'):
    from .generation import _generate_one
    from .metrics import evaluate
    from .models import load_generator
    from transformers import AutoTokenizer, set_seed
    bundle = load_adaptive_inputs(stage2, submission, private_diagnostics)
    identity = runtime_identity(bundle, models, adapter)
    tokenizer = AutoTokenizer.from_pretrained(Path(models) / 'generator', local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    identity['context_limit'] = min(identity['context_limit'], int(tokenizer.model_max_length))
    ladder = token_ladder(identity['context_limit'])
    root = Path(output)
    marker = root / 'identity.json'
    if marker.exists():
        _require(read_json(marker) == identity, 'Adaptive identity differs; choose a new output directory')
    else:
        _require(not root.exists() or not any(root.iterdir()), 'Adaptive output directory is not empty')
        write_json(marker, identity)
    split = 'private' if mode == 'adaptive_private' else 'dev'
    modes = ['audit', 'heuristic'] if split == 'dev' else [
        'audit' if bundle['private']['audit'] is not None else 'heuristic']
    if mode == 'adaptive_private':
        _require((root / 'dev.status.json').is_file()
                 and read_json(root / 'dev.status.json').get('status') == 'complete',
                 'Finish both audit and heuristic dev screens before private generation')
    if mode == 'adaptive_audit':
        report = {'source': bundle['source'], 'counts': {}, 'queues': {}}
        for part in ('dev', 'private'):
            methods = ['heuristic'] + (['audit'] if bundle[part]['audit'] is not None else [])
            for method in methods:
                queue = candidate_queue(bundle[part], bundle['config'], tokenizer, method)
                report['queues'][f'{part}_{method}'] = queue
                report['counts'][f'{part}_{method}'] = dict(Counter(v['group'] for v in queue.values()))
        write_json(root / 'audit.json', report)
        return {'status': 'complete', 'report': 'audit.json'}
    _require(mode in {'adaptive_dev', 'adaptive_private'}, 'Unknown adaptive mode')
    import torch
    _require(str(device).startswith('cuda') and torch.cuda.is_available(), 'Adaptive generation requires CUDA')
    set_seed(bundle['config']['seed'])
    smoke_path = root / f'{split}.smoke.json'
    smoke = not smoke_path.exists()
    budget = SessionBudget(min(5, max_items) if smoke else max_items)
    model = None

    def generate(config, key, questions, records):
        nonlocal model
        if model is None:
            model, _ = load_generator(config, models, device, adapter)
        return _generate_one(config, key, questions, records, model, tokenizer, device, 'generate')

    summaries = {}
    for detection in modes:
        data = bundle[split]
        target = root / split / detection
        queue = candidate_queue(data, bundle['config'], tokenizer, detection)
        if split == 'private':
            dev = root / 'dev' / detection
            decision = read_json(dev / 'decision.json')
            _require(decision['status'] == 'complete', 'Matching dev screen is incomplete')
            groups = decision['enabled_groups']
            skipped = {k: {**v, 'skip_reason': 'dev_group_did_not_pass'}
                       for k, v in queue.items() if v['group'] not in groups}
            write_json(target / 'skipped.json', skipped)
            queue = {k: v for k, v in queue.items() if v['group'] in groups}
        write_json(target / 'candidates.json', queue)
        result = run_queue(data, queue, bundle['config'], tokenizer, generate, target,
                           {**identity, 'detection': detection}, ladder, budget, split)
        if smoke and (len(budget.touched) >= min(5, max_items) or result is not None):
            write_json(smoke_path, {'status': 'complete', 'ids': sorted(k for _, k in budget.touched)})
        if result is None:
            status = {'status': 'paused', 'split': split, 'detection': detection,
                      'reason': 'smoke_or_session_budget', 'new_ids': len(budget.touched)}
            write_json(root / f'{split}.status.json', status)
            return status
        if split == 'dev':
            if should_pause():
                status = {'status': 'paused', 'split': split, 'reason': 'before_scoring'}
                write_json(root / f'{split}.status.json', status)
                return status
            summaries[detection] = evaluate_dev(data, result, queue, target, evaluate)
        else:
            _package(result, data['questions'], root / 'submission_adaptive.zip')
            summaries[detection] = read_json(target / 'status.json')
    status = {'status': 'complete', 'split': split, 'summary': summaries,
              'submission_zip': 'submission_adaptive.zip' if split == 'private' else None}
    write_json(root / f'{split}.status.json', status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['adaptive_audit', 'adaptive_dev', 'adaptive_private'], required=True)
    for name in ('stage2', 'submission', 'models', 'adapter', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--private-diagnostics')
    parser.add_argument('--max-items', type=int, default=50)
    parser.add_argument('--device', default='cuda:0')
    args = vars(parser.parse_args())
    output = Path(args['output'])
    try:
        status = run(**args)
    except Exception as exc:
        write_json(output / 'status.json', {'status': 'failed', 'error': str(exc)})
        raise
    write_json(output / 'status.json', status)


if __name__ == '__main__':
    main()
