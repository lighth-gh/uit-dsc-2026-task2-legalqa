"""Main 04 v2: one private generation per flagged ID, always export partial progress."""
import argparse
import copy
import time
from pathlib import Path

from .adaptive import attempt_config, candidate_queue, check_candidate, runtime_identity
from .adaptive_inputs import load_adaptive_inputs
from .io import Journal, digest, read_json, write_json
from .prompts import pack_prompt
from .repair import _package, _require


POLICY = {'version': 1, 'max_new_tokens': 3072, 'max_input_tokens': 4096,
          'attempts_per_id': 1, 'max_seconds_per_id': 150, 'reserve_seconds': 60}


def export(predictions, questions, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _package(predictions, questions, output / 'submission.zip')


def run_queue(data, queue, config, tokenizer, generate, output, identity, deadline,
              clock=time.time):
    """Commit each decision; untouched/rejected IDs retain the exact baseline."""
    output = Path(output)
    journal = Journal(output / 'attempts.jsonl', {**identity, 'queue': queue})
    _require(set(journal.records) <= set(queue), 'Unknown journal ID')
    merged = copy.deepcopy(data['predictions'])
    accepted = 0
    for key, row in journal.records.items():
        if row['check']['accepted']:
            merged[key] = row['check']['prediction']
            accepted += int(merged[key] != data['predictions'][key])
    export(merged, data['questions'], output)
    reason, error = 'complete', None

    def status():
        result = {'status': reason, 'attempted': len(journal.records), 'candidates': len(queue),
                  'changed': accepted, 'remaining': len(queue) - len(journal.records),
                  'total_ids': len(merged), 'submission_zip': str(output / 'submission.zip')}
        if error:
            result['error'] = error
        write_json(output / 'status.json', result)
        return result

    try:
        for key, item in queue.items():
            if key in journal.records:
                continue
            remaining = deadline - clock() - POLICY['reserve_seconds']
            if remaining < 10:
                reason = 'deadline'
                break
            cfg = attempt_config(config, item['group'], POLICY['max_new_tokens'])
            question = data['questions'][key]['question']
            prompt_ids, packed = pack_prompt(question, data['records'][key]['contexts'], tokenizer, cfg)
            _require(len(prompt_ids) + POLICY['max_new_tokens'] <= identity['context_limit'],
                     'Prompt/output exceeds model window')
            print(f"[{len(journal.records)+1}/{len(queue)}] {key} {item['group']} "
                  f"contexts={len(packed)} input={len(prompt_ids)}/4096 output<=3072 "
                  f"remaining={remaining:.0f}s", flush=True)
            value = generate(cfg, key, data['questions'], data['records'],
                             min(POLICY['max_seconds_per_id'], remaining))
            checked = check_candidate(value, question, packed)
            journal.append(key, {'generation': value, 'check': checked,
                                 'prompt_hash': digest(prompt_ids), 'group': item['group']})
            if checked['accepted']:
                merged[key] = checked['prediction']
                accepted += int(merged[key] != data['predictions'][key])
            export(merged, data['questions'], output)
            print(f"  route={value['audit']['route']} changed={accepted} "
                  f"accepted={checked['accepted']} rejected={checked['rejected']}", flush=True)
            reason = 'running'
            status()
        else:
            reason = 'complete'
    except (Exception, KeyboardInterrupt) as exc:
        reason, error = 'stopped_error', f'{type(exc).__name__}: {exc}'
        print(f'Stopped; keeping completed answers: {error}', flush=True)
    finally:
        export(merged, data['questions'], output)
        result = status()
    return result


def run(stage2, submission, models, adapter, output, deadline, private_diagnostics=None):
    # Export the verified full baseline before tokenizer/model setup.
    bundle = load_adaptive_inputs(stage2, submission, private_diagnostics)
    data, output = bundle['private'], Path(output)
    # Re-running an existing session must not temporarily replace its improved ZIP.
    if not (output / 'submission.zip').exists():
        export(data['predictions'], data['questions'], output)
    from transformers import AutoTokenizer, set_seed
    from .models import load_generator
    from .generation import _generate_one
    import torch
    identity = {**runtime_identity(bundle, models, adapter), 'deadline_policy': POLICY}
    tokenizer = AutoTokenizer.from_pretrained(Path(models) / 'generator', local_files_only=True)
    tokenizer.pad_token, tokenizer.padding_side = tokenizer.eos_token, 'left'
    identity['context_limit'] = min(identity['context_limit'], int(tokenizer.model_max_length))
    _require(identity['context_limit'] >= 7168, 'Model must support 4096 input + 3072 output tokens')
    marker = output / 'identity.json'
    if marker.exists():
        _require(read_json(marker) == identity, 'Inputs/code changed; use a new OUTPUT directory')
    else:
        write_json(marker, identity)
    detection = 'audit' if data['audit'] is not None else 'heuristic'
    queue = candidate_queue(data, bundle['config'], tokenizer, detection)
    # Actual/reconstructed fallback first, clear unfinished answers next, near-limit last.
    queue = dict(sorted(queue.items(), key=lambda kv: (
        0 if kv[1]['group'] == 'fallback' else
        1 if 'unfinished_lead_in' in kv[1]['reasons'] or 'audit_token_limit' in kv[1]['reasons'] else 2,
        kv[0])))
    write_json(output / 'candidates.json', queue)
    print(f'Private IDs={len(data["questions"])}; candidates={len(queue)}; detection={detection}; '
          'one attempt per ID; no dev/scoring/retrieval', flush=True)
    _require(torch.cuda.is_available(), 'CUDA GPU required; baseline ZIP is available')
    set_seed(bundle['config']['seed'])
    model = None

    def generate(cfg, key, questions, records, seconds):
        nonlocal model
        if model is None:
            model, _ = load_generator(cfg, models, 'cuda:0', adapter)
        seconds = min(seconds, deadline - time.time() - POLICY['reserve_seconds'])
        if seconds <= 0:
            raise TimeoutError('Deadline reached during model loading')
        return _generate_one(cfg, key, questions, records, model, tokenizer, 'cuda:0', 'generate',
                             generation_overrides={'max_time': seconds})

    return run_queue(data, queue, bundle['config'], tokenizer, generate, output, identity, deadline)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('stage2', 'submission', 'models', 'adapter', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--private-diagnostics')
    args = vars(parser.parse_args())
    try:
        print(run(**args), flush=True)
    except Exception as exc:
        write_json(Path(args['output']) / 'error.json', {'error': f'{type(exc).__name__}: {exc}'})
        raise


if __name__ == '__main__':
    main()
