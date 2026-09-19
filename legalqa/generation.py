import os
import time
import multiprocessing
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import ExitStack
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from .io import Journal, digest, file_hash, load_questions, read_json, source_hash, validate_predictions, write_json
from .models import load_generator, model_lock
from .prompts import (answer_flags, clean_answer, complete_truncated_answer,
                      citation_context_conflict, extractive_fallback, pack_prompt,
                      refusal_evidence_support)
from .retrieval import read_retrieval
from .runtime import should_pause


def adapter_identity(path):
    if not path:
        return None
    files = sorted(Path(path).glob("adapter*"))
    if not any(p.suffix == ".safetensors" for p in files):
        raise ValueError("Adapter must contain a Safetensors checkpoint")
    return {p.name: file_hash(p) for p in files if p.is_file()}


def generation_devices(device, multi_gpu, mode, cuda):
    if not multi_gpu or mode != "generate" or not str(device).startswith("cuda"):
        return [device]
    count = cuda.device_count()
    if not count:
        raise RuntimeError("CUDA requested but unavailable. Enable a Kaggle GPU.")
    first = int(str(device).split(":", 1)[1]) if ":" in str(device) else cuda.current_device()
    if not 0 <= first < count:
        raise ValueError(f"Requested {device}, but only {count} CUDA devices are visible")
    return [f"cuda:{i}" for i in ([first] + [i for i in range(count) if i != first])[:2]]


def _init_generation_worker(c, root, device, adapter):
    # Spawn gives each replica its own CUDA/bitsandbytes state. Do not fork CUDA.
    import torch
    from transformers import set_seed
    global _worker_state
    torch.cuda.set_device(device)
    set_seed(c["seed"])
    model, tokenizer = load_generator(c, root, device, adapter)
    _worker_state = c, device, model, tokenizer
    print(f"Generation worker ready: {device} ({torch.cuda.get_device_name(device)}), "
          f"allocated={torch.cuda.memory_allocated(device) / 2**30:.2f} GiB", flush=True)


def _generate_worker(key, question, record):
    if should_pause():  # Model loading may have consumed the remaining budget.
        return None
    c, device, model, tokenizer = _worker_state
    return _generate_one(c, key, {key: question}, {key: record}, model, tokenizer, device, "generate")


def _dispatch_answers(keys, executors, task, arguments):
    """One in-flight question per replica; only the parent writes the journal."""
    pending, submitted, failure = {}, 0, None
    keys = iter(keys)

    def submit(executor):
        nonlocal submitted, failure
        if failure is not None or should_pause(submitted):
            return
        key = next(keys, None)
        if key is not None:
            try:
                future = executor.submit(task, key, *arguments(key))
            except Exception as error:
                failure = failure or error
                return
            pending[future] = (executor, key)
            submitted += 1  # The session cap is global, including in-flight work.

    for executor in executors:
        submit(executor)
    while pending:
        completed, _ = wait(pending, return_when=FIRST_COMPLETED)
        available = []
        for future in completed:
            executor, key = pending.pop(future)
            available.append(executor)
            try:
                value = future.result()
            except Exception as error:
                failure = failure or error
            else:
                if value is not None:
                    yield key, value
        # Persist successful in-flight answers even if the other replica failed.
        for executor in available:
            submit(executor)
    if failure is not None:
        raise failure


def _parallel_answers(c, root, adapter, devices, keys, questions, records):
    with ExitStack() as stack:
        executors = [stack.enter_context(ProcessPoolExecutor(
            max_workers=1, mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_generation_worker, initargs=(c, root, device, adapter)))
            for device in devices]
        yield from _dispatch_answers(keys, executors, _generate_worker,
                                     lambda key: (questions[key], records[key]))


def _generate_one(c, key, questions, records, model, tokenizer, device, mode, *, generation_overrides=None):
    import torch
    start = time.perf_counter()
    prompt_ids, packed = pack_prompt(questions[key]["question"], records[key]["contexts"], tokenizer,c)
    hit_limit = False
    completion_tokens = 0
    raw = ""
    if mode == "generate":
        ids = torch.tensor([prompt_ids],device=device)
        with torch.inference_mode():
            generation = c["generation"]
            options = dict(do_sample=False, num_beams=1, max_new_tokens=c["generation"]["max_new_tokens"],
                repetition_penalty=float(generation.get("repetition_penalty", 1.0)),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id, use_cache=True)
            if "no_repeat_ngram_size" in generation:
                options["no_repeat_ngram_size"] = int(generation["no_repeat_ngram_size"])
            options.update(generation_overrides or {})
            sequences = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), **options)
        new = sequences[0,len(prompt_ids):].tolist()
        completion_tokens = len(new)
        hit_limit = len(new) >= c["generation"]["max_new_tokens"] and new[-1] != tokenizer.eos_token_id
        raw = tokenizer.decode(new,skip_special_tokens=True)
        answer = clean_answer(raw)
        evidence = "\n".join(p["text"] for p in packed)
        flags = answer_flags(answer,evidence)
        refusal_support = refusal_evidence_support(questions[key]["question"], packed)
        entity_conflict = citation_context_conflict(questions[key]["question"],answer,packed)
        invalid = flags["empty"] or flags["artifact"] or bool(flags["unsupported_document_numbers"])
        # Preserve a grounded, complete prefix when the token budget is reached. A raw
        # parent dump is reserved for invalid output or a refusal despite strong evidence.
        if entity_conflict["conflict"]:
            answer = extractive_fallback(questions[key]["question"],packed,refusal_support)
            route = "entity_guard_fallback"
        elif hit_limit and not invalid and not flags["refusal"]:
            completed = complete_truncated_answer(answer)
            if completed:
                answer = completed
                route = "generated_truncated"
            else:
                answer = extractive_fallback(questions[key]["question"],packed,refusal_support)
                route = "source_fallback"
        elif invalid or (flags["refusal"] and refusal_support["strong"]):
            answer = extractive_fallback(questions[key]["question"],packed,refusal_support)
            route = "source_fallback"
        elif flags["refusal"]:
            # Keep an honest refusal when retrieval itself does not contain enough
            # matching evidence; dumping an unrelated source would be misleading.
            route = "generated_refusal"
        else:
            route = "generated"
    else:
        refusal_support = refusal_evidence_support(questions[key]["question"],packed)
        entity_conflict = {"conflict":False,"conflicting_context_index":None,"copied_phrase_words":0}
        answer = extractive_fallback(questions[key]["question"],packed,refusal_support)
        route,flags = "extractive_baseline",{}
    try:
        packed_context_tokens = [len(tokenizer(p["text"], add_special_tokens=False)["input_ids"])
                                 for p in packed]
    except TypeError:  # Minimal mocked tokenizers in CPU orchestration tests need not tokenize text.
        packed_context_tokens = [None for _ in packed]
    audit = {"route": route, "device": str(device), "raw_answer": raw, "flags_before_fallback": flags,
        "refusal_evidence_support": refusal_support if mode == "generate" else None,
        "entity_conflict": entity_conflict,
        "hit_token_limit": hit_limit, "input_tokens": len(prompt_ids), "output_tokens": completion_tokens,
        "answer_words": len(answer.split()), "context_parent_ids": [p["parent_id"] for p in packed],
        "context_document_ids": [p.get("doc_id") for p in packed],
        "packed_contexts": len(packed),
        "packed_context_tokens": packed_context_tokens,
        "generation_settings": {
            "repetition_penalty": float(c["generation"].get("repetition_penalty", 1.0)),
            "no_repeat_ngram_size": int(c["generation"].get("no_repeat_ngram_size", 0)),
            "contexts_k": int(c["generation"].get(
                "contexts_k", c.get("retrieval", {}).get("parents_k", len(packed) or 4)
            )),
            "complete_legal_units": bool(c["generation"].get("complete_legal_units", False)),
            "same_document_as_top": bool(c["generation"].get("same_document_as_top", False)),
        },
        "seconds": time.perf_counter()-start}
    return {"prediction": {"answer": answer}, "audit": audit}


def generate(c, questions_path, retrieval_path, root, output, device, adapter=None, mode="generate",
             multi_gpu=False):
    import torch
    from transformers import AutoTokenizer, set_seed
    set_seed(c["seed"])
    questions = load_questions(questions_path)
    records, retrieval_id = read_retrieval(retrieval_path, questions, c, root, expected_mode="full")
    identity = {"questions_hash": digest(questions), "retrieval": retrieval_id,
                "retrieval_file_hash": file_hash(retrieval_path), "models": model_lock(c,root),
                "config": c, "code": source_hash(), "adapter": adapter_identity(adapter), "mode": mode}
    out = Path(output)
    journal = Journal(out.with_suffix(".checkpoint.jsonl"),identity)
    keys = [k for k in questions if k not in journal.records]
    if keys and not should_pause():
        devices = generation_devices(device, multi_gpu, mode, torch.cuda if mode == "generate" else None)
        print(f"Generation devices: {devices}; replicas={len(devices)}; "
              f"max_input_tokens={c['generation']['max_input_tokens']}; "
              f"max_new_tokens={c['generation']['max_new_tokens']}; "
              f"contexts_k={c['generation'].get('contexts_k', c['retrieval']['parents_k'])}; "
              f"repetition_penalty={c['generation'].get('repetition_penalty', 1.0)}; "
              f"session_cap={os.environ.get('LEGALQA_MAX_ITEMS', '0')}", flush=True)

        def serial_answers():
            model = None
            if mode == "generate":
                model, tokenizer = load_generator(c,root,devices[0],adapter)
            else:
                tokenizer = AutoTokenizer.from_pretrained(Path(root)/"generator",local_files_only=True)
            for pos, key in enumerate(keys):
                if should_pause(pos):
                    break
                yield key, _generate_one(c, key, questions, records, model, tokenizer, devices[0], mode)

        answers = (_parallel_answers(c, root, adapter, devices, keys, questions, records)
                   if len(devices) > 1 else serial_answers())
        for pos, (key, value) in enumerate(answers):
            journal.append(key, value)
            print(f"Answered {key}: device={value['audit']['device']} route={value['audit']['route']}", flush=True)
            if (pos+1)%10 == 0 or pos+1==len(keys):
                print(f"Answered: {len(journal.records)}/{len(questions)}",flush=True)
    predictions = {k:value["prediction"] for k,value in journal.records.items()}
    if set(predictions) != set(questions):
        write_json(out.with_suffix(".partial.json"), predictions)
        return {"status":"paused", "samples":len(predictions), "total":len(questions)}
    validate_predictions(predictions,questions)
    predictions = {k: predictions[k] for k in questions}
    write_json(out,predictions)
    write_json(out.with_suffix(".audit.json"),{k:journal.records[k]["audit"] for k in questions})
    write_json(out.with_suffix(".manifest.json"), {"identity": identity, "prediction_hash": digest(predictions),
                                                   "adapter_path": str(adapter) if adapter else None})
    return {"samples":len(predictions),"output":str(out)}


def package_submission(prediction_path, questions_path, output, filename="submission.json"):
    if Path(filename).name != filename or not filename.endswith(".json"):
        raise ValueError("Submission filename must be a JSON basename without directory components")
    predictions, questions = read_json(prediction_path), load_questions(questions_path)
    validate_predictions(predictions,questions)
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    temp = output.with_name(output.name+".tmp")
    with ZipFile(temp,"w",compression=ZIP_DEFLATED) as z:
        z.write(prediction_path,arcname=filename)
    os.replace(temp,output)
    return {"samples":len(questions),"zip":str(output),"members":[filename]}
