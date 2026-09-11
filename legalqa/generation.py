import os
import time
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from .io import Journal, digest, file_hash, load_questions, read_json, source_hash, validate_predictions, write_json
from .models import load_generator, model_lock
from .prompts import (answer_flags, clean_answer, complete_truncated_answer,
                      extractive_fallback, pack_prompt, refusal_evidence_support)
from .retrieval import read_retrieval


def adapter_identity(path):
    if not path:
        return None
    files = sorted(Path(path).glob("adapter*"))
    if not any(p.suffix == ".safetensors" for p in files):
        raise ValueError("Adapter must contain a Safetensors checkpoint")
    return {p.name: file_hash(p) for p in files if p.is_file()}


def generate(c, questions_path, retrieval_path, root, output, device, adapter=None, mode="generate"):
    import torch
    from transformers import AutoTokenizer, set_seed
    set_seed(c["seed"])
    questions = load_questions(questions_path)
    records, retrieval_id = read_retrieval(retrieval_path, questions, c, root)
    identity = {"questions_hash": digest(questions), "retrieval": retrieval_id,
                "retrieval_file_hash": file_hash(retrieval_path), "models": model_lock(c,root),
                "config": c, "code": source_hash(), "adapter": adapter_identity(adapter), "mode": mode}
    out = Path(output)
    journal = Journal(out.with_suffix(".checkpoint.jsonl"),identity)
    keys = [k for k in questions if k not in journal.records]
    if keys:
        if mode == "generate":
            model, tokenizer = load_generator(c,root,device,adapter)
        else:
            tokenizer = AutoTokenizer.from_pretrained(Path(root)/"generator",local_files_only=True)
        for pos,key in enumerate(keys):
            start = time.perf_counter()
            prompt_ids, packed = pack_prompt(questions[key]["question"], records[key]["contexts"], tokenizer,c)
            hit_limit = False
            completion_tokens = 0
            raw = ""
            if mode == "generate":
                ids = torch.tensor([prompt_ids],device=device)
                with torch.inference_mode():
                    sequences = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                        do_sample=False, num_beams=1, max_new_tokens=c["generation"]["max_new_tokens"],
                        repetition_penalty=1.0, eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id, use_cache=True)
                new = sequences[0,len(prompt_ids):].tolist()
                completion_tokens = len(new)
                hit_limit = len(new) >= c["generation"]["max_new_tokens"] and new[-1] != tokenizer.eos_token_id
                raw = tokenizer.decode(new,skip_special_tokens=True)
                answer = clean_answer(raw)
                evidence = "\n".join(p["text"] for p in packed)
                flags = answer_flags(answer,evidence)
                refusal_support = refusal_evidence_support(questions[key]["question"], packed)
                invalid = flags["empty"] or flags["artifact"] or bool(flags["unsupported_document_numbers"])
                # Preserve a grounded, complete prefix when the token budget is reached. A raw
                # parent dump is reserved for invalid output or a refusal despite strong evidence.
                if hit_limit and not invalid and not flags["refusal"]:
                    completed = complete_truncated_answer(answer)
                    if completed:
                        answer = completed
                        route = "generated_truncated"
                    else:
                        answer = extractive_fallback(packed)
                        route = "source_fallback"
                elif invalid or (flags["refusal"] and refusal_support["strong"]):
                    answer = extractive_fallback(packed)
                    route = "source_fallback"
                elif flags["refusal"]:
                    # Keep an honest refusal when retrieval itself does not contain enough
                    # matching evidence; dumping an unrelated source would be misleading.
                    route = "generated_refusal"
                else:
                    route = "generated"
            else:
                answer, route, flags = extractive_fallback(packed), "extractive_baseline", {}
            audit = {"route": route, "raw_answer": raw, "flags_before_fallback": flags,
                "refusal_evidence_support": refusal_support if mode == "generate" else None,
                "hit_token_limit": hit_limit, "input_tokens": len(prompt_ids), "output_tokens": completion_tokens,
                "answer_words": len(answer.split()), "context_parent_ids": [p["parent_id"] for p in packed],
                "seconds": time.perf_counter()-start}
            journal.append(key,{"prediction":{"answer":answer},"audit":audit})
            if (pos+1)%10 == 0 or pos+1==len(keys):
                print(f"Answered: {len(journal.records)}/{len(questions)}",flush=True)
    predictions = {k:value["prediction"] for k,value in journal.records.items()}
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
