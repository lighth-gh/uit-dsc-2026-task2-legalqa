"""Stage 4 V2: compare CPU policies, then optionally validate selective GPU repair."""
import argparse
import copy
import re
from pathlib import Path

from .io import Journal, digest, file_hash, read_json, source_hash, validate_predictions, write_json
from .repair import (EXTENDED_POLICY, _incomplete, _package, _require, deduplicate_answer,
                     load_diagnostics, repair_predictions, repetition_ratio, run_repair)


GPU_RECIPE = {"version": 2, "repetition_penalty": 1.0, "no_repeat_ngram_size": 0,
              "system_suffix": (
                  "Trong câu trả lời này, được phép chép nguyên văn các cụm từ từ tài liệu. "
                  "Giữ đúng số hiệu, tên gọi, số liệu và từ phủ định. Không đổi từ để né lặp. "
                  "Chỉ tránh lặp nguyên cả câu hoặc đoạn từ ba lần liên tiếp; "
                  "vẫn trình bày đủ điều kiện, ngoại lệ và danh sách."),
              "min_dev_gain": .001, "min_dev_changes": 2}


def _identity(root, identity):
    marker = root / "identity.json"
    if marker.exists():
        _require(read_json(marker) == identity, "Stage 4 V2 identity differs; choose a new output directory")
    else:
        _require(not root.exists() or not any(root.iterdir()), "Output directory is not empty")
        write_json(marker, identity)


def _severe(predictions):
    return sum(repetition_ratio(v["answer"]) >= .5 for v in predictions.values())


def compare_variant(control_metrics, candidate_metrics, control, candidate, *, gpu=False):
    changed = [k for k in control if control[k] != candidate[k]]
    delta = candidate_metrics["meteor"] - control_metrics["meteor"]
    accepted = (len(changed) >= (GPU_RECIPE["min_dev_changes"] if gpu else 1)
                and delta >= (GPU_RECIPE["min_dev_gain"] if gpu else -1e-12)
                and _severe(candidate) <= _severe(control))
    return {"accepted": accepted, "changed_dev_ids": changed,
            "meteor_delta": delta, "rougeL_delta": candidate_metrics["rougeL"] - control_metrics["rougeL"],
            "severe_before": _severe(control), "severe_after": _severe(candidate),
            "per_question": {k: {metric: candidate_metrics["per_question"][k][metric]
                                  - control_metrics["per_question"][k][metric]
                                  for metric in ("meteor", "rougeL")} for k in changed},
            "note": "One fixed policy for all IDs; dev100 is not independent validation."}


def _publish(root, bundle, predictions, variant, metrics, status="complete"):
    for split in ("dev", "public"):
        validate_predictions(predictions[split], bundle[split]["questions"])
        write_json(root / f"{split}.selected.json", predictions[split])
    name = "submission_selected.zip"
    _package(predictions["public"], bundle["public"]["questions"], root / name)
    write_json(root / "repair.metrics.json", metrics)
    write_json(root / "repair.manifest.json", {
        "status": status, "selected_variant": variant, "submission_zip": name,
        "files": {p: file_hash(root / p) for p in
                  (name, "identity.json", "dev.selected.json", "public.selected.json", "repair.metrics.json")},
        "source": bundle["source"],
        "verification": bundle["verification"], "public_count": len(predictions["public"])})


def run_cpu(diagnostics, output, *, audit_only=False):
    from .metrics import evaluate
    bundle, root = load_diagnostics(diagnostics), Path(output)
    _identity(root, {"source": bundle["source"], "code": source_hash(),
                     "policy": EXTENDED_POLICY, "audit_only": audit_only})
    (root / "submission_selected.zip").unlink(missing_ok=True)
    write_json(root / "repair.manifest.json", {"status": "running", "submission_zip": None})
    # V1 is the known submission control, not just the pre-repair Stage 3 baseline.
    legacy = run_repair(diagnostics, root / "conservative", audit_only=audit_only)
    original, candidate, audits, queues = {}, {}, {}, {}
    for split in ("dev", "public"):
        original[split] = bundle[split]["predictions"]
        candidate[split], audits[split], queues[split] = repair_predictions(
            original[split], bundle[split]["audit"], EXTENDED_POLICY)
        write_json(root / f"{split}.candidate.json", candidate[split])
    write_json(root / "repair.audit.json", audits)
    write_json(root / "repair.candidate_unresolved.json", queues)
    if audit_only:
        write_json(root / "repair.unresolved.json", queues)
        write_json(root / "repair.metrics.json", {"status": "not_scored"})
        write_json(root / "repair.manifest.json", {"status": "audit_only", "submission_zip": None})
        return bundle, None, None
    prior = read_json(root / "conservative/repair.metrics.json")
    legacy_repaired = legacy["decision"]["accepted"]
    incumbent = {split: read_json(root / "conservative" /
                  f"{'dev' if split == 'dev' else 'submission'}.{'repaired' if legacy_repaired else 'original'}.json")
                 for split in ("dev", "public")}
    score_path = root / "conservative" / f"dev.{'repaired' if legacy_repaired else 'original'}.metrics.json"
    control_metrics = read_json(score_path)
    evaluate(root / "dev.candidate.json", root / "conservative/dev.references.json",
             root / "dev.candidate.metrics.json")
    candidate_metrics = read_json(root / "dev.candidate.metrics.json")
    decision = compare_variant(control_metrics, candidate_metrics, incumbent["dev"], candidate["dev"])
    decision["public_severe_nonincreasing"] = _severe(candidate["public"]) <= _severe(incumbent["public"])
    decision["accepted"] &= decision["public_severe_nonincreasing"]
    selected = candidate if decision["accepted"] else incumbent
    variant = "cpu_v2" if decision["accepted"] else ("cpu_v1" if legacy_repaired else "original")
    selected_metrics = candidate_metrics if decision["accepted"] else control_metrics
    report = {"status": "scored", "baseline": prior["baseline"],
              "cpu_v1": {k: control_metrics[k] for k in ("meteor", "rougeL")},
              "cpu_v2": {k: candidate_metrics[k] for k in ("meteor", "rougeL")},
              "selected": {k: selected_metrics[k] for k in ("meteor", "rougeL")},
              "decision": decision, "selected_variant": variant,
              "public_changed_vs_original": sum(selected["public"][k] != original["public"][k] for k in original["public"]),
              "public_severe_before": _severe(original["public"]), "public_severe_after": _severe(selected["public"])}
    write_json(root / "dev.selected.metrics.json", selected_metrics)
    write_json(root / "cpu.metrics.json", report)
    selected_queue = queues if decision["accepted"] else read_json(root / "conservative/repair.unresolved.json")
    write_json(root / "repair.unresolved.json", selected_queue)
    _publish(root, bundle, selected, variant, report)
    print(report, flush=True)
    return bundle, selected, selected_metrics


NUMBERED_ITEM = re.compile(r"^\s*(?P<label>\d+|[a-zđ])[.)]\s+(?P<body>.+)$", re.I)


def loop_reasons(text):
    """Detect exact block loops and runaway renumbering without reference answers."""
    reasons = []
    _, changes = deduplicate_answer(text, EXTENDED_POLICY)
    if any(change["unit"] == "numbered_loop" for change in changes):
        reasons.append("numbered_loop")
    if any(change["unit"] != "numbered_loop" for change in changes):
        reasons.append("exact_adjacent_loop")
    if repetition_ratio(text) >= .25:
        reasons.append("repeated_long_lines")

    run = []
    for line in text.splitlines() + [""]:
        match = NUMBERED_ITEM.match(line)
        if not match:
            run = []
            continue
        label = match.group("label").casefold()
        body = " ".join(match.group("body").casefold().split())
        numeric = int(label) if label.isdigit() else ord(label) - ord("a") + 1
        if run and (numeric != run[-1][0] + 1 or body != run[-1][1]):
            run = []
        run.append((numeric, body))
        if len(run) >= 4 and len(body) >= EXTENDED_POLICY["min_chars"]:
            reasons.append("runaway_incrementing_list")
            break
    return sorted(set(reasons))


def gpu_candidates(data, predictions, *, detection_predictions=None, loops_only=False,
                   contexts_k=None, same_document_as_top=False):
    """Select without gold; optionally restrict regeneration to detected loop failures."""
    from .prompts import refusal_evidence_support, select_contexts
    keys, skipped = [], {}
    for key, value in predictions.items():
        row, text = data["audit"][key], value["answer"]
        detection_text = (detection_predictions or predictions)[key]["answer"]
        loops = loop_reasons(detection_text)
        _, edits, _ = repair_predictions({key: value}, {key: row}, EXTENDED_POLICY)
        blocked = bool(edits.get(key, {}).get("blocked"))
        flagged = bool(loops) if loops_only else (
            bool(loops) or _incomplete(text) or blocked or row["hit_token_limit"]
            or repetition_ratio(text) >= .25 or "fallback" in row["route"]
        )
        if not flagged:
            continue
        contexts = data["records"][key]["contexts"]
        if contexts_k is not None:
            contexts = select_contexts(contexts, contexts_k,
                                       same_document_as_top=same_document_as_top)
        support = refusal_evidence_support(data["questions"][key]["question"], contexts)
        if not support["strong"]:
            skipped[key] = "cached_evidence_not_strong; review retrieval before regeneration"
        else:
            keys.append(key)
    # Stable order and all eligible IDs: no silent top-k/sample reduction.
    return sorted(keys), skipped


def accept_generated(original, value):
    row = value["audit"]
    pred, edits, _ = repair_predictions({"item": value["prediction"]}, {"item": row}, EXTENDED_POLICY)
    text = pred["item"]["answer"]
    reasons = []
    if row["route"] != "generated" or row["hit_token_limit"]:
        reasons.append("not_a_complete_generation")
    if _incomplete(text) or repetition_ratio(text) >= .25 or edits.get("item", {}).get("blocked"):
        reasons.append("incomplete_or_repetitive")
    if not (row.get("refusal_evidence_support") or {}).get("strong"):
        reasons.append("packed_evidence_not_strong")
    if row.get("entity_conflict", {}).get("conflict"):
        reasons.append("entity_conflict")
    flags = row.get("flags_before_fallback", {})
    if any(flags.get(k) for k in ("empty", "artifact", "refusal", "unsupported_document_numbers")):
        reasons.append("generation_guard")
        reasons.extend(f"guard:{key}" for key in
                       ("empty", "artifact", "refusal", "unsupported_document_numbers") if flags.get(key))
    if row["hit_token_limit"]:
        reasons.append("hit_token_limit")
    return (original if reasons else pred["item"]), reasons


def run_gpu(bundle, selected, control_metrics, root, models, adapter, *, max_items=50, device="cuda:0"):
    from .generation import _generate_one, adapter_identity
    from .metrics import evaluate
    from .models import load_generator, model_lock
    from .runtime import should_pause
    import torch
    from transformers import set_seed
    root, models, adapter = Path(root), Path(models), Path(adapter)
    _require(max_items > 0, "max_items must be positive")
    _require(str(device).startswith("cuda") and torch.cuda.is_available(), "GPU mode requires CUDA")
    expected = bundle["dev"]["prediction_manifest"]["identity"]
    _require(adapter_identity(adapter) == expected["adapter"], "Selected adapter hash mismatch")
    lock = model_lock(bundle["config"], models)
    _require(lock == expected["models"], "Model lock differs from the Stage 3 generator/tokenizer snapshot")
    gpu_root = root / "gpu"
    generator_files = {p.name: file_hash(p) for p in sorted((models / "generator").iterdir())
                       if p.is_file() and p.suffix in {".json", ".safetensors", ".model", ".txt"}}
    _require(any(name.endswith(".safetensors") for name in generator_files), "Missing generator weights")
    identity = {"source": bundle["source"], "code": source_hash(), "recipe": GPU_RECIPE,
                "adapter": expected["adapter"], "models": lock, "generator_files": generator_files,
                "cpu_predictions": digest(selected), "scorer": control_metrics["metric_identity"]}
    _identity(gpu_root, identity)
    inference_config = copy.deepcopy(bundle["config"])
    inference_config["generation"]["system_suffix"] = GPU_RECIPE["system_suffix"]
    # Never expose a stale GPU ZIP after a paused/failed rerun. CPU remains available.
    report = read_json(root / "cpu.metrics.json")
    report["gpu"] = {"status": "running"}
    _publish(root, bundle, selected, report["selected_variant"], report, "running")
    jobs = {split: gpu_candidates(bundle[split], selected[split]) for split in ("dev", "public")}
    write_json(gpu_root / "candidates.json", {s: {"ids": v[0], "skipped": v[1]} for s, v in jobs.items()})
    print(f"GPU candidates: dev={len(jobs['dev'][0])}, public={len(jobs['public'][0])}; "
          f"max_input_tokens={bundle['config']['generation']['max_input_tokens']}; "
          f"max_new_tokens={bundle['config']['generation']['max_new_tokens']}; recipe={GPU_RECIPE}", flush=True)
    merged = copy.deepcopy(selected)
    effective_audits = {s: dict(bundle[s]["audit"]) for s in ("dev", "public")}
    model = tokenizer = None
    completed = 0
    set_seed(bundle["config"]["seed"])
    for split in ("dev", "public"):
        keys, _ = jobs[split]
        journal = Journal(gpu_root / f"{split}.checkpoint.jsonl", {**identity, "split": split, "ids": keys})
        _require(set(journal.records) <= set(keys), "Unknown GPU journal IDs")
        for key in keys:
            if key not in journal.records:
                if completed >= max_items or should_pause():
                    write_json(gpu_root / "status.json", {"status": "paused", "split": split,
                               "completed": len(journal.records), "total": len(keys)})
                    report["gpu"] = {"status": "paused", "split": split,
                                     "completed": len(journal.records), "total": len(keys)}
                    _publish(root, bundle, selected, report["selected_variant"], report, "paused")
                    return
                if model is None:
                    model, tokenizer = load_generator(bundle["config"], models, device, adapter)
                if should_pause():
                    # Re-enter the pause branch without generating after slow model loading.
                    write_json(gpu_root / "status.json", {"status": "paused", "split": split})
                    report["gpu"] = {"status": "paused", "split": split}
                    _publish(root, bundle, selected, report["selected_variant"], report, "paused")
                    return
                value = _generate_one(inference_config, key, bundle[split]["questions"],
                                      bundle[split]["records"], model, tokenizer, device, "generate",
                                      generation_overrides={k: GPU_RECIPE[k] for k in
                                                            ("repetition_penalty", "no_repeat_ngram_size")})
                prediction, rejected = accept_generated(selected[split][key], value)
                journal.append(key, {"prediction": prediction, "rejected": rejected, "generation": value})
                completed += 1
                print(f"GPU {split} {len(journal.records)}/{len(keys)}: {key}; rejected={rejected}", flush=True)
            saved = journal.records[key]
            verified, rejected = accept_generated(selected[split][key], saved["generation"])
            _require(verified == saved["prediction"] and rejected == saved["rejected"], "GPU journal content mismatch")
            merged[split][key] = saved["prediction"]
            if not rejected:
                effective_audits[split][key] = saved["generation"]["audit"]
        write_json(gpu_root / f"{split}.candidate.json", merged[split])
        if split == "dev":
            evaluate(gpu_root / "dev.candidate.json", root / "conservative/dev.references.json",
                     gpu_root / "dev.metrics.json")
            scores = read_json(gpu_root / "dev.metrics.json")
            decision = compare_variant(control_metrics, scores, selected["dev"], merged["dev"], gpu=True)
            write_json(gpu_root / "decision.json", decision)
            report["gpu"] = {"status": "dev_accepted" if decision["accepted"] else "rejected",
                             "decision": decision, "metrics": {k: scores[k] for k in ("meteor", "rougeL")}}
            if not decision["accepted"]:
                _publish(root, bundle, selected, report["selected_variant"], report)
                write_json(gpu_root / "status.json", {"status": "rejected"})
                return
    _require(_severe(merged["public"]) <= _severe(selected["public"]), "GPU increased severe public repetition")
    report["gpu"]["status"] = "complete"
    report["gpu"]["public_changed"] = sum(merged["public"][k] != selected["public"][k] for k in selected["public"])
    report["selected"] = report["gpu"]["metrics"]
    report["selected_variant"] = f"gpu_v{GPU_RECIPE['version']}"
    queues = {s: repair_predictions(merged[s], effective_audits[s], EXTENDED_POLICY)[2]
              for s in ("dev", "public")}
    write_json(root / "repair.unresolved.json", queues)
    _publish(root, bundle, merged, report["selected_variant"], report)
    write_json(gpu_root / "status.json", {"status": "complete"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--models")
    parser.add_argument("--adapter")
    parser.add_argument("--max-items", type=int, default=50)
    args = parser.parse_args()
    if args.gpu and (args.audit_only or not args.models or not args.adapter):
        parser.error("--gpu requires --models and --adapter, without --audit-only")
    bundle, selected, metrics = run_cpu(args.diagnostics, args.output, audit_only=args.audit_only)
    if args.gpu:
        run_gpu(bundle, selected, metrics, args.output, args.models, args.adapter, max_items=args.max_items)


if __name__ == "__main__":
    main()
