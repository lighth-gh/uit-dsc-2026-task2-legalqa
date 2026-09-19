"""Locked-baseline and bounded v8 inference/retrieval ablations."""
import argparse
import copy
import json
from pathlib import Path

from .io import Journal, digest, file_hash, read_json, source_hash, write_json


FOCUSED_REGEN_SUFFIX = (
    "Đây là lượt sửa một câu trả lời bị lặp. Chỉ trả lời các ý trực tiếp liên quan đến câu hỏi, "
    "mỗi ý đúng một lần; không tự tạo danh sách khoản hoặc điểm tăng dần. Trước hết xác định đúng "
    "chủ thể và phạm vi áp dụng, rồi ưu tiên trích đoạn xếp đầu nếu nó trực tiếp trả lời. "
    "Giữ nguyên thuật ngữ, số hiệu, điều kiện, ngoại lệ và số liệu cần thiết; không bỏ ý pháp lý "
    "chỉ vì có từ ngữ cần lặp. Không trộn quy định từ văn bản hoặc thực thể khác."
)

FOCUSED_GENERATION = {
    "contexts_k": 2,
    "same_document_as_top": True,
    "complete_legal_units": True,
    "system_suffix": FOCUSED_REGEN_SUFFIX,
}

INFERENCE_VARIANTS = {
    "g0_penalty_100": {"generation": {**FOCUSED_GENERATION, "repetition_penalty": 1.0}},
    "g1_penalty_103": {"generation": {**FOCUSED_GENERATION, "repetition_penalty": 1.03}},
    "g1_penalty_105": {"generation": {**FOCUSED_GENERATION, "repetition_penalty": 1.05}},
}

RETRIEVAL_VARIANTS = {
    "r1_pool_64": {"retrieval": {"pool_k": 64}},
    "r2_intent_query": {"retrieval": {"intent_query_expansion": True}},
    "r3_adjacent_articles": {"retrieval": {
        "adjacent_articles": 1, "adjacent_seed_k": 8, "adjacent_reserve": 8,
    }},
    "r4_lexical_weight_1": {"retrieval": {"lexical_score_weight": 1.0}},
    "r5_scope_penalty": {"retrieval": {"scope_mismatch_penalty": 0.75}},
}


def patched_config(base, patch):
    result = copy.deepcopy(base)
    for section, values in patch.items():
        if section not in result or not isinstance(result[section], dict):
            raise ValueError(f"Unknown config section: {section}")
        for key, value in values.items():
            result[section][key] = value
    return result


def write_ablation_configs(base_path, output):
    base = read_json(base_path)
    root = Path(output)
    manifest = {"baseline_config_hash": digest(base), "pipeline_code": source_hash(),
                "inference": {}, "retrieval": {}}
    for group, variants in (("inference", INFERENCE_VARIANTS), ("retrieval", RETRIEVAL_VARIANTS)):
        for name, patch in variants.items():
            path = root / group / f"{name}.json"
            config = patched_config(base, patch)
            write_json(path, config)
            manifest[group][name] = {"patch": patch, "config_hash": digest(config),
                                     "path": path.relative_to(root).as_posix()}
    write_json(root / "manifest.json", manifest)
    return manifest


def _verify_manifest_files(root, manifest):
    missing, changed = [], []
    for relative, expected in manifest.get("files", {}).items():
        path = root / relative
        if not path.is_file():
            missing.append(relative)
        elif file_hash(path) != expected:
            changed.append(relative)
    if missing or changed:
        raise ValueError(f"Baseline artifact mismatch; missing={missing}, changed={changed}")


def lock_baseline(root, output, reported_public_score=0.5599):
    """Verify the accepted Stage 4 artifact and write an immutable comparison record."""
    root = Path(root)
    manifest_path = root / "repair.manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") != "complete":
        raise ValueError("Baseline Stage 4 is not complete")
    verification = manifest.get("verification", {})
    if any(verification.get(key) != "passed" for key in
           ("crc", "hashes", "schema_and_ids", "journal", "identity")):
        raise ValueError("Baseline verification is incomplete")
    if verification.get("public_count") != 1000:
        raise ValueError("Baseline must contain exactly 1,000 public IDs")
    _verify_manifest_files(root, manifest)
    metrics = read_json(root / "repair.metrics.json")
    selected_json = root / "submission.selected.json"
    if not selected_json.is_file():
        selected_json = root / "public.selected.json"
    if not selected_json.is_file():
        raise ValueError("Cannot find selected public JSON")
    dev_metrics = metrics.get("candidate") or metrics.get("selected")
    if not dev_metrics:
        raise ValueError("Baseline metrics do not identify the selected dev score")
    source = manifest.get("identity", {}).get("source") or manifest.get("source")
    record = {
        "status": "locked",
        "artifact_root": str(root.resolve()),
        "source": source,
        "selected_adapter": (source or {}).get("selected_adapter"),
        "selected_public_json": selected_json.name,
        "selected_public_sha256": file_hash(selected_json),
        "submission_zip": manifest.get("submission_zip"),
        "submission_zip_sha256": file_hash(root / manifest["submission_zip"]),
        "dev100": {key: float(dev_metrics[key]) for key in ("meteor", "rougeL")},
        "public": {"score": float(reported_public_score),
                   "provenance": "user-reported leaderboard score; not recomputable locally"},
        "verification": verification,
        "baseline_manifest_sha256": file_hash(manifest_path),
    }
    if record["selected_adapter"] != "epoch-01":
        raise ValueError("The requested v8 baseline must use epoch-01")
    write_json(output, record)
    return record


def _selected_predictions(root, split):
    root = Path(root)
    choices = [root / f"{split}.selected.json"]
    if split == "public":
        choices += [root / "submission.selected.json", root / "submission.repaired.json"]
    else:
        choices += [root / "dev.repaired.json"]
    path = next((value for value in choices if value.is_file()), None)
    if path is None:
        raise ValueError(f"Missing selected {split} predictions under {root}")
    return read_json(path)


def _screen_decision(control_metrics, candidate_metrics, control, candidate):
    from .repair import repetition_ratio
    changed = [key for key in control if control[key] != candidate[key]]
    severe_before = sum(repetition_ratio(row["answer"]) >= .5 for row in control.values())
    severe_after = sum(repetition_ratio(row["answer"]) >= .5 for row in candidate.values())
    meteor_delta = candidate_metrics["meteor"] - control_metrics["meteor"]
    rouge_delta = candidate_metrics["rougeL"] - control_metrics["rougeL"]
    return {"passes_screen": bool(changed) and meteor_delta >= .001 and rouge_delta >= -.005
            and severe_after <= severe_before,
            "changed_ids": changed, "meteor_delta": meteor_delta, "rougeL_delta": rouge_delta,
            "severe_before": severe_before, "severe_after": severe_after,
            "rule": "delta METEOR >= 0.001; delta ROUGE-L >= -0.005; severe repetition non-increasing"}


def postprocess_candidate(predictions_path, audit_path, output):
    """Apply the same conservative CPU V1 repair to a newly generated candidate."""
    from .repair import POLICY, repair_predictions
    predictions, audits = read_json(predictions_path), read_json(audit_path)
    if set(predictions) != set(audits):
        raise ValueError("Prediction/audit IDs differ")
    repaired, changes, unresolved = repair_predictions(predictions, audits, POLICY)
    write_json(output, repaired)
    output = Path(output)
    write_json(output.with_suffix(".repair.audit.json"), changes)
    write_json(output.with_suffix(".repair.unresolved.json"), unresolved)
    return {"samples": len(repaired),
            "changed": sum(repaired[key] != predictions[key] for key in predictions),
            "unresolved": len(unresolved), "policy": POLICY}


def run_inference_ablation(diagnostics, baseline, models, adapter, variant, output, *,
                           split="dev", max_items=50, device="cuda:0", dev_result=None):
    """Regenerate only inference-detectable error candidates using one fixed variant."""
    if variant not in INFERENCE_VARIANTS:
        raise ValueError(f"Unknown inference variant: {variant}")
    if split not in {"dev", "public"}:
        raise ValueError("split must be dev or public")
    if split == "public":
        if not dev_result:
            raise ValueError("Public ablation requires --dev-result from the same accepted variant")
        dev_identity = read_json(Path(dev_result) / "identity.json")
        dev_decision = read_json(Path(dev_result) / "decision.json")
        if dev_identity.get("variant") != variant or not dev_decision.get("passes_screen"):
            raise ValueError("The matching dev variant did not pass the screening rule")
    from .generation import _generate_one, adapter_identity
    from .metrics import evaluate
    from .models import load_generator, model_lock
    from .repair import load_diagnostics
    from .repair_v2 import accept_generated, gpu_candidates, loop_reasons
    from .runtime import should_pause
    import torch
    from transformers import set_seed

    bundle = load_diagnostics(diagnostics)
    control = _selected_predictions(baseline, split)
    if split == "public" and dev_identity.get("source") != bundle["source"]:
        raise ValueError("Dev result belongs to a different Stage 3 source")
    expected = bundle["dev"]["prediction_manifest"]["identity"]
    if adapter_identity(adapter) != expected["adapter"]:
        raise ValueError("Selected adapter hash mismatch")
    if model_lock(bundle["config"], models) != expected["models"]:
        raise ValueError("Model lock differs from Stage 3")
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Inference ablation requires CUDA")
    config = patched_config(bundle["config"], INFERENCE_VARIANTS[variant])
    detection_predictions = bundle[split]["predictions"]
    keys, skipped = gpu_candidates(bundle[split], control,
                                   detection_predictions=detection_predictions,
                                   loops_only=True,
                                   contexts_k=int(config["generation"]["contexts_k"]),
                                   same_document_as_top=bool(
                                       config["generation"].get("same_document_as_top", False)))
    candidate_reasons = {key: loop_reasons(detection_predictions[key]["answer"]) for key in keys}
    root = Path(output)
    identity = {"source": bundle["source"], "baseline_predictions": digest(control),
                "variant": variant, "patch": INFERENCE_VARIANTS[variant],
                "config_hash": digest(config), "code": source_hash(), "split": split,
                "ids": keys, "candidate_reasons": candidate_reasons,
                "selection_policy": "loop_signals_only_from_original_predictions",
                "adapter": expected["adapter"], "models": expected["models"]}
    marker = root / "identity.json"
    if marker.exists() and read_json(marker) != identity:
        raise ValueError("Ablation identity differs; choose a new output directory")
    if not marker.exists():
        if root.exists() and any(root.iterdir()):
            raise ValueError("Ablation output directory is not empty")
        write_json(marker, identity)
    write_json(root / "candidates.json", {"ids": keys, "reasons": candidate_reasons,
                                           "skipped": skipped})
    journal = Journal(root / f"{split}.checkpoint.jsonl", identity)
    merged, effective_audit = copy.deepcopy(control), {}
    model = tokenizer = None
    completed = 0
    set_seed(config["seed"])
    for key in keys:
        if key not in journal.records:
            if completed >= max_items or should_pause():
                write_json(root / "status.json", {"status": "paused", "split": split,
                           "completed": len(journal.records), "total": len(keys)})
                return {"status": "paused", "completed": len(journal.records), "total": len(keys)}
            if model is None:
                model, tokenizer = load_generator(config, models, device, adapter)
            value = _generate_one(config, key, bundle[split]["questions"], bundle[split]["records"],
                                  model, tokenizer, device, "generate")
            prediction, rejected = accept_generated(control[key], value)
            journal.append(key, {"prediction": prediction, "rejected": rejected,
                                 "generation": value})
            completed += 1
        saved = journal.records[key]
        verified, reasons = accept_generated(control[key], saved["generation"])
        if verified != saved["prediction"] or reasons != saved["rejected"]:
            raise ValueError("Ablation journal content mismatch")
        merged[key] = saved["prediction"]
        effective_audit[key] = saved["generation"]["audit"]
    write_json(root / f"{split}.candidate.json", merged)
    write_json(root / f"{split}.candidate.audit.json", effective_audit)
    result = {"status": "complete", "split": split, "variant": variant,
              "changed": sum(merged[key] != control[key] for key in control)}
    if split == "dev":
        reference_path = root / "dev.references.json"
        write_json(reference_path, bundle["dev"]["references"])
        control_path = root / "dev.baseline.json"
        write_json(control_path, control)
        evaluate(control_path, reference_path, root / "dev.baseline.metrics.json")
        candidate_path = root / "dev.candidate.json"
        evaluate(candidate_path, reference_path, root / "dev.candidate.metrics.json")
        candidate_metrics = read_json(root / "dev.candidate.metrics.json")
        baseline_metrics = read_json(root / "dev.baseline.metrics.json")
        metric_choices = [Path(baseline) / "dev.selected.metrics.json",
                          Path(baseline) / "dev.repaired.metrics.json",
                          Path(baseline) / "conservative/dev.repaired.metrics.json"]
        metric_path = next((path for path in metric_choices if path.is_file()), None)
        if metric_path is None:
            raise ValueError("Missing selected baseline dev metrics")
        stored_metrics = read_json(metric_path)
        for metric in ("meteor", "rougeL"):
            if abs(baseline_metrics[metric] - stored_metrics[metric]) > 1e-10:
                raise ValueError(f"Selected baseline {metric} was not reproduced")
        decision = _screen_decision(baseline_metrics, candidate_metrics, control, merged)
        write_json(root / "decision.json", decision)
        result["decision"] = decision
    write_json(root / "status.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    lock = sub.add_parser("lock-baseline")
    lock.add_argument("--root", required=True); lock.add_argument("--output", required=True)
    lock.add_argument("--public-score", type=float, default=.5599)
    configs = sub.add_parser("write-configs")
    configs.add_argument("--base", required=True); configs.add_argument("--output", required=True)
    postprocess = sub.add_parser("postprocess")
    postprocess.add_argument("--predictions", required=True); postprocess.add_argument("--audit", required=True)
    postprocess.add_argument("--output", required=True)
    inference = sub.add_parser("inference")
    inference.add_argument("--diagnostics", required=True); inference.add_argument("--baseline", required=True)
    inference.add_argument("--models", required=True); inference.add_argument("--adapter", required=True)
    inference.add_argument("--variant", choices=sorted(INFERENCE_VARIANTS), required=True)
    inference.add_argument("--output", required=True); inference.add_argument("--split", choices=["dev", "public"], default="dev")
    inference.add_argument("--max-items", type=int, default=50); inference.add_argument("--device", default="cuda:0")
    inference.add_argument("--dev-result")
    args = parser.parse_args()
    if args.command == "lock-baseline":
        result = lock_baseline(args.root, args.output, args.public_score)
    elif args.command == "write-configs":
        result = write_ablation_configs(args.base, args.output)
    elif args.command == "postprocess":
        result = postprocess_candidate(args.predictions, args.audit, args.output)
    else:
        result = run_inference_ablation(args.diagnostics, args.baseline, args.models, args.adapter,
                                        args.variant, args.output, split=args.split,
                                        max_items=args.max_items, device=args.device,
                                        dev_result=args.dev_result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
