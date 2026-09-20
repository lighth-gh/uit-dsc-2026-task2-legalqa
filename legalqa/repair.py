"""Stage 4 CPU repair: verified artifacts, exact repetition removal, measured export.

Run with ``python -m legalqa.repair --diagnostics INPUT.zip --output OUTPUT``.
Gold answers are used only by evaluation, never by the repair policy.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from .io import ROOT, _unique_pairs, digest, file_hash, read_json, validate_predictions, write_json


POLICY = {"version": 1, "min_chars": 60, "min_words": 8,
          "min_repeats": 3, "max_block_units": 12}
EXTENDED_POLICY = {**POLICY, "version": 2, "numbered_min_repeats": 6}
PUBLIC = "submissions/public/submission"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def diagnostics_zip_from_directory(source, destination):
    """Support Kaggle datasets that unpack uploaded ZIPs; copy diagnostic text only."""
    source, destination = Path(source).resolve(), Path(destination)
    manifest = read_json(source / "stage3_manifest.json")
    names = ["stage3_manifest.json"] + [n for n in manifest["files"]
                                             if PurePosixPath(n).suffix in {".json", ".jsonl", ".txt"}]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination.with_suffix(".zip.tmp"), "w", compression=ZIP_DEFLATED) as archive:
        for name in sorted(names):
            relative = PurePosixPath(name)
            _require(not relative.is_absolute() and ".." not in relative.parts
                     and ":" not in name and "\\" not in name, "Invalid diagnostic path")
            path = (source / name).resolve()
            _require(path.is_relative_to(source) and path.is_file(), f"Missing/unsafe diagnostic: {name}")
            info = ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    destination.with_suffix(".zip.tmp").replace(destination)
    return destination


def load_diagnostics(path, expected_public_count=None):
    """Read in place, without extracting or executing anything from the archive."""
    path = Path(path)
    with ZipFile(path) as archive:
        names = archive.namelist()
        _require(len(names) == len(set(names)), "Duplicate archive member")
        for name in names:
            member = PurePosixPath(name)
            _require(not member.is_absolute() and ".." not in member.parts
                     and "\\" not in name and ":" not in name, "Invalid archive path")
        _require(archive.testzip() is None, "ZIP CRC failed")

        def read(name):
            _require(name in names, f"Missing required diagnostic: {name}")
            return json.loads(archive.read(name), object_pairs_hook=_unique_pairs)

        manifest = read("stage3_manifest.json")
        _require(manifest.get("stage") == 3 and manifest.get("schema") == 2
                 and manifest.get("quality_version") == "v8"
                 and manifest.get("status") == "complete"
                 and manifest.get("progress", {}).get("complete") is True,
                 "Expected a completed Stage 3 V8 snapshot")
        omitted = []
        for name, info in manifest["files"].items():
            if name not in names:
                # Stage 3's diagnostics exporter intentionally omits tensor weights.
                _require(name == "selected_adapter/adapter_model.safetensors",
                         f"Missing manifest file: {name}")
                omitted.append(name)
                continue
            data = archive.read(name)
            _require(len(data) == info["size"]
                     and hashlib.sha256(data).hexdigest() == info["sha256"],
                     f"Manifest hash/size mismatch: {name}")
        _require(set(names) == set(manifest["files"]) - set(omitted)
                 | {"stage3_manifest.json"}, "Archive contains unmanifested files")
        selection = read("selection.json")
        label = selection["label"]
        _require(bool(re.fullmatch(r"epoch-\d+", label)), "Unexpected selected adapter label")
        config = read("config.json")
        session = read("session.json")
        _require(digest(config) == session["config_hash"], "Session/config mismatch")
        _require(session["source_hash"] == manifest["source_hash"], "Session/code mismatch")
        splits = {}
        for split, prefix, question_file in [
            ("public", PUBLIC, "data/test.questions.json"),
            ("dev", f"dev100.{label}", "data/dev100.questions.json"),
        ]:
            pred, audit = read(prefix + ".json"), read(prefix + ".audit.json")
            questions = read(question_file)
            validate_predictions(pred, questions)
            _require(all(isinstance(v.get("question"), str) and v["question"].strip()
                         for v in questions.values()), "Invalid question text")
            _require(set(audit) == set(questions), f"{split}: audit ID mismatch")
            pm = read(prefix + ".manifest.json")
            identity = pm["identity"]
            _require(digest(pred) == pm["prediction_hash"], f"{split}: prediction hash mismatch")
            _require(digest(questions) == identity["questions_hash"], f"{split}: question hash mismatch")
            _require(identity["code"] == manifest["source_hash"]
                     and identity["config"] == config, f"{split}: config/code mismatch")
            _require(identity == read(prefix + ".checkpoint.jsonl.meta.json"),
                     f"{split}: journal identity mismatch")
            journal = {}
            for line in archive.read(prefix + ".checkpoint.jsonl").splitlines(keepends=True):
                _require(line.endswith(b"\n"), "Incomplete final journal line")
                row = json.loads(line, object_pairs_hook=_unique_pairs)
                _require(row["id"] not in journal, "Duplicate journal ID")
                journal[row["id"]] = row["value"]
            _require(set(journal) == set(pred), f"{split}: journal ID mismatch")
            _require(all(row["prediction"] == pred[k] and row["audit"] == audit[k]
                         for k, row in journal.items()), f"{split}: journal content mismatch")
            retrieval_name = ("public" if split == "public" else "dev100") + ".retrieval.json"
            retrieval = read(retrieval_name)
            _require(hashlib.sha256(archive.read(retrieval_name)).hexdigest()
                     == identity["retrieval_file_hash"], f"{split}: retrieval hash mismatch")
            _require(retrieval["identity"] == identity["retrieval"], f"{split}: retrieval identity mismatch")
            records = retrieval["records"]
            _require(set(records) == set(questions), f"{split}: retrieval ID mismatch")
            for k, record in records.items():
                _require(record["question"] == questions[k]["question"], "Retrieval/question mismatch")
                contexts = record["contexts"]
                _require(contexts and all(c["text"].strip() for c in contexts), "Empty retrieval evidence")
                _require(set(audit[k]["context_parent_ids"]) <= {c["parent_id"] for c in contexts},
                         "Audit references an unknown parent")
            splits[split] = {"predictions": pred, "audit": audit, "questions": questions,
                             "prediction_manifest": pm, "records": records}
        if expected_public_count is None:
            expected_public_count = len(splits["public"]["questions"])
        _require(expected_public_count > 0, "Empty test question set")
        _require(len(splits["public"]["predictions"]) == expected_public_count
                 == manifest["progress"]["answers"], "Unexpected public answer count")
        dev_manifest = splits["dev"]["prediction_manifest"]
        _require(selection["prediction_manifest"] == dev_manifest,
                 "Selection/dev prediction mismatch")
        _require(dev_manifest["identity"]["adapter"]
                 == splits["public"]["prediction_manifest"]["identity"]["adapter"],
                 "Public does not use the selected adapter")
        references = read("data/dev100.references.json")
        references = {k: v["answer"] if isinstance(v, dict) else v for k, v in references.items()}
        _require(set(references) == set(splits["dev"]["questions"])
                 and all(isinstance(v, str) and v.strip() for v in references.values()),
                 "Invalid dev references")
        stored_metrics = read(f"dev100.{label}.metrics.json")
        _require(digest(references) == stored_metrics["reference_hash"] == selection["reference_hash"],
                 "Dev reference hash mismatch")
        _require(stored_metrics["prediction_hash"] == dev_manifest["prediction_hash"],
                 "Stored metrics refer to different predictions")
        _require(all(abs(stored_metrics[k] - selection[k]) <= 1e-12 for k in ("meteor", "rougeL")),
                 "Selection/metrics mismatch")
        splits["dev"].update(references=references, stored_metrics=stored_metrics)
    return {**splits, "config": config, "source": {"diagnostics_sha256": file_hash(path), "stage3_code": manifest["source_hash"],
            "selected_adapter": label, "omitted_weights": omitted},
            "verification": {"crc": "passed", "hashes": "passed", "schema_and_ids": "passed",
                             "journal": "passed", "identity": "passed",
                             "public_count": expected_public_count}}


def repetition_ratio(text):
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) >= 35]
    return (len(lines) - len(set(lines))) / len(lines) if lines else 0.0


def _collapse(text, sentence_mode, policy):
    pattern = r"\S[\s\S]*?(?:[.!?](?=\s|$)|$)" if sentence_mode else r"[^\r\n]+"
    units = [(m.start(), m.end(), m.group().strip()) for m in re.finditer(pattern, text)
             if m.group().strip()]
    deletions, i = [], 0
    while i < len(units):
        matched = False
        for width in range(1, min(policy["max_block_units"], (len(units) - i) // policy["min_repeats"]) + 1):
            block = [u[2] for u in units[i:i + width]]
            joined = " ".join(block)
            if len(joined) < policy["min_chars"] or len(joined.split()) < policy["min_words"]:
                continue
            end = i + width
            while end + width <= len(units) and [u[2] for u in units[end:end + width]] == block:
                end += width
            repeats = (end - i) // width
            if repeats >= policy["min_repeats"]:
                deletions.append({"start": units[i + width - 1][1], "end": units[end - 1][1],
                                  "repeats": repeats, "unit": "sentence" if sentence_mode else "line",
                                  "block_units": width})
                i, matched = end, True
                break
        if not matched:
            i += 1
    result = text
    for change in reversed(deletions):
        result = result[:change["start"]] + result[change["end"]:]
    return result, deletions


def _line_content(text):
    """Ignore list markers only; quantities, citations and negations remain exact."""
    text = re.sub(r"^\s*(?:(?:\d+|[a-zA-ZđĐ]{1,2})[.)]\s+|[-*•●]\s+)", "", text)
    return " ".join(text.split()).strip()


def _collapse_numbered(text, policy):
    units = list(re.finditer(r"[^\r\n]+", text))
    deletions, i = [], 0
    marker = re.compile(r"^\s*(?:\d+|[a-zA-ZđĐ]{1,2})[.)]\s+")
    while i < len(units):
        first = units[i]
        content = _line_content(first.group())
        end = i + 1
        if marker.match(first.group()) and len(content.split()) >= 5 and len(content) >= 25:
            while (end < len(units) and marker.match(units[end].group())
                   and _line_content(units[end].group()) == content):
                end += 1
            if end - i >= policy["numbered_min_repeats"]:
                deletions.append({"start": first.end(), "end": units[end - 1].end(),
                                  "repeats": end - i, "unit": "numbered_loop", "block_units": 1})
        i = end
    for change in reversed(deletions):
        text = text[:change["start"]] + text[change["end"]:]
    return text, deletions


def deduplicate_answer(text, policy=None):
    """Only identical adjacent blocks (>=3 copies); never fuzzy-match legal clauses."""
    policy = dict(POLICY if policy is None else policy)
    _require(policy["min_repeats"] >= 3 and policy["min_chars"] >= 60
             and policy["min_words"] >= 8 and policy["max_block_units"] >= 1,
             "Unsafe repetition policy")
    _require(policy["version"] in (1, 2) and
             (policy["version"] == 1 or policy.get("numbered_min_repeats", 0) >= 6),
             "Unsafe numbered repetition policy")
    result, changes = text, []
    for sentence_mode in (False, True):
        result, removed = _collapse(result, sentence_mode, policy)
        changes.extend(removed)
    if policy["version"] == 2:
        result, removed = _collapse_numbered(result, policy)
        changes.extend(removed)
    return result, changes


def _incomplete(text):
    return not text.strip() or text.rstrip().endswith(":") or bool(
        re.search(r"(?:như sau|bao gồm|cụ thể là)\s*[.:;]?\s*$", text, re.I))


def repair_predictions(predictions, audit, policy=None):
    """No references, model calls or raw rejected generations enter this function."""
    _require(set(predictions) == set(audit), "Repair/audit ID mismatch")
    candidate, records, unresolved = {}, {}, {}
    for key, value in predictions.items():
        before, row = value["answer"], audit[key]
        proposed, changes = deduplicate_answer(before, policy)
        reasons = []
        ratio = repetition_ratio(before)
        if ratio >= .25:
            reasons.append("repetition_screen")
        if changes:
            reasons.extend(sorted({"numbered_loop" if c["unit"] == "numbered_loop"
                                   else "exact_adjacent_loop" for c in changes}))
        if row["hit_token_limit"]:
            reasons.append("token_limit_review")
        if "fallback" in row["route"]:
            reasons.append("fallback_review")
        if _incomplete(before):
            reasons.append("unfinished_lead_in")
        blocked = []
        if changes and _incomplete(proposed):
            blocked.append("dedup_leaves_unfinished_answer")
        if changes and len(proposed) < .2 * len(before) and len(proposed.split()) < 40:
            blocked.append("dedup_leaves_too_little_content")
        after = before if blocked else proposed
        candidate[key] = {"answer": after}
        remaining = list(blocked)
        if repetition_ratio(after) >= .25:
            remaining.append("repetition_remaining")
        if row["hit_token_limit"]:
            remaining.append("check_answer_completeness")
        if "fallback" in row["route"]:
            remaining.append("check_context_relevance")
        if _incomplete(after):
            remaining.append("unfinished_lead_in")
        if reasons:
            records[key] = {"reasons": reasons, "original_route": row["route"],
                            "before": before, "after": after, "proposed": proposed,
                            "changed": after != before, "blocked": blocked,
                            "removed_blocks": changes, "before_words": len(before.split()),
                            "after_words": len(after.split()), "repetition_before": ratio,
                            "repetition_after": repetition_ratio(after)}
        if remaining:
            support = row.get("refusal_evidence_support") or {}
            unresolved[key] = {"reasons": sorted(set(remaining)),
                               "action": "review_context_first" if "fallback" in row["route"] else
                                         "review_for_selective_regeneration" if blocked or _incomplete(after)
                                         else "review_completeness_or_repetition",
                               "context_parent_ids": row["context_parent_ids"],
                               "evidence_support_strong": support.get("strong"),
                               "regenerate_automatically": False}
    validate_predictions(candidate, predictions)
    return candidate, records, unresolved


def _implementation_identity():
    paths = [ROOT / "legalqa" / name for name in ("repair.py", "io.py", "metrics.py")]
    paths += [ROOT / "vendor/scoring.py", *sorted((ROOT / "vendor/rouge_score").glob("*.py"))]
    return {p.relative_to(ROOT).as_posix(): file_hash(p) for p in paths}


def _score_dev(root, stored):
    from .metrics import evaluate
    print("Reproducing selected dev100 baseline with BTC scorer...", flush=True)
    evaluate(root / "dev.original.json", root / "dev.references.json", root / "dev.original.metrics.json")
    baseline = read_json(root / "dev.original.metrics.json")
    for key in ("scoring", "rouge_files", "nltk"):
        _require(baseline["metric_identity"][key] == stored["metric_identity"][key],
                 f"Scorer identity drift: {key}")
    _require(set(baseline["per_question"]) == set(stored["per_question"]), "Stored metric ID mismatch")
    for key in ("meteor", "rougeL"):
        _require(abs(baseline[key] - stored[key]) < 1e-10, f"Baseline {key} not reproduced")
        _require(all(abs(v[key] - stored["per_question"][k][key]) < 1e-10
                     for k, v in baseline["per_question"].items()), f"Per-item {key} not reproduced")
    print("Baseline reproduced. Scoring the repaired dev100 candidate...", flush=True)
    evaluate(root / "dev.repaired.json", root / "dev.references.json", root / "dev.repaired.metrics.json")
    return baseline, read_json(root / "dev.repaired.metrics.json")


def acceptance_decision(baseline, candidate, original, repaired):
    deltas = {k: candidate[k] - baseline[k] for k in ("meteor", "rougeL")}
    before = sum(repetition_ratio(v["answer"]) >= .5 for v in original.values())
    after = sum(repetition_ratio(v["answer"]) >= .5 for v in repaired.values())
    changed = [k for k in original if original[k] != repaired[k]]
    reduction = sum(len(original[k]["answer"]) - len(repaired[k]["answer"]) for k in changed)
    accepted = bool(changed) and deltas["meteor"] >= -1e-12 and after <= before and reduction > 0
    return {"accepted": accepted, "selected": "repaired" if accepted else "original",
            "delta": deltas, "changed_dev_ids": changed, "severe_repetition_before": before,
            "severe_repetition_after": after,
            "rule": "METEOR non-decreasing; exact-loop content reduced; severe repetition non-increasing",
            "note": "Dev100 was already used for checkpoint selection; this is not independent validation."}


def _package(predictions, questions, path):
    validate_predictions(predictions, questions)
    payload = (json.dumps(predictions, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    temporary = path.with_suffix(".zip.tmp")
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
        info_name = "submission.json"
        archive.writestr(info_name, payload)
    with ZipFile(temporary) as archive:
        _require(archive.namelist() == ["submission.json"] and archive.testzip() is None,
                 "Export ZIP verification failed")
        exported = json.loads(archive.read("submission.json"), object_pairs_hook=_unique_pairs)
        _require(exported == predictions, "Export changed predictions")
        validate_predictions(exported, questions)
    temporary.replace(path)


def run_repair(diagnostics, output, *, audit_only=False, expected_public_count=None):
    print("Verifying Stage 3 diagnostics...", flush=True)
    bundle = load_diagnostics(diagnostics, expected_public_count)
    root = Path(output)
    identity = {"source": bundle["source"], "implementation": _implementation_identity(),
                "policy": POLICY, "audit_only": audit_only, "expected_public_count": expected_public_count}
    marker = root / "repair.identity.json"
    if marker.exists():
        _require(read_json(marker) == identity, "Stage 4 identity differs; choose a new output directory")
    else:
        _require(not root.exists() or not any(root.iterdir()), "Output is not an empty Stage 4 directory")
        write_json(marker, identity)
    # A same-identity retry must not expose an old ZIP if scoring fails this time.
    for name in ("submission_repaired.zip", "submission_original.zip"):
        (root / name).unlink(missing_ok=True)
    write_json(root / "repair.manifest.json", {"status": "running", "identity": identity})
    predictions, audits, queues = {}, {}, {}
    for split in ("dev", "public"):
        data = bundle[split]
        pred, audit, queue = repair_predictions(data["predictions"], data["audit"])
        predictions[split], audits[split], queues[split] = pred, audit, queue
        prefix = "submission" if split == "public" else "dev"
        write_json(root / f"{prefix}.original.json", data["predictions"])
        write_json(root / f"{prefix}.repaired.json", pred)
    write_json(root / "dev.references.json", bundle["dev"]["references"])
    write_json(root / "repair.audit.json", audits)
    write_json(root / "repair.candidate_unresolved.json", queues)
    summary = {split: {"questions": len(predictions[split]), "screened": len(audits[split]),
                       "changed_cpu": sum(row["changed"] for row in audits[split].values()),
                       "blocked_edits": sum(bool(row["blocked"]) for row in audits[split].values()),
                       "unresolved": len(queues[split]), "generated_gpu": 0}
               for split in ("dev", "public")}
    if audit_only:
        decision = {"accepted": False, "selected": "original", "reason": "audit_only_not_scored"}
        metrics = {"status": "not_scored", "decision": decision, "summary": summary}
    else:
        try:
            baseline, candidate = _score_dev(root, bundle["dev"]["stored_metrics"])
        except Exception as error:
            write_json(root / "repair.manifest.json", {"status": "failed", "identity": identity,
                       "error": str(error), "submission_zip": None})
            raise
        decision = acceptance_decision(baseline, candidate, bundle["dev"]["predictions"], predictions["dev"])
        changed_ids = decision["changed_dev_ids"]
        comparison = [{"id": k, **{m: candidate["per_question"][k][m] - baseline["per_question"][k][m]
                                   for m in ("meteor", "rougeL")}} for k in changed_ids]
        metrics = {"status": "scored", "baseline": {m: baseline[m] for m in ("meteor", "rougeL")},
                   "candidate": {m: candidate[m] for m in ("meteor", "rougeL")},
                   "metric_identity": baseline["metric_identity"], "decision": decision,
                   "changed_dev_deltas": comparison, "summary": summary}
    # Candidates remain inspectable even when rejected; only the selected variant is packaged.
    selected = predictions["public"] if decision["accepted"] else bundle["public"]["predictions"]
    selected_queues = queues
    if not decision["accepted"]:
        selected_queues = {}
        for split in ("dev", "public"):
            selected_queues[split] = {
                k: {"reasons": row["reasons"], "action": "review_original",
                    "context_parent_ids": bundle[split]["audit"][k]["context_parent_ids"],
                    "regenerate_automatically": False} for k, row in audits[split].items()}
    for split in summary:
        summary[split]["candidate_unresolved"] = summary[split]["unresolved"]
        summary[split]["unresolved"] = len(selected_queues[split])
    write_json(root / "repair.unresolved.json", {"variant": decision["selected"], **selected_queues})
    write_json(root / "submission.selected.json", selected)
    write_json(root / "repair.metrics.json", metrics)
    zip_name = None
    if not audit_only:
        zip_name = "submission_repaired.zip" if decision["accepted"] else "submission_original.zip"
        _package(selected, bundle["public"]["questions"], root / zip_name)
    outputs = {p.name: file_hash(p) for p in root.iterdir()
               if p.is_file() and p.name not in ("repair.manifest.json",) and not p.name.endswith(".tmp")}
    result = {"status": "audit_only" if audit_only else "complete", "identity": identity,
              "verification": bundle["verification"], "decision": decision, "summary": summary,
              "submission_zip": zip_name, "files": outputs}
    write_json(root / "repair.manifest.json", result)
    print(json.dumps({k: result[k] for k in ("status", "decision", "summary", "submission_zip")},
                     ensure_ascii=False, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-only", action="store_true", help="Inspect without scoring or packaging")
    args = parser.parse_args()
    run_repair(args.diagnostics, args.output, audit_only=args.audit_only)


if __name__ == "__main__":
    main()
