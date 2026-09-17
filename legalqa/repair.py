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


POLICY = {"version": 2, "min_chars": 60, "min_words": 8,
          "min_repeats": 3, "max_block_units": 12,
          "min_chars_2rep": 100, "min_words_2rep": 15,
          "allow_2_repeats_for_large_blocks": True,
          "clean_dangling_colons": True,
          "clean_trailing_bare_bullets": True,
          "enable_extended_dedup": True,
          "min_words_floor": 160,
          "keep_single_lead_in_on_blocked": True}
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


def load_diagnostics(path, expected_public_count=1000):
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
                             "prediction_manifest": pm}
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
    return {**splits, "source": {"diagnostics_sha256": file_hash(path), "stage3_code": manifest["source_hash"],
            "selected_adapter": label, "omitted_weights": omitted},
            "verification": {"crc": "passed", "hashes": "passed", "schema_and_ids": "passed",
                             "journal": "passed", "identity": "passed",
                             "public_count": expected_public_count}}


def repetition_ratio(text):
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) >= 35]
    return (len(lines) - len(set(lines))) / len(lines) if lines else 0.0


def _collapse(text, mode, policy):
    if mode is True or mode == "sentence":
        pattern = r"\S[\s\S]*?(?:[.!?](?=\s|$)|$)"
        unit_name = "sentence"
    elif mode == "clause":
        pattern = r"\S[\s\S]*?(?:[.!?;](?=\s|$)|$)"
        unit_name = "clause"
    else:
        pattern = r"[^\r\n]+"
        unit_name = "line"
    units = [(m.start(), m.end(), m.group().strip()) for m in re.finditer(pattern, text)
             if m.group().strip()]
    deletions, i = [], 0
    eff_min_rep = 2 if policy.get("allow_2_repeats_for_large_blocks") else policy["min_repeats"]
    while i < len(units):
        matched = False
        max_width = min(policy["max_block_units"], (len(units) - i) // eff_min_rep)
        for width in range(1, max_width + 1):
            block = [u[2] for u in units[i:i + width]]
            joined = " ".join(block)
            wc = len(joined.split())
            cc = len(joined)
            if cc < policy["min_chars"] or wc < policy["min_words"]:
                continue
            end = i + width
            while end + width <= len(units) and [u[2] for u in units[end:end + width]] == block:
                end += width
            repeats = (end - i) // width
            req_repeats = policy["min_repeats"]
            if policy.get("allow_2_repeats_for_large_blocks") and cc >= policy.get("min_chars_2rep", 100) and wc >= policy.get("min_words_2rep", 15):
                req_repeats = 2
            if repeats >= req_repeats:
                deletions.append({"start": units[i + width - 1][1], "end": units[end - 1][1],
                                  "repeats": repeats, "unit": unit_name,
                                  "block_units": width})
                i, matched = end, True
                break
        if not matched:
            i += 1
    result = text
    for change in reversed(deletions):
        result = result[:change["start"]] + result[change["end"]:]
    return result, deletions


BULLET_RE = re.compile(
    r'^(?:(?:(?:Theo\s+)?(?:Khoản|Điều|Điểm|Mục|Phần)\s*\d+[a-z]?'
    r'(?:\s*Điều\s*\d+[a-z]?)?'
    r'(?:\s*quy\s*định(?:\s*về[^\n:]*)?(?:\s*như\s*sau)?)?'
    r'[\.\:\)]?|\d+[\.\)\:]|[a-zđ][\.\)\:]|[\-\*\•])\s*)+',
    re.IGNORECASE
)

LEAD_IN_RE = re.compile(
    r'^(?:(?:Căn\s+cứ\s+theo|Theo)\s+)?(?:Khoản|Điều|Điểm|Mục)\s*\d+[a-z]?'
    r'(?:\s*Điều\s*\d+[a-z]?)?'
    r'(?:\s*Luật|\s*Nghị\s*định|\s*Thông\s*tư|\s*Quyết\s*định)?'
    r'[\s\S]*?(?:quy\s*định(?:\s*về[^\n:]*)?(?:\s*như\s*sau)?)?\s*[:.]?$',
    re.IGNORECASE
)


def extract_bullet_body(line):
    m = BULLET_RE.match(line.strip())
    if m:
        return line.strip()[m.end():].strip(), m.group(0).strip()
    return line.strip(), ""


def dedup_enumerated_extreme(text, min_repeats=3, min_body_chars=25):
    lines = text.splitlines(keepends=True)
    if len(lines) < 4:
        return text, []
    new_lines = []
    changes = []
    i = 0
    while i < len(lines):
        line = lines[i]
        body, bullet = extract_bullet_body(line)
        if len(body) >= min_body_chars:
            j = i + 1
            matching_count = 1
            while j < len(lines):
                next_body, next_bullet = extract_bullet_body(lines[j])
                if next_body == body and (next_bullet or bullet):
                    matching_count += 1
                    j += 1
                elif not lines[j].strip():
                    if j + 1 < len(lines):
                        nb, nbul = extract_bullet_body(lines[j + 1])
                        if nb == body and (nbul or bullet):
                            matching_count += 1
                            j += 2
                            continue
                    break
                else:
                    break
            if matching_count >= min_repeats:
                new_lines.append(lines[i])
                changes.append({"start": len("".join(new_lines)), "end": len("".join(new_lines)) + len("".join(lines[i + 1:j])),
                                "repeats": matching_count, "unit": "enumerated_extreme", "block_units": 1})
                i = j
                continue
        new_lines.append(lines[i])
        i += 1
    return "".join(new_lines), changes


def dedup_incremental_loops(text):
    lines = text.splitlines(keepends=True)
    if len(lines) < 6:
        return text, []
    changes, i, new_lines = [], 0, []
    while i < len(lines):
        matched = False
        max_w = min(35, (len(lines) - i) // 2)
        for w in range(max_w, 1, -1):
            block = [l.strip() for l in lines[i:i + w]]
            if sum(len(l) for l in block) < 60:
                continue
            step = w + 1
            if i + step + w <= len(lines) and LEAD_IN_RE.match(lines[i + w].strip()):
                next_block = [l.strip() for l in lines[i + step:i + step + w]]
                if next_block == block:
                    k = i + step
                    reps = 2
                    while k + step <= len(lines):
                        if LEAD_IN_RE.match(lines[k + w].strip()) and [l.strip() for l in lines[k + step:k + step + w]] == block:
                            k += step
                            reps += 1
                        else:
                            break
                    new_lines.extend(lines[i:i + w])
                    changes.append({"start": len("".join(new_lines)), "end": len("".join(new_lines)) + len("".join(lines[i + w:k + w])),
                                    "repeats": reps, "unit": "incremental_loop", "block_units": w})
                    i = k + w
                    matched = True
                    break
        if not matched:
            new_lines.append(lines[i])
            i += 1
    return "".join(new_lines), changes


def drop_duplicate_tail(text, min_lines=5):
    lines = [l for l in text.splitlines(keepends=True)]
    stripped = [l.strip() for l in lines]
    max_k = min(len(lines) // 2, 40)
    for k in range(max_k, min_lines - 1, -1):
        tail = stripped[-k:]
        if not tail or not any(len(x) >= 20 for x in tail):
            continue
        earlier = stripped[:-k]
        for start_idx in range(len(earlier) - len(tail) + 1):
            if earlier[start_idx:start_idx + len(tail)] == tail:
                res = "".join(lines[:-k]).rstrip()
                return res, [{"start": len(res), "end": len(text), "repeats": 2, "unit": "tail_duplicate", "block_units": k}]
    return text, []


def clean_trailing_bare_bullet(text):
    """Strip dangling trailing bullet markers (e.g. '\n49.' or '\n2.') without following text."""
    m = re.search(r'(?:\n|\A)\s*(?:\d+[\.\)]|[a-zđ][\.\)]|[\-\*\•])\s*$', text)
    if m:
        cleaned = text[:m.start()].rstrip()
        if len(cleaned.split()) >= 20:
            return cleaned, True
    return text, False


def deduplicate_answer(text, policy=None):
    """Only identical adjacent blocks (>=3 copies, or >=2 copies for blocks >=100 chars); never fuzzy-match legal clauses."""
    policy = {**POLICY, **(policy or {})}
    _require(policy["min_repeats"] >= 3 and policy["min_chars"] >= 60
             and policy["min_words"] >= 8 and policy["max_block_units"] >= 1,
             "Unsafe repetition policy")
    result, changes = text, []
    for mode in ("line", "sentence", "clause"):
        result, removed = _collapse(result, mode, policy)
        changes.extend(removed)
    if policy.get("enable_extended_dedup", True):
        p_large = dict(policy)
        p_large["max_block_units"] = max(policy.get("max_block_units", 12), 50)
        for mode in ("line", "sentence", "clause"):
            result, removed = _collapse(result, mode, p_large)
            changes.extend(removed)
        result, incr_removed = dedup_incremental_loops(result)
        changes.extend(incr_removed)
        result, enum_removed = dedup_enumerated_extreme(result, min_repeats=3, min_body_chars=25)
        changes.extend(enum_removed)
        result, tail_removed = drop_duplicate_tail(result, min_lines=5)
        changes.extend(tail_removed)
    return result, changes


def clean_dangling_tail(text):
    """Strip trailing dangling colons / headers from truncated long answers."""
    stripped = text.rstrip()
    if not stripped.endswith(":"):
        return text, False
    lines = [l.rstrip() for l in stripped.splitlines()]
    changed = False
    while len(lines) >= 2 and lines[-1].endswith(":"):
        prev_lines = lines[:-1]
        prev_text = "\n".join(prev_lines).rstrip()
        words_prev = len(prev_text.split())
        if words_prev >= 15 and len(lines[-1].strip()) <= 250:
            lines = prev_lines
            changed = True
        else:
            break
    if changed:
        result = "\n".join(lines).rstrip()
        result = re.sub(r"[\s,:;]+$", "", result)
        if result and not result.endswith((".", "!", "?")):
            result += "."
        return result, True
    result = re.sub(r"[\s,:;]+$", "", stripped) + "."
    return result, True


def _incomplete(text):
    return not text.strip() or text.rstrip().endswith(":") or bool(
        re.search(r"(?:như sau|bao gồm|cụ thể là)\s*[.:;]?\s*$", text, re.I))


def repair_predictions(predictions, audit, policy=None):
    """No references, model calls or raw rejected generations enter this function."""
    _require(set(predictions) == set(audit), "Repair/audit ID mismatch")
    policy = {**POLICY, **(policy or {})}
    candidate, records, unresolved = {}, {}, {}
    for key, value in predictions.items():
        before, row = value["answer"], audit[key]
        proposed, changes = deduplicate_answer(before, policy)
        if policy.get("clean_dangling_colons", True):
            proposed, tail_cleaned = clean_dangling_tail(proposed)
            if tail_cleaned:
                changes.append({"start": len(proposed), "end": len(before),
                                "repeats": 1, "unit": "tail_clean", "block_units": 1})
        reasons = []
        ratio = repetition_ratio(before)
        if ratio >= .25:
            reasons.append("repetition_screen")
        if changes:
            reasons.append("exact_adjacent_loop")
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
        floor = policy.get("min_words_floor", 160)
        if changes and len(proposed.split()) < floor and len(before.split()) >= floor:
            blocked.append("dedup_leaves_below_length_floor")
        if blocked:
            if policy.get("keep_single_lead_in_on_blocked", True) and (_incomplete(before) or repetition_ratio(before) >= 0.5):
                after = proposed
            else:
                after = before
        else:
            after = proposed
        if policy.get("clean_trailing_bare_bullets", True):
            after, bullet_cleaned = clean_trailing_bare_bullet(after)
            if bullet_cleaned:
                changes.append({"start": len(after), "end": len(before),
                                "repeats": 1, "unit": "bare_bullet_clean", "block_units": 1})
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


def run_repair(diagnostics, output, *, audit_only=False, expected_public_count=1000):
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


def run_repair_submission(submission_path, output, *, audit_only=False, questions_path=None, policy=None):
    """Repair a standalone submission.zip or submission.json without requiring full Stage 3 diagnostics."""
    sub_path = Path(submission_path)
    _require(sub_path.exists(), f"Submission file not found: {sub_path}")
    if sub_path.suffix.lower() == ".zip":
        with ZipFile(sub_path) as z:
            _require("submission.json" in z.namelist(), "Missing submission.json inside zip")
            predictions = json.loads(z.read("submission.json"), object_pairs_hook=_unique_pairs)
    else:
        predictions = read_json(sub_path)

    questions = None
    if questions_path and Path(questions_path).exists():
        questions = read_json(questions_path)
        validate_predictions(predictions, questions)
    else:
        validate_predictions(predictions, predictions)

    audit_mock = {k: {"route": "generated", "hit_token_limit": False, "context_parent_ids": []} for k in predictions}
    policy = {**POLICY, **(policy or {})}
    repaired, records, unresolved = repair_predictions(predictions, audit_mock, policy=policy)

    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "submission.original.json", predictions)
    write_json(root / "submission.repaired.json", repaired)
    write_json(root / "repair.audit.json", records)
    write_json(root / "repair.unresolved.json", unresolved)

    changed_count = sum(predictions[k]["answer"] != repaired[k]["answer"] for k in predictions)
    chars_before = sum(len(predictions[k]["answer"]) for k in predictions)
    chars_after = sum(len(repaired[k]["answer"]) for k in repaired)
    rep_before = sum(repetition_ratio(predictions[k]["answer"]) >= 0.5 for k in predictions)
    rep_after = sum(repetition_ratio(repaired[k]["answer"]) >= 0.5 for k in repaired)
    colons_before = sum(predictions[k]["answer"].strip().endswith(":") for k in predictions)
    colons_after = sum(repaired[k]["answer"].strip().endswith(":") for k in repaired)

    summary = {
        "questions": len(predictions),
        "changed": changed_count,
        "chars_reduced": chars_before - chars_after,
        "severe_repetition_before": rep_before,
        "severe_repetition_after": rep_after,
        "colons_before": colons_before,
        "colons_after": colons_after,
    }
    write_json(root / "repair.summary.json", summary)

    zip_name = None
    if not audit_only:
        zip_name = "submission_repaired.zip"
        _package(repaired, questions or predictions, root / zip_name)

    outputs = {p.name: file_hash(p) for p in root.iterdir()
               if p.is_file() and p.name not in ("repair.manifest.json",) and not p.name.endswith(".tmp")}
    result = {"status": "complete", "submission_zip": zip_name, "summary": summary, "files": outputs}
    write_json(root / "repair.manifest.json", result)
    print(json.dumps({k: result[k] for k in ("status", "summary", "submission_zip")},
                     ensure_ascii=False, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", help="Path to Stage 3 diagnostics ZIP or folder")
    parser.add_argument("--submission", help="Path to standalone submission.zip or submission.json to repair on local")
    parser.add_argument("--questions", help="Optional path to questions JSON for validation")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--audit-only", action="store_true", help="Inspect without scoring or packaging")
    args = parser.parse_args()
    if args.diagnostics:
        run_repair(args.diagnostics, args.output, audit_only=args.audit_only)
    elif args.submission:
        run_repair_submission(args.submission, args.output, audit_only=args.audit_only, questions_path=args.questions)
    else:
        parser.error("Must provide either --diagnostics or --submission")


if __name__ == "__main__":
    main()
