import ast
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from legalqa.io import digest, read_json
from legalqa.repair import (PUBLIC, acceptance_decision, deduplicate_answer,
                           diagnostics_zip_from_directory, load_diagnostics,
                           repair_predictions, run_repair)


CLAUSE = "Người sử dụng lao động phải thông báo đầy đủ cho người lao động trước khi thực hiện thủ tục."
DETAIL = "Hồ sơ được nộp tại cơ quan có thẩm quyền. Người nộp hồ sơ phải giữ lại giấy tiếp nhận để đối chiếu."


def audit_row(**updates):
    return {"route": "generated", "hit_token_limit": False, "context_parent_ids": ["doc:1"],
            "raw_answer": "RAW MUST NEVER BE RESTORED", "refusal_evidence_support": {"strong": True},
            **updates}


def dump(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def fixture(path, edit=None):
    """Small complete snapshot exercising provenance, not just prediction validation."""
    import hashlib
    config = {"generation": {"max_new_tokens": 1536}}
    references = {"a": CLAUSE, "b": DETAIL}
    questions = {"a": {"question": "Thủ tục được thực hiện như thế nào?"},
                 "b": {"question": "Nộp hồ sơ ở đâu?"}}
    pred = {"a": {"answer": (CLAUSE + "\n") * 3 + DETAIL}, "b": {"answer": DETAIL}}
    audit = {k: audit_row() for k in pred}
    files = {"config.json": dump(config), "session.json": dump({"config_hash": digest(config),
             "source_hash": "stage3-code"}), "data/dev100.references.json": dump(references)}
    for split, prefix in [("public", PUBLIC), ("dev100", "dev100.epoch-01")]:
        retrieval = {"identity": {"questions_hash": digest(questions)}, "records": {
            k: {"question": q["question"], "contexts": [{"parent_id": "doc:1", "text": CLAUSE}]}
            for k, q in questions.items()}}
        files[f"{split}.retrieval.json"] = dump(retrieval)
        identity = {"questions_hash": digest(questions), "config": config, "code": "stage3-code",
                    "adapter": {"adapter_model.safetensors": "weight-hash"},
                    "retrieval": retrieval["identity"],
                    "retrieval_file_hash": hashlib.sha256(dump(retrieval)).hexdigest()}
        pm = {"identity": identity, "prediction_hash": digest(pred)}
        files[prefix + ".json"] = dump(pred)
        files[prefix + ".audit.json"] = dump(audit)
        files[prefix + ".manifest.json"] = dump(pm)
        files[prefix + ".checkpoint.jsonl.meta.json"] = dump(identity)
        files[prefix + ".checkpoint.jsonl"] = b"".join(
            (json.dumps({"id": k, "value": {"prediction": pred[k], "audit": audit[k]}},
                        ensure_ascii=False) + "\n").encode() for k in pred)
    files["data/test.questions.json"] = dump(questions)
    files["data/dev100.questions.json"] = dump(questions)
    metrics = {"meteor": .5, "rougeL": .5, "prediction_hash": digest(pred),
               "reference_hash": digest(references), "per_question": {
                   k: {"meteor": .5, "rougeL": .5} for k in pred}}
    files["dev100.epoch-01.metrics.json"] = dump(metrics)
    files["selection.json"] = dump({"label": "epoch-01", "meteor": .5, "rougeL": .5,
                                   "reference_hash": digest(references), "prediction_manifest": pm})
    if edit:
        edit(files)
    manifest = {"stage": 3, "schema": 2, "quality_version": "v8", "status": "complete",
                "source_hash": "stage3-code", "progress": {"complete": True, "answers": 2},
                "files": {k: {"size": len(v), "sha256": hashlib.sha256(v).hexdigest()}
                          for k, v in files.items()}}
    manifest["files"]["selected_adapter/adapter_model.safetensors"] = {"size": 99, "sha256": "omitted"}
    files["stage3_manifest.json"] = dump(manifest)
    with ZipFile(path, "w") as z:
        for k, v in files.items():
            z.writestr(k, v)
    return files


class RepairTests(unittest.TestCase):
    def test_exact_multiline_loop_and_unchanged_tail(self):
        block = CLAUSE + "\n" + DETAIL + "\n"
        result, changes = deduplicate_answer(block * 4 + "Kết thúc.")
        self.assertEqual(result, block + "Kết thúc.")
        self.assertTrue(changes)

    def test_exact_sentence_loop_on_one_line(self):
        text = " ".join([CLAUSE] * 4) + " " + DETAIL
        result, _ = deduplicate_answer(text)
        self.assertEqual(result, CLAUSE + " " + DETAIL)

    def test_preserves_two_copies_numbers_negations_and_conditions(self):
        for text in [CLAUSE + "\n" + CLAUSE,
                     "\n".join(f"{n}. Mức tiền phạt áp dụng cho trường hợp này là {n} triệu đồng theo quy định."
                               for n in (1, 2, 3)),
                     "\n".join([CLAUSE, CLAUSE.replace("phải", "không phải"),
                                CLAUSE + " Trừ trường hợp đã có thỏa thuận."])]:
            self.assertEqual(deduplicate_answer(text), (text, []))

    def test_intro_loop_is_blocked_and_queued(self):
        lead = "Căn cứ theo khoản 8 Điều 3 Quyết định 909/QĐ-BNV năm 2016 quy định như sau:"
        before = {"a": {"answer": "\n".join([lead] * 45)}}
        result, audit, queue = repair_predictions(before, {"a": audit_row(hit_token_limit=True)})
        self.assertEqual(result, {"a": {"answer": lead.rstrip(":") + "."}})
        self.assertIn("dedup_leaves_unfinished_answer", audit["a"]["blocked"])
        self.assertFalse(queue["a"]["regenerate_automatically"])

    def test_two_repeats_large_block_deduplication(self):
        long_block = CLAUSE + " " + DETAIL
        result, changes = deduplicate_answer(long_block + "\n" + long_block)
        self.assertEqual(result, long_block)
        self.assertTrue(changes)

    def test_semicolon_clause_deduplication(self):
        semi_block = "Khoản a quy định người lao động được nghỉ phép hàng năm; Khoản b quy định người sử dụng lao động phải thanh toán tiền lương;"
        result, changes = deduplicate_answer(semi_block + " " + semi_block + " " + semi_block)
        self.assertEqual(result.strip(), semi_block.strip())
        self.assertTrue(changes)

    def test_dangling_tail_cleaned(self):
        long_answer = (CLAUSE + "\n") * 3 + "Năng lực chuyên môn:"
        result, audit, _ = repair_predictions({"a": {"answer": long_answer}}, {"a": audit_row()})
        self.assertFalse(result["a"]["answer"].endswith(":"))
        self.assertTrue(result["a"]["answer"].endswith("."))

    def test_token_limit_does_not_force_edit_or_restore_raw(self):
        before = {"a": {"answer": DETAIL}, "b": {"answer": CLAUSE}}
        result, _, queue = repair_predictions(before, {"a": audit_row(hit_token_limit=True),
            "b": audit_row(route="source_fallback")})
        self.assertEqual(result, before)
        self.assertIn("check_answer_completeness", queue["a"]["reasons"])
        self.assertIn("check_context_relevance", queue["b"]["reasons"])

    def test_snapshot_integrity_and_intentionally_missing_weights(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "input.zip"
            fixture(path)
            bundle = load_diagnostics(path, 2)
            self.assertEqual(bundle["verification"]["journal"], "passed")
            self.assertEqual(bundle["source"]["omitted_weights"], ["selected_adapter/adapter_model.safetensors"])
            with self.assertRaisesRegex(ValueError, "count"):
                load_diagnostics(path, 1000)

    def test_corrupt_hash_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "input.zip"
            files = fixture(path)
            files[PUBLIC + ".json"] = files[PUBLIC + ".json"] + b" "
            with ZipFile(path, "w") as z:
                for k, v in files.items(): z.writestr(k, v)
            with self.assertRaisesRegex(ValueError, "hash/size"):
                load_diagnostics(path, 2)

    def test_journal_corruption_rejected_even_with_updated_file_hash(self):
        def edit(files):
            rows = files[PUBLIC + ".checkpoint.jsonl"].splitlines()
            files[PUBLIC + ".checkpoint.jsonl"] = rows[0] + b"\n" + rows[0] + b"\n"
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "input.zip"
            fixture(path, edit)
            with self.assertRaisesRegex(ValueError, "Duplicate journal"):
                load_diagnostics(path, 2)

    def test_missing_required_diagnostic_is_not_treated_like_weights(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "input.zip"
            files = fixture(path)
            del files["dev100.epoch-01.audit.json"]
            with ZipFile(path, "w") as z:
                for k, v in files.items(): z.writestr(k, v)
            with self.assertRaisesRegex(ValueError, "Missing manifest file"):
                load_diagnostics(path, 2)

    def test_unpacked_kaggle_input_is_deterministic(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            files = fixture(root / "input.zip")
            source = root / "dataset"
            for k, v in files.items():
                p = source / k; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(v)
            a = diagnostics_zip_from_directory(source, root / "a.zip")
            b = diagnostics_zip_from_directory(source, root / "b.zip")
            self.assertEqual(a.read_bytes(), b.read_bytes())
            load_diagnostics(a, 2)

    def test_metric_drop_rejects_even_if_rouge_improves(self):
        original = {"a": {"answer": (CLAUSE + "\n") * 4 + DETAIL}}
        repaired, _, _ = repair_predictions(original, {"a": audit_row()})
        decision = acceptance_decision({"meteor": .5, "rougeL": .4},
                                       {"meteor": .49, "rougeL": .6}, original, repaired)
        self.assertFalse(decision["accepted"])

    def test_export_accept_reject_and_audit_only(self):
        baseline = {"meteor": .5, "rougeL": .5, "metric_identity": {},
                    "per_question": {k: {"meteor": .5, "rougeL": .5} for k in ("a", "b")}}
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root = Path(d); fixture(root / "input.zip")
            for score, accepted in [(.6, True), (.4, False)]:
                out = root / str(score)
                candidate = {**baseline, "meteor": score}
                with patch("legalqa.repair._score_dev", return_value=(baseline, candidate)):
                    result = run_repair(root / "input.zip", out, expected_public_count=2)
                self.assertEqual(result["decision"]["accepted"], accepted)
                with ZipFile(out / result["submission_zip"]) as z:
                    self.assertEqual(z.namelist(), ["submission.json"])
                    exported = json.loads(z.read("submission.json"))
                selected = read_json(out / ("submission.repaired.json" if accepted else "submission.original.json"))
                self.assertEqual(exported, selected)
                self.assertEqual(exported["b"], {"answer": DETAIL})
                if not accepted:
                    self.assertIn("a", read_json(out / "repair.unresolved.json")["public"])
            audit = run_repair(root / "input.zip", root / "audit", audit_only=True, expected_public_count=2)
            self.assertIsNone(audit["submission_zip"])
            self.assertEqual(list((root / "audit").glob("*.zip")), [])

    def test_retry_cannot_publish_stale_zip_or_mix_identities(self):
        baseline = {"meteor": .5, "rougeL": .5, "metric_identity": {},
                    "per_question": {k: {"meteor": .5, "rougeL": .5} for k in ("a", "b")}}
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root = Path(d); fixture(root / "input.zip"); out = root / "output"
            with patch("legalqa.repair._score_dev", return_value=(baseline, {**baseline, "meteor": .6})):
                run_repair(root / "input.zip", out, expected_public_count=2)
            with patch("legalqa.repair._score_dev", side_effect=RuntimeError("scorer failed")):
                with self.assertRaisesRegex(RuntimeError, "scorer failed"):
                    run_repair(root / "input.zip", out, expected_public_count=2)
            self.assertFalse((out / "submission_repaired.zip").exists())
            self.assertEqual(read_json(out / "repair.manifest.json")["status"], "failed")
            with self.assertRaisesRegex(ValueError, "identity differs"):
                run_repair(root / "input.zip", out, audit_only=True, expected_public_count=2)

    def test_notebook_cells_compile_and_bundle_matches_sources(self):
        import base64
        import hashlib
        from legalqa.io import ROOT
        nb = read_json(ROOT / "legalqa_main_04_repair_submit.ipynb")
        namespace = {}
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                ast.parse(source)
                if cell["id"] == "s4bundle":
                    exec(source, namespace)
        payload = base64.b64decode(namespace["BUNDLE_B64"])
        self.assertEqual(hashlib.sha256(payload).hexdigest(), namespace["BUNDLE_SHA256"])
        with ZipFile(io.BytesIO(payload)) as z:
            for name in z.namelist():
                self.assertEqual(z.read(name), (ROOT / name).read_bytes().replace(b"\r\n", b"\n"))


if __name__ == "__main__":
    unittest.main()
