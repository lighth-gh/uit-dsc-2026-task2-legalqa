import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch
from zipfile import ZipFile

from legalqa.io import read_json, write_json
from legalqa.repair import EXTENDED_POLICY, deduplicate_answer, repair_predictions
from legalqa.repair_v2 import (GPU_RECIPE, accept_generated, compare_variant, gpu_candidates,
                               loop_reasons, run_cpu, run_gpu)
from test_repair import CLAUSE, DETAIL, audit_row, fixture


class RepairV2Tests(unittest.TestCase):
    def test_numbered_loop_only_deletes_identical_content(self):
        text = "\n".join(f"{i}) {CLAUSE}" for i in range(1, 11)) + "\n" + DETAIL
        result, changes = deduplicate_answer(text, EXTENDED_POLICY)
        self.assertEqual(result, "1) " + CLAUSE + "\n" + DETAIL)
        self.assertEqual(changes[-1]["unit"], "numbered_loop")
        self.assertEqual(deduplicate_answer(result, EXTENDED_POLICY), (result, []))

    def test_preserves_money_conditions_negation_and_separate_sections(self):
        variants = ["\n".join(f"{i}) Mức tiền phạt áp dụng là {i} triệu đồng đối với hành vi này." for i in range(1, 10)),
                    "\n".join(f"{i}) " + (CLAUSE if i % 2 else CLAUSE.replace("phải", "không phải")) for i in range(1, 10)),
                    "\n".join(f"{i}) {CLAUSE} Trừ trường hợp thứ {i}." for i in range(1, 10)),
                    "\n".join(f"{i}) {CLAUSE}" for i in range(1, 6)),
                    "1. Điều kiện áp dụng A:\n" + CLAUSE + "\n2. Điều kiện áp dụng B:\n" + CLAUSE]
        for text in variants:
            self.assertEqual(deduplicate_answer(text, EXTENDED_POLICY), (text, []))

    def test_preserves_recap_and_blocks_empty_lead(self):
        text = CLAUSE + "\n" + DETAIL + "\nNhư vậy, hồ sơ gồm:\n" + CLAUSE + "\n" + DETAIL
        self.assertEqual(deduplicate_answer(text, EXTENDED_POLICY), (text, []))
        loop = "\n".join(f"{i}) Các yêu cầu về hồ sơ được quy định cụ thể như sau:" for i in range(1, 12))
        pred = {"a": {"answer": loop}}
        result, audit, _ = repair_predictions(pred, {"a": audit_row()}, EXTENDED_POLICY)
        self.assertEqual(result, pred)
        self.assertIn("dedup_leaves_unfinished_answer", audit["a"]["blocked"])

    def test_gpu_gate_uses_whole_dev_metric_and_minimum_changes(self):
        old = {k: {"answer": CLAUSE} for k in ("a", "b")}
        new = {k: {"answer": DETAIL} for k in old}
        base = {"meteor": .6, "rougeL": .5, "per_question": {k: {"meteor": .6, "rougeL": .5} for k in old}}
        for delta, accepted in [(-.01, False), (.0005, False), (.005, True)]:
            candidate = {**base, "meteor": .6 + delta, "rougeL": .7}
            self.assertEqual(compare_variant(base, candidate, old, new, gpu=True)["accepted"], accepted)
        self.assertFalse(compare_variant(base, {**base, "meteor": .7}, old,
                                         {"a": new["a"], "b": old["b"]}, gpu=True)["accepted"])

    def test_generated_guards_never_restore_raw_or_unsafe_answer(self):
        original = {"answer": CLAUSE}
        good = {"prediction": {"answer": DETAIL}, "audit": audit_row()}
        self.assertEqual(accept_generated(original, good), (good["prediction"], []))
        for updates in [{"route": "source_fallback"}, {"hit_token_limit": True},
                        {"refusal_evidence_support": {"strong": False}},
                        {"entity_conflict": {"conflict": True}},
                        {"flags_before_fallback": {"unsupported_document_numbers": ["fake"]}}]:
            bad = {**good, "audit": audit_row(**updates)}
            selected, rejected = accept_generated(original, bad)
            self.assertEqual(selected, original)
            self.assertTrue(rejected)

    def test_generation_overrides_are_opt_in_and_keep_stage3_defaults(self):
        from legalqa.generation import _generate_one
        class Sequence:
            def __getitem__(self, key):
                return SimpleNamespace(tolist=lambda: [9, 2])
        fake_torch = SimpleNamespace(tensor=lambda *a, **k: object(), ones_like=lambda value: object(),
                                     inference_mode=contextlib.nullcontext)
        model = SimpleNamespace(generate=Mock(return_value=Sequence()))
        tokenizer = SimpleNamespace(eos_token_id=2, pad_token_id=2, decode=lambda *a, **k: DETAIL)
        with patch.dict("sys.modules", {"torch": fake_torch}), \
             patch("legalqa.generation.pack_prompt", return_value=([1], [{"text": DETAIL, "parent_id": "doc:1"}])), \
             patch("legalqa.generation.answer_flags", return_value={"empty": False, "artifact": False,
                   "unsupported_document_numbers": [], "refusal": False}), \
             patch("legalqa.generation.refusal_evidence_support", return_value={"strong": True}), \
             patch("legalqa.generation.citation_context_conflict", return_value={"conflict": False}):
            args = ({"generation": {"max_new_tokens": 1536}}, "a", {"a": {"question": "Thủ tục?"}},
                    {"a": {"contexts": []}}, model, tokenizer, "cuda:0", "generate")
            result = _generate_one(*args)
            self.assertEqual(result["audit"]["route"], "generated")
            self.assertEqual(model.generate.call_args.kwargs["repetition_penalty"], 1.0)
            self.assertNotIn("no_repeat_ngram_size", model.generate.call_args.kwargs)
            _generate_one(*args, generation_overrides={"repetition_penalty": 1.0, "no_repeat_ngram_size": 0})
            self.assertEqual(model.generate.call_args.kwargs["no_repeat_ngram_size"], 0)
            self.assertEqual(model.generate.call_args.kwargs["repetition_penalty"], 1.0)
            self.assertEqual(model.generate.call_args.kwargs["max_new_tokens"], 1536)

    def test_repair_prompt_preserves_original_system_and_respects_budget(self):
        from legalqa.io import config
        from legalqa.prompts import SYSTEM, messages, pack_prompt
        from test_core import TinyTokenizer
        contexts = [{"parent_id": "doc:1", "text": " ".join([CLAUSE, DETAIL] * 30)}]
        original = messages("Hồ sơ?", contexts)
        self.assertEqual(original[0]["content"], SYSTEM)
        custom = messages("Hồ sơ?", contexts, system_suffix=GPU_RECIPE["system_suffix"])
        self.assertTrue(custom[0]["content"].startswith(SYSTEM + " "))
        self.assertEqual(custom[1], original[1])
        settings = config()
        settings["generation"]["max_input_tokens"] = 700
        tokenizer = TinyTokenizer()
        normal_ids, normal_contexts = pack_prompt("Hồ sơ?", contexts, tokenizer, settings)
        settings["generation"]["system_suffix"] = GPU_RECIPE["system_suffix"]
        repaired_ids, packed = pack_prompt("Hồ sơ?", contexts, tokenizer, settings)
        self.assertLessEqual(len(repaired_ids), 700)
        self.assertIn("Không đổi từ để né lặp.", tokenizer.decode(repaired_ids))
        self.assertEqual([p["parent_id"] for p in packed], ["doc:1"])
        self.assertLess(len(packed[0]["text"]), len(normal_contexts[0]["text"]))
        settings["generation"].pop("system_suffix")
        self.assertEqual(pack_prompt("Hồ sơ?", contexts, tokenizer, settings), (normal_ids, normal_contexts))

    def test_candidate_selection_has_no_gold_and_skips_weak_evidence(self):
        predictions = {k: {"answer": CLAUSE} for k in ("a", "b", "c")}
        data = {"audit": {k: audit_row(hit_token_limit=k != "c") for k in predictions},
                "questions": {k: {"question": "Thủ tục?"} for k in predictions},
                "records": {k: {"contexts": [{"text": DETAIL}]} for k in predictions}}
        with patch("legalqa.prompts.refusal_evidence_support", side_effect=[{"strong": True}, {"strong": False}]):
            ids, skipped = gpu_candidates(data, predictions)
        self.assertEqual(ids, ["a"])
        self.assertEqual(list(skipped), ["b"])

    def test_loop_only_selection_uses_original_prediction_and_ignores_other_errors(self):
        loop = "\n".join(f"{i}) {CLAUSE}" for i in range(1, 7))
        selected = {"loop": {"answer": CLAUSE}, "token": {"answer": CLAUSE}}
        original = {"loop": {"answer": loop}, "token": {"answer": CLAUSE}}
        data = {"audit": {key: audit_row(hit_token_limit=True) for key in selected},
                "questions": {key: {"question": "Thủ tục?"} for key in selected},
                "records": {key: {"contexts": [{"text": DETAIL}]} for key in selected}}
        self.assertIn("runaway_incrementing_list", loop_reasons(loop))
        with patch("legalqa.prompts.refusal_evidence_support", return_value={"strong": True}):
            ids, skipped = gpu_candidates(data, selected, detection_predictions=original,
                                          loops_only=True)
        self.assertEqual(ids, ["loop"])
        self.assertEqual(skipped, {})

    def test_focused_prompt_keeps_only_top_document_contexts(self):
        from legalqa.io import config
        from legalqa.prompts import pack_prompt
        from test_core import TinyTokenizer
        settings = config()
        settings["generation"].update({"contexts_k": 2, "same_document_as_top": True,
                                       "complete_legal_units": True})
        contexts = [
            {"parent_id": "right:1", "doc_id": "right", "text": CLAUSE},
            {"parent_id": "wrong:1", "doc_id": "wrong", "text": DETAIL},
            {"parent_id": "right:2", "doc_id": "right", "text": DETAIL},
        ]
        _, packed = pack_prompt("Thủ tục?", contexts, TinyTokenizer(), settings)
        self.assertEqual([row["parent_id"] for row in packed], ["right:1", "right:2"])

    def test_cpu_v2_cannot_replace_better_v1(self):
        from legalqa.repair import load_diagnostics, run_repair
        def score(path, refs, out):
            write_json(out, {**baseline, "meteor": .51})
        baseline = {"meteor": .5, "rougeL": .5, "metric_identity": {},
                    "per_question": {k: {"meteor": .5, "rougeL": .5} for k in ("a", "b")}}
        def legacy(*args, **kwargs):
            result = run_repair(*args, **kwargs, expected_public_count=2)
            target = Path(args[1])
            write_json(target / "dev.original.metrics.json", baseline)
            write_json(target / "dev.repaired.metrics.json", {**baseline, "meteor": .6})
            return result
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root = Path(d); fixture(root / "input.zip")
            with patch("legalqa.repair_v2.load_diagnostics", side_effect=lambda p: load_diagnostics(p, 2)), \
                 patch("legalqa.repair_v2.run_repair", side_effect=legacy), \
                 patch("legalqa.repair._score_dev", return_value=(baseline, {**baseline, "meteor": .6})), \
                 patch("legalqa.metrics.evaluate", side_effect=score):
                run_cpu(root / "input.zip", root / "out")
            self.assertEqual(read_json(root / "out/repair.manifest.json")["selected_variant"], "cpu_v1")
            with ZipFile(root / "out/submission_selected.zip") as z:
                self.assertEqual(z.namelist(), ["submission.json"])
            with patch("legalqa.repair_v2.load_diagnostics", side_effect=lambda p: load_diagnostics(p, 2)), \
                 patch("legalqa.repair_v2.run_repair", side_effect=RuntimeError("scorer failed")):
                with self.assertRaisesRegex(RuntimeError, "scorer failed"):
                    run_cpu(root / "input.zip", root / "out")
            self.assertFalse((root / "out/submission_selected.zip").exists())
            self.assertEqual(read_json(root / "out/repair.manifest.json")["status"], "running")


class GPUWorkflowTests(unittest.TestCase):
    def prepare(self, root):
        selected = {"dev": {k: {"answer": CLAUSE} for k in ("a", "b")}, "public": {"p": {"answer": CLAUSE}}}
        expected = {"adapter": {"adapter_model.safetensors": "expected"}, "models": {"models": {}}}
        bundle = {"source": {"diagnostics_sha256": "source"}, "verification": {},
                  "config": {"seed": 2026, "generation": {"max_input_tokens": 7680, "max_new_tokens": 1536}}}
        for split, rows in selected.items():
            bundle[split] = {"predictions": rows, "questions": {k: {"question": "Thủ tục?"} for k in rows},
                             "audit": {k: audit_row() for k in rows},
                             "records": {k: {"contexts": [{"text": DETAIL}]} for k in rows},
                             "prediction_manifest": {"identity": expected}}
        control = {"meteor": .6, "rougeL": .5, "metric_identity": {},
                   "per_question": {k: {"meteor": .6, "rougeL": .5} for k in selected["dev"]}}
        write_json(root / "identity.json", {})
        write_json(root / "cpu.metrics.json", {"selected_variant": "cpu_v2"})
        models = root / "models"; (models / "generator").mkdir(parents=True)
        (models / "generator/model.safetensors").write_bytes(b"test weights only")
        return bundle, selected, control, expected, models

    def workflow(self, dev_gain, *, resume=False, corrupt=False, old_recipe=False):
        with tempfile.TemporaryDirectory() as d, contextlib.ExitStack() as stack:
            root = Path(d); bundle, selected, control, expected, models = self.prepare(root)
            def evaluate(pred, refs, output):
                write_json(output, {**control, "meteor": control["meteor"] + dev_gain})
            stack.enter_context(patch.dict("sys.modules", {
                "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
                "transformers": SimpleNamespace(set_seed=lambda seed: None)}))
            stack.enter_context(patch("legalqa.generation.adapter_identity", return_value=expected["adapter"]))
            stack.enter_context(patch("legalqa.models.model_lock", return_value=expected["models"]))
            stack.enter_context(patch("legalqa.models.load_generator", return_value=(object(), object())))
            stack.enter_context(patch("legalqa.runtime.should_pause", return_value=False))
            stack.enter_context(patch("legalqa.repair_v2.gpu_candidates", side_effect=lambda data, pred: (sorted(pred), {})))
            stack.enter_context(patch("legalqa.metrics.evaluate", side_effect=evaluate))
            generate = stack.enter_context(patch("legalqa.generation._generate_one", return_value={
                "prediction": {"answer": DETAIL}, "audit": audit_row()}))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            run_gpu(bundle, selected, control, root, models, root / "adapter", max_items=1 if resume else 50)
            if resume:
                self.assertEqual(read_json(root / "repair.manifest.json")["status"], "paused")
                self.assertEqual(generate.call_count, 1)
                self.assertFalse((root / "gpu/public.checkpoint.jsonl").exists())
                if old_recipe:
                    path = root / "gpu/identity.json"
                    identity = read_json(path)
                    identity["recipe"]["version"] = 1
                    write_json(path, identity)
                    with self.assertRaisesRegex(ValueError, "identity differs"):
                        run_gpu(bundle, selected, control, root, models, root / "adapter")
                    self.assertEqual(generate.call_count, 1)
                    return
                if corrupt:
                    path = root / "gpu/dev.checkpoint.jsonl"
                    row = json.loads(path.read_text(encoding="utf-8"))
                    row["value"]["prediction"]["answer"] = "corrupt"
                    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "journal content"):
                        run_gpu(bundle, selected, control, root, models, root / "adapter")
                    return
                run_gpu(bundle, selected, control, root, models, root / "adapter")
            self.assertEqual(generate.call_count, 3 if dev_gain >= .001 else 2)
            manifest = read_json(root / "repair.manifest.json")
            self.assertEqual(manifest["selected_variant"], "gpu_v2" if dev_gain >= .001 else "cpu_v2")
            if dev_gain < .001:
                self.assertFalse((root / "gpu/public.checkpoint.jsonl").exists())
            for call in generate.call_args_list:
                self.assertEqual(call.kwargs["generation_overrides"], {"repetition_penalty": 1.0, "no_repeat_ngram_size": 0})
                self.assertEqual(call.args[0]["generation"]["system_suffix"], GPU_RECIPE["system_suffix"])
                self.assertNotIn("references", call.kwargs)
            self.assertNotIn("system_suffix", bundle["config"]["generation"])

    def test_gpu_dev_rejection_prevents_public_generation(self):
        self.workflow(-.01)

    def test_gpu_resume_and_complete_export(self):
        self.workflow(.005, resume=True)

    def test_corrupt_gpu_journal_rejected(self):
        self.workflow(.005, resume=True, corrupt=True)

    def test_previous_recipe_cannot_be_resumed(self):
        self.workflow(.005, resume=True, old_recipe=True)


if __name__ == "__main__":
    unittest.main()
