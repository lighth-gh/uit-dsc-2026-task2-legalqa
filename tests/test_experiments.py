import tempfile
import unittest
from pathlib import Path

from legalqa.experiments import (INFERENCE_VARIANTS, RETRIEVAL_VARIANTS,
                                 _screen_decision, lock_baseline, patched_config, postprocess_candidate,
                                 write_ablation_configs)
from legalqa.io import config, file_hash, read_json, write_json


class ExperimentTests(unittest.TestCase):
    def test_variants_change_only_the_declared_section(self):
        base = config()
        for variants in (INFERENCE_VARIANTS, RETRIEVAL_VARIANTS):
            for name, patch in variants.items():
                with self.subTest(name=name):
                    candidate = patched_config(base, patch)
                    for section in base:
                        if section not in patch:
                            self.assertEqual(candidate[section], base[section])
                    for section, values in patch.items():
                        for key, value in values.items():
                            self.assertEqual(candidate[section][key], value)

    def test_write_configs_has_distinct_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = write_ablation_configs(Path(__file__).parents[1] / "config.json", root)
            self.assertEqual(set(manifest["inference"]), set(INFERENCE_VARIANTS))
            hashes = [row["config_hash"] for group in ("inference", "retrieval")
                      for row in manifest[group].values()]
            self.assertEqual(len(hashes), len(set(hashes)))
            self.assertTrue(all((root / row["path"]).is_file()
                                for group in ("inference", "retrieval")
                                for row in manifest[group].values()))

    def test_lock_baseline_verifies_bytes_and_records_reported_score(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_json(root / "submission.selected.json", {"1": {"answer": "ok"}})
            write_json(root / "submission.zip", {"fixture": True})
            write_json(root / "repair.metrics.json", {"candidate": {"meteor": .62, "rougeL": .59}})
            files = {name: file_hash(root / name) for name in
                     ("submission.selected.json", "submission.zip", "repair.metrics.json")}
            manifest = {"status": "complete", "submission_zip": "submission.zip",
                        "identity": {"source": {"selected_adapter": "epoch-01"}},
                        "verification": {"crc": "passed", "hashes": "passed",
                            "schema_and_ids": "passed", "journal": "passed",
                            "identity": "passed", "public_count": 1000}, "files": files}
            write_json(root / "repair.manifest.json", manifest)
            result = lock_baseline(root, root / "lock.json", .5599)
            self.assertEqual(result["status"], "locked")
            self.assertEqual(result["public"]["score"], .5599)
            (root / "submission.selected.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                lock_baseline(root, root / "other.json", .5599)

    def test_screen_rule_requires_material_gain_and_no_regression(self):
        control = {"a": {"answer": "câu trả lời gốc"}}
        candidate = {"a": {"answer": "câu trả lời mới"}}
        metrics = {"meteor": .62, "rougeL": .59}
        passed = _screen_decision(metrics, {"meteor": .631, "rougeL": .586}, control, candidate)
        self.assertTrue(passed["passes_screen"])
        failed = _screen_decision(metrics, {"meteor": .625, "rougeL": .60}, control, candidate)
        self.assertFalse(failed["passes_screen"])

    def test_postprocess_rejects_misaligned_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_json(root / "pred.json", {"a": {"answer": "đáp án"}})
            write_json(root / "audit.json", {"b": {}})
            with self.assertRaisesRegex(ValueError, "IDs differ"):
                postprocess_candidate(root / "pred.json", root / "audit.json", root / "out.json")


if __name__ == "__main__":
    unittest.main()
