import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.auto_stage1 import run


class FakeKaggle:
    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.pushed = []
        self.current = {}

    def kernels_push(self, folder):
        metadata = json.loads((Path(folder) / "kernel-metadata.json").read_text(encoding="utf-8"))
        self.pushed.append(metadata)
        self.current[metadata["id"]] = len(self.pushed) - 1
        return SimpleNamespace(ref=metadata["id"], version_number=1)

    def kernels_status(self, kernel):
        return SimpleNamespace(status="COMPLETE", failure_message="")

    def kernels_output(self, kernel, path, **kwargs):
        (Path(path) / "legalqa_main_stage1_v8").mkdir(parents=True)
        marker = Path(path) / "legalqa_main_stage1_v8" / "stage1_manifest.json"
        marker.write_text(json.dumps({"schema": 2, "stage": 1,
                                      "status": self.outcomes[self.current[kernel]]}), encoding="utf-8")
        return [str(marker)], ""


class AutoStage1Tests(unittest.TestCase):
    def test_paused_output_is_attached_automatically_and_complete_stops(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            notebook = root / "stage1.ipynb"
            notebook.write_text("{}", encoding="utf-8")
            state_path = root / "runs" / "state.json"
            api = FakeKaggle(["paused", "complete"])
            state = run(api, owner="example", slug_prefix="legalqa-test", notebook=notebook,
                        state_path=state_path, poll_seconds=1, sleep=lambda _: None)
            self.assertTrue(state["done"])
            self.assertEqual(len(api.pushed), 2)
            first, second = api.pushed
            self.assertEqual(first["kernel_sources"], [])
            self.assertEqual(second["kernel_sources"], [first["id"]])
            self.assertNotEqual(first["id"], second["id"])
            self.assertTrue(first["is_private"])
            run(api, owner="example", slug_prefix="legalqa-test", notebook=notebook,
                state_path=state_path, poll_seconds=1, sleep=lambda _: None)
            self.assertEqual(len(api.pushed), 2)

    def test_failed_kaggle_run_does_not_launch_next_version(self):
        class Errored(FakeKaggle):
            def kernels_status(self, kernel):
                return SimpleNamespace(status="ERROR", failure_message="GPU unavailable")

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            notebook = root / "stage1.ipynb"
            notebook.write_text("{}", encoding="utf-8")
            api = Errored([])
            state_path = root / "state.json"
            with self.assertRaisesRegex(RuntimeError, "GPU unavailable"):
                run(api, owner="example", slug_prefix="legalqa-test", notebook=notebook,
                    state_path=state_path, sleep=lambda _: None)
            self.assertEqual(len(api.pushed), 1)
            self.assertEqual(json.loads(state_path.read_text(encoding="utf-8"))["active"]["kernel"],
                             api.pushed[0]["id"])

    def test_controller_restart_polls_existing_session_without_repush(self):
        class RunningOnce(FakeKaggle):
            first = True

            def kernels_status(self, kernel):
                if self.first:
                    self.first = False
                    return SimpleNamespace(status="RUNNING", failure_message="")
                return super().kernels_status(kernel)

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            notebook = root / "stage1.ipynb"
            notebook.write_text("{}", encoding="utf-8")
            state_path = root / "state.json"
            api = RunningOnce(["complete"])
            with self.assertRaisesRegex(RuntimeError, "controller stopped"):
                run(api, owner="example", slug_prefix="legalqa-test", notebook=notebook,
                    state_path=state_path, poll_seconds=1,
                    sleep=lambda _: (_ for _ in ()).throw(RuntimeError("controller stopped")))
            self.assertEqual(len(api.pushed), 1)
            state = run(api, owner="example", slug_prefix="legalqa-test", notebook=notebook,
                        state_path=state_path, poll_seconds=1, sleep=lambda _: None)
            self.assertTrue(state["done"])
            self.assertEqual(len(api.pushed), 1)
