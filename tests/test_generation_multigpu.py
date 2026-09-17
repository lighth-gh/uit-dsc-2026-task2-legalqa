import multiprocessing
import os
import tempfile
import threading
import types
import unittest
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

from legalqa.generation import _dispatch_answers, generate, generation_devices
from legalqa.io import Journal, config, read_json, write_json


def process_answer(key):
    # A real spawned-process smoke test without CUDA/model dependencies.
    return {"key": key, "pid": os.getpid()}


class MultiGpuGenerationTests(unittest.TestCase):
    def test_visible_gpu_selection_and_single_device_fallback(self):
        cuda = Mock()
        for count, expected in [(1, ["cuda:0"]), (2, ["cuda:0", "cuda:1"]),
                                (4, ["cuda:0", "cuda:1"])]:
            cuda.device_count.return_value = count
            self.assertEqual(generation_devices("cuda:0", True, "generate", cuda), expected)
        self.assertEqual(generation_devices("cuda:1", True, "generate", cuda), ["cuda:1", "cuda:0"])
        self.assertEqual(generation_devices("cuda:1", False, "generate", None), ["cuda:1"])
        self.assertEqual(generation_devices("cpu", True, "generate", None), ["cpu"])
        self.assertEqual(generation_devices("cuda:0", True, "extractive", None), ["cuda:0"])
        cuda.device_count.return_value = 1
        with self.assertRaisesRegex(ValueError, "visible"):
            generation_devices("cuda:1", True, "generate", cuda)
        cuda.device_count.return_value = 0
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            generation_devices("cuda:0", True, "generate", cuda)

    def test_real_spawn_two_processes_global_odd_cap(self):
        with patch.dict(os.environ, {"LEGALQA_MAX_ITEMS": "3", "LEGALQA_DEADLINE": "0"}), ExitStack() as stack:
            pools = [stack.enter_context(ProcessPoolExecutor(
                max_workers=1, mp_context=multiprocessing.get_context("spawn"))) for _ in range(2)]
            values = dict(_dispatch_answers([str(i) for i in range(7)], pools, process_answer, lambda _: ()))
        self.assertEqual(set(values), {"0", "1", "2"})
        self.assertEqual(len({v["pid"] for v in values.values()}), 2)
        self.assertNotIn(os.getpid(), {v["pid"] for v in values.values()})

    def test_deadline_drains_inflight_without_dispatching_more(self):
        barrier = threading.Barrier(2)
        paused = threading.Event()

        def answer(key):
            barrier.wait(timeout=5)
            paused.set()
            return key

        with patch("legalqa.generation.should_pause", side_effect=lambda *_: paused.is_set()), ExitStack() as stack:
            pools = [stack.enter_context(ThreadPoolExecutor(max_workers=1)) for _ in range(2)]
            values = dict(_dispatch_answers(["0", "1", "2", "3"], pools, answer, lambda _: ()))
        self.assertEqual(values, {"0": "0", "1": "1"})

    def test_worker_failure_preserves_successful_inflight_checkpoint(self):
        barrier = threading.Barrier(2)

        def answer(key):
            barrier.wait(timeout=5)
            if key == "bad":
                raise RuntimeError("simulated CUDA failure")
            return {"prediction": {"answer": "saved"}}

        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {"LEGALQA_MAX_ITEMS": "2", "LEGALQA_DEADLINE": "0"}))
            pools = [stack.enter_context(ThreadPoolExecutor(max_workers=1)) for _ in range(2)]
            path = Path(folder)/"checkpoint.jsonl"
            journal = Journal(path, {"test": True})
            with self.assertRaisesRegex(RuntimeError, "CUDA failure"):
                for key, value in _dispatch_answers(["bad", "good", "later"], pools, answer, lambda _: ()):
                    journal.append(key, value)
            self.assertEqual(set(Journal(path, {"test": True}).records), {"good"})

    def test_expired_budget_does_not_launch_workers(self):
        pool = Mock()
        with patch("legalqa.generation.should_pause", return_value=True):
            self.assertEqual(list(_dispatch_answers(["0"], [pool], process_answer, lambda _: ())), [])
        pool.submit.assert_not_called()

    def test_two_gpu_generation_pause_resume_order_and_no_duplicates(self):
        # Exercise the real coordinator, journal and publisher with fake GPU workers.
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            root = Path(folder)
            qa, cache, output = root/"qa.json", root/"cache.json", root/"submission.json"
            questions = {str(i): {"question": f"question {i}"} for i in range(7)}
            records = {k: {"contexts": []} for k in questions}
            write_json(qa, questions)
            write_json(cache, {})
            cuda = Mock()
            cuda.device_count.return_value = 2
            stack.enter_context(patch.dict("sys.modules", {
                "torch": types.SimpleNamespace(cuda=cuda),
                "transformers": types.SimpleNamespace(AutoTokenizer=Mock(), set_seed=lambda _: None)}))
            stack.enter_context(patch("legalqa.generation.read_retrieval", return_value=(records, {})))
            stack.enter_context(patch("legalqa.generation.model_lock", return_value={}))
            seen, initializers = [], []

            def pool_factory(**kwargs):
                self.assertEqual(kwargs["mp_context"].get_start_method(), "spawn")
                initializers.append(kwargs["initargs"])
                return ThreadPoolExecutor(max_workers=kwargs["max_workers"])

            def answer(key, question, record):
                seen.append(key)
                self.assertEqual(question, questions[key])
                self.assertEqual(record, records[key])
                return {"prediction": {"answer": f"answer {key}"},
                        "audit": {"route": "generated", "device": "fixture"}}

            stack.enter_context(patch("legalqa.generation.ProcessPoolExecutor", side_effect=pool_factory))
            stack.enter_context(patch("legalqa.generation._generate_worker", side_effect=answer))
            stack.enter_context(patch.dict(os.environ, {"LEGALQA_MAX_ITEMS": "3", "LEGALQA_DEADLINE": "0"}))
            first = generate(config(), qa, cache, "models", output, "cuda:0", multi_gpu=True)
            self.assertEqual(first["status"], "paused")
            self.assertEqual(len(read_json(output.with_suffix(".partial.json"))), 3)
            self.assertFalse(output.exists())
            self.assertEqual([args[2] for args in initializers], ["cuda:0", "cuda:1"])
            with patch.dict(os.environ, {"LEGALQA_MAX_ITEMS": "0"}):
                final = generate(config(), qa, cache, "models", output, "cuda:0", multi_gpu=True)
            self.assertEqual(final["samples"], 7)
            self.assertEqual(list(read_json(output)), list(questions))
            self.assertEqual(sorted(seen), sorted(questions))
            self.assertEqual(set(read_json(output.with_suffix(".audit.json"))), set(questions))
            self.assertTrue(output.with_suffix(".manifest.json").exists())
            generate(config(), qa, cache, "models", output, "cuda:0", multi_gpu=True)
            self.assertEqual(len(initializers), 4)  # Completed runs load no replicas.


if __name__ == "__main__":
    unittest.main()
