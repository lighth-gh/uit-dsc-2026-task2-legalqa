from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from legalqa_baseline.hardware import (
    MODEL_IDENTITY_FILENAME,
    cuda_supports_bfloat16,
    recommended_cuda_dtype,
    resolve_model_identity,
)


class _FakeCuda:
    def __init__(self, capability: tuple[int, int]) -> None:
        self.capability = capability

    def get_device_capability(self, _device: int) -> tuple[int, int]:
        return self.capability


class _FakeTorch:
    bfloat16 = "bfloat16"
    float16 = "float16"

    def __init__(self, capability: tuple[int, int]) -> None:
        self.cuda = _FakeCuda(capability)


class HardwareTests(unittest.TestCase):
    def test_t4_uses_float16(self) -> None:
        torch_module = _FakeTorch((7, 5))
        self.assertFalse(cuda_supports_bfloat16(torch_module))
        self.assertEqual(recommended_cuda_dtype(torch_module), "float16")

    def test_ampere_can_use_bfloat16(self) -> None:
        torch_module = _FakeTorch((8, 0))
        self.assertTrue(cuda_supports_bfloat16(torch_module))
        self.assertEqual(recommended_cuda_dtype(torch_module), "bfloat16")

    def test_local_snapshot_resolves_to_stable_hub_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory) / "saved-model"
            model_dir.mkdir()
            (model_dir / MODEL_IDENTITY_FILENAME).write_text(
                json.dumps({"repo_id": "owner/model-name"}),
                encoding="utf-8",
            )
            self.assertEqual(resolve_model_identity(str(model_dir)), "owner/model-name")
            self.assertEqual(resolve_model_identity("owner/model-name"), "owner/model-name")


if __name__ == "__main__":
    unittest.main()
