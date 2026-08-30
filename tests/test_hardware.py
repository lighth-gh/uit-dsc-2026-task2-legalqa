from __future__ import annotations

import unittest

from legalqa_baseline.hardware import cuda_supports_bfloat16, recommended_cuda_dtype


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


if __name__ == "__main__":
    unittest.main()
