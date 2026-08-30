from __future__ import annotations

from typing import Any


def cuda_supports_bfloat16(torch_module: Any, device: Any = 0) -> bool:
    """Return whether the CUDA device has native BF16 support.

    ``torch.cuda.is_bf16_supported()`` can report true for software/emulated
    support on older GPUs.  Use CUDA compute capability as the conservative
    gate so a memory-constrained T4 is not assigned BF16 by auto-detection.
    """
    try:
        device_index = getattr(device, "index", None)
        if device_index is None:
            device_index = 0
        major, _minor = torch_module.cuda.get_device_capability(device_index)
        return int(major) >= 8
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False


def recommended_cuda_dtype(torch_module: Any, device: Any = 0) -> Any:
    """Choose a safe reduced-precision dtype for the given CUDA device."""
    return (
        torch_module.bfloat16
        if cuda_supports_bfloat16(torch_module, device=device)
        else torch_module.float16
    )
