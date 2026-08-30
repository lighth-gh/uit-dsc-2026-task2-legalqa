from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MODEL_IDENTITY_FILENAME = ".legalqa_model.json"


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


def resolve_model_identity(model_name_or_path: str) -> str:
    """Return a stable repo ID for either a Hub ID or portable local snapshot."""
    model_path = Path(model_name_or_path)
    marker_path = model_path / MODEL_IDENTITY_FILENAME
    if model_path.is_dir() and marker_path.is_file():
        try:
            payload = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            payload = None
        if isinstance(payload, dict):
            repo_id = payload.get("repo_id")
            if isinstance(repo_id, str) and repo_id.strip():
                return repo_id.strip()
    return str(model_name_or_path)
