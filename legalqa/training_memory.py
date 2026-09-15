"""Training-only memory changes; preserve tokens, targets and DDP loss scaling."""
from types import MethodType


def enable_fused_loss(model):
    # Patch only this instance, before PEFT/DDP wrap it. Generation in other
    # stages continues to use the original Transformers forward method.
    if model.config.model_type != "qwen2":
        raise ValueError("Fused training loss is validated only for Qwen2")
    if model.lm_head.bias is not None or model.lm_head.weight.requires_grad:
        raise ValueError("QLoRA fused loss requires the audited frozen, bias-free lm_head")
    try:
        from liger_kernel.transformers.model.qwen2 import lce_forward
    except ImportError as error:
        raise RuntimeError("Install requirements.txt: QLoRA requires liger-kernel==0.5.10") from error
    model.forward = MethodType(lce_forward, model)
    print("QLoRA loss=liger_fused_linear_cross_entropy; full targets; no CPU offload", flush=True)


def compact_samples(samples):
    """Replace Python integer lists one row at a time, without a second dataset."""
    import numpy as np
    for row in samples:
        for key in ("input_ids", "labels", "attention_mask"):
            row[key] = np.asarray(row[key], dtype=np.int32 if key != "attention_mask" else np.uint8)
    return samples


def padded_batch(batch, pad_token_id):
    import numpy as np
    length = max(len(row["input_ids"]) for row in batch)
    values = {}
    for key, pad in (("input_ids", pad_token_id), ("attention_mask", 0), ("labels", -100)):
        value = np.full((len(batch), length), pad, dtype=np.int64)
        for i, row in enumerate(batch):
            value[i, :len(row[key])] = row[key]
        values[key] = value
    return values
