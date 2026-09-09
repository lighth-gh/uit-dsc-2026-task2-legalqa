import gc
import json
from pathlib import Path

from .io import ROOT, digest, file_hash, read_json, write_json

EXPECTED = {"generator": 3_085_938_688, "embedding": 117_653_760, "reranker": 567_755_777}


def require_approved(c):
    allowed = set(read_json(ROOT / "assets" / "approved_models.json")["model_ids"])
    for role, model_id in c["models"].items():
        if model_id not in allowed:
            raise ValueError(f"Model absent from the supplied approval spreadsheet: {role}={model_id}")
    # This implementation's pooling, heads and prompt are specific to these three architectures.
    supported = {"embedding": "intfloat/multilingual-e5-small", "reranker": "AITeamVN/Vietnamese_Reranker",
                 "generator": "AITeamVN/Vi-Qwen2-3B-RAG"}
    if c["models"] != supported:
        raise ValueError("Changing model families requires implementing and validating their inference contracts")


def fetch_models(c, root):
    """Only downloads public model files, never uses remote model inference."""
    from huggingface_hub import HfApi, snapshot_download
    require_approved(c)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "models.lock.json"
    lock = read_json(lock_path) if lock_path.exists() else {"models": {}}
    for role, model_id in c["models"].items():
        previous = lock["models"].get(role)
        if previous and previous["model_id"] != model_id:
            raise ValueError("Model lock disagrees with config; use a new model directory")
        revision = previous["revision"] if previous else HfApi().model_info(model_id).sha
        target = root / role
        snapshot_download(model_id, revision=revision, local_dir=target,
                          allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "LICENSE*", "README.md"],
                          ignore_patterns=["onnx/*", "openvino/*"])
        if not list(target.glob("*.safetensors")):
            raise ValueError(f"No Safetensors checkpoint downloaded for {role}")
        lock["models"][role] = {"model_id": model_id, "revision": revision,
                                 "config_sha256": file_hash(target / "config.json")}
        write_json(lock_path, lock)
    return audit_models(c, root)


def model_lock(c, root):
    require_approved(c)
    root = Path(root)
    lock = read_json(root / "models.lock.json")
    for role, model_id in c["models"].items():
        entry = lock["models"].get(role, {})
        if entry.get("model_id") != model_id or entry.get("config_sha256") != file_hash(root/role/"config.json"):
            raise ValueError(f"Model identity/config mismatch: {role}")
    return lock


def audit_models(c, root):
    import torch
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification
    from peft import LoraConfig, TaskType, get_peft_model
    lock = model_lock(c, root)
    classes = {"embedding": AutoModel, "reranker": AutoModelForSequenceClassification, "generator": AutoModelForCausalLM}
    rows = {}
    adapter_count = 0
    for role, cls in classes.items():
        cfg = AutoConfig.from_pretrained(Path(root)/role, local_files_only=True, trust_remote_code=False)
        # Construct full unquantized architecture on the meta device. Includes embeddings and heads.
        with torch.device("meta"):
            model = cls.from_config(cfg, trust_remote_code=False)
            model.tie_weights()
            count = sum(p.numel() for p in model.parameters())
            if count != EXPECTED[role]:
                raise ValueError(f"Architecture changed for {role}: got {count}, expected {EXPECTED[role]}")
            rows[role] = {**lock["models"][role], "base_parameters": count}
            if role == "generator":
                t = c["training"]
                lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=t["lora_rank"], lora_alpha=t["lora_alpha"],
                                  lora_dropout=t["lora_dropout"], target_modules=t["target_modules"], bias="none")
                adapted = get_peft_model(model, lora)
                adapter_count = sum(p.numel() for p in adapted.parameters()) - count
        del model
    total = sum(row["base_parameters"] for row in rows.values())
    report = {"models": rows, "base_total": total, "lora_additional": adapter_count,
              "total_with_unmerged_lora": total+adapter_count, "limit_exclusive": c["parameter_limit"],
              "passes": total+adapter_count < c["parameter_limit"], "config_hash": digest(c)}
    if not report["passes"]:
        raise ValueError(f"Total parameter budget violated: {total+adapter_count}")
    write_json(Path(root)/"parameter_audit.json", report)
    return report


def dtype_for(device):
    import torch
    if str(device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable. Enable a Kaggle GPU or explicitly use --device cpu.")
        return torch.float16
    return torch.float32


class Encoder:
    def __init__(self, root, device="cuda:0"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.device = torch, device
        self.tokenizer = AutoTokenizer.from_pretrained(Path(root)/"embedding", local_files_only=True, use_fast=True)
        self.model = AutoModel.from_pretrained(Path(root)/"embedding", local_files_only=True,
                                              torch_dtype=dtype_for(device)).to(device).eval()

    def encode(self, texts, kind="query", batch_size=32):
        import numpy as np
        torch = self.torch
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        result = []
        for start in range(0, len(texts), batch_size):
            values = [f"{kind}: {text}" for text in texts[start:start+batch_size]]
            if kind == "passage":
                lengths = [len(x) for x in self.tokenizer(values, add_special_tokens=True, truncation=False)["input_ids"]]
                if max(lengths) > 512:
                    raise ValueError("An E5 passage exceeds 512 tokens; rebuild with a smaller child/header window")
            inputs = self.tokenizer(values, padding=True, truncation=True, max_length=512,
                                    return_tensors="pt").to(self.device)
            with torch.inference_mode():
                out = self.model(**inputs).last_hidden_state.float()
                mask = inputs["attention_mask"].unsqueeze(-1)
                vectors = (out*mask).sum(1) / mask.sum(1).clamp(min=1)
                vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
            values = vectors.cpu().numpy().astype(np.float32)
            if not np.isfinite(values).all():
                raise ValueError("Embedding contains NaN/Inf")
            result.append(values)
        return np.concatenate(result)


class Reranker:
    def __init__(self, root, device="cuda:0", batch_size=8, max_tokens=768):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch, self.device, self.batch_size, self.max_tokens = torch, device, batch_size, max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(Path(root)/"reranker", local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(Path(root)/"reranker", local_files_only=True,
                       torch_dtype=dtype_for(device), attn_implementation="sdpa").to(device).eval()

    def score(self, question, passages):
        result = []
        # Cross-encoder pair tokenization, not encode(question) dot encode(passage).
        for start in range(0, len(passages), self.batch_size):
            pairs = [[question, p] for p in passages[start:start+self.batch_size]]
            inputs = self.tokenizer(pairs, padding=True, truncation=True, max_length=self.max_tokens,
                                    return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                scores = self.model(**inputs).logits.float().view(-1)
            if not self.torch.isfinite(scores).all():
                raise ValueError("Reranker contains NaN/Inf")
            result.extend(scores.cpu().tolist())
        return result


def release(*objects):
    import torch
    for obj in objects:
        if hasattr(obj, "model"):
            del obj.model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_generator(c, root, device, adapter=None, training=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    path = Path(root)/"generator"
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" if training else "left"
    dtype = dtype_for(device)
    kwargs = dict(local_files_only=True, trust_remote_code=False, torch_dtype=dtype,
                  attn_implementation="sdpa", device_map={"": device})
    if c["generation"]["load_in_4bit"]:
        if not str(device).startswith("cuda"):
            raise ValueError("Set generation.load_in_4bit=false for CPU inference")
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    if adapter:
        ac = read_json(Path(adapter)/"adapter_config.json")
        t = c["training"]
        if ac.get("r") != t["lora_rank"] or set(ac.get("target_modules", [])) != set(t["target_modules"]):
            raise ValueError("Adapter architecture is outside the audited parameter budget")
        if ac.get("modules_to_save") or ac.get("bias", "none") != "none" or ac.get("rank_pattern") or ac.get("use_dora"):
            raise ValueError("Additional adapter parameters require a new explicit audit")
        model = PeftModel.from_pretrained(model, adapter, local_files_only=True)
    model.config.use_cache = not training
    if not training:
        model.eval()
    return model, tokenizer
