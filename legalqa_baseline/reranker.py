from __future__ import annotations

import math
import sys
from typing import Any

from .hardware import recommended_cuda_dtype

class VietnameseReranker:
    """Mô hình Cross-Encoder Tái xếp hạng (AITeamVN/Vietnamese_Reranker)."""

    def __init__(
        self,
        model_name_or_path: str = "AITeamVN/Vietnamese_Reranker",
        device: str = "auto",
        batch_size: int = 4,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.device_setting = device
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        self.batch_size = batch_size
        self._tokenizer: Any = None
        self._model: Any = None
        self._device: Any = None

    @staticmethod
    def _usable_length_limit(value: Any) -> int | None:
        """Normalize real model limits while ignoring Transformers' huge sentinel."""
        try:
            limit = int(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if limit <= 0 or limit >= 1_000_000_000:
            return None
        return limit

    def _maximum_sequence_length(self) -> int | None:
        limits: list[int] = []

        tokenizer_limit = self._usable_length_limit(
            getattr(self._tokenizer, "model_max_length", None)
        )
        if tokenizer_limit is not None:
            limits.append(tokenizer_limit)

        config = getattr(self._model, "config", None)
        model_limit = self._usable_length_limit(
            getattr(config, "max_position_embeddings", None)
        )
        if model_limit is not None:
            # RoBERTa/XLM-R reserve position indices through ``pad_token_id + 1``.
            model_type = str(getattr(config, "model_type", "")).lower()
            if model_type in {"roberta", "xlm-roberta"}:
                try:
                    padding_idx = int(getattr(config, "pad_token_id", None))
                except (TypeError, ValueError, OverflowError):
                    padding_idx = None
                if padding_idx is not None and padding_idx >= 0:
                    model_limit -= padding_idx + 1
            if model_limit > 0:
                limits.append(model_limit)

        return min(limits) if limits else None

    def _load_model(self) -> None:
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Chưa cài đặt PyTorch hoặc Transformers. Vui lòng chạy:\n"
                "pip install -r requirements-generator.txt"
            ) from exc

        if self.device_setting == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(self.device_setting)

        print(
            f"[Reranker] Tải mô hình {self.model_name_or_path} lên {self._device}...",
            file=sys.stderr,
            flush=True,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=False,
        )
        model_dtype = None
        if self._device.type == "cuda":
            dtype_device = self._device
            if self._device.index is None:
                dtype_device = torch.device("cuda", torch.cuda.current_device())
            model_dtype = recommended_cuda_dtype(torch, device=dtype_device)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path,
            torch_dtype=model_dtype,
            trust_remote_code=False,
        )
        num_labels = getattr(getattr(model, "config", None), "num_labels", None)
        if num_labels is not None and int(num_labels) != 1:
            raise ValueError(
                "Reranker model must return exactly one relevance score per pair; "
                f"got num_labels={num_labels}"
            )
        self._model = model.to(self._device)
        self._model.eval()

    def rerank(
        self,
        question: str,
        candidates: list[dict[str, Any]],
        top_k: int = 3,
        max_length: int = 2304,
    ) -> list[dict[str, Any]]:
        """Tái xếp hạng danh sách candidate chunks và chọn Top-K chunks liên quan nhất."""
        if top_k <= 0 or max_length <= 0:
            raise ValueError("top_k and max_length must be greater than 0")
        if not candidates:
            return []

        self._load_model()
        import torch

        maximum_length = self._maximum_sequence_length()
        if maximum_length is not None and max_length > maximum_length:
            raise ValueError(
                f"max_length={max_length} exceeds the reranker limit "
                f"of {maximum_length} tokens"
            )

        pairs = []
        for c in candidates:
            name = str(c.get("name") or "").strip()
            text = str(c.get("text") or "").strip()
            passage = f"{name}: {text}" if name else text
            pairs.append([question, passage])

        all_scores: list[float] = []
        for i in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[i : i + self.batch_size]
            encoded = self._tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self._device)

            with torch.inference_mode():
                outputs = self._model(**encoded)
                logits = outputs.logits
                if logits.dim() == 2 and logits.shape[1] == 1:
                    score_tensor = logits.squeeze(1)
                elif logits.dim() == 1:
                    score_tensor = logits
                else:
                    raise ValueError(
                        "Reranker model must return logits shaped [batch] or "
                        f"[batch, 1]; got {tuple(logits.shape)}"
                    )
                batch_scores = score_tensor.float().cpu().tolist()

            if not isinstance(batch_scores, list) or len(batch_scores) != len(batch_pairs):
                score_count = len(batch_scores) if isinstance(batch_scores, list) else 1
                raise ValueError(
                    "Reranker returned an invalid number of scores: "
                    f"expected {len(batch_pairs)}, got {score_count}"
                )

            for raw_score in batch_scores:
                try:
                    score = float(raw_score)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError("Reranker returned a non-numeric score") from exc
                if not math.isfinite(score):
                    raise ValueError(f"Reranker returned a non-finite score: {score}")
                all_scores.append(score)

        if len(all_scores) != len(candidates):
            raise ValueError(
                "Reranker returned an invalid number of scores: "
                f"expected {len(candidates)}, got {len(all_scores)}"
            )

        scored_candidates = []
        for index, cand in enumerate(candidates):
            item = dict(cand)
            item["rerank_score"] = all_scores[index]
            scored_candidates.append(item)

        # Sắp xếp giảm dần theo điểm reranker
        scored_candidates.sort(key=lambda item: item["rerank_score"], reverse=True)
        return scored_candidates[:top_k]
