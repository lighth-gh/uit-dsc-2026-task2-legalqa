from __future__ import annotations

import sys
from typing import Any


class VietnameseReranker:
    """Mô hình Cross-Encoder Tái xếp hạng (AITeamVN/Vietnamese_Reranker)."""

    def __init__(
        self,
        model_name_or_path: str = "AITeamVN/Vietnamese_Reranker",
        device: str = "auto",
        batch_size: int = 32,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.device_setting = device
        self.batch_size = batch_size
        self._tokenizer: Any = None
        self._model: Any = None
        self._device: Any = None

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
            trust_remote_code=True,
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
        ).to(self._device)
        self._model.eval()

    def rerank(
        self,
        question: str,
        candidates: list[dict[str, Any]],
        top_k: int = 3,
        max_length: int = 512,
    ) -> list[dict[str, Any]]:
        """Tái xếp hạng danh sách candidate chunks và chọn Top-K chunks liên quan nhất."""
        if not candidates:
            return []

        self._load_model()
        import torch

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
                    batch_scores = logits.squeeze(1).cpu().tolist()
                elif logits.dim() == 1:
                    batch_scores = logits.cpu().tolist()
                else:
                    batch_scores = logits[:, 0].cpu().tolist()

            if isinstance(batch_scores, float):
                all_scores.append(batch_scores)
            else:
                all_scores.extend(batch_scores)

        scored_candidates = []
        for cand, score in zip(candidates, all_scores):
            item = dict(cand)
            item["rerank_score"] = float(score)
            scored_candidates.append(item)

        # Sắp xếp giảm dần theo điểm reranker
        scored_candidates.sort(key=lambda item: item["rerank_score"], reverse=True)
        return scored_candidates[:top_k]
