from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from .storage import iter_contexts
from .text import chunk_passage


class VietnameseEmbeddingModel:
    """Mô hình tạo vector embedding tiếng Việt (AITeamVN/Vietnamese_Embedding_v2)."""

    def __init__(
        self,
        model_name_or_path: str = "AITeamVN/Vietnamese_Embedding_v2",
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
            from transformers import AutoModel, AutoTokenizer
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
            f"[DenseEmbedding] Tải mô hình {self.model_name_or_path} lên {self._device}...",
            file=sys.stderr,
            flush=True,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
        )
        self._model = AutoModel.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
        ).to(self._device)
        self._model.eval()

    def encode(
        self,
        texts: list[str],
        batch_size: int | None = None,
        max_length: int = 512,
    ) -> Any:
        """Sinh ma trận embedding (N, D) đã chuẩn hóa L2 (dùng cho Cosine Similarity)."""
        self._load_model()
        import numpy as np
        import torch
        import torch.nn.functional as F

        bs = batch_size or self.batch_size
        all_embeddings: list[np.ndarray] = []

        for i in range(0, len(texts), bs):
            batch_texts = [str(t or "") for t in texts[i : i + bs]]
            encoded = self._tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self._device)

            with torch.inference_mode():
                outputs = self._model(**encoded)
                # Mean pooling có xét attention mask
                token_embeddings = outputs[0]
                input_mask_expanded = (
                    encoded["attention_mask"]
                    .unsqueeze(-1)
                    .expand(token_embeddings.size())
                    .float()
                )
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                mean_pooled = sum_embeddings / sum_mask
                # L2 normalize để Inner Product tương đương Cosine Similarity
                normalized = F.normalize(mean_pooled, p=2, dim=1)
                all_embeddings.append(normalized.cpu().numpy())

        if not all_embeddings:
            return np.empty((0, 768), dtype=np.float32)
        return np.vstack(all_embeddings).astype(np.float32)


class DenseVectorIndex:
    """Quản lý Index tìm kiếm vector Dense (FAISS IndexFlatIP hoặc NumPy Cosine Search)."""

    def __init__(
        self,
        vectors: Any | None = None,
        metadata: list[dict[str, Any]] | None = None,
        faiss_index: Any | None = None,
    ) -> None:
        self.vectors = vectors
        self.metadata = metadata or []
        self.faiss_index = faiss_index

    @classmethod
    def load(cls, path: str | Path) -> "DenseVectorIndex":
        """Nạp Dense index từ đĩa (.faiss hoặc .npy + .meta.json)."""
        import numpy as np

        base_path = Path(path)
        meta_path = (
            base_path.with_suffix(".meta.json")
            if base_path.suffix in (".faiss", ".npy")
            else base_path.parent / f"{base_path.name}.meta.json"
        )
        if not meta_path.is_file():
            # Thử tìm file metadata cùng tên
            meta_path = base_path.with_name(f"{base_path.stem}.meta.json")

        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        faiss_file = base_path if base_path.suffix == ".faiss" else base_path.with_suffix(".faiss")
        npy_file = base_path if base_path.suffix == ".npy" else base_path.with_suffix(".npy")

        faiss_index = None
        try:
            import faiss
            if faiss_file.is_file():
                faiss_index = faiss.read_index(str(faiss_file))
        except ImportError:
            pass

        vectors = None
        if npy_file.is_file():
            vectors = np.load(str(npy_file))
        elif faiss_index is not None:
            # Nếu chỉ có faiss index
            pass

        return cls(vectors=vectors, metadata=meta, faiss_index=faiss_index)

    def save(self, path: str | Path) -> None:
        """Lưu Dense index xuống đĩa."""
        import numpy as np

        base_path = Path(path)
        base_path.parent.mkdir(parents=True, exist_ok=True)

        # Lưu metadata
        meta_path = base_path.with_suffix(".meta.json")
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False)

        # Lưu numpy vectors
        if self.vectors is not None:
            npy_path = base_path.with_suffix(".npy")
            np.save(str(npy_path), self.vectors)

        # Lưu faiss index nếu có
        if self.faiss_index is not None:
            try:
                import faiss
                faiss_path = base_path.with_suffix(".faiss")
                faiss.write_index(self.faiss_index, str(faiss_path))
            except ImportError:
                pass

    def search(self, query_vector: Any, top_k: int = 50) -> list[dict[str, Any]]:
        """Tìm Top-K chunks tương đồng nhất với vector câu hỏi."""
        import numpy as np

        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)

        # Chuẩn hóa query vector
        norm = np.linalg.norm(query_vector, axis=1, keepdims=True)
        query_vector = query_vector / np.maximum(norm, 1e-9)

        if self.faiss_index is not None:
            scores, indices = self.faiss_index.search(query_vector.astype(np.float32), int(top_k))
            matched_indices = indices[0]
            matched_scores = scores[0]
        elif self.vectors is not None:
            # Cosine similarity bằng ma trận tích vô hướng
            scores = np.dot(self.vectors, query_vector[0])
            top_k_indices = np.argpartition(-scores, min(top_k, len(scores) - 1))[:top_k]
            top_k_indices = top_k_indices[np.argsort(-scores[top_k_indices])]
            matched_indices = top_k_indices
            matched_scores = scores[top_k_indices]
        else:
            return []

        results: list[dict[str, Any]] = []
        for rank, (idx, score) in enumerate(zip(matched_indices, matched_scores), start=1):
            if 0 <= idx < len(self.metadata):
                item = dict(self.metadata[idx])
                item["dense_score"] = float(score)
                item["dense_rank"] = rank
                results.append(item)
        return results


def build_dense_index(
    contexts_path: str | Path,
    output_index_path: str | Path,
    embedding_model_name: str = "AITeamVN/Vietnamese_Embedding_v2",
    device: str = "auto",
    batch_size: int = 64,
    max_chunk_words: int = 620,
    overlap_words: int = 100,
    force: bool = False,
) -> dict[str, Any]:
    """Xây dựng Dense Vector Index từ kho văn bản selected-contexts."""
    import numpy as np

    output_path = Path(output_index_path)
    meta_path = output_path.with_suffix(".meta.json")
    if meta_path.exists() and not force:
        raise FileExistsError(f"Index đã tồn tại tại {output_path}; thêm --force để xây lại.")

    started = time.time()
    print(f"[BuildDense] Khởi tạo encoder: {embedding_model_name}...", file=sys.stderr, flush=True)
    encoder = VietnameseEmbeddingModel(
        model_name_or_path=embedding_model_name,
        device=device,
        batch_size=batch_size,
    )

    print(f"[BuildDense] Đọc và chunking văn bản từ {contexts_path}...", file=sys.stderr, flush=True)
    metadata: list[dict[str, Any]] = []
    chunk_texts: list[str] = []

    doc_count = 0
    for context in iter_contexts(contexts_path):
        doc_count += 1
        passage = str(context.get("passage") or "")
        chunks = chunk_passage(passage, max_words=max_chunk_words, overlap_words=overlap_words)
        if not chunks:
            continue
        context_id = str(context.get("id", ""))
        name = str(context.get("name") or "")
        link = str(context.get("link") or "")
        for chunk_no, text in enumerate(chunks):
            # Tạo chuỗi đại diện ngữ nghĩa cho embedding (kết hợp tiêu đề văn bản + đoạn luật)
            dense_input = f"{name}: {text}" if name else text
            chunk_texts.append(dense_input)
            metadata.append({
                "context_id": context_id,
                "chunk_no": chunk_no,
                "name": name,
                "link": link,
                "text": text,
            })

    total_chunks = len(chunk_texts)
    print(
        f"[BuildDense] Đã tạo {total_chunks:,} chunks từ {doc_count:,} văn bản. Đang mã hóa vector...",
        file=sys.stderr,
        flush=True,
    )

    # Encode theo batch
    vectors = encoder.encode(chunk_texts, batch_size=batch_size)

    # Xây dựng FAISS Index nếu có
    faiss_index = None
    try:
        import faiss
        dim = vectors.shape[1]
        faiss_index = faiss.IndexFlatIP(dim)
        faiss_index.add(vectors)
        print(f"[BuildDense] Đã xây dựng FAISS IndexFlatIP (dim={dim})", file=sys.stderr, flush=True)
    except ImportError:
        print("[BuildDense] FAISS chưa cài đặt, lưu trữ dưới dạng NumPy matrix.", file=sys.stderr, flush=True)

    dense_index = DenseVectorIndex(vectors=vectors, metadata=metadata, faiss_index=faiss_index)
    dense_index.save(output_path)
    elapsed = time.time() - started

    stats = {
        "documents": doc_count,
        "chunks": total_chunks,
        "embedding_dim": int(vectors.shape[1]),
        "elapsed_seconds": round(elapsed, 2),
        "output_path": str(output_path.resolve()),
    }
    print(f"[BuildDense] Hoàn thành sau {elapsed:.2f}s: {stats}", file=sys.stderr, flush=True)
    return stats
