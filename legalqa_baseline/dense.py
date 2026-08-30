from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from .hardware import recommended_cuda_dtype
from .storage import iter_contexts
from .text import chunk_passage


DENSE_SCHEMA_VERSION = 3
VALID_SIMILARITIES = {"cosine", "dot_product"}


def _tensor_to_float32_numpy(tensor: Any) -> Any:
    """Convert CUDA FP16/BF16 tensors to a NumPy-compatible FP32 array."""
    return tensor.float().cpu().numpy()


class VietnameseEmbeddingModel:
    """Mô hình tạo vector embedding tiếng Việt (AITeamVN/Vietnamese_Embedding_v2)."""

    def __init__(
        self,
        model_name_or_path: str = "AITeamVN/Vietnamese_Embedding_v2",
        device: str = "auto",
        batch_size: int = 8,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.device_setting = device
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        self.batch_size = batch_size
        self._tokenizer: Any = None
        self._model: Any = None
        self._device: Any = None
        self._device_ids: list[int] = []
        self._hidden_size = 0
        self._multi_gpu = False

    def _load_model(self) -> None:
        if self._model is not None:
            return

        # Must be set before importing Torch in a fresh CLI process.
        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
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
            trust_remote_code=False,
        )
        model_dtype = None
        if self._device.type == "cuda":
            model_dtype = recommended_cuda_dtype(torch, device=self._device)
        base_model = AutoModel.from_pretrained(
            self.model_name_or_path,
            torch_dtype=model_dtype,
            trust_remote_code=False,
        ).to(self._device)
        base_model.eval()
        self._hidden_size = int(getattr(base_model.config, "hidden_size", 0))

        # Return only the CLS tensor so DataParallel does not need to gather a
        # Transformers ModelOutput object (which varies between versions).
        class _CLSEncoder(torch.nn.Module):
            def __init__(self, encoder: Any) -> None:
                super().__init__()
                self.encoder = encoder

            def forward(self, **kwargs: Any) -> Any:
                outputs = self.encoder(**kwargs)
                return outputs[0][:, 0]

        cls_encoder = _CLSEncoder(base_model)
        if self._device.type == "cuda":
            selected_index = getattr(self._device, "index", None)
            if selected_index is None:
                self._device_ids = list(range(torch.cuda.device_count()))
            else:
                self._device_ids = [int(selected_index)]

        if len(self._device_ids) > 1:
            self._model = torch.nn.DataParallel(
                cls_encoder,
                device_ids=self._device_ids,
                output_device=self._device_ids[0],
            )
            self._multi_gpu = True
            gpu_names = [
                f"{gpu_id}:{torch.cuda.get_device_name(gpu_id)}"
                for gpu_id in self._device_ids
            ]
            print(
                "[DenseEmbedding] Multi-GPU DataParallel active on "
                f"{', '.join(gpu_names)} | total batch={self.batch_size} "
                f"(~{max(1, self.batch_size // len(self._device_ids))}/GPU)",
                file=sys.stderr,
                flush=True,
            )
        else:
            self._model = cls_encoder
            if self._device.type == "cuda":
                gpu_id = self._device_ids[0] if self._device_ids else 0
                print(
                    f"[DenseEmbedding] Using one GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}",
                    file=sys.stderr,
                    flush=True,
                )
        self._model.eval()

    def encode(
        self,
        texts: list[str],
        batch_size: int | None = None,
        max_length: int = 2048,
        normalize_embeddings: bool = False,
        show_progress: bool | None = None,
    ) -> Any:
        """Sinh embedding (N, D); chỉ L2-normalize khi được yêu cầu."""
        self._load_model()
        import time
        import numpy as np
        import torch
        import torch.nn.functional as F

        bs = self.batch_size if batch_size is None else batch_size
        if bs <= 0 or max_length <= 0:
            raise ValueError("batch_size and max_length must be greater than 0")
        all_embeddings: list[np.ndarray] = []
        total_texts = len(texts)
        if show_progress is None:
            show_progress = total_texts >= 50

        started = time.time()
        last_log = started
        reported_parallel_forward = False

        i = 0
        effective_bs = bs
        while i < total_texts:
            current_bs = min(effective_bs, total_texts - i)
            batch_texts = [str(t or "") for t in texts[i : i + current_bs]]
            try:
                encoded = self._tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(self._device)

                with torch.inference_mode():
                    # _CLSEncoder already performs the model-specific CLS
                    # pooling and returns a plain tensor for DataParallel.
                    embeddings = self._model(**encoded)
                    if normalize_embeddings:
                        embeddings = F.normalize(embeddings, p=2, dim=1)
                    # NumPy does not support torch.bfloat16. Convert explicitly
                    # so reduced-precision CUDA inference remains portable.
                    all_embeddings.append(_tensor_to_float32_numpy(embeddings))
                    if self._multi_gpu and not reported_parallel_forward:
                        allocations = ", ".join(
                            f"GPU {gpu_id}={torch.cuda.memory_allocated(gpu_id) / 1024**3:.2f} GiB"
                            for gpu_id in self._device_ids
                        )
                        print(
                            f"[DenseEmbedding] First multi-GPU forward OK ({allocations}).",
                            file=sys.stderr,
                            flush=True,
                        )
                        reported_parallel_forward = True
            except RuntimeError as exc:
                is_cuda_oom = (
                    self._device.type == "cuda"
                    and "out of memory" in str(exc).lower()
                )
                if not is_cuda_oom or current_bs <= 1:
                    raise
                next_bs = max(1, current_bs // 2)
                for gpu_id in self._device_ids or [0]:
                    with torch.cuda.device(gpu_id):
                        torch.cuda.empty_cache()
                print(
                    f"[DenseEmbedding] CUDA OOM with batch={current_bs}; retrying batch={next_bs}.",
                    file=sys.stderr,
                    flush=True,
                )
                effective_bs = next_bs
                continue

            i += current_bs
            done = i
            now = time.time()
            if show_progress and (now - last_log >= 10.0 or done == total_texts or done <= bs * 2):
                elapsed = now - started
                speed = done / max(elapsed, 0.001)
                eta = (total_texts - done) / max(speed, 0.001)
                pct = (done / total_texts) * 100
                print(
                    f"[DenseEmbedding] Đang tạo vector: {done:,}/{total_texts:,} ({pct:.1f}%) | "
                    f"Tốc độ: {speed:.1f} chunks/s | Đã chạy: {elapsed:.0f}s | ETA: {eta:.0f}s",
                    file=sys.stderr,
                    flush=True,
                )
                last_log = now

        if not all_embeddings:
            return np.empty((0, self._hidden_size), dtype=np.float32)
        return np.vstack(all_embeddings).astype(np.float32)


class DenseVectorIndex:
    """Quản lý Dense index FAISS/NumPy cho cosine hoặc dot product."""

    def __init__(
        self,
        vectors: Any | None = None,
        metadata: list[dict[str, Any]] | None = None,
        faiss_index: Any | None = None,
        manifest: dict[str, Any] | None = None,
        similarity: str = "cosine",
    ) -> None:
        self.vectors = vectors
        self.metadata = metadata or []
        self.faiss_index = faiss_index
        self.manifest = manifest or {}
        if similarity not in VALID_SIMILARITIES:
            raise ValueError(f"Similarity không hỗ trợ: {similarity}")
        self.similarity = similarity

    @classmethod
    def load(
        cls,
        path: str | Path,
        expected_model_name: str | None = None,
    ) -> "DenseVectorIndex":
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

        if not meta_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy Dense metadata: {meta_path}")
        with meta_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            # Index schema v1 luôn dùng vector đã L2-normalize.
            meta = payload
            manifest: dict[str, Any] = {
                "schema_version": 1,
                "similarity": "cosine",
                "legacy": True,
            }
        elif isinstance(payload, dict) and isinstance(payload.get("items"), list):
            meta = payload["items"]
            manifest = dict(payload.get("manifest") or {})
        else:
            raise ValueError(f"Dense metadata không hợp lệ: {meta_path}")

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
        if faiss_index is None and npy_file.is_file():
            # Loading as a regular ndarray avoids leaving an open mmap handle on
            # Windows, which otherwise prevents replacing or cleaning the index.
            vectors = np.load(str(npy_file), allow_pickle=False)
        if faiss_index is None and vectors is None:
            raise FileNotFoundError(
                f"Dense metadata tồn tại nhưng thiếu vector index: {faiss_file} / {npy_file}"
            )

        if vectors is not None and vectors.ndim != 2:
            raise ValueError(f"Dense vectors must be a 2D matrix, got shape={vectors.shape}")

        expected_count = len(meta)
        actual_count = int(faiss_index.ntotal) if faiss_index is not None else int(vectors.shape[0])
        if actual_count != expected_count:
            raise ValueError(
                f"Dense index/metadata lệch số lượng: vectors={actual_count}, metadata={expected_count}"
            )
        actual_dim = int(faiss_index.d) if faiss_index is not None else int(vectors.shape[1])
        manifest_dim = manifest.get("dimension")
        if manifest_dim is not None and int(manifest_dim) != actual_dim:
            raise ValueError(
                f"Dense dimension không khớp manifest: index={actual_dim}, manifest={manifest_dim}"
            )
        manifest_model = manifest.get("embedding_model")
        if expected_model_name and not manifest_model:
            raise ValueError(
                "Legacy Dense index has no embedding model/pooling metadata; rebuild the index"
            )
        if expected_model_name and manifest_model and expected_model_name != manifest_model:
            raise ValueError(
                f"Dense index dùng model {manifest_model!r}, không phải {expected_model_name!r}"
            )
        if expected_model_name and int(manifest.get("schema_version") or 0) < DENSE_SCHEMA_VERSION:
            raise ValueError("Dense index uses an old pooling schema; rebuild it with CLS pooling")
        if expected_model_name and manifest.get("pooling") != "cls":
            raise ValueError("Dense index does not use CLS pooling; rebuild the index")
        similarity = str(manifest.get("similarity") or "cosine")
        return cls(
            vectors=vectors,
            metadata=meta,
            faiss_index=faiss_index,
            manifest=manifest,
            similarity=similarity,
        )

    def save(self, path: str | Path) -> None:
        """Lưu Dense index xuống đĩa."""
        import numpy as np

        base_path = Path(path)
        base_path.parent.mkdir(parents=True, exist_ok=True)

        if self.vectors is None and self.faiss_index is None:
            raise ValueError("Không có vector/FAISS index để lưu")

        meta_path = base_path.with_suffix(".meta.json")
        # Ghi vector trước; metadata là marker hoàn tất và được ghi sau cùng.
        if self.vectors is not None:
            npy_path = base_path.with_suffix(".npy")
            npy_tmp = npy_path.with_suffix(npy_path.suffix + ".tmp")
            with npy_tmp.open("wb") as f:
                np.save(f, self.vectors)
            npy_tmp.replace(npy_path)

        # Lưu faiss index nếu có
        if self.faiss_index is not None:
            try:
                import faiss
                faiss_path = base_path.with_suffix(".faiss")
                faiss_tmp = faiss_path.with_suffix(faiss_path.suffix + ".tmp")
                faiss.write_index(self.faiss_index, str(faiss_tmp))
                faiss_tmp.replace(faiss_path)
            except ImportError:
                pass

        meta_tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
        with meta_tmp.open("w", encoding="utf-8") as f:
            json.dump(
                {"manifest": self.manifest, "items": self.metadata},
                f,
                ensure_ascii=False,
            )
        meta_tmp.replace(meta_path)

    def validate_against_bm25(self, bm25_metadata: dict[str, str]) -> None:
        """Từ chối ghép hai index được dựng từ corpus/chunk config khác nhau."""
        mismatches = []
        for key in ("chunks", "max_chunk_words", "overlap_words", "corpus_sha256"):
            dense_value = self.manifest.get(key)
            bm25_value = bm25_metadata.get(key)
            if (
                dense_value is not None
                and bm25_value is not None
                and str(dense_value) != str(bm25_value)
            ):
                mismatches.append(f"{key}: dense={dense_value}, bm25={bm25_value}")
        if mismatches:
            raise ValueError("Dense/BM25 index không tương thích: " + "; ".join(mismatches))

    def search(self, query_vector: Any, top_k: int = 50) -> list[dict[str, Any]]:
        """Tìm Top-K chunks tương đồng nhất với vector câu hỏi."""
        import numpy as np

        if top_k <= 0:
            raise ValueError("top_k phải lớn hơn 0")

        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)

        if query_vector.ndim != 2 or query_vector.shape[0] != 1:
            raise ValueError("query_vector phải có shape (D,) hoặc (1, D)")
        if self.similarity == "cosine":
            norm = np.linalg.norm(query_vector, axis=1, keepdims=True)
            query_vector = query_vector / np.maximum(norm, 1e-9)

        if self.faiss_index is not None:
            if query_vector.shape[1] != int(self.faiss_index.d):
                raise ValueError(
                    f"Query dimension {query_vector.shape[1]} != index dimension {self.faiss_index.d}"
                )
            k = min(int(top_k), int(self.faiss_index.ntotal))
            if k == 0:
                return []
            scores, indices = self.faiss_index.search(query_vector.astype(np.float32), k)
            matched_indices = indices[0]
            matched_scores = scores[0]
        elif self.vectors is not None:
            if len(self.vectors) == 0:
                return []
            if query_vector.shape[1] != int(self.vectors.shape[1]):
                raise ValueError(
                    f"Query dimension {query_vector.shape[1]} != index dimension {self.vectors.shape[1]}"
                )
            # Cả cosine (vector đã chuẩn hóa) và dot product đều dùng tích vô hướng.
            scores = np.dot(self.vectors, query_vector[0])
            k = min(int(top_k), len(scores))
            top_k_indices = np.argpartition(-scores, k - 1)[:k]
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
    batch_size: int = 8,
    max_chunk_words: int = 620,
    overlap_words: int = 100,
    embedding_max_length: int = 2048,
    force: bool = False,
) -> dict[str, Any]:
    """Xây dựng Dense Vector Index từ kho văn bản selected-contexts."""
    import numpy as np

    output_path = Path(output_index_path)
    meta_path = output_path.with_suffix(".meta.json")
    index_paths = [
        meta_path,
        output_path.with_suffix(".npy"),
        output_path.with_suffix(".faiss"),
    ]
    if any(path.exists() for path in index_paths) and not force:
        raise FileExistsError(f"Index đã tồn tại tại {output_path}; thêm --force để xây lại.")
    if batch_size <= 0 or embedding_max_length <= 0:
        raise ValueError("batch_size và embedding_max_length phải lớn hơn 0")

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
    corpus_hasher = hashlib.sha256()

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
            corpus_hasher.update(context_id.encode("utf-8"))
            corpus_hasher.update(b"\0")
            corpus_hasher.update(str(chunk_no).encode("ascii"))
            corpus_hasher.update(b"\0")
            corpus_hasher.update(name.encode("utf-8"))
            corpus_hasher.update(b"\0")
            corpus_hasher.update(link.encode("utf-8"))
            corpus_hasher.update(b"\0")
            corpus_hasher.update(text.encode("utf-8"))
            corpus_hasher.update(b"\0")
            metadata.append({
                "context_id": context_id,
                "chunk_no": chunk_no,
                "name": name,
                "link": link,
                "text": text,
            })

    total_chunks = len(chunk_texts)
    if total_chunks == 0:
        raise ValueError("Corpus không tạo được chunk nào để xây Dense index")
    print(
        f"[BuildDense] Đã tạo {total_chunks:,} chunks từ {doc_count:,} văn bản. Bắt đầu tạo vector ngữ nghĩa...",
        file=sys.stderr,
        flush=True,
    )

    # Encode theo batch
    vectors = encoder.encode(
        chunk_texts,
        batch_size=batch_size,
        max_length=embedding_max_length,
        normalize_embeddings=False,
    )

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

    manifest = {
        "schema_version": DENSE_SCHEMA_VERSION,
        "embedding_model": embedding_model_name,
        "dimension": int(vectors.shape[1]),
        "similarity": "dot_product",
        "pooling": "cls",
        "documents": doc_count,
        "chunks": total_chunks,
        "max_chunk_words": max_chunk_words,
        "overlap_words": overlap_words,
        "embedding_max_length": embedding_max_length,
        "corpus_sha256": corpus_hasher.hexdigest(),
    }
    dense_index = DenseVectorIndex(
        vectors=vectors,
        metadata=metadata,
        faiss_index=faiss_index,
        manifest=manifest,
        similarity="dot_product",
    )
    temp_output = output_path.parent / f".{output_path.stem}.{uuid.uuid4().hex}.building"
    temp_paths = {
        suffix: temp_output.with_suffix(suffix)
        for suffix in (".npy", ".faiss", ".meta.json")
    }
    final_paths = {
        suffix: output_path.with_suffix(suffix)
        for suffix in (".npy", ".faiss", ".meta.json")
    }
    try:
        dense_index.save(temp_output)
        # Vector trước, metadata marker hoàn tất sau cùng.
        for suffix in (".npy", ".faiss"):
            if temp_paths[suffix].is_file():
                temp_paths[suffix].replace(final_paths[suffix])
            elif force and final_paths[suffix].is_file():
                final_paths[suffix].unlink()
        temp_paths[".meta.json"].replace(final_paths[".meta.json"])
    finally:
        for path in temp_paths.values():
            if path.is_file():
                path.unlink()
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
