from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from .hardware import recommended_cuda_dtype, resolve_model_identity
from .storage import iter_contexts
from .text import chunk_passage


DENSE_SCHEMA_VERSION = 4
DENSE_BUILD_CHECKPOINT_SCHEMA_VERSION = 2
VALID_SIMILARITIES = {"cosine", "dot_product"}
VALID_NORMALIZATIONS = {"none", "l2"}


def _tensor_to_float32_numpy(tensor: Any) -> Any:
    """Convert CUDA FP16/BF16 tensors to a NumPy-compatible FP32 array."""
    return tensor.float().cpu().numpy()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def _save_numpy_atomic(path: Path, array: Any, np_module: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        np_module.save(handle, array)
    tmp_path.replace(path)


def _close_numpy_memmap(array: Any) -> None:
    mmap_handle = getattr(array, "_mmap", None)
    if mmap_handle is not None:
        mmap_handle.close()


def _validate_embedding_matrix(
    vectors: Any,
    np_module: Any,
    *,
    expected_rows: int | None = None,
    require_l2: bool = False,
) -> None:
    """Reject malformed, non-finite, or unexpectedly unnormalized vectors."""
    if getattr(vectors, "ndim", None) != 2:
        raise ValueError(f"Embedding matrix must be 2D, got shape={getattr(vectors, 'shape', None)}")
    if expected_rows is not None and int(vectors.shape[0]) != expected_rows:
        raise ValueError(
            f"Embedding row count mismatch: got={vectors.shape[0]}, expected={expected_rows}"
        )
    if int(vectors.shape[1]) <= 0:
        raise ValueError("Embedding dimension must be greater than 0")
    if not bool(np_module.isfinite(vectors).all()):
        raise ValueError("Embedding matrix contains NaN or infinity")
    if require_l2 and len(vectors):
        norms = np_module.linalg.norm(vectors, axis=1)
        max_error = float(np_module.max(np_module.abs(norms - 1.0)))
        if max_error > 1e-3:
            raise ValueError(
                f"Embedding matrix is not L2-normalized (max norm error={max_error:.6f})"
            )


def _clear_dense_checkpoint(checkpoint_dir: Path) -> None:
    """Remove only files created inside one explicitly scoped checkpoint dir."""
    if not checkpoint_dir.is_dir():
        return
    for path in checkpoint_dir.iterdir():
        if path.is_file() and (
            path.name == "state.json"
            or path.name.startswith("part-")
            or path.suffix == ".tmp"
        ):
            path.unlink()
    try:
        checkpoint_dir.rmdir()
    except OSError:
        pass


class VietnameseEmbeddingModel:
    """Mô hình tạo vector embedding tiếng Việt (AITeamVN/Vietnamese_Embedding_v2)."""

    def __init__(
        self,
        model_name_or_path: str = "AITeamVN/Vietnamese_Embedding_v2",
        device: str = "auto",
        batch_size: int = 8,
        revision: str | None = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.model_revision = revision
        self.device_setting = device
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        self.batch_size = batch_size
        self._tokenizer: Any = None
        self._model: Any = None
        self._parallel_model: Any = None
        self._device: Any = None
        self._device_ids: list[int] = []
        self._hidden_size = 0
        self._multi_gpu = False
        self._parallel_forward_reported = False

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
        revision_kwargs = (
            {"revision": self.model_revision}
            if self.model_revision
            else {}
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=False,
            **revision_kwargs,
        )
        model_dtype = None
        if self._device.type == "cuda":
            model_dtype = recommended_cuda_dtype(torch, device=self._device)
        base_model = AutoModel.from_pretrained(
            self.model_name_or_path,
            torch_dtype=model_dtype,
            trust_remote_code=False,
            **revision_kwargs,
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
        self._model = cls_encoder
        if self._device.type == "cuda":
            selected_index = getattr(self._device, "index", None)
            if selected_index is None:
                self._device_ids = list(range(torch.cuda.device_count()))
            else:
                self._device_ids = [int(selected_index)]

        if len(self._device_ids) > 1:
            self._parallel_model = torch.nn.DataParallel(
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
            if self._device.type == "cuda":
                gpu_id = self._device_ids[0] if self._device_ids else 0
                print(
                    f"[DenseEmbedding] Using one GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}",
                    file=sys.stderr,
                    flush=True,
                )
        self._model.eval()
        if self._parallel_model is not None:
            self._parallel_model.eval()

    def encode(
        self,
        texts: list[str],
        batch_size: int | None = None,
        max_length: int = 2048,
        normalize_embeddings: bool = True,
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
                    use_parallel = (
                        self._parallel_model is not None
                        and current_bs >= len(self._device_ids)
                    )
                    active_model = self._parallel_model if use_parallel else self._model
                    # _CLSEncoder already performs the model-specific CLS
                    # pooling and returns a plain tensor for DataParallel.
                    embeddings = active_model(**encoded)
                    if normalize_embeddings:
                        # The released SentenceTransformer recipe ends with a
                        # Normalize module. Normalize in FP32 to reproduce it
                        # accurately even when T4 inference uses FP16 weights.
                        embeddings = F.normalize(embeddings.float(), p=2, dim=1)
                    # NumPy does not support torch.bfloat16. Convert explicitly
                    # so reduced-precision CUDA inference remains portable.
                    all_embeddings.append(_tensor_to_float32_numpy(embeddings))
                    if use_parallel and not self._parallel_forward_reported:
                        allocations = ", ".join(
                            f"GPU {gpu_id}={torch.cuda.memory_allocated(gpu_id) / 1024**3:.2f} GiB"
                            for gpu_id in self._device_ids
                        )
                        print(
                            f"[DenseEmbedding] First multi-GPU forward OK ({allocations}).",
                            file=sys.stderr,
                            flush=True,
                        )
                        self._parallel_forward_reported = True
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
        default_normalization = "l2" if similarity == "cosine" else "none"
        self.normalization = str(
            self.manifest.get("normalization") or default_normalization
        )
        if self.normalization not in VALID_NORMALIZATIONS:
            raise ValueError(f"Normalization không hỗ trợ: {self.normalization}")

    @classmethod
    def load(
        cls,
        path: str | Path,
        expected_model_name: str | None = None,
        expected_model_revision: str | None = None,
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
        expected_model_identity = (
            resolve_model_identity(expected_model_name, expected_model_revision)
            if expected_model_name
            else None
        )
        if expected_model_name and not manifest_model:
            raise ValueError(
                "Legacy Dense index has no embedding model/pooling metadata; rebuild the index"
            )
        if (
            expected_model_identity
            and manifest_model
            and expected_model_identity != manifest_model
        ):
            raise ValueError(
                f"Dense index dùng model {manifest_model!r}, không phải {expected_model_identity!r}"
            )
        if expected_model_name and int(manifest.get("schema_version") or 0) < DENSE_SCHEMA_VERSION:
            raise ValueError(
                "Dense index uses an old embedding recipe; rebuild it with CLS + L2 normalization"
            )
        if expected_model_name and manifest.get("pooling") != "cls":
            raise ValueError("Dense index does not use CLS pooling; rebuild the index")
        if expected_model_name and manifest.get("normalization") != "l2":
            raise ValueError("Dense index does not use L2 normalization; rebuild the index")
        if expected_model_name and manifest.get("similarity") != "dot_product":
            raise ValueError("Dense index does not use dot-product search; rebuild the index")
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
        if self.similarity == "cosine" or self.normalization == "l2":
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
    resume: bool = False,
    checkpoint_chunks: int = 4096,
    embedding_model_revision: str | None = None,
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
    if batch_size <= 0 or embedding_max_length <= 0 or checkpoint_chunks <= 0:
        raise ValueError(
            "batch_size, embedding_max_length và checkpoint_chunks phải lớn hơn 0"
        )

    started = time.time()
    print(f"[BuildDense] Khởi tạo encoder: {embedding_model_name}...", file=sys.stderr, flush=True)
    encoder = VietnameseEmbeddingModel(
        model_name_or_path=embedding_model_name,
        revision=embedding_model_revision,
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

    corpus_sha256 = corpus_hasher.hexdigest()
    model_identity = resolve_model_identity(
        embedding_model_name,
        embedding_model_revision,
    )
    checkpoint_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "corpus_sha256": corpus_sha256,
                "embedding_model": model_identity,
                "embedding_max_length": embedding_max_length,
                "max_chunk_words": max_chunk_words,
                "overlap_words": overlap_words,
                "total_chunks": total_chunks,
                "pooling": "cls",
                "normalization": "l2",
                "similarity": "dot_product",
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    checkpoint_dir = output_path.parent / f"{output_path.name}.dense-checkpoint"
    checkpoint_state_path = checkpoint_dir / "state.json"
    checkpoint_parts: list[dict[str, Any]] = []
    completed_chunks = 0
    resumed_chunks = 0

    if resume and checkpoint_state_path.is_file():
        try:
            state = json.loads(checkpoint_state_path.read_text(encoding="utf-8"))
            if not isinstance(state, dict):
                raise ValueError("state must be an object")
            if state.get("schema_version") != DENSE_BUILD_CHECKPOINT_SCHEMA_VERSION:
                raise ValueError("checkpoint schema mismatch")
            if state.get("fingerprint") != checkpoint_fingerprint:
                raise ValueError("checkpoint fingerprint mismatch")
            raw_parts = state.get("parts")
            if not isinstance(raw_parts, list):
                raise ValueError("checkpoint parts must be a list")
            expected_start = 0
            expected_dimension: int | None = None
            for raw_part in raw_parts:
                if not isinstance(raw_part, dict):
                    raise ValueError("invalid checkpoint part")
                file_name = str(raw_part.get("file") or "")
                if not file_name or Path(file_name).name != file_name:
                    raise ValueError("unsafe checkpoint part name")
                part_start = int(raw_part.get("start"))
                part_end = int(raw_part.get("end"))
                if part_start != expected_start or not part_start < part_end <= total_chunks:
                    raise ValueError("non-contiguous checkpoint parts")
                part_path = checkpoint_dir / file_name
                part_array = np.load(part_path, mmap_mode="r", allow_pickle=False)
                try:
                    _validate_embedding_matrix(
                        part_array,
                        np,
                        expected_rows=part_end - part_start,
                        require_l2=True,
                    )
                    part_shape = tuple(part_array.shape)
                finally:
                    _close_numpy_memmap(part_array)
                if expected_dimension is None:
                    expected_dimension = int(part_shape[1])
                elif int(part_shape[1]) != expected_dimension:
                    raise ValueError("checkpoint embedding dimension mismatch")
                checkpoint_parts.append(
                    {"file": file_name, "start": part_start, "end": part_end}
                )
                expected_start = part_end
            completed_chunks = expected_start
            if int(state.get("completed_chunks", -1)) != completed_chunks:
                raise ValueError("checkpoint completion marker mismatch")
            resumed_chunks = completed_chunks
            print(
                f"[BuildDense] Resume checkpoint: {completed_chunks:,}/{total_chunks:,} chunks đã có.",
                file=sys.stderr,
                flush=True,
            )
        except (OSError, EOFError, ValueError, TypeError, KeyError) as exc:
            print(
                f"[BuildDense] Bỏ checkpoint không tương thích ({exc}); xây lại từ đầu.",
                file=sys.stderr,
                flush=True,
            )
            _clear_dense_checkpoint(checkpoint_dir)
            checkpoint_parts = []
            completed_chunks = 0
    elif checkpoint_dir.exists():
        _clear_dense_checkpoint(checkpoint_dir)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(
        checkpoint_state_path,
        {
            "schema_version": DENSE_BUILD_CHECKPOINT_SCHEMA_VERSION,
            "fingerprint": checkpoint_fingerprint,
            "total_chunks": total_chunks,
            "completed_chunks": completed_chunks,
            "parts": checkpoint_parts,
        },
    )

    while completed_chunks < total_chunks:
        part_start = completed_chunks
        part_end = min(total_chunks, part_start + checkpoint_chunks)
        part_vectors = encoder.encode(
            chunk_texts[part_start:part_end],
            batch_size=batch_size,
            max_length=embedding_max_length,
            normalize_embeddings=True,
        )
        part_vectors = np.asarray(part_vectors, dtype=np.float32)
        _validate_embedding_matrix(
            part_vectors,
            np,
            expected_rows=part_end - part_start,
            require_l2=True,
        )
        part_file_name = f"part-{part_start:09d}-{part_end:09d}.npy"
        _save_numpy_atomic(
            checkpoint_dir / part_file_name,
            part_vectors,
            np,
        )
        checkpoint_parts.append(
            {"file": part_file_name, "start": part_start, "end": part_end}
        )
        completed_chunks = part_end
        _write_json_atomic(
            checkpoint_state_path,
            {
                "schema_version": DENSE_BUILD_CHECKPOINT_SCHEMA_VERSION,
                "fingerprint": checkpoint_fingerprint,
                "total_chunks": total_chunks,
                "completed_chunks": completed_chunks,
                "parts": checkpoint_parts,
            },
        )
        print(
            f"[BuildDense] Đã lưu checkpoint: {completed_chunks:,}/{total_chunks:,} chunks.",
            file=sys.stderr,
            flush=True,
        )

    part_arrays = [
        np.load(checkpoint_dir / part["file"], mmap_mode="r", allow_pickle=False)
        for part in checkpoint_parts
    ]
    try:
        vectors = np.concatenate(part_arrays, axis=0).astype(np.float32, copy=False)
    finally:
        for part_array in part_arrays:
            _close_numpy_memmap(part_array)
    del part_arrays
    _validate_embedding_matrix(
        vectors,
        np,
        expected_rows=total_chunks,
        require_l2=True,
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
        "embedding_model": model_identity,
        "dimension": int(vectors.shape[1]),
        "similarity": "dot_product",
        "pooling": "cls",
        "normalization": "l2",
        "documents": doc_count,
        "chunks": total_chunks,
        "max_chunk_words": max_chunk_words,
        "overlap_words": overlap_words,
        "embedding_max_length": embedding_max_length,
        "corpus_sha256": corpus_sha256,
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
    _clear_dense_checkpoint(checkpoint_dir)
    elapsed = time.time() - started

    stats = {
        "documents": doc_count,
        "chunks": total_chunks,
        "embedding_dim": int(vectors.shape[1]),
        "resumed_chunks": resumed_chunks,
        "elapsed_seconds": round(elapsed, 2),
        "output_path": str(output_path.resolve()),
    }
    print(f"[BuildDense] Hoàn thành sau {elapsed:.2f}s: {stats}", file=sys.stderr, flush=True)
    return stats
