import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from legalqa_baseline.dense import (
    DenseVectorIndex,
    VietnameseEmbeddingModel,
    _tensor_to_float32_numpy,
    _validate_embedding_matrix,
    build_dense_index,
)
from legalqa_baseline.pipeline import LegalQABaseline, reciprocal_rank_fusion
from legalqa_baseline.reranker import VietnameseReranker


class MockEmbeddingModel:
    def __init__(self, dim: int = 4) -> None:
        self.dim = dim

    def encode(self, texts: list[str], batch_size: int = 32, **_: Any) -> Any:
        import numpy as np
        # Sinh mock vector dựa trên hash chuỗi để có tính xác định
        vecs = []
        for text in texts:
            val = float(hash(text) % 100) / 100.0
            vec = np.array([val, 1.0 - val, val * 0.5, 0.5], dtype=np.float32)
            vec = vec / np.linalg.norm(vec)
            vecs.append(vec)
        return np.vstack(vecs).astype(np.float32)


class MockReranker:
    def rerank(
        self,
        question: str,
        candidates: list[dict[str, Any]],
        top_k: int = 3,
        max_length: int = 2304,
    ) -> list[dict[str, Any]]:
        scored = []
        for idx, c in enumerate(candidates):
            item = dict(c)
            # Giả lập điểm rerank: chunk nào chứa từ khóa câu hỏi hoặc chunk đầu được điểm cao
            score = 10.0 - idx
            if "phạt" in item.get("text", ""):
                score += 5.0
            item["rerank_score"] = float(score)
            scored.append(item)
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_k]


class MockGenerator:
    def generate(self, context: str, question: str) -> str:
        return f"Đáp án được sinh dựa trên ngữ cảnh: {context[:50]}..."


class MockSearchIndex:
    def search_contexts(self, question: str, top_k: int = 50) -> list[dict[str, Any]]:
        # Giả lập 50 candidate BM25
        results = []
        for i in range(min(top_k, 50)):
            results.append({
                "context_id": f"bm25_doc_{i}",
                "chunk_no": i,
                "name": f"Văn bản BM25 số {i}",
                "link": f"https://law.vn/{i}",
                "text": f"Nội dung luật BM25 {i} quy định mức phạt.",
                "bm25_score": float(-50 + i),
            })
        return results

    def search_train(self, question: str, top_k: int = 5, exclude_id: str | None = None) -> list[dict[str, Any]]:
        return []


class TestDenseRAG(unittest.TestCase):
    def test_0_bfloat_tensor_is_cast_before_numpy(self) -> None:
        class FakeTensor:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def float(self) -> "FakeTensor":
                self.calls.append("float")
                return self

            def cpu(self) -> "FakeTensor":
                self.calls.append("cpu")
                return self

            def numpy(self) -> str:
                self.calls.append("numpy")
                return "array"

        tensor = FakeTensor()
        self.assertEqual(_tensor_to_float32_numpy(tensor), "array")
        self.assertEqual(tensor.calls, ["float", "cpu", "numpy"])

    def test_1_embedding_encoding(self) -> None:
        """1. Kiểm tra sinh vector và chuẩn hóa L2 của embedding."""
        import numpy as np

        mock_model = MockEmbeddingModel(dim=4)
        vectors = mock_model.encode(["Quy định mức phạt", "Thủ tục hải quan"])
        self.assertEqual(vectors.shape, (2, 4))
        # Kiểm tra chuẩn L2 = 1.0
        norms = np.linalg.norm(vectors, axis=1)
        for norm in norms:
            self.assertAlmostEqual(norm, 1.0, places=5)

    def test_2_dense_index_search(self) -> None:
        """2. Kiểm tra dot product trên các vector đã L2-normalize và lưu/tải index."""
        import numpy as np

        mock_model = MockEmbeddingModel(dim=4)
        metadata = [
            {"context_id": "1", "chunk_no": 0, "name": "Luật A", "text": "Quy định xử phạt giao thông"},
            {"context_id": "2", "chunk_no": 0, "name": "Luật B", "text": "Thủ tục cấp phép"},
        ]
        vectors = mock_model.encode([m["text"] for m in metadata])
        index = DenseVectorIndex(
            vectors=vectors,
            metadata=metadata,
            manifest={
                "schema_version": 4,
                "embedding_model": "mock-embedding",
                "dimension": 4,
                "similarity": "dot_product",
                "pooling": "cls",
                "normalization": "l2",
            },
            similarity="dot_product",
        )

        q_vec = mock_model.encode(["Xử phạt giao thông"])[0]
        results = index.search(q_vec, top_k=2)
        self.assertEqual(len(results), 2)
        self.assertIn("dense_score", results[0])
        self.assertEqual(results[0]["dense_rank"], 1)

        # Kiểm tra lưu và nạp từ đĩa
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_dense.npy"
            index.save(save_path)
            loaded_index = DenseVectorIndex.load(
                save_path,
                expected_model_name="mock-embedding",
            )
            loaded_results = loaded_index.search(q_vec, top_k=1)
            self.assertEqual(len(loaded_results), 1)
            self.assertEqual(loaded_results[0]["name"], results[0]["name"])
            with self.assertRaisesRegex(ValueError, "không phải"):
                DenseVectorIndex.load(save_path, expected_model_name="wrong-model")
            loaded_index.manifest.update(
                {"chunks": 2, "max_chunk_words": 620, "overlap_words": 100}
            )
            loaded_index.validate_against_bm25(
                {"chunks": "2", "max_chunk_words": "620", "overlap_words": "100"}
            )
            with self.assertRaisesRegex(ValueError, "không tương thích"):
                loaded_index.validate_against_bm25({"chunks": "3"})
            loaded_index.manifest["schema_version"] = 3
            loaded_index.save(save_path)
            with self.assertRaisesRegex(ValueError, "old embedding recipe"):
                DenseVectorIndex.load(
                    save_path,
                    expected_model_name="mock-embedding",
                )

            loaded_index.manifest["schema_version"] = 4
            loaded_index.manifest.pop("normalization")
            loaded_index.save(save_path)
            with self.assertRaisesRegex(ValueError, "L2 normalization"):
                DenseVectorIndex.load(
                    save_path,
                    expected_model_name="mock-embedding",
                )

            loaded_index.manifest["normalization"] = "l2"
            loaded_index.manifest["similarity"] = "cosine"
            loaded_index.save(save_path)
            with self.assertRaisesRegex(ValueError, "dot-product search"):
                DenseVectorIndex.load(
                    save_path,
                    expected_model_name="mock-embedding",
                )

    def test_3_rrf_fusion(self) -> None:
        """3. Kiểm tra thuật toán RRF k=60 hợp nhất thứ hạng từ BM25 và Dense."""
        bm25_res = [
            {"context_id": "A", "chunk_no": 0, "name": "Doc A", "text": "Content A"},
            {
                "context_id": "B",
                "chunk_no": 0,
                "name": "Doc B",
                "text": "Content B",
                "dense_score": 0.9,
                "dense_rank": 1,
            },
        ]
        dense_res = [
            {"context_id": "B", "chunk_no": 0, "name": "Doc B", "text": "Content B"},
            {"context_id": "C", "chunk_no": 0, "name": "Doc C", "text": "Content C"},
        ]
        fused = reciprocal_rank_fusion(bm25_res, dense_res, rrf_k=60, top_k=3)
        self.assertEqual(len(fused), 3)
        # Doc B xuất hiện ở cả BM25 (rank 2) và Dense (rank 1) -> RRF score cao nhất: 1/(60+2) + 1/(60+1)
        self.assertEqual(fused[0]["context_id"], "B")
        self.assertIn("rrf_score", fused[0])
        self.assertEqual(fused[0]["dense_score"], 0.9)
        self.assertGreater(fused[0]["rrf_score"], fused[1]["rrf_score"])

    def test_4_vietnamese_reranker(self) -> None:
        """4. Kiểm tra chấm điểm cross-encoder và lọc Top-3 chunks."""
        mock_reranker = MockReranker()
        candidates = [
            {"context_id": "1", "chunk_no": 0, "name": "Doc 1", "text": "Văn bản thường"},
            {"context_id": "2", "chunk_no": 0, "name": "Doc 2", "text": "Văn bản mức xử phạt cao"},
            {"context_id": "3", "chunk_no": 0, "name": "Doc 3", "text": "Văn bản khác"},
            {"context_id": "4", "chunk_no": 0, "name": "Doc 4", "text": "Văn bản bổ sung"},
        ]
        reranked = mock_reranker.rerank("Mức phạt là bao nhiêu?", candidates, top_k=3)
        self.assertEqual(len(reranked), 3)
        # Doc 2 có chứa chữ 'phạt' nên được ưu tiên lên đầu
        self.assertEqual(reranked[0]["context_id"], "2")
        self.assertIn("rerank_score", reranked[0])

    def test_5_bm25_fallback(self) -> None:
        """5. Kiểm tra tự động fallback về BM25 khi không có Dense Index hoặc Reranker."""
        mock_index = MockSearchIndex()
        mock_gen = MockGenerator()
        # Không truyền dense_index hay reranker
        pipeline = LegalQABaseline(
            index=mock_index,  # type: ignore[arg-type]
            generator=mock_gen,
            dense_index=None,
            embedding_model=None,
            reranker=None,
            bm25_top_k=50,
            rrf_top_k=50,
            rerank_top_k=3,
        )
        pred = pipeline.predict_one("Câu hỏi kiểm tra fallback?", mode="rag")
        self.assertEqual(pred.route, "rag")
        self.assertEqual(pred.evidence["num_contexts"], 3)
        self.assertTrue(pred.answer.startswith("Đáp án được sinh"))

    def test_6_hybrid_pipeline_smoke(self) -> None:
        """6. Kiểm thử tích hợp toàn diện Hybrid RAG: BM25 + Dense FAISS + RRF + Reranker + Generator."""
        import numpy as np

        mock_search_index = MockSearchIndex()
        mock_model = MockEmbeddingModel(dim=4)
        mock_dense_meta = [
            {
                "context_id": f"dense_doc_{i}",
                "chunk_no": i,
                "name": f"Văn bản Dense số {i}",
                "link": f"https://law.vn/dense/{i}",
                "text": f"Quy định chi tiết điều khoản Dense {i} phạt nặng.",
            }
            for i in range(50)
        ]
        dense_vectors = mock_model.encode([m["text"] for m in mock_dense_meta])
        dense_index = DenseVectorIndex(vectors=dense_vectors, metadata=mock_dense_meta)

        mock_reranker = MockReranker()
        mock_generator = MockGenerator()

        pipeline = LegalQABaseline(
            index=mock_search_index,  # type: ignore[arg-type]
            generator=mock_generator,
            dense_index=dense_index,
            embedding_model=mock_model,
            reranker=mock_reranker,
            bm25_top_k=50,
            dense_top_k=50,
            rrf_k=60,
            rrf_top_k=50,
            rerank_top_k=3,
        )

        pred = pipeline.predict_one("Hành vi vi phạm bị phạt như thế nào?", mode="rag")
        self.assertEqual(pred.route, "rag")
        self.assertEqual(pred.evidence["num_contexts"], 3)
        top_contexts = pred.evidence["top_contexts"]
        self.assertEqual(len(top_contexts), 3)
        self.assertIn("rerank_score", top_contexts[0])
        self.assertIn("rrf_score", top_contexts[0])
        print("[Smoke Test OK] Generated answer:", pred.answer)

    def test_7_dense_error_fallback_is_explicit(self) -> None:
        class BadEmbedding:
            def encode(self, texts: list[str], **_: Any) -> Any:
                raise RuntimeError("dense boom")

        pipeline = LegalQABaseline(
            index=MockSearchIndex(),  # type: ignore[arg-type]
            generator=MockGenerator(),
            dense_index=object(),
            embedding_model=BadEmbedding(),
            allow_retrieval_fallback=True,
        )
        pred = pipeline.predict_one("Câu hỏi fallback", mode="rag")
        self.assertEqual(pred.route, "rag")

        strict_pipeline = LegalQABaseline(
            index=MockSearchIndex(),  # type: ignore[arg-type]
            generator=MockGenerator(),
            dense_index=object(),
            embedding_model=BadEmbedding(),
        )
        with self.assertRaisesRegex(RuntimeError, "Dense search thất bại"):
            strict_pipeline.predict_one("Câu hỏi strict", mode="rag")

    def test_7b_internal_encoder_type_error_is_not_retried(self) -> None:
        class InternalTypeErrorEmbedding:
            calls = 0

            def encode(self, texts: list[str]) -> Any:
                type(self).calls += 1
                raise TypeError("internal encoder bug")

        embedding = InternalTypeErrorEmbedding()
        pipeline = LegalQABaseline(
            index=MockSearchIndex(),  # type: ignore[arg-type]
            generator=MockGenerator(),
            dense_index=object(),
            embedding_model=embedding,
            allow_retrieval_fallback=True,
        )
        prediction = pipeline.predict_one("Câu hỏi fallback", mode="rag")
        self.assertEqual(prediction.route, "rag")
        self.assertEqual(InternalTypeErrorEmbedding.calls, 1)

    def test_7c_empty_reranker_result_uses_explicit_fallback(self) -> None:
        class EmptyReranker:
            def rerank(
                self,
                question: str,
                candidates: list[dict[str, Any]],
                top_k: int = 3,
                max_length: int = 2304,
            ) -> list[dict[str, Any]]:
                return []

        pipeline = LegalQABaseline(
            index=MockSearchIndex(),  # type: ignore[arg-type]
            generator=MockGenerator(),
            reranker=EmptyReranker(),
            allow_retrieval_fallback=True,
        )
        prediction = pipeline.predict_one("Mức phạt?", mode="rag")
        self.assertEqual(prediction.route, "rag")
        self.assertEqual(prediction.evidence["num_contexts"], 3)

        strict_pipeline = LegalQABaseline(
            index=MockSearchIndex(),  # type: ignore[arg-type]
            generator=MockGenerator(),
            reranker=EmptyReranker(),
        )
        with self.assertRaisesRegex(RuntimeError, "Reranker thất bại"):
            strict_pipeline.predict_one("Mức phạt?", mode="rag")

    def test_8_dense_only_candidate_has_confidence(self) -> None:
        import numpy as np

        class EmptyBM25(MockSearchIndex):
            def search_contexts(self, question: str, top_k: int = 50) -> list[dict[str, Any]]:
                return []

        metadata = [
            {
                "context_id": "dense-only",
                "chunk_no": 0,
                "name": "Luật Dense",
                "link": "",
                "text": "Quy định xử phạt từ Dense.",
            }
        ]
        dense_index = DenseVectorIndex(
            vectors=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            metadata=metadata,
        )

        class FixedEmbedding:
            normalize_flags: list[bool] = []

            def encode(
                self,
                texts: list[str],
                normalize_embeddings: bool = False,
                **_: Any,
            ) -> Any:
                type(self).normalize_flags.append(normalize_embeddings)
                return np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

        pipeline = LegalQABaseline(
            index=EmptyBM25(),  # type: ignore[arg-type]
            generator=MockGenerator(),
            dense_index=dense_index,
            embedding_model=FixedEmbedding(),
        )
        pred = pipeline.predict_one("Mức phạt?", mode="rag")
        self.assertEqual(pred.evidence["top_contexts"][0]["context_id"], "dense-only")
        self.assertGreater(pred.confidence, 0.0)
        self.assertEqual(FixedEmbedding.normalize_flags, [True])

    def test_9_context_top_k_controls_prompt_evidence(self) -> None:
        pipeline = LegalQABaseline(
            index=MockSearchIndex(),  # type: ignore[arg-type]
            generator=MockGenerator(),
            reranker=MockReranker(),
            rrf_top_k=50,
            rerank_top_k=3,
            context_top_k=2,
        )
        pred = pipeline.predict_one("Mức phạt?", mode="rag")
        self.assertEqual(pred.evidence["num_contexts"], 2)

    def test_10_dense_build_resumes_from_atomic_parts(self) -> None:
        import numpy as np

        chunks = [f"chunk {index} with enough deterministic content" for index in range(5)]

        class InterruptingEncoder:
            calls = 0
            normalize_flags: list[bool] = []

            def __init__(self, **_: Any) -> None:
                pass

            def encode(
                self,
                texts: list[str],
                normalize_embeddings: bool = False,
                **_: Any,
            ) -> Any:
                type(self).calls += 1
                type(self).normalize_flags.append(normalize_embeddings)
                if type(self).calls == 2:
                    raise RuntimeError("simulated interruption")
                rows = np.zeros((len(texts), 4), dtype=np.float32)
                rows[:, 0] = 1.0
                return rows

        class CompletingEncoder:
            encoded_texts: list[str] = []
            normalize_flags: list[bool] = []

            def __init__(self, **_: Any) -> None:
                pass

            def encode(
                self,
                texts: list[str],
                normalize_embeddings: bool = False,
                **_: Any,
            ) -> Any:
                type(self).encoded_texts.extend(texts)
                type(self).normalize_flags.append(normalize_embeddings)
                rows = np.zeros((len(texts), 4), dtype=np.float32)
                rows[:, 0] = 1.0
                return rows

        contexts = [
            {"id": "doc-1", "name": "Document", "link": "", "passage": "text"}
        ]

        def fake_contexts(_path: Any) -> Any:
            return iter(contexts)

        def fake_chunks(*_args: Any, **_kwargs: Any) -> list[str]:
            return chunks

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dense"
            with (
                patch("legalqa_baseline.dense.iter_contexts", side_effect=fake_contexts),
                patch("legalqa_baseline.dense.chunk_passage", side_effect=fake_chunks),
                patch("legalqa_baseline.dense.VietnameseEmbeddingModel", InterruptingEncoder),
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    build_dense_index(
                        "contexts.zip",
                        output,
                        embedding_model_name="owner/model",
                        checkpoint_chunks=2,
                        resume=True,
                    )

            checkpoint_dir = Path(f"{output}.dense-checkpoint")
            state = json.loads((checkpoint_dir / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["schema_version"], 2)
            self.assertEqual(state["completed_chunks"], 2)
            self.assertEqual(len(state["parts"]), 1)

            with (
                patch("legalqa_baseline.dense.iter_contexts", side_effect=fake_contexts),
                patch("legalqa_baseline.dense.chunk_passage", side_effect=fake_chunks),
                patch("legalqa_baseline.dense.VietnameseEmbeddingModel", CompletingEncoder),
            ):
                stats = build_dense_index(
                    "contexts.zip",
                    output,
                    embedding_model_name="owner/model",
                    checkpoint_chunks=2,
                    resume=True,
                )

            self.assertEqual(stats["resumed_chunks"], 2)
            self.assertTrue(all(InterruptingEncoder.normalize_flags))
            self.assertTrue(all(CompletingEncoder.normalize_flags))
            self.assertEqual(
                CompletingEncoder.encoded_texts,
                [f"Document: {chunk}" for chunk in chunks[2:]],
            )
            self.assertTrue(output.with_suffix(".meta.json").is_file())
            self.assertTrue(output.with_suffix(".npy").is_file())
            self.assertFalse(checkpoint_dir.exists())
            payload = json.loads(
                output.with_suffix(".meta.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["manifest"]["schema_version"], 4)
            self.assertEqual(payload["manifest"]["normalization"], "l2")

    def test_11_embedding_validation_rejects_raw_or_non_finite_vectors(self) -> None:
        import numpy as np

        valid = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        _validate_embedding_matrix(valid, np, expected_rows=2, require_l2=True)

        with self.assertRaisesRegex(ValueError, "not L2-normalized"):
            _validate_embedding_matrix(
                np.array([[2.0, 0.0]], dtype=np.float32),
                np,
                require_l2=True,
            )
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            _validate_embedding_matrix(
                np.array([[np.nan, 0.0]], dtype=np.float32),
                np,
                require_l2=True,
            )


if __name__ == "__main__":
    unittest.main()
