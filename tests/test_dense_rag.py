import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from legalqa_baseline.dense import DenseVectorIndex, VietnameseEmbeddingModel
from legalqa_baseline.pipeline import LegalQABaseline, reciprocal_rank_fusion
from legalqa_baseline.reranker import VietnameseReranker


class MockEmbeddingModel:
    def __init__(self, dim: int = 4) -> None:
        self.dim = dim

    def encode(self, texts: list[str], batch_size: int = 32) -> Any:
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
        self, question: str, candidates: list[dict[str, Any]], top_k: int = 3
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
        """2. Kiểm tra tìm kiếm cosine vector qua DenseVectorIndex và lưu/tải index."""
        import numpy as np

        mock_model = MockEmbeddingModel(dim=4)
        metadata = [
            {"context_id": "1", "chunk_no": 0, "name": "Luật A", "text": "Quy định xử phạt giao thông"},
            {"context_id": "2", "chunk_no": 0, "name": "Luật B", "text": "Thủ tục cấp phép"},
        ]
        vectors = mock_model.encode([m["text"] for m in metadata])
        index = DenseVectorIndex(vectors=vectors, metadata=metadata)

        q_vec = mock_model.encode(["Xử phạt giao thông"])[0]
        results = index.search(q_vec, top_k=2)
        self.assertEqual(len(results), 2)
        self.assertIn("dense_score", results[0])
        self.assertEqual(results[0]["dense_rank"], 1)

        # Kiểm tra lưu và nạp từ đĩa
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_dense.npy"
            index.save(save_path)
            loaded_index = DenseVectorIndex.load(save_path)
            loaded_results = loaded_index.search(q_vec, top_k=1)
            self.assertEqual(len(loaded_results), 1)
            self.assertEqual(loaded_results[0]["name"], results[0]["name"])

    def test_3_rrf_fusion(self) -> None:
        """3. Kiểm tra thuật toán RRF k=60 hợp nhất thứ hạng từ BM25 và Dense."""
        bm25_res = [
            {"context_id": "A", "chunk_no": 0, "name": "Doc A", "text": "Content A"},
            {"context_id": "B", "chunk_no": 0, "name": "Doc B", "text": "Content B"},
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
            rrf_top_k=12,
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
            rrf_top_k=12,
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


if __name__ == "__main__":
    unittest.main()
