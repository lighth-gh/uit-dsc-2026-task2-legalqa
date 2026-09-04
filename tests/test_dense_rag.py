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
    DENSE_BUILD_CHECKPOINT_SCHEMA_VERSION,
    DENSE_SCHEMA_VERSION,
    DenseVectorIndex,
    VietnameseEmbeddingModel,
    _tensor_to_float32_numpy,
    _validate_embedding_matrix,
    build_dense_index,
)
from legalqa_baseline.pipeline import (
    LegalQABaseline,
    _apply_legal_signal_boost,
    _apply_reranker_legal_guardrails,
    prediction_audit_record,
    reciprocal_rank_fusion,
)
from legalqa_baseline.reranker import VietnameseReranker
from legalqa_baseline.storage import (
    CORPUS_HASH_VERSION,
    CorpusHasher,
    SearchIndex,
    build_index,
)


class MockEmbeddingModel:
    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.seen_batches: list[list[str]] = []

    def encode(self, texts: list[str], batch_size: int = 32, **_: Any) -> Any:
        import numpy as np
        self.seen_batches.append(list(texts))
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
                "schema_version": DENSE_SCHEMA_VERSION,
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
                {
                    "corpus_hash_version": CORPUS_HASH_VERSION,
                    "chunks": 2,
                    "max_chunk_words": 620,
                    "overlap_words": 100,
                    "corpus_sha256": "test-hash",
                }
            )
            loaded_index.validate_against_bm25(
                {
                    "corpus_hash_version": CORPUS_HASH_VERSION,
                    "chunks": "2",
                    "max_chunk_words": "620",
                    "overlap_words": "100",
                    "corpus_sha256": "test-hash",
                }
            )
            with self.assertRaisesRegex(ValueError, "không tương thích"):
                loaded_index.validate_against_bm25({"chunks": "3"})
            loaded_index.manifest["schema_version"] = DENSE_SCHEMA_VERSION - 1
            loaded_index.save(save_path)
            with self.assertRaisesRegex(ValueError, "old embedding recipe"):
                DenseVectorIndex.load(
                    save_path,
                    expected_model_name="mock-embedding",
                )

            loaded_index.manifest["schema_version"] = DENSE_SCHEMA_VERSION
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

    def test_3b_exact_legal_signals_boost_after_rrf(self) -> None:
        candidates = [
            {
                "context_id": "generic",
                "chunk_no": 0,
                "name": "Văn bản chung",
                "text": "Quy định chung về lương và chế độ lao động.",
                "rrf_score": 0.0200,
            },
            {
                "context_id": "exact",
                "chunk_no": 0,
                "name": "Văn bản đúng",
                "text": "Mức lương cơ sở là 1.800.000 đồng mỗi tháng.",
                "rrf_score": 0.0188,
            },
        ]

        boosted = _apply_legal_signal_boost(
            "Mức lương cơ sở có phải là 1,8 triệu đồng không?",
            candidates,
        )

        self.assertEqual(boosted[0]["context_id"], "exact")
        self.assertEqual(boosted[0]["rrf_score"], 0.0188)
        self.assertGreater(boosted[0]["legal_signal_boost"], 0.0)
        self.assertLessEqual(boosted[0]["legal_signal_boost"], 0.006)
        self.assertEqual(boosted[0]["rrf_rank_before_boost"], 2)
        self.assertEqual(boosted[0]["rrf_rank_after_boost"], 1)
        self.assertEqual(
            boosted[0]["legal_signal_matches"]["money_amounts_vnd"],
            [1_800_000],
        )

    def test_3c_heading_focus_keeps_land_compensation_article_in_pool(self) -> None:
        candidates = [
            {
                "context_id": "crop",
                "chunk_no": 127,
                "name": "Luật Đất đai",
                "text": "Điều 100. Bồi thường đối với cây trồng, vật nuôi 1. Khi Nhà nước thu hồi đất...",
                "rrf_score": 0.0164,
            },
            {
                "context_id": "land",
                "chunk_no": 115,
                "name": "Luật Đất đai",
                "text": "Điều 89. Nguyên tắc bồi thường về đất khi Nhà nước thu hồi đất 1. Người sử dụng đất...",
                "rrf_score": 0.0130,
            },
        ]

        boosted = _apply_legal_signal_boost(
            "Giá trị bồi thường đối với phần đất Nhà nước thu hồi được xác định như thế nào?",
            candidates,
        )

        self.assertEqual(boosted[0]["context_id"], "land")
        self.assertGreaterEqual(
            boosted[0]["legal_signal_matches"]["heading_query_coverage"],
            0.8,
        )

    def test_3d_guardrail_distinguishes_complaint_limitation_from_protest_term(self) -> None:
        candidates = [
            {
                "context_id": "wrong",
                "chunk_no": 316,
                "name": "Bộ luật Tố tụng dân sự 2004",
                "text": "Điều 308. Thời hạn kháng nghị theo thủ tục tái thẩm là một năm.",
                "rerank_score": 1.31,
                "rrf_rank_after_boost": 20,
            },
            {
                "context_id": "correct",
                "chunk_no": 306,
                "name": "Bộ luật Tố tụng hình sự 2015",
                "text": "Điều 471. Thời hiệu khiếu nại là 15 ngày kể từ ngày nhận được quyết định, hành vi tố tụng.",
                "rerank_score": -2.95,
                "rrf_rank_after_boost": 2,
            },
        ]

        ranked = _apply_reranker_legal_guardrails(
            "Thời hiệu khiếu nại thông báo không kháng nghị theo thủ tục tái thẩm đối với bản án hình sự là bao lâu?",
            candidates,
        )

        self.assertEqual(ranked[0]["context_id"], "correct")
        self.assertEqual(ranked[0]["rerank_rank_after_guardrail"], 1)
        self.assertGreater(ranked[0]["final_rerank_score"], ranked[1]["final_rerank_score"])
        self.assertGreater(
            ranked[0]["rerank_guardrail_components"]["exact_focus"],
            0.0,
        )

    def test_3e_guardrail_honors_explicit_civil_procedure_scope(self) -> None:
        question = "Quy định về nghĩa vụ chứng minh trong vụ án dân sự như thế nào?"
        candidates = [
            {
                "context_id": "consumer",
                "chunk_no": 73,
                "name": "Luật Bảo vệ quyền lợi người tiêu dùng",
                "text": (
                    "Điều 69. Nghĩa vụ chứng minh của người tiêu dùng. "
                    "Tranh chấp được giải quyết theo pháp luật tố tụng dân sự."
                ),
                "rerank_score": 4.10,
                "rrf_rank_after_boost": 2,
            },
            {
                "context_id": "civil-procedure",
                "chunk_no": 49,
                "name": "Bộ luật Tố tụng dân sự 2015",
                "text": "Điều 91. Nghĩa vụ chứng minh. Đương sự có yêu cầu Tòa án bảo vệ quyền lợi phải đưa ra chứng cứ.",
                "rerank_score": 2.54,
                "rrf_rank_after_boost": 1,
            },
        ]

        ranked = _apply_reranker_legal_guardrails(question, candidates)

        self.assertEqual(ranked[0]["context_id"], "civil-procedure")
        self.assertIn(
            "vụ án dân sự",
            ranked[0]["legal_signal_matches"]["scope_phrases"],
        )

    def test_3f_guardrail_prefers_governing_code_over_unscoped_case_narrative(self) -> None:
        question = (
            "Nguyên đơn có phải làm lại đơn khởi kiện trong trường hợp vụ án dân sự "
            "có bản án phúc thẩm tuyên hủy bản án sơ thẩm vì vi phạm tố tụng không?"
        )
        candidates = [
            {
                "context_id": "case",
                "chunk_no": 19,
                "name": "Quyết định công bố án lệ",
                "text": "Bị đơn yêu cầu Hội đồng xét xử phúc thẩm hủy bản án sơ thẩm.",
                "rerank_score": 0.73,
                "rrf_rank_after_boost": 18,
            },
            {
                "context_id": "code",
                "chunk_no": 136,
                "name": "Bộ luật Tố tụng dân sự 2015",
                "text": (
                    "Điều 310. Hủy bản án sơ thẩm và chuyển hồ sơ vụ án cho Tòa án cấp sơ thẩm "
                    "giải quyết lại khi có vi phạm nghiêm trọng về thủ tục tố tụng."
                ),
                "rerank_score": -4.18,
                "rrf_rank_after_boost": 3,
            },
        ]

        ranked = _apply_reranker_legal_guardrails(question, candidates)

        self.assertEqual(ranked[0]["context_id"], "code")

    def test_3g_id_80189_exact_raw_leader_is_not_flipped_by_generic_heading(self) -> None:
        question = "Mẫu thông báo thay đổi người đại diện theo pháp luật"
        shared_exact = {
            "document_references": [],
            "article_references": [],
            "document_names": [],
            "money_amounts_vnd": [],
            "years": [],
            "plan_names": [],
            "form_names": [],
            "long_phrase": "người đại diện theo pháp luật",
            "long_phrase_tokens": 6,
            "focus_phrases": [],
            "scope_phrases": [],
            "scope_requested": False,
        }
        candidates = [
            {
                "context_id": "230689",
                "chunk_no": 19,
                "name": "Thông tư 12/2012/TT-BTP",
                "text": "Thông báo thay đổi người đại diện theo pháp luật.",
                "rerank_score": -1.0,
                "rrf_rank_after_boost": 20,
                "exact_phrase_matches": 1,
                "legal_signal_matches": {
                    **shared_exact,
                    "heading_overlap_tokens": 0,
                    "heading_query_coverage": 0.0,
                },
            },
            {
                "context_id": "wrong-heading",
                "chunk_no": 5,
                "name": "Biểu mẫu không đúng đối tượng",
                "text": "Thay đổi người đại diện theo pháp luật của cơ sở lưu trú.",
                "rerank_score": -1.1,
                "rrf_rank_after_boost": 1,
                "legal_signal_matches": {
                    **shared_exact,
                    "heading_overlap_tokens": 5,
                    "heading_query_coverage": 0.71,
                },
            },
        ]

        ranked = _apply_reranker_legal_guardrails(question, candidates)

        self.assertEqual(ranked[0]["context_id"], "230689")
        wrong = next(item for item in ranked if item["context_id"] == "wrong-heading")
        self.assertLessEqual(wrong["rerank_guardrail_components"]["heading"], 0.6)
        self.assertLess(wrong["final_rerank_score"], ranked[0]["final_rerank_score"])
        protected_by = wrong["rerank_guardrail_protected_by"]
        if protected_by is not None:
            self.assertEqual(protected_by["context_id"], "230689")

    def test_3h_controlled_zone_phrase_can_rescue_the_traffic_chunk(self) -> None:
        candidates = [
            {
                "context_id": "wrong-border-sign",
                "chunk_no": 5,
                "name": "Thông tư về khu vực biên giới",
                "text": "Mẫu biển báo khu vực biên giới, vành đai biên giới và vùng cấm.",
                "rerank_score": -1.48,
                "rrf_rank_after_boost": 3,
            },
            {
                "context_id": "174131",
                "chunk_no": 158,
                "name": "Quy chuẩn báo hiệu đường bộ",
                "text": (
                    "Để báo cấm, hạn chế hoặc chỉ dẫn có hiệu lực cho tất cả các tuyến "
                    "đường trong một khu vực, đặt biển Bắt đầu vào khu vực. Từ ZONE được "
                    "biểu thị ở phía trên."
                ),
                "rerank_score": -5.0,
                "rrf_rank_after_boost": 1,
                "exact_phrase_matches": 1,
            },
        ]

        ranked = _apply_reranker_legal_guardrails(
            "Biển báo ZONE hiện nay bao gồm những loại biển báo nào?",
            candidates,
        )

        self.assertEqual(ranked[0]["context_id"], "174131")
        self.assertEqual(
            ranked[0]["rerank_guardrail_components"]["exact_retrieval_phrase"],
            4.0,
        )

    def test_3i_covid_infection_focus_beats_generic_prevention_phrase(self) -> None:
        question = (
            "Các biện pháp dự phòng cho nhân viên y tế để tránh tình trạng "
            "lây nhiễm COVID-19 như thế nào?"
        )
        candidates = [
            {
                "context_id": "violence",
                "chunk_no": 19,
                "name": "Hướng dẫn an toàn cho nhân viên y tế",
                "text": (
                    "Nhân viên lo lắng bị lây nhiễm. Bệnh nhân COVID-19 có thể gây "
                    "bạo hành. Các biện pháp dự phòng bạo hành tại nơi làm việc."
                ),
                "rerank_score": 9.68,
                "rrf_rank_after_boost": 13,
            },
            {
                "context_id": "infection",
                "chunk_no": 11,
                "name": "Hướng dẫn an toàn cho nhân viên y tế",
                "text": (
                    "DỰ PHÒNG LÂY NHIỄM SARS-COV-2. Các biện pháp dự phòng gồm "
                    "phương tiện bảo vệ cá nhân và kiểm soát nguồn lây."
                ),
                "rerank_score": 9.38,
                "rrf_rank_after_boost": 3,
            },
        ]

        ranked = _apply_reranker_legal_guardrails(question, candidates)

        self.assertEqual(ranked[0]["context_id"], "infection")
        self.assertEqual(
            ranked[0]["rerank_guardrail_components"]["exact_focus"],
            6.0,
        )
        self.assertEqual(
            ranked[1]["rerank_guardrail_components"]["exact_focus"],
            0.0,
        )

    def test_3h_exact_form_article_document_and_long_phrase_are_strong_signals(self) -> None:
        question = (
            "Mẫu số 16/TP-TTTM tại Điều 91 Bộ luật Tố tụng dân sự "
            "về nghĩa vụ chứng minh"
        )
        candidate = {
            "context_id": "exact",
            "chunk_no": 91,
            "name": "Bộ luật Tố tụng dân sự",
            "text": (
                "Mẫu số 16/TP-TTTM. Điều 91 Bộ luật Tố tụng dân sự "
                "quy định về nghĩa vụ chứng minh."
            ),
            "rerank_score": -2.0,
            "rrf_rank_after_boost": 2,
        }

        ranked = _apply_reranker_legal_guardrails(question, [candidate])
        components = ranked[0]["rerank_guardrail_components"]

        self.assertGreater(components["exact_form"], 0.0)
        self.assertGreater(components["exact_article"], 0.0)
        self.assertGreater(components["exact_document_name"], 0.0)
        self.assertGreater(components["exact_long_phrase"], 0.0)

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
        self.assertEqual(pred.route, "generated_512")
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
        self.assertEqual(pred.route, "generated_512")
        self.assertEqual(pred.evidence["num_contexts"], 3)
        top_contexts = pred.evidence["top_contexts"]
        self.assertEqual(len(top_contexts), 3)
        self.assertIn("rerank_score", top_contexts[0])
        self.assertIn("rrf_score", top_contexts[0])
        trace = pred.evidence["retrieval_trace"]
        self.assertEqual(len(trace["bm25"]["candidates"]), 50)
        self.assertEqual(len(trace["dense"]["candidates"]), 50)
        self.assertEqual(len(trace["rrf"]["candidates"]), 50)
        self.assertEqual(len(trace["reranker_pool"]["candidates"]), 20)
        self.assertEqual(len(trace["reranker_top"]["candidates"]), 3)
        first_reranked = trace["reranker_pool"]["candidates"][0]
        self.assertEqual(first_reranked["rank"], 1)
        self.assertEqual(first_reranked["document_id"], first_reranked["context_id"])
        self.assertIsInstance(first_reranked["chunk_no"], int)
        self.assertIsInstance(first_reranked["title"], str)
        self.assertIsInstance(first_reranked["score"], float)
        self.assertIn("bm25_score", first_reranked)
        self.assertIn("dense_score", first_reranked)
        self.assertIn("rrf_score", first_reranked)
        self.assertIn("legal_signal_boost", first_reranked)
        self.assertIn("boosted_rrf_score", first_reranked)
        self.assertIn("legal_signal_matches", first_reranked)
        self.assertIn("rerank_score", first_reranked)

        audit = prediction_audit_record("trace-sample", pred)
        self.assertEqual(audit["retrieval_trace"], trace)
        json.dumps(audit, ensure_ascii=False, allow_nan=False)
        print("[Smoke Test OK] Generated answer:", pred.answer)

    def test_6b_retrieval_only_uses_shared_path_without_generator(self) -> None:
        class ForbiddenGenerator:
            calls = 0

            def generate(self, **_: Any) -> str:
                type(self).calls += 1
                raise AssertionError("retrieval-only must not call generator")

        pipeline = LegalQABaseline(
            index=MockSearchIndex(),  # type: ignore[arg-type]
            generator=ForbiddenGenerator(),
            reranker=MockReranker(),
            bm25_top_k=50,
            rrf_top_k=50,
            reranker_candidate_k=20,
            rerank_top_k=3,
            context_top_k=3,
        )

        diagnostic = pipeline.retrieve_only("Mức phạt được quy định thế nào?")

        self.assertEqual(ForbiddenGenerator.calls, 0)
        self.assertEqual(diagnostic["status"], "ok")
        self.assertEqual(len(diagnostic["diagnostic_candidates"]["top50"]), 50)
        self.assertEqual(len(diagnostic["diagnostic_candidates"]["top20"]), 20)
        self.assertEqual(len(diagnostic["diagnostic_candidates"]["top3"]), 3)
        self.assertEqual(diagnostic["stage_seconds"]["generation"], 0.0)
        self.assertIn("text_preview", diagnostic["diagnostic_candidates"]["top3"][0])

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
        self.assertEqual(pred.route, "generated_512")

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
        self.assertEqual(prediction.route, "generated_512")
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
        self.assertEqual(prediction.route, "generated_512")
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

    def test_9b_reranker_scores_only_optimized_candidate_pool(self) -> None:
        class RecordingReranker(MockReranker):
            candidate_count = 0
            seen_max_length = 0

            def rerank(
                self,
                question: str,
                candidates: list[dict[str, Any]],
                top_k: int = 3,
                max_length: int = 1024,
            ) -> list[dict[str, Any]]:
                type(self).candidate_count = len(candidates)
                type(self).seen_max_length = max_length
                return super().rerank(question, candidates, top_k, max_length)

        pipeline = LegalQABaseline(
            index=MockSearchIndex(),  # type: ignore[arg-type]
            generator=MockGenerator(),
            reranker=RecordingReranker(),
            rrf_top_k=50,
            reranker_candidate_k=20,
            rerank_top_k=3,
            reranker_max_length=1024,
        )
        pred = pipeline.predict_one("Mức phạt?", mode="rag")

        self.assertEqual(RecordingReranker.candidate_count, 20)
        self.assertEqual(RecordingReranker.seen_max_length, 1024)
        self.assertEqual(pred.evidence["reranker_candidates"], 20)
        self.assertEqual(
            len(pred.evidence["retrieval_trace"]["reranker_pool"]["candidates"]),
            20,
        )
        self.assertIn("reranker", pred.evidence["stage_seconds"])

    def test_9c_dense_query_uses_controlled_retrieval_expansion(self) -> None:
        mock_model = MockEmbeddingModel(dim=4)
        dense_meta = [
            {
                "context_id": "power-plan-viii",
                "chunk_no": 0,
                "name": "Quy hoạch điện VIII",
                "link": "",
                "text": "Thủ tướng phê duyệt Quy hoạch điện VIII.",
            }
        ]
        dense_vectors = mock_model.encode([dense_meta[0]["text"]])
        dense_index = DenseVectorIndex(vectors=dense_vectors, metadata=dense_meta)
        mock_model.seen_batches.clear()
        pipeline = LegalQABaseline(
            index=MockSearchIndex(),  # type: ignore[arg-type]
            generator=MockGenerator(),
            dense_index=dense_index,
            embedding_model=mock_model,
            reranker=MockReranker(),
        )

        pred = pipeline.predict_one(
            "Thủ tướng chính thức phê duyệt Quy hoạch điện 8?",
            mode="rag",
        )

        self.assertIn("Quy hoạch điện VIII", mock_model.seen_batches[-1][0])
        query_trace = pred.evidence["retrieval_trace"]["query"]
        self.assertIn("Quy hoạch điện VIII", query_trace["expanded"])
        self.assertIn("quy hoạch điện viii", query_trace["priority_phrases"])

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
            self.assertEqual(
                state["schema_version"],
                DENSE_BUILD_CHECKPOINT_SCHEMA_VERSION,
            )
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
            self.assertEqual(
                payload["manifest"]["schema_version"],
                DENSE_SCHEMA_VERSION,
            )
            self.assertEqual(payload["manifest"]["normalization"], "l2")
            self.assertEqual(
                payload["manifest"]["corpus_hash_version"],
                CORPUS_HASH_VERSION,
            )
            expected_hasher = CorpusHasher()
            for chunk_no, chunk in enumerate(chunks):
                expected_hasher.update(
                    "doc-1",
                    chunk_no,
                    "Document",
                    "",
                    chunk,
                )
            self.assertEqual(
                payload["manifest"]["corpus_sha256"],
                expected_hasher.hexdigest(),
            )

    def test_10b_bm25_and_dense_use_the_same_corpus_hash_contract(self) -> None:
        import numpy as np

        class FixedEncoder:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                pass

            def encode(self, texts: list[str], **_kwargs: Any) -> Any:
                vectors = np.zeros((len(texts), 4), dtype=np.float32)
                vectors[:, 0] = 1.0
                return vectors

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contexts = root / "contexts"
            contexts.mkdir()
            (contexts / "context_1.json").write_text(
                json.dumps(
                    {
                        "id": 123,
                        "name": "Document",
                        "link": "https://example.test/a\u0000b",
                        "passage": "Điều 1. Nội dung pháp lý ngắn.",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            train = root / "train.json"
            train.write_text(
                json.dumps(
                    {"1": {"question": "Câu hỏi?", "answer": "Câu trả lời."}},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            bm25_path = root / "bm25.sqlite"
            dense_path = root / "dense"
            build_index(contexts, train, bm25_path)
            with patch(
                "legalqa_baseline.dense.VietnameseEmbeddingModel",
                FixedEncoder,
            ):
                build_dense_index(
                    contexts,
                    dense_path,
                    embedding_model_name="owner/model",
                )

            with SearchIndex(bm25_path) as index:
                bm25_metadata = index.metadata()
            dense_payload = json.loads(
                dense_path.with_suffix(".meta.json").read_text(encoding="utf-8")
            )
            dense_manifest = dense_payload["manifest"]
            for key in (
                "corpus_hash_version",
                "chunks",
                "max_chunk_words",
                "overlap_words",
                "corpus_sha256",
            ):
                self.assertEqual(str(dense_manifest[key]), bm25_metadata[key])

    def test_10c_dense_resume_discards_parts_when_corpus_order_changes(self) -> None:
        import numpy as np

        class InterruptingEncoder:
            calls = 0

            def __init__(self, **_kwargs: Any) -> None:
                pass

            def encode(self, texts: list[str], **_kwargs: Any) -> Any:
                type(self).calls += 1
                if type(self).calls == 2:
                    raise RuntimeError("simulated interruption")
                vectors = np.zeros((len(texts), 4), dtype=np.float32)
                vectors[:, 0] = 1.0
                return vectors

        class RecordingEncoder:
            encoded_texts: list[str] = []

            def __init__(self, **_kwargs: Any) -> None:
                pass

            def encode(self, texts: list[str], **_kwargs: Any) -> Any:
                type(self).encoded_texts.extend(texts)
                vectors = np.zeros((len(texts), 4), dtype=np.float32)
                vectors[:, 0] = 1.0
                return vectors

        contexts = [
            {
                "id": "alpha",
                "name": "Alpha",
                "link": "",
                "passage": "alpha chunk",
            },
            {
                "id": "beta",
                "name": "Beta",
                "link": "",
                "passage": "beta chunk",
            },
        ]

        def fake_contexts(_path: Any) -> Any:
            return iter(contexts)

        def fake_chunks(passage: str, **_kwargs: Any) -> list[str]:
            return [passage]

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dense"
            with (
                patch(
                    "legalqa_baseline.dense.iter_contexts",
                    side_effect=fake_contexts,
                ),
                patch(
                    "legalqa_baseline.dense.chunk_passage",
                    side_effect=fake_chunks,
                ),
                patch(
                    "legalqa_baseline.dense.VietnameseEmbeddingModel",
                    InterruptingEncoder,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    build_dense_index(
                        "contexts.zip",
                        output,
                        embedding_model_name="owner/model",
                        checkpoint_chunks=1,
                        resume=True,
                    )

            checkpoint_dir = Path(f"{output}.dense-checkpoint")
            state = json.loads(
                (checkpoint_dir / "state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(state["completed_chunks"], 1)
            self.assertIn("ordered_corpus_sha256", state)

            contexts.reverse()
            with (
                patch(
                    "legalqa_baseline.dense.iter_contexts",
                    side_effect=fake_contexts,
                ),
                patch(
                    "legalqa_baseline.dense.chunk_passage",
                    side_effect=fake_chunks,
                ),
                patch(
                    "legalqa_baseline.dense.VietnameseEmbeddingModel",
                    RecordingEncoder,
                ),
            ):
                stats = build_dense_index(
                    "contexts.zip",
                    output,
                    embedding_model_name="owner/model",
                    checkpoint_chunks=1,
                    resume=True,
                )

            self.assertEqual(stats["resumed_chunks"], 0)
            self.assertEqual(
                RecordingEncoder.encoded_texts,
                ["Beta: beta chunk", "Alpha: alpha chunk"],
            )

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
