from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
import zipfile
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from legalqa_baseline.metrics import answer_token_f1, meteor_exact, rouge_l_f1
from legalqa_baseline.pipeline import (
    LegalQABaseline,
    Prediction,
    prediction_audit_record,
    question_similarity,
    reciprocal_rank_fusion,
)
from legalqa_baseline.storage import (
    SearchIndex,
    build_index,
    iter_contexts,
    load_qa,
    write_predictions,
)
from legalqa_baseline.text import (
    _split_on_headings,
    best_excerpt,
    chunk_passage,
    expand_retrieval_query,
    legal_retrieval_signal_matches,
    query_terms,
    retrieval_priority_phrases,
    retrieval_query_aliases,
)


class TextTests(unittest.TestCase):
    def test_article_chunking(self) -> None:
        text = "Mở đầu văn bản.\nĐiều 1. Quy định chung " + "nội dung " * 700
        chunks = chunk_passage(text, max_words=120, overlap_words=20)
        self.assertGreater(len(chunks), 5)
        self.assertEqual(chunks[0], "Mở đầu văn bản.")
        self.assertTrue(any(chunk.startswith("Điều 1") for chunk in chunks))

    def test_chunking_preserves_two_short_articles(self) -> None:
        text = (
            "Điều 1. Cấm hút thuốc.\n"
            "Điều 2. Quyết định này có hiệu lực ngay."
        )

        chunks = chunk_passage(text)
        chunked_text = "\n".join(chunks)

        self.assertIn("Điều 1. Cấm hút thuốc.", chunked_text)
        self.assertIn("Điều 2. Quyết định này có hiệu lực ngay.", chunked_text)

    def test_chunking_preserves_short_article_between_long_articles(self) -> None:
        short_article = "Điều 2. Cấm hành vi thử nghiệm đặc biệt."
        text = "\n".join(
            (
                "Điều 1. " + "Nội dung quy định thứ nhất. " * 12,
                short_article,
                "Điều 3. " + "Nội dung quy định thứ ba. " * 12,
            )
        )

        chunks = chunk_passage(text)

        self.assertIn(short_article, "\n".join(chunks))

    def test_query_terms(self) -> None:
        terms = query_terms("Mức phạt theo khoản 3 Điều 17 là bao nhiêu?")
        self.assertIn("phạt", terms)
        self.assertIn("17", terms)

    def test_controlled_retrieval_query_expansion(self) -> None:
        power_question = "Thủ tướng phê duyệt Quy hoạch điện 8?"
        power_expanded = expand_retrieval_query(power_question)
        self.assertIn("Quy hoạch điện VIII", power_expanded)
        self.assertIn("8", query_terms(power_expanded))
        self.assertIn("quy hoạch điện viii", retrieval_priority_phrases(power_question))
        self.assertIn("quy hoạch điện 8", retrieval_priority_phrases(power_question))

        salary_question = "Mức lương cơ sở tăng lên 1,8 triệu đồng?"
        self.assertIn("1.800.000 đồng", retrieval_query_aliases(salary_question))
        self.assertIn("mức lương cơ sở", retrieval_priority_phrases(salary_question))
        self.assertIn(
            "1,8 triệu",
            retrieval_query_aliases("Mức lương cơ sở là 1.800.000 đồng"),
        )

        prrs_question = (
            "Phương pháp ELISA chẩn đoán hội chứng rối loạn sinh sản và hô hấp "
            "ở lợn gồm bao nhiêu bước?"
        )
        self.assertIn("PRRS", retrieval_query_aliases(prrs_question))
        self.assertIn("bệnh tai xanh", retrieval_query_aliases(prrs_question))

        zone_question = "Biển báo ZONE hiện nay bao gồm những loại biển báo nào?"
        self.assertIn("Bắt đầu vào khu vực", retrieval_query_aliases(zone_question))
        self.assertIn("bắt đầu vào khu vực", retrieval_priority_phrases(zone_question))

        pccc_report_question = (
            "Thực hiện báo cáo phương tiện phòng cháy chữa cháy như thế nào "
            "và tại đâu?"
        )
        pccc_aliases = retrieval_query_aliases(pccc_report_question)
        self.assertIn(
            "Trình tự báo cáo và cơ quan tiếp nhận báo cáo",
            pccc_aliases,
        )
        self.assertIn(
            "trình tự báo cáo và cơ quan tiếp nhận báo cáo",
            retrieval_priority_phrases(pccc_report_question),
        )
        self.assertEqual(
            retrieval_query_aliases("Ai phải lập phương án phòng cháy chữa cháy?"),
            [],
        )

        fund_question = "Sử dụng Quỹ bảo hiểm tai nạn lao động, bệnh nghề nghiệp?"
        self.assertIn(
            "sử dụng quỹ bảo hiểm tai nạn lao động bệnh nghề nghiệp",
            retrieval_priority_phrases(fund_question),
        )

        form_question = "Mẫu kế hoạch giáo dục của giáo viên mới nhất?"
        self.assertIn(
            "kế hoạch giáo dục của giáo viên",
            retrieval_priority_phrases(form_question),
        )

        legal_number = "Điều 8 Nghị định 12/2024/NĐ-CP quy định thế nào?"
        self.assertEqual(expand_retrieval_query(legal_number), legal_number)
        self.assertNotIn("VIII", expand_retrieval_query(legal_number))

    def test_exact_legal_retrieval_signals(self) -> None:
        question = (
            "Theo Nghị định 12/2024/NĐ-CP, mức lương cơ sở năm 2024 "
            "có phải là 1,8 triệu đồng không?"
        )
        candidate = (
            "Nghị định 12/2024/NĐ-CP quy định mức lương cơ sở năm 2024 "
            "là 1.800.000 đồng/tháng."
        )
        matches = legal_retrieval_signal_matches(question, candidate)
        self.assertEqual(matches["document_references"], ["nghị định 12/2024/nđ-cp"])
        self.assertEqual(matches["money_amounts_vnd"], [1_800_000])
        self.assertEqual(matches["years"], ["2024"])

        covid_question = (
            "Các biện pháp dự phòng cho nhân viên y tế để tránh tình trạng "
            "lây nhiễm COVID-19 như thế nào?"
        )
        covid_match = legal_retrieval_signal_matches(
            covid_question,
            "DỰ PHÒNG LÂY NHIỄM SARS-COV-2. Các biện pháp dự phòng đối với cơ sở y tế.",
        )
        self.assertIn("lây nhiễm covid 19", covid_match["focus_phrases"])
        unrelated_match = legal_retrieval_signal_matches(
            covid_question,
            "Nhân viên lo lắng bị lây nhiễm. Bệnh nhân COVID-19 có thể bị bạo hành.",
        )
        self.assertEqual(unrelated_match["focus_phrases"], [])
        self.assertGreaterEqual(matches["long_phrase_tokens"], 4)

        plan_matches = legal_retrieval_signal_matches(
            "Nội dung Quy hoạch điện 8 là gì?",
            "Quy hoạch điện VIII xác định các nhiệm vụ trọng tâm.",
        )
        self.assertEqual(plan_matches["plan_names"], ["quy hoạch điện viii"])

        form_matches = legal_retrieval_signal_matches(
            "Mẫu thông báo thay đổi người đại diện theo pháp luật ở đâu?",
            "Mẫu thông báo thay đổi người đại diện theo pháp luật của doanh nghiệp.",
        )
        self.assertTrue(form_matches["form_names"])

        citation_matches = legal_retrieval_signal_matches(
            "Điều 91 Bộ luật Tố tụng dân sự quy định thế nào?",
            "Bộ luật Tố tụng dân sự. Điều 91. Nghĩa vụ chứng minh.",
        )
        self.assertEqual(citation_matches["article_references"], ["điều 91"])
        self.assertEqual(
            citation_matches["document_names"],
            ["bộ luật tố tụng dân sự"],
        )

        ordinary_notice = legal_retrieval_signal_matches(
            "Mẫu thông báo thay đổi người đại diện theo pháp luật",
            "Thông báo thay đổi người đại diện theo pháp luật của doanh nghiệp.",
        )
        self.assertEqual(ordinary_notice["document_names"], [])

    def test_excerpt_limit(self) -> None:
        text = ("khác " * 300) + ("kiểm dịch động vật " * 100)
        excerpt = best_excerpt(text, "kiểm dịch động vật", max_words=100)
        self.assertLessEqual(len(excerpt.split()), 100)
        self.assertIn("kiểm dịch", excerpt)

    def test_split_on_headings_preserves_preamble_and_headings(self) -> None:
        text = "Lời mở đầu văn bản luật.\nĐiều 1. Phạm vi điều chỉnh\nNội dung điều 1.\nPhụ lục 1. Biểu mẫu\nNội dung phụ lục."
        pieces = _split_on_headings(text)
        self.assertEqual(len(pieces), 3)
        self.assertEqual(pieces[0], "Lời mở đầu văn bản luật.")
        self.assertTrue(pieces[1].startswith("Điều 1"))
        self.assertTrue(pieces[2].startswith("Phụ lục 1"))


class MetricTests(unittest.TestCase):
    def test_identical_scores(self) -> None:
        text = "a b c d"
        self.assertAlmostEqual(rouge_l_f1(text, text), 1.0)
        self.assertGreater(meteor_exact(text, text), 0.99)
        self.assertAlmostEqual(answer_token_f1(text, text), 1.0)

    def test_question_similarity(self) -> None:
        same = question_similarity("Mức xử phạt là bao nhiêu?", "Mức xử phạt là bao nhiêu?")
        other = question_similarity("Mức xử phạt là bao nhiêu?", "Thủ tục cấp hộ chiếu")
        self.assertGreater(same, other)

    def test_question_similarity_keeps_legal_disambiguators(self) -> None:
        self.assertLess(
            question_similarity(
                "\u0110i\u1ec1u 5 x\u1eed ph\u1ea1t xe m\u00e1y",
                "\u0110i\u1ec1u 6 x\u1eed ph\u1ea1t xe m\u00e1y",
            ),
            0.72,
        )
        self.assertLess(
            question_similarity(
                "Ngh\u0129a v\u1ee5 ng\u01b0\u1eddi \u0111\u01b0\u1ee3c thi h\u00e0nh \u00e1n",
                "Ngh\u0129a v\u1ee5 ng\u01b0\u1eddi ph\u1ea3i thi h\u00e0nh \u00e1n",
            ),
            0.72,
        )


class PipelineRoutingTests(unittest.TestCase):
    class EmptyIndex:
        def search_contexts(self, question: str, top_k: int = 12) -> list[dict[str, object]]:
            return []

        def search_train(
            self,
            question: str,
            top_k: int = 5,
            exclude_id: str | None = None,
        ) -> list[dict[str, object]]:
            return [{
                "sample_id": "wrong",
                "question": "thu tuc cap ho chieu",
                "answer": "wrong answer",
                "bm25_score": -1.0,
            }]

    def test_hybrid_does_not_return_knn_below_threshold(self) -> None:
        pipeline = LegalQABaseline(index=self.EmptyIndex())  # type: ignore[arg-type]
        prediction = pipeline.predict_one("muc phat giao thong", mode="hybrid")
        self.assertEqual(prediction.route, "fallback")

    def test_blank_question_returns_fallback_without_retrieval(self) -> None:
        class TrackingIndex(self.EmptyIndex):
            def search_contexts(self, question: str, top_k: int = 12) -> list[dict[str, object]]:
                raise AssertionError("blank question must not query contexts")

            def search_train(
                self,
                question: str,
                top_k: int = 5,
                exclude_id: str | None = None,
            ) -> list[dict[str, object]]:
                raise AssertionError("blank question must not query train")

        prediction = LegalQABaseline(index=TrackingIndex()).predict_one(" \t", mode="rag")  # type: ignore[arg-type]
        self.assertEqual(prediction.route, "fallback")

    def test_partial_dense_configuration_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cùng nhau"):
            LegalQABaseline(index=self.EmptyIndex(), dense_index=object())  # type: ignore[arg-type]

    def test_blank_knn_answers_are_skipped(self) -> None:
        class BlankAnswerIndex(self.EmptyIndex):
            def search_train(
                self,
                question: str,
                top_k: int = 5,
                exclude_id: str | None = None,
            ) -> list[dict[str, object]]:
                return [{
                    "sample_id": "blank",
                    "question": question,
                    "answer": "",
                    "bm25_score": -1.0,
                }]

        prediction = LegalQABaseline(index=BlankAnswerIndex()).predict_one("same", mode="hybrid")  # type: ignore[arg-type]
        self.assertEqual(prediction.route, "fallback")

    def test_final_pipeline_cleaner_covers_knn_and_empty_answers(self) -> None:
        class DirtyKnnIndex(self.EmptyIndex):
            def search_train(
                self,
                question: str,
                top_k: int = 5,
                exclude_id: str | None = None,
            ) -> list[dict[str, object]]:
                return [{
                    "sample_id": "dirty",
                    "question": question,
                    "answer": (
                        "**Kết quả** theo Luật số 123257 và "
                        "Nghi-dinh-74-2015-ND-CP-noi-dung-289989."
                    ),
                    "bm25_score": -1.0,
                }]

        prediction = LegalQABaseline(index=DirtyKnnIndex()).predict_one(  # type: ignore[arg-type]
            "Trách nhiệm của cơ quan là gì?",
            mode="hybrid_rag",
        )
        self.assertEqual(prediction.route, "knn_exact")
        self.assertNotIn("**", prediction.answer)
        self.assertNotIn("123257", prediction.answer)
        self.assertNotIn("Nghi-dinh-", prediction.answer)
        self.assertTrue(prediction.evidence["answer_cleaning"]["schema_valid"])
        audit = prediction_audit_record("dirty", prediction)
        self.assertFalse(audit["has_markdown"])
        self.assertFalse(audit["has_document_slug"])
        self.assertFalse(audit["has_fake_document_number"])

        pipeline = LegalQABaseline(index=self.EmptyIndex())  # type: ignore[arg-type]
        pipeline._extractive = lambda question: Prediction(" ** ** ", "extractive", 1.0, {})  # type: ignore[method-assign]
        empty_prediction = pipeline.predict_one("Câu hỏi", mode="extractive")
        self.assertEqual(empty_prediction.route, "fallback")
        self.assertTrue(empty_prediction.answer)
        self.assertFalse(empty_prediction.evidence["answer_cleaning"]["schema_valid"])

    def test_id_129859_exact_normalized_question_uses_train_answer(self) -> None:
        question = (
            "Cơ quan chủ trì soạn thảo văn bản quy phạm pháp luật có những trách nhiệm "
            "gì trong việc lồng ghép vấn đề bình đẳng giới vào văn bản?"
        )

        class ExactIndex(self.EmptyIndex):
            def search_contexts(self, question: str, top_k: int = 12) -> list[dict[str, object]]:
                raise AssertionError("exact normalized KNN must not call RAG retrieval")

            def search_train(
                self,
                question: str,
                top_k: int = 5,
                exclude_id: str | None = None,
            ) -> list[dict[str, object]]:
                return [{
                    "sample_id": "88019",
                    "question": question.upper().replace("?", " ? "),
                    "answer": "Đáp án đúng từ train cho trách nhiệm của cơ quan chủ trì soạn thảo.",
                    "bm25_score": -20.0,
                }]

        prediction = LegalQABaseline(
            index=ExactIndex(),  # type: ignore[arg-type]
            knn_threshold=1.0,
        ).predict_one(
            question,
            mode="hybrid_rag",
        )

        self.assertEqual(prediction.route, "knn_exact")
        self.assertEqual(prediction.evidence["sample_id"], "88019")
        self.assertEqual(prediction.evidence["knn_match_type"], "exact_normalized")

    def test_near_duplicate_requires_point_nine_and_matching_guards(self) -> None:
        query = "Hồ sơ cấp giấy phép cho doanh nghiệp gồm những gì?"

        class NearIndex(self.EmptyIndex):
            def search_train(
                self,
                question: str,
                top_k: int = 5,
                exclude_id: str | None = None,
            ) -> list[dict[str, object]]:
                return [{
                    "sample_id": "near",
                    "question": "Hồ sơ cấp giấy phép cho doanh nghiệp gồm những giấy tờ gì?",
                    "answer": "Hồ sơ gồm đơn đề nghị và giấy tờ của doanh nghiệp.",
                    "bm25_score": -9.0,
                }]

        pipeline = LegalQABaseline(
            index=NearIndex(),  # type: ignore[arg-type]
            guarded_knn_threshold=0.90,
        )
        prediction = pipeline._guarded_knn(query, exclude_id=None)

        self.assertIsNotNone(prediction)
        assert prediction is not None
        self.assertEqual(prediction.route, "knn_guarded")
        self.assertGreaterEqual(prediction.confidence, 0.90)
        self.assertEqual(prediction.evidence["knn_match_type"], "near_duplicate")

    def test_near_duplicate_below_point_nine_falls_back_to_rag(self) -> None:
        query = "Hồ sơ cấp giấy phép cho doanh nghiệp gồm những gì?"

        class BelowThresholdIndex(self.EmptyIndex):
            def search_train(
                self,
                question: str,
                top_k: int = 5,
                exclude_id: str | None = None,
            ) -> list[dict[str, object]]:
                return [{
                    "sample_id": "below-threshold",
                    "question": "Hồ sơ đăng ký giấy phép cho doanh nghiệp gồm các giấy tờ nào?",
                    "answer": "Không được phép lấy đáp án dưới ngưỡng.",
                    "bm25_score": -8.0,
                }]

        pipeline = LegalQABaseline(index=BelowThresholdIndex())  # type: ignore[arg-type]
        pipeline._rag = lambda question, retrieval_only=False, exclude_id=None: Prediction(  # type: ignore[method-assign]
            "Đáp án từ RAG.",
            "generated_512",
            0.8,
            {},
        )
        prediction = pipeline.predict_one(query, mode="hybrid_rag")

        self.assertEqual(prediction.route, "generated_512")
        self.assertEqual(prediction.answer, "Đáp án từ RAG.")

    def test_guarded_knn_rejects_intent_level_subject_and_legal_scope_mismatch(self) -> None:
        pairs = (
            (
                "Nguyên tắc cấp giấy phép hoạt động giáo dục nghề nghiệp cho trường cao đẳng theo quy định hiện hành là gì?",
                "Hồ sơ cấp giấy phép hoạt động giáo dục nghề nghiệp cho trường cao đẳng theo quy định hiện hành là gì?",
            ),
            (
                "Điều kiện cấp giấy phép đào tạo nghề cho người học trình độ cao đẳng theo quy định hiện hành là gì?",
                "Điều kiện cấp giấy phép đào tạo nghề cho người học trình độ trung cấp theo quy định hiện hành là gì?",
            ),
            (
                "Điều kiện xử phạt vi phạm hành chính đối với cá nhân kinh doanh theo quy định hiện hành là gì?",
                "Điều kiện xử phạt vi phạm hành chính đối với tổ chức kinh doanh theo quy định hiện hành là gì?",
            ),
            (
                "Thời hạn kháng nghị trong tố tụng dân sự theo thủ tục hiện hành được xác định như thế nào?",
                "Thời hạn kháng nghị trong tố tụng hình sự theo thủ tục hiện hành được xác định như thế nào?",
            ),
        )
        for query, candidate_question in pairs:
            with self.subTest(query=query, candidate=candidate_question):
                class GuardIndex(self.EmptyIndex):
                    def search_train(
                        self,
                        question: str,
                        top_k: int = 5,
                        exclude_id: str | None = None,
                    ) -> list[dict[str, object]]:
                        return [{
                            "sample_id": "wrong-scope",
                            "question": candidate_question,
                            "answer": "Không được phép dùng đáp án này.",
                            "bm25_score": -15.0,
                        }]

                pipeline = LegalQABaseline(index=GuardIndex())  # type: ignore[arg-type]
                pipeline._rag = lambda question, retrieval_only=False, exclude_id=None: Prediction(  # type: ignore[method-assign]
                    "Đáp án từ RAG.",
                    "generated_512",
                    0.8,
                    {},
                )
                prediction = pipeline.predict_one(query, mode="hybrid_rag")
                self.assertEqual(prediction.route, "generated_512")
                self.assertEqual(prediction.answer, "Đáp án từ RAG.")

    def test_rrf_validates_and_deduplicates_candidates(self) -> None:
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([], [], rrf_k=0)
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([], [], top_k=0)
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([{"text": "missing identity"}], [])

        fused = reciprocal_rank_fusion(
            [
                {"context_id": "doc", "chunk_no": 0, "text": "first"},
                {"context_id": "doc", "chunk_no": 0, "text": "duplicate"},
            ],
            [],
        )
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]["text"], "first")


class IoTests(unittest.TestCase):
    def test_schema_and_submission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.json"
            source.write_text(
                json.dumps({"1": {"question": "Câu hỏi?", "answer": None}}, ensure_ascii=False),
                encoding="utf-8",
            )
            self.assertEqual(load_qa(source)["1"]["question"], "Câu hỏi?")
            output = root / "prediction.json"
            write_predictions(output, {"1": "Câu trả lời"})
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["1"]["answer"], "Câu trả lời")

    def test_submission_writer_rejects_empty_or_dirty_answer_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "prediction.json"
            invalid_answers = (
                "",
                "**Nội dung Markdown**",
                "Luật số 123257 quy định nội dung này.",
                "Nghi-dinh-74-2015-ND-CP-noi-dung-289989",
            )
            for answer in invalid_answers:
                with self.subTest(answer=answer):
                    with self.assertRaises(ValueError):
                        write_predictions(output, {"1": answer})
            with self.assertRaises(ValueError):
                write_predictions(output, {"1": {"answer": "sai schema"}})  # type: ignore[dict-item]


    def test_build_index_rejects_missing_train_answers_before_creating_db(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contexts = root / "contexts"
            contexts.mkdir()
            (contexts / "context_1.json").write_text(
                json.dumps({"id": "ctx", "name": "Doc", "link": "", "passage": ""}),
                encoding="utf-8",
            )
            train = root / "train.json"
            train.write_text(
                json.dumps({"1": {"question": "Question", "answer": None}}),
                encoding="utf-8",
            )
            database = root / "index.sqlite"
            with self.assertRaisesRegex(ValueError, "thiếu answer"):
                build_index(contexts, train, database)
            self.assertFalse(database.exists())

    def test_failed_force_rebuild_preserves_previous_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contexts = root / "contexts"
            contexts.mkdir()
            context_path = contexts / "context_1.json"
            context_path.write_text(
                json.dumps(
                    {
                        "id": "old",
                        "name": "Old document",
                        "link": "",
                        "passage": "oldmarker legal text " * 10,
                    }
                ),
                encoding="utf-8",
            )
            train = root / "train.json"
            train.write_text(
                json.dumps({"1": {"question": "Question", "answer": "Answer"}}),
                encoding="utf-8",
            )
            database = root / "index.sqlite"
            build_index(contexts, train, database)

            with closing(sqlite3.connect(database)) as connection:
                old_metadata = dict(connection.execute("SELECT key, value FROM metadata"))

            replacement_contexts = [
                {
                    "id": f"new-{index}",
                    "name": "New document",
                    "link": "",
                    "passage": "newmarker replacement text " * 10,
                }
                for index in range(250)
            ]
            replacement_contexts.append([])  # type: ignore[arg-type]

            with (
                patch(
                    "legalqa_baseline.storage.iter_contexts",
                    return_value=iter(replacement_contexts),
                ),
                self.assertRaises(AttributeError),
            ):
                build_index(contexts, train, database, force=True)

            with closing(sqlite3.connect(database)) as connection:
                self.assertEqual(
                    dict(connection.execute("SELECT key, value FROM metadata")),
                    old_metadata,
                )
                old_rows = connection.execute(
                    "SELECT context_id FROM contexts_fts WHERE contexts_fts MATCH ?",
                    ("oldmarker",),
                ).fetchall()
                new_rows = connection.execute(
                    "SELECT context_id FROM contexts_fts WHERE contexts_fts MATCH ?",
                    ("newmarker",),
                ).fetchall()
            self.assertEqual(old_rows, [("old",)])
            self.assertEqual(new_rows, [])
            self.assertEqual(list(root.glob(f".{database.name}.*.building*")), [])

    def test_successful_force_rebuild_atomically_replaces_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contexts = root / "contexts"
            contexts.mkdir()
            context_path = contexts / "context_1.json"
            train = root / "train.json"
            database = root / "index.sqlite"
            train.write_text(
                json.dumps({"1": {"question": "Question", "answer": "Answer"}}),
                encoding="utf-8",
            )

            def write_context(context_id: str, marker: str) -> None:
                context_path.write_text(
                    json.dumps(
                        {
                            "id": context_id,
                            "name": "Document",
                            "link": "",
                            "passage": f"{marker} legal text " * 10,
                        }
                    ),
                    encoding="utf-8",
                )

            write_context("old", "oldmarker")
            build_index(contexts, train, database)
            write_context("new", "newmarker")
            build_index(contexts, train, database, force=True)

            with closing(sqlite3.connect(database)) as connection:
                rows = connection.execute(
                    "SELECT context_id FROM contexts_fts WHERE contexts_fts MATCH ?",
                    ("newmarker",),
                ).fetchall()
                old_rows = connection.execute(
                    "SELECT context_id FROM contexts_fts WHERE contexts_fts MATCH ?",
                    ("oldmarker",),
                ).fetchall()
            self.assertEqual(rows, [("new",)])
            self.assertEqual(old_rows, [])

    def test_build_index_refuses_orphan_target_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contexts = root / "contexts"
            contexts.mkdir()
            (contexts / "context_1.json").write_text(
                json.dumps(
                    {
                        "id": "new",
                        "name": "New document",
                        "link": "",
                        "passage": "newmarker legal text " * 10,
                    }
                ),
                encoding="utf-8",
            )
            train = root / "train.json"
            train.write_text(
                json.dumps({"1": {"question": "Question", "answer": "Answer"}}),
                encoding="utf-8",
            )
            database = root / "index.sqlite"
            orphan_journal = Path(f"{database}-journal")
            orphan_contents = b"orphan rollback journal"
            orphan_journal.write_bytes(orphan_contents)

            with self.assertRaisesRegex(RuntimeError, "sidecar"):
                build_index(contexts, train, database)

            self.assertFalse(database.exists())
            self.assertEqual(orphan_journal.read_bytes(), orphan_contents)
            self.assertEqual(list(root.glob(f".{database.name}.*.building*")), [])

    def test_iter_contexts_contract_parity_between_dir_and_zip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contexts_dir = root / "contexts_dir"
            contexts_dir.mkdir()
            zip_path = root / "contexts.zip"

            doc1 = {"id": "1", "name": "Doc 1", "passage": "Nội dung điều 1 và điều khoản luật 1." * 5}
            doc2 = {"id": "2", "name": "Doc 2", "passage": "Nội dung điều 2 và quy định chung 2." * 5}
            manifest = {"version": "1.0", "dataset": "legalqa", "description": "metadata file"}
            extra_info = {"note": "extra metadata file"}

            # Ghi ra thư mục: gồm context_1.json, context_2.json, manifest.json, dataset_info.json, .hidden.json
            (contexts_dir / "context_1.json").write_text(json.dumps(doc1), encoding="utf-8")
            (contexts_dir / "context_2.json").write_text(json.dumps(doc2), encoding="utf-8")
            (contexts_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (contexts_dir / "dataset_info.json").write_text(json.dumps(extra_info), encoding="utf-8")
            (contexts_dir / ".hidden_context.json").write_text(json.dumps(doc1), encoding="utf-8")

            # Đóng gói zip: gồm context_1, context_2, manifest, dataset_info, và file macOS __MACOSX
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("selected-contexts/context_2.json", json.dumps(doc2))
                zf.writestr("selected-contexts/context_1.json", json.dumps(doc1))
                zf.writestr("manifest.json", json.dumps(manifest))
                zf.writestr("selected-contexts/dataset_info.json", json.dumps(extra_info))
                zf.writestr("__MACOSX/selected-contexts/._context_1.json", b"junk macos metadata")
                zf.writestr("selected-contexts/.hidden_context.json", json.dumps(doc1))

            dir_docs = list(iter_contexts(contexts_dir))
            zip_docs = list(iter_contexts(zip_path))

            # Cả hai nguồn phải đọc đúng 2 contexts, bỏ qua manifest, metadata và shadow files
            self.assertEqual(len(dir_docs), 2)
            self.assertEqual(len(zip_docs), 2)
            self.assertEqual([d["id"] for d in dir_docs], ["1", "2"])
            self.assertEqual([d["id"] for d in zip_docs], ["1", "2"])

            # Kiểm tra build_index tạo ra cùng corpus_sha256 từ cả 2 nguồn
            train_path = root / "train.json"
            train_path.write_text(json.dumps({"1": {"question": "Q?", "answer": "A."}}), encoding="utf-8")

            db_dir = root / "from_dir.sqlite"
            db_zip = root / "from_zip.sqlite"
            build_index(contexts_dir, train_path, db_dir)
            build_index(zip_path, train_path, db_zip)

            with SearchIndex(db_dir) as idx_dir, SearchIndex(db_zip) as idx_zip:
                meta_dir = idx_dir.metadata()
                meta_zip = idx_zip.metadata()
                self.assertEqual(meta_dir["documents"], meta_zip["documents"])
                self.assertEqual(meta_dir["chunks"], meta_zip["chunks"])
                self.assertEqual(meta_dir["corpus_sha256"], meta_zip["corpus_sha256"])

    def test_load_qa_rejects_blank_or_whitespace_question(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bad_empty = root / "empty_q.json"
            bad_whitespace = root / "whitespace_q.json"
            valid_qa = root / "valid_qa.json"

            bad_empty.write_text(json.dumps({"1": {"question": "", "answer": "Ans"}}), encoding="utf-8")
            bad_whitespace.write_text(json.dumps({"2": {"question": "   \n\t  ", "answer": "Ans"}}), encoding="utf-8")
            valid_qa.write_text(json.dumps({"3": {"question": "Câu hỏi hợp lệ?", "answer": "Ans"}}), encoding="utf-8")

            with self.assertRaises(ValueError):
                load_qa(bad_empty)
            with self.assertRaises(ValueError):
                load_qa(bad_whitespace)

            loaded = load_qa(valid_qa)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded["3"]["question"], "Câu hỏi hợp lệ?")


if __name__ == "__main__":
    unittest.main()
