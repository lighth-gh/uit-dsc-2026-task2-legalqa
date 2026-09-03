from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from legalqa_baseline.pipeline import LegalQABaseline
from legalqa_baseline.storage import SearchIndex, build_index
from legalqa_baseline.text import (
    LONG_ANSWER_PATTERNS,
    build_extractive_answer,
    build_focused_extractive_answer,
    clean_answer,
    deduplicate_overlaps,
    is_heading_only_answer,
    is_long_answer_question,
    is_long_form_question,
    is_structured_extractive_question,
    merge_adjacent_chunks,
    needs_extended_generation_retry,
    output_artifact_flags,
    possibly_cut,
)


class DummyGenerator:
    def __init__(self) -> None:
        self.called_count = 0

    def generate(self, context: str, question: str) -> str:
        self.called_count += 1
        return f"LLM Answer for {question}"


class DummyReranker:
    def __init__(self, score: float = 3.0) -> None:
        self.score = score

    def rerank(
        self,
        question: str,
        candidates: list[dict],
        top_k: int = 3,
        max_length: int = 2304,
    ) -> list[dict]:
        output = []
        for candidate in candidates[:top_k]:
            item = dict(candidate)
            item["rerank_score"] = self.score
            output.append(item)
        return output


class TestLongAnswerPatterns(unittest.TestCase):
    def test_patterns_presence(self) -> None:
        expected = [
            "thủ tục", "trình tự", "hồ sơ", "bao gồm", "các trường hợp",
            "các bước", "điều kiện", "quyền và nghĩa vụ", "trách nhiệm",
            "nội dung và phương pháp", "biểu mẫu",
        ]
        for pat in expected:
            with self.subTest(pattern=pat):
                self.assertTrue(is_long_answer_question(f"Cho biết {pat} theo quy định?"))

    def test_is_long_answer_question_positive(self) -> None:
        test_cases = [
            "Hãy cho biết mẫu số 01 quy định như thế nào?",
            "Biểu mẫu báo cáo tình hình sử dụng lao động gồm những gì?",
            "Quy định tại Phụ Lục II ban hành kèm theo?",
            "Hãy LIỆT KÊ các trường hợp miễn thuế nhập khẩu.",
            "Hồ sơ gồm những giấy tờ tài liệu gì theo quy định?",
            "Nội dung Điều 15 Luật Doanh nghiệp gồm những gì?",
            "Nội dung khoản 2 Điều 10 quy định về vấn đề gì?",
            "Các trường hợp nào bị thu hồi giấy phép?",
            "Điều kiện để được hưởng trợ cấp thất nghiệp là gì?",
            "Quyền và nghĩa vụ của người lao động được quy định ra sao?",
            "Đối tượng bao gồm những ai?",
            "Hãy cung cấp nguyên văn Điều 15 của Luật Doanh nghiệp.",
            "Toàn văn điều luật này quy định thế nào?",
        ]
        for q in test_cases:
            with self.subTest(question=q):
                self.assertTrue(is_long_answer_question(q))

    def test_is_long_answer_question_negative(self) -> None:
        negatives = [
            "Ai là người đại diện theo pháp luật?",
            "Thời hạn nộp đơn là bao nhiêu ngày?",
            "Mức phạt tối đa đối với cá nhân là bao nhiêu?",
            "",
            "   ",
        ]
        for q in negatives:
            with self.subTest(question=q):
                self.assertFalse(is_long_answer_question(q))

    def test_smoke30_structured_fallbacks_route_directly_to_extractive(self) -> None:
        structured_questions = {
            "97935": (
                "Việc cho vay vốn ưu đãi để mua nhà ở xã hội thông qua Ngân hàng "
                "Chính sách xã hội được thực hiện như thế nào?"
            ),
            "83665": (
                "Hàng dự trữ quốc gia trong quá trình nhập kho, xuất kho và lưu kho "
                "phải tuân thủ các yêu cầu thế nào?"
            ),
            "63093": (
                "Khi vi phạm pháp luật về đất đai sẽ bị xử phạt hành chính bằng "
                "các hình thức nào?"
            ),
            "76135": (
                "Người học ngành bảo vệ thực vật trình độ cao đẳng sau khi tốt "
                "nghiệp có thể làm những công việc nào?"
            ),
            "94131": "Tiêu chuẩn của công dân để được tham gia nghĩa vụ quân sự",
            "120127": "Thời gian hưởng chế độ ốm đau theo quy định pháp luật",
        }
        for sample_id, question in structured_questions.items():
            with self.subTest(sample_id=sample_id):
                self.assertTrue(is_long_form_question(question))

    def test_smoke30_short_synthesis_questions_stay_on_llm_512(self) -> None:
        short_questions = {
            "62147": (
                "Bộ Nội vụ đang lấy ý kiến Nghị định quy định tăng mức lương cơ sở "
                "đối với CB, CC, VC và lực lượng vũ trang lên 1,8 triệu đồng?"
            ),
            "103999": (
                "Chi cục Thuế gửi cho từng hộ khoán thông báo về việc dự kiến doanh "
                "thu, mức thuế khoán khi nào?"
            ),
            "59823": "Nhà nước khuyến khích kinh tế tuần hoàn ra sao?",
        }
        for sample_id, question in short_questions.items():
            with self.subTest(sample_id=sample_id):
                self.assertFalse(is_long_form_question(question))

    def test_is_long_answer_non_string(self) -> None:
        self.assertFalse(is_long_answer_question(None))  # type: ignore[arg-type]
        self.assertFalse(is_long_answer_question(123))  # type: ignore[arg-type]
        self.assertTrue(is_long_form_question("Trình tự gồm các bước nào?"))

    def test_synthesis_question_is_not_direct_extractive(self) -> None:
        self.assertTrue(is_structured_extractive_question("Hồ sơ gồm những gì?"))
        self.assertFalse(
            is_structured_extractive_question(
                "Hãy phân tích và so sánh điều kiện của hai thủ tục này."
            )
        )
        self.assertTrue(
            needs_extended_generation_retry("Hãy liệt kê danh sách hồ sơ cần nộp.")
        )
        self.assertFalse(
            needs_extended_generation_retry("Hồ sơ này có hợp lệ không?")
        )

    def test_release_smoke_token_limit_questions_are_retry_eligible(self) -> None:
        questions = {
            "63093": (
                "Khi vi phạm pháp luật về đất đai sẽ bị xử phạt hành chính "
                "bằng các hình thức nào?"
            ),
            "55463": (
                "Thực hiện báo cáo phương tiện phòng cháy chữa cháy như thế nào "
                "và tại đâu?"
            ),
            "67397": (
                "Những quy định chung về kỹ thuật đối với quá trình xây dựng, "
                "khai thác và sử dụng công trình tàu điện ngầm là gì?"
            ),
            "138443": (
                "Các biện pháp dự phòng cho nhân viên y tế để tránh tình trạng "
                "lây nhiễm COVID-19 như thế nào?"
            ),
        }
        for sample_id, question in questions.items():
            with self.subTest(sample_id=sample_id):
                self.assertTrue(needs_extended_generation_retry(question))

    def test_validation100_list_and_process_token_limits_are_retry_eligible(self) -> None:
        questions = {
            "46443": "Thời gian học tập và chương trình đào tạo hệ đại học được quy định như thế nào?",
            "130171": "Viên chức Cảng vụ hàng không nữ sẽ có trang phục như thế nào?",
            "104693": "Trách nhiệm của Hội đồng tư vấn thuế được đề cập ra sao?",
            "12451": "Lực lượng cảnh sát đường thủy thực hiện tuần tra, kiểm soát như thế nào?",
            "36675": "Nguyên tắc phân vị trí các loại đất nông nghiệp, phi nông nghiệp?",
            "45085": "Sản lượng sản xuất và nhập khẩu thuốc lá được giới hạn thế nào?",
            "62387": "Điều kiện cấp chứng chỉ thiết kế, giám sát công trình giao thông?",
            "140001": "Cán bộ kiểm tra có nhiệm vụ và quyền hạn như thế nào?",
            "77119": "Hệ thống định mức xây dựng gồm những định mức nào?",
            "75715": "Quy trình cưỡng chế được thực hiện theo mấy bước?",
            "75603": "Một số quy định khác gồm những gì?",
            "5017": "Hồ sơ được quy định như thế nào?",
        }
        for sample_id, question in questions.items():
            with self.subTest(sample_id=sample_id):
                self.assertTrue(needs_extended_generation_retry(question))

    def test_scalar_and_yes_no_questions_do_not_request_larger_budget(self) -> None:
        questions = (
            "Nếu không ghi nhãn hàng hóa thì bị xử phạt bao nhiêu tiền?",
            "Mức đóng BHXH tự nguyện được tính ra sao?",
            "Ai là người có nghĩa vụ nộp thuế tài nguyên?",
            "Hồ sơ này có hợp lệ không?",
            "Điều kiện này có được áp dụng hay không?",
        )
        for question in questions:
            with self.subTest(question=question):
                self.assertFalse(needs_extended_generation_retry(question))


class TestMergeAdjacentChunks(unittest.TestCase):
    def test_prompt_context_order_is_best_previous_next_then_other_docs(self) -> None:
        best = {"context_id": "doc", "chunk_no": 5, "text": "best"}
        previous = {"context_id": "doc", "chunk_no": 4, "text": "previous"}
        following = {"context_id": "doc", "chunk_no": 6, "text": "following"}
        other = {"context_id": "other", "chunk_no": 2, "text": "other"}

        ordered = LegalQABaseline._prioritize_prompt_chunks(
            [best, previous, best, following],
            [best, other],
        )

        self.assertEqual(
            [(item["context_id"], item["chunk_no"]) for item in ordered],
            [("doc", 5), ("doc", 4), ("doc", 6), ("other", 2)],
        )

    def test_raw_chunk_merge_respects_configured_word_limit(self) -> None:
        chunks = [
            {
                "context_id": "doc",
                "chunk_no": number,
                "text": " ".join(
                    f"chunk{number}_word{word}{'.' if word % 20 == 19 else ''}"
                    for word in range(400)
                ),
            }
            for number in range(3)
        ]
        pipeline = object.__new__(LegalQABaseline)
        pipeline.max_long_answer_words = 640
        merged = pipeline._merge_raw_chunks(chunks)

        self.assertLessEqual(len(merged.split()), 640)
        self.assertIn("chunk0_word0", merged)
        self.assertFalse(possibly_cut(merged))

    def test_merge_ordering_and_dedup(self) -> None:
        chunks = [
            {"chunk_no": 2, "text": "Đoạn 3 kết thúc."},
            {"chunk_no": 0, "text": "Đoạn 1 mở đầu."},
            {"chunk_no": 1, "text": "Đoạn 2 tiếp theo."},
        ]
        merged = merge_adjacent_chunks(chunks, max_words=100)
        self.assertEqual(
            merged, "Đoạn 1 mở đầu.\n\nĐoạn 2 tiếp theo.\n\nĐoạn 3 kết thúc."
        )

    def test_merge_max_words_truncation(self) -> None:
        chunks = [
            {"chunk_no": 0, "text": "w1 w2 w3 w4 w5"},
            {"chunk_no": 1, "text": "w6 w7 w8 w9 w10"},
        ]
        merged = merge_adjacent_chunks(chunks, max_words=7)
        words = merged.split()
        self.assertEqual(len(words), 5)
        self.assertEqual(" ".join(words), "w1 w2 w3 w4 w5")

    def test_deduplicate_configured_word_overlap(self) -> None:
        merged = deduplicate_overlaps(
            [
                "Mục 1. a b c d e",
                "c d e Mục 2. f g",
                "Mục 2. f g Mục 3. h",
            ]
        )
        self.assertEqual(merged.count("c d e"), 1)
        self.assertEqual(merged.count("Mục 2. f g"), 1)
        self.assertIn("Mục 3. h", merged)

    def test_build_extractive_answer_accepts_chunk_index_alias(self) -> None:
        answer = build_extractive_answer(
            [
                {"chunk_index": 2, "text": "c d e Mục 3."},
                {"chunk_index": 1, "text": "Mục 2. a b c d e"},
            ]
        )
        self.assertEqual(answer, "Mục 2. a b c d e\n\nMục 3.")

    def test_id_80189_raw_answer_starts_at_matching_form_title(self) -> None:
        chunks = [
            {
                "chunk_no": 20,
                "text": (
                    "PHỤ LỤC I THÔNG BÁO THAY ĐỔI NHÂN SỰ. "
                    "Danh mục có Mẫu thông báo thay đổi người đại diện theo pháp luật. "
                    "Nội dung của biểu mẫu nhân sự không liên quan."
                ),
            },
            {
                "chunk_no": 21,
                "text": (
                    "MẪU THÔNG BÁO THAY ĐỔI NGƯỜI ĐẠI DIỆN THEO PHÁP LUẬT "
                    "Tên doanh nghiệp: ... Kính gửi: Phòng Đăng ký kinh doanh."
                ),
            },
            {
                "chunk_no": 22,
                "text": "Họ và tên người đại diện mới: ... Chữ ký: ...",
            },
        ]

        answer = build_focused_extractive_answer(
            "Mẫu thông báo thay đổi người đại diện theo pháp luật",
            chunks,
            best_chunk_no=20,
        )

        self.assertTrue(
            answer.startswith(
                "MẪU THÔNG BÁO THAY ĐỔI NGƯỜI ĐẠI DIỆN THEO PHÁP LUẬT"
            )
        )
        self.assertNotIn("THÔNG BÁO THAY ĐỔI NHÂN SỰ", answer)
        self.assertIn("Họ và tên người đại diện mới", answer)

    def test_focused_raw_fallback_does_not_blindly_join_both_neighbours(self) -> None:
        chunks = [
            {"chunk_no": 4, "text": "Nội dung chunk trước."},
            {"chunk_no": 5, "text": "Nội dung chunk tốt nhất."},
            {"chunk_no": 6, "text": "Nội dung chunk sau."},
        ]
        answer = build_focused_extractive_answer(
            "Các trường hợp được áp dụng gồm những gì?",
            chunks,
            best_chunk_no=5,
        )

        self.assertIn("chunk tốt nhất", answer)
        self.assertNotIn("chunk trước", answer)
        self.assertNotIn("chunk sau", answer)

    def test_focused_raw_answer_can_start_at_numbered_article(self) -> None:
        chunks = [
            {"chunk_no": 8, "text": "Điều 11. Quy định không liên quan."},
            {
                "chunk_no": 9,
                "text": "Điều 12. Hồ sơ gồm đơn đề nghị và bản sao giấy tờ.",
            },
            {"chunk_no": 10, "text": "1. Đơn đề nghị. 2. Bản sao giấy tờ."},
        ]
        answer = build_focused_extractive_answer(
            "Nội dung Điều 12 quy định gì?",
            chunks,
            best_chunk_no=8,
        )

        self.assertTrue(answer.startswith("Điều 12"))
        self.assertNotIn("Điều 11", answer)

    def test_id_31969_heading_only_expands_to_refund_conditions(self) -> None:
        chunks = [
            {"chunk_no": 88, "text": "THU NHẬP TÍNH THUẾ"},
            {
                "chunk_no": 89,
                "text": (
                    "Điều 28. Hoàn thuế. 1. Việc hoàn thuế thu nhập cá nhân áp dụng "
                    "đối với cá nhân đã đăng ký và có mã số thuế tại thời điểm nộp hồ sơ. "
                    "2. Cá nhân được hoàn số thuế đã nộp thừa theo đúng quy định pháp luật."
                ),
            },
        ]
        answer = build_focused_extractive_answer(
            "Điều kiện hoàn thuế thu nhập cá nhân đối với người lao động chưa đến mức phải nộp thuế là gì?",
            chunks,
            best_chunk_no=88,
        )
        self.assertFalse(is_heading_only_answer(answer))
        self.assertIn("mã số thuế", answer)

    def test_id_123257_heading_only_expands_to_classification_conditions(self) -> None:
        chunks = [
            {"chunk_no": 16, "text": "XẾP LOẠI"},
            {
                "chunk_no": 17,
                "text": (
                    "Điều 13. Tiêu chuẩn xếp loại học kỳ và cả năm học. "
                    "Loại trung bình nếu có đủ các tiêu chuẩn sau đây: điểm trung bình "
                    "các môn học từ 5,0 trở lên và không có môn học nào dưới mức tối thiểu. "
                    "Hạnh kiểm phải được xếp từ loại trung bình trở lên."
                ),
            },
        ]
        answer = build_focused_extractive_answer(
            "Điều kiện để học sinh THPT được xếp loại trung bình",
            chunks,
            best_chunk_no=16,
        )
        self.assertFalse(is_heading_only_answer(answer))
        self.assertIn("Loại trung bình", answer)

    def test_id_35853_stops_before_unrelated_export_section(self) -> None:
        chunks = [
            {
                "chunk_no": 44,
                "text": (
                    "Điều 42. Điều kiện buôn bán phân bón. 1. Tổ chức, cá nhân buôn bán "
                    "phân bón phải có cửa hàng, địa điểm giao dịch hợp pháp. 2. Người trực "
                    "tiếp buôn bán phải được tập huấn chuyên môn về phân bón. "
                    "Mục 3. XUẤT KHẨU VÀ NHẬP KHẨU PHÂN BÓN."
                ),
            },
            {"chunk_no": 45, "text": "Điều 43. Xuất khẩu phân bón thực hiện theo pháp luật ngoại thương."},
        ]
        answer = build_focused_extractive_answer(
            "Điều kiện buôn bán phân bón",
            chunks,
            best_chunk_no=44,
        )
        self.assertIn("địa điểm giao dịch hợp pháp", answer)
        self.assertNotIn("XUẤT KHẨU", answer)
        self.assertNotIn("Điều 43", answer)

    def test_id_129215_starts_at_method_heading_not_mid_procedure(self) -> None:
        chunks = [
            {
                "chunk_no": 8,
                "text": (
                    "Phần cuối của quy trình nuôi cấy tế bào không liên quan. "
                    "E.1. Phương pháp ELISA chẩn đoán hội chứng rối loạn sinh sản và "
                    "hô hấp ở lợn. Bước 1. Chuẩn bị mẫu xét nghiệm và phiến phản ứng. "
                    "Bước 2. Pha loãng huyết thanh theo hướng dẫn kỹ thuật."
                ),
            },
            {
                "chunk_no": 9,
                "text": (
                    "Bước 3. Ủ phiến phản ứng trong thời gian quy định. "
                    "Bước 4. Rửa phiến và bổ sung cộng hợp. Bước 5. Đọc kết quả xét nghiệm."
                ),
            },
        ]
        answer = build_focused_extractive_answer(
            "Phương pháp ELISA dùng để chẩn đoán hội chứng rối loạn sinh sản và hô hấp ở lợn có bao nhiêu bước thực hiện?",
            chunks,
            best_chunk_no=9,
        )
        self.assertTrue(answer.startswith("E.1. Phương pháp ELISA"))
        self.assertNotIn("nuôi cấy tế bào", answer)
        self.assertIn("Bước 1", answer)

    def test_long_extractive_is_bounded_and_ends_at_sentence(self) -> None:
        text = " ".join(
            f"Từ thứ {number} thuộc nội dung điều kiện này{'.' if number % 18 == 17 else ''}"
            for number in range(260)
        )
        answer = build_focused_extractive_answer(
            "Các điều kiện này bao gồm nội dung gì?",
            [{"chunk_no": 1, "text": text}],
            best_chunk_no=1,
            max_words=620,
        )
        self.assertLessEqual(len(answer.split()), 620)
        self.assertTrue(answer.endswith("."))
        self.assertFalse(possibly_cut(answer))

    def test_clean_answer_only_removes_prefix(self) -> None:
        answer = "Dựa trên ngữ cảnh được cung cấp: Điều 1. Nội dung đầy đủ."
        self.assertEqual(clean_answer(answer), "Điều 1. Nội dung đầy đủ.")

    def test_clean_answer_removes_markdown_opening_slug_and_fake_law_number(self) -> None:
        answer = (
            'Câu trả lời cho câu hỏi "Ai có thẩm quyền?" là:\n\n'
            "**Điều 10. Thủ tướng Chính phủ có thẩm quyền.**\n"
            "Nguồn: Nghi-dinh-74-2015-ND-CP-ve-phong-khong-nhan-dan-289989.\n"
            "Nội dung này thuộc Luật An toàn, vệ sinh lao động số 281961."
        )

        cleaned = clean_answer(answer)

        self.assertTrue(cleaned.startswith("Điều 10."))
        self.assertNotIn("**", cleaned)
        self.assertNotIn("Nghi-dinh-", cleaned)
        self.assertNotIn("281961", cleaned)
        self.assertEqual(output_artifact_flags(cleaned), set())

    def test_clean_answer_only_keeps_document_number_from_trusted_metadata(self) -> None:
        answer = (
            "Theo **Nghị định 74/2015/NĐ-CP** và Nghị định 90/2017/NĐ-CP, "
            "Quyết định 1500/QĐ-BTC-2020 cũng được nhắc đến."
        )
        metadata = (
            "Nghi-dinh-74-2015-ND-CP-ve-phong-khong-nhan-dan-289989",
        )

        cleaned = clean_answer(answer, trusted_metadata=metadata)

        self.assertIn("Nghị định 74/2015/NĐ-CP", cleaned)
        self.assertNotIn("90/2017/NĐ-CP", cleaned)
        self.assertNotIn("1500/QĐ-BTC-2020", cleaned)
        self.assertEqual(output_artifact_flags(cleaned), set())

    def test_clean_answer_validates_schema_and_non_empty_result(self) -> None:
        with self.assertRaisesRegex(TypeError, "answer"):
            clean_answer({"answer": "không hợp lệ"})  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "rỗng"):
            clean_answer(" ** ** ")

    def test_clean_answer_removes_prompt_id_but_keeps_real_page_reference(self) -> None:
        cleaned = clean_answer("Xem nội dung tại trang 15. (Văn bản 1)")
        self.assertEqual(cleaned, "Xem nội dung tại trang 15.")
        self.assertEqual(output_artifact_flags(cleaned), set())

    def test_possibly_cut_detects_dangling_generation(self) -> None:
        self.assertTrue(possibly_cut("Hồ sơ bao gồm các giấy tờ sau:"))
        self.assertTrue(possibly_cut("Người nộp chuẩn bị đơn đề nghị và"))
        self.assertFalse(possibly_cut("Hồ sơ được nộp trong 03 ngày."))

    def test_merge_empty_and_invalid(self) -> None:
        self.assertEqual(merge_adjacent_chunks([]), "")
        self.assertEqual(merge_adjacent_chunks([{"chunk_no": 0, "text": ""}]), "")
        with self.assertRaises(ValueError):
            merge_adjacent_chunks([{"chunk_no": 0, "text": "abc"}], max_words=0)


class TestSearchIndexGetContextChunks(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test.db"
        contexts_dir = Path(self.temp_dir.name) / "contexts"
        contexts_dir.mkdir(parents=True)
        train_file = Path(self.temp_dir.name) / "train.json"

        ctx1 = {
            "id": "ctx_001",
            "name": "Nghị định 01/2026",
            "link": "https://example.com/01",
            "passage": (
                "Điều 1. Phạm vi điều chỉnh.\n"
                "Nghị định này quy định về hoạt động ABC.\n\n"
                "Điều 2. Đối tượng áp dụng.\n"
                "Bao gồm các cơ quan tổ chức cá nhân có liên quan.\n\n"
                "Điều 3. Hồ sơ gồm các loại giấy tờ sau đây."
            ),
        }
        import json
        with (contexts_dir / "context_001.json").open("w", encoding="utf-8") as f:
            json.dump(ctx1, f)
        with train_file.open("w", encoding="utf-8") as f:
            json.dump({"1": {"question": "Mẫu số 01 là gì?", "answer": "Đáp án mẫu 01"}}, f)

        build_index(
            contexts_path=contexts_dir,
            train_path=train_file,
            db_path=self.db_path,
            max_chunk_words=50,
            overlap_words=10,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_get_all_chunks_for_context(self) -> None:
        with SearchIndex(self.db_path) as index:
            chunks = index.get_context_chunks("ctx_001")
            self.assertGreater(len(chunks), 1)
            # Verify ordering by chunk_no
            chunk_nos = [int(c["chunk_no"]) for c in chunks]
            self.assertEqual(chunk_nos, sorted(chunk_nos))

    def test_get_specific_chunk_nos(self) -> None:
        with SearchIndex(self.db_path) as index:
            chunks = index.get_context_chunks("ctx_001", chunk_nos=[0, 1])
            self.assertEqual(len(chunks), 2)
            self.assertEqual([int(c["chunk_no"]) for c in chunks], [0, 1])

    def test_get_nonexistent_context(self) -> None:
        with SearchIndex(self.db_path) as index:
            chunks = index.get_context_chunks("ctx_999")
            self.assertEqual(chunks, [])
            empty = index.get_context_chunks("")
            self.assertEqual(empty, [])


class TestPipelineLongAnswerRouting(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test.db"
        contexts_dir = Path(self.temp_dir.name) / "contexts"
        contexts_dir.mkdir(parents=True)
        train_file = Path(self.temp_dir.name) / "train.json"

        ctx1 = {
            "id": "ctx_law",
            "name": "Luật Quy định về Biểu mẫu",
            "link": "https://example.com/law",
            "passage": (
                "Điều 10. Hồ sơ gồm có:\n"
                "1. Đơn đề nghị theo Mẫu số 01 ban hành kèm theo nghị định này.\n"
                "2. Bản sao chứng thực căn cước công dân hoặc giấy tờ tương đương.\n\n"
                "Điều 11. Trình tự và thủ tục nộp hồ sơ:\n"
                "1. Người nộp gửi trực tiếp hoặc qua bưu điện đến cơ quan có thẩm quyền.\n"
                "2. Thời hạn xử lý là 03 ngày làm việc kể từ khi nhận đủ hồ sơ hợp lệ.\n\n"
                "Điều 12. Điều kiện tiếp nhận và phê duyệt hồ sơ."
            ),
        }
        import json
        with (contexts_dir / "context_law.json").open("w", encoding="utf-8") as f:
            json.dump(ctx1, f)
        with train_file.open("w", encoding="utf-8") as f:
            json.dump({"s1": {"question": "Câu hỏi không liên quan?", "answer": "Đáp án 1"}}, f)

        build_index(
            contexts_path=contexts_dir,
            train_path=train_file,
            db_path=self.db_path,
            max_chunk_words=50,
            overlap_words=10,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_rag_routes_to_extractive_long_for_patterns(self) -> None:
        generator = DummyGenerator()
        reranker = DummyReranker()
        with SearchIndex(self.db_path) as index:
            pipeline = LegalQABaseline(
                index=index,
                generator=generator,
                reranker=reranker,
                enable_long_answer_extractive=True,
                max_long_answer_words=800,
            )
            question = "Hãy cho biết Mẫu số 01 và hồ sơ gồm những gì?"
            pred = pipeline.predict_one(question, mode="rag")
            self.assertEqual(pred.route, "extractive_long")
            # Generator must NOT have been called
            self.assertEqual(generator.called_count, 0)
            self.assertIn("Mẫu số 01", pred.answer)
            self.assertIn("merged_chunk_nos", pred.evidence)

    def test_rag_calls_generator_when_not_long_pattern(self) -> None:
        generator = DummyGenerator()
        reranker = DummyReranker()
        with SearchIndex(self.db_path) as index:
            pipeline = LegalQABaseline(
                index=index,
                generator=generator,
                reranker=reranker,
                enable_long_answer_extractive=True,
            )
            question = "Cơ quan nào có thẩm quyền giải quyết?"
            pred = pipeline.predict_one(question, mode="rag")
            self.assertEqual(pred.route, "generated_512")
            self.assertEqual(generator.called_count, 1)

    def test_disabled_long_answer_extractive(self) -> None:
        generator = DummyGenerator()
        reranker = DummyReranker()
        with SearchIndex(self.db_path) as index:
            pipeline = LegalQABaseline(
                index=index,
                generator=generator,
                reranker=reranker,
                enable_long_answer_extractive=False,
            )
            question = "Hãy liệt kê hồ sơ gồm những gì theo mẫu số 01?"
            pred = pipeline.predict_one(question, mode="rag")
            self.assertEqual(pred.route, "generated_512")
            self.assertEqual(generator.called_count, 1)

    def test_raw_reranker_below_two_uses_generator(self) -> None:
        generator = DummyGenerator()
        reranker = DummyReranker(score=1.99)
        with SearchIndex(self.db_path) as index:
            pipeline = LegalQABaseline(
                index=index,
                generator=generator,
                reranker=reranker,
                enable_long_answer_extractive=True,
            )
            pred = pipeline.predict_one(
                "Hãy cho biết Mẫu số 01 và hồ sơ gồm những gì?",
                mode="rag",
            )
            self.assertEqual(pred.route, "generated_512")
            self.assertEqual(generator.called_count, 1)


if __name__ == "__main__":
    unittest.main()
