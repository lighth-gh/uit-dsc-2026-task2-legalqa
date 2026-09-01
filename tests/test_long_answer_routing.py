from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from legalqa_baseline.pipeline import LegalQABaseline
from legalqa_baseline.storage import SearchIndex, build_index
from legalqa_baseline.text import (
    LONG_ANSWER_PATTERNS,
    is_long_answer_question,
    merge_adjacent_chunks,
)


class DummyGenerator:
    def __init__(self) -> None:
        self.called_count = 0

    def generate(self, context: str, question: str) -> str:
        self.called_count += 1
        return f"LLM Answer for {question}"


class DummyReranker:
    def rerank(
        self,
        question: str,
        candidates: list[dict],
        top_k: int = 3,
        max_length: int = 2304,
    ) -> list[dict]:
        return candidates[:top_k]


class TestLongAnswerPatterns(unittest.TestCase):
    def test_patterns_presence(self) -> None:
        expected = [
            "mẫu", "biểu mẫu", "phụ lục",
            "liệt kê", "bao gồm những",
            "nội dung điều", "nội dung khoản",
            "các trường hợp", "điều kiện",
            "hồ sơ gồm", "quyền và nghĩa vụ",
            "điều luật", "nguyên văn", "toàn văn", "quy định đầy đủ",
        ]
        for pat in expected:
            self.assertIn(pat, LONG_ANSWER_PATTERNS)

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
            "Thời hạn nộp hồ sơ là bao nhiêu ngày?",
            "Mức phạt tối đa đối với cá nhân là bao nhiêu?",
            "",
            "   ",
        ]
        for q in negatives:
            with self.subTest(question=q):
                self.assertFalse(is_long_answer_question(q))

    def test_is_long_answer_non_string(self) -> None:
        self.assertFalse(is_long_answer_question(None))  # type: ignore[arg-type]
        self.assertFalse(is_long_answer_question(123))  # type: ignore[arg-type]


class TestMergeAdjacentChunks(unittest.TestCase):
    def test_raw_chunk_merge_has_no_800_word_limit(self) -> None:
        chunks = [
            {
                "context_id": "doc",
                "chunk_no": number,
                "text": " ".join(f"chunk{number}_word{word}" for word in range(400)),
            }
            for number in range(3)
        ]

        merged = LegalQABaseline._merge_raw_chunks(chunks)

        self.assertEqual(len(merged.split()), 1200)
        self.assertIn("chunk0_word0", merged)
        self.assertIn("chunk2_word399", merged)

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
        self.assertEqual(len(words), 7)
        self.assertEqual(" ".join(words), "w1 w2 w3 w4 w5 w6 w7")

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
            self.assertEqual(pred.route, "rag")
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
            self.assertEqual(pred.route, "rag")
            self.assertEqual(generator.called_count, 1)


if __name__ == "__main__":
    unittest.main()
