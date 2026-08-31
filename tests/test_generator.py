from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any

from legalqa_baseline.generator import (
    RAG_TEMPLATE,
    SYSTEM_PROMPT,
    build_chat_messages,
    build_user_prompt,
    format_raw_qwen_prompt,
)
from legalqa_baseline.pipeline import LegalQABaseline, Prediction


class MockGenerator:
    def __init__(self) -> None:
        self.last_context: str | None = None
        self.last_question: str | None = None

    def generate(self, context: str, question: str) -> str:
        self.last_context = context
        self.last_question = question
        return f"Câu trả lời được sinh từ model cho câu hỏi: {question}"


class MockSearchIndex:
    def search_contexts(self, question: str, top_k: int = 12) -> list[dict[str, Any]]:
        return [
            {
                "context_id": "100",
                "chunk_no": 0,
                "name": "Luật Xử lý vi phạm hành chính",
                "link": "https://example.com/law",
                "text": "Điều 12 quy định mức xử phạt từ 2 triệu đến 5 triệu đồng.",
                "bm25_score": -15.5,
            },
            {
                "context_id": "101",
                "chunk_no": 1,
                "name": "Nghị định hướng dẫn",
                "link": "https://example.com/decree",
                "text": "Khoản 2 Điều 5 quy định về thẩm quyền xử phạt.",
                "bm25_score": -12.0,
            },
        ]

    def search_train(self, question: str, top_k: int = 5, exclude_id: str | None = None) -> list[dict[str, Any]]:
        return [
            {
                "sample_id": "999",
                "question": "Mức phạt vi phạm hành chính?",
                "answer": "Mức phạt từ 2 đến 5 triệu đồng theo Điều 12.",
                "bm25_score": -10.0,
            }
        ]


class GeneratorPromptTests(unittest.TestCase):
    def test_prompt_constants(self) -> None:
        self.assertIn("trợ lý hỏi đáp pháp luật tiếng Việt", SYSTEM_PROMPT)
        self.assertIn("### Ngữ cảnh:", RAG_TEMPLATE)
        self.assertIn("### Câu hỏi:", RAG_TEMPLATE)
        self.assertIn("### Trả lời:", RAG_TEMPLATE)

    def test_build_user_prompt(self) -> None:
        context = "Điều 1. Quy định chung..."
        question = "Quy định chung là gì?"
        prompt = build_user_prompt(context, question)
        self.assertIn(context, prompt)
        self.assertIn(question, prompt)
        self.assertTrue(prompt.endswith("### Trả lời:"))

    def test_build_chat_messages(self) -> None:
        context = "Ngữ cảnh mẫu"
        question = "Câu hỏi mẫu"
        messages = build_chat_messages(context, question)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], SYSTEM_PROMPT)
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn(question, messages[1]["content"])

    def test_format_raw_qwen_prompt(self) -> None:
        context = "Ngữ cảnh luật"
        question = "Ai có thẩm quyền?"
        raw = format_raw_qwen_prompt(context, question)
        self.assertIn("<|im_start|>system\n", raw)
        self.assertIn("<|im_end|>\n<|im_start|>user\n", raw)
        self.assertIn("<|im_end|>\n<|im_start|>assistant\n", raw)


class RAGPipelineTests(unittest.TestCase):
    def test_pipeline_rag_mode(self) -> None:
        mock_index = MockSearchIndex()
        mock_gen = MockGenerator()
        pipeline = LegalQABaseline(
            index=mock_index,  # type: ignore[arg-type]
            generator=mock_gen,
            context_top_k=2,
        )

        pred = pipeline.predict_one("Mức xử phạt Điều 12?", mode="rag")
        self.assertEqual(pred.route, "rag")
        self.assertIn("Câu trả lời được sinh từ model", pred.answer)
        self.assertIsNotNone(mock_gen.last_context)
        self.assertIn("Luật Xử lý vi phạm hành chính", str(mock_gen.last_context))
        self.assertEqual(pred.evidence["num_contexts"], 2)

    def test_pipeline_hybrid_rag_mode(self) -> None:
        mock_index = MockSearchIndex()
        mock_gen = MockGenerator()
        pipeline = LegalQABaseline(
            index=mock_index,  # type: ignore[arg-type]
            generator=mock_gen,
            knn_threshold=0.99,  # Ngưỡng cao nên sẽ chuyển sang RAG
        )

        pred = pipeline.predict_one("Câu hỏi ngẫu nhiên không trùng", mode="hybrid_rag")
        self.assertEqual(pred.route, "rag")


    def test_pipeline_rejects_empty_generator_answer(self) -> None:
        class EmptyGenerator:
            def generate(self, context: str, question: str) -> None:
                return None

        pipeline = LegalQABaseline(
            index=MockSearchIndex(),  # type: ignore[arg-type]
            generator=EmptyGenerator(),
        )
        with self.assertRaisesRegex(RuntimeError, "Generator trả về answer"):
            pipeline.predict_one("Mức phạt?", mode="rag")


if __name__ == "__main__":
    unittest.main()
