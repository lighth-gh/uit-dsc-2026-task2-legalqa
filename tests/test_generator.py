from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from legalqa_baseline.generator import (
    GenerationTokenLimitReached,
    RAG_TEMPLATE,
    SYSTEM_PROMPT,
    ViQwenRAGGenerator,
    build_chat_messages,
    build_user_prompt,
    format_raw_qwen_prompt,
)
from legalqa_baseline.pipeline import LegalQABaseline


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


class _FakeInferenceMode:
    def __enter__(self) -> _FakeInferenceMode:
        return self

    def __exit__(self, *args: object) -> bool:
        return False


class _FakeTensor:
    def __init__(self, values: list[int]) -> None:
        self.values = list(values)

    @property
    def shape(self) -> tuple[int, int]:
        return (1, len(self.values))

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, key: int | slice) -> int | _FakeTensor:
        if isinstance(key, slice):
            return _FakeTensor(self.values[key])
        return self.values[key]

    def to(self, device: object) -> _FakeTensor:
        return self


class _FakeBatchEncoding(dict[str, _FakeTensor]):
    def __getattr__(self, name: str) -> _FakeTensor:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def to(self, device: object) -> _FakeBatchEncoding:
        for value in self.values():
            value.to(device)
        return self


class _RecordingTokenizer:
    eos_token_id = 151645
    pad_token_id = 151643
    chat_template: str | None = "configured"

    def __init__(self) -> None:
        # Match the production tokenizer setup so a regression to truncating the
        # complete serialized conversation from the left remains observable.
        self.truncation_side = "left"
        self.tokenize_calls: list[str] = []
        self.add_special_tokens_calls: list[bool] = []

    @staticmethod
    def _render_chat(messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
        prompt = "".join(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            for message in messages
        )
        if add_generation_prompt:
            prompt += "<|im_start|>assistant\n"
        return prompt

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        return_dict: bool = False,
        return_tensors: str | None = None,
        truncation: bool = False,
        max_length: int | None = None,
        **kwargs: Any,
    ) -> str | _FakeTensor | _FakeBatchEncoding:
        prompt = self._render_chat(messages, add_generation_prompt)
        if not tokenize:
            return prompt
        encoded = self(
            prompt,
            return_tensors=return_tensors,
            truncation=truncation,
            max_length=max_length,
            add_special_tokens=False,
        )
        if return_dict:
            return encoded
        return encoded["input_ids"]

    def encode(self, text: str, add_special_tokens: bool = False, **kwargs: Any) -> list[int]:
        return [ord(char) for char in text]

    def decode(
        self,
        token_ids: _FakeTensor | list[int],
        skip_special_tokens: bool = True,
        **kwargs: Any,
    ) -> str:
        values = token_ids.values if isinstance(token_ids, _FakeTensor) else token_ids
        return "".join(chr(token_id) for token_id in values)

    def __call__(
        self,
        text: str,
        *,
        return_tensors: str | None = None,
        truncation: bool = False,
        max_length: int | None = None,
        add_special_tokens: bool = True,
        **kwargs: Any,
    ) -> _FakeBatchEncoding:
        values = self.encode(text, add_special_tokens=add_special_tokens)
        if truncation and max_length is not None and len(values) > max_length:
            if self.truncation_side == "left":
                values = values[-max_length:]
            else:
                values = values[:max_length]
        retained_text = self.decode(values)
        self.tokenize_calls.append(retained_text)
        self.add_special_tokens_calls.append(add_special_tokens)
        return _FakeBatchEncoding(
            {
                "input_ids": _FakeTensor(values),
                "attention_mask": _FakeTensor([1] * len(values)),
            }
        )


class _MissingChatTemplateTokenizer(_RecordingTokenizer):
    chat_template = None

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> Any:
        raise ValueError("tokenizer.chat_template is not set")


class _MissingDefaultChatTemplateTokenizer(_RecordingTokenizer):
    chat_template = {"tool_use": "configured only for tools"}

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("apply_chat_template must not be called without a default template")


class _RecordingModel:
    def __init__(
        self,
        *,
        max_position_embeddings: int = 4096,
        eos_token_id: int | list[int] | tuple[int, ...] = (151645, 151643),
        pad_token_id: int = 151643,
    ) -> None:
        self.device = "cpu"
        self.config = SimpleNamespace(max_position_embeddings=max_position_embeddings)
        self.generation_config = SimpleNamespace(
            eos_token_id=list(eos_token_id) if isinstance(eos_token_id, tuple) else eos_token_id,
            pad_token_id=pad_token_id,
        )
        self.generate_kwargs: dict[str, Any] | None = None
        self.seen_prompt: str | None = None

    def generate(self, **kwargs: Any) -> list[list[int]]:
        self.generate_kwargs = dict(kwargs)
        input_ids = kwargs["input_ids"]
        values = input_ids.values
        self.seen_prompt = "".join(chr(token_id) for token_id in values)
        return [values + [ord(char) for char in "model answer"]]


def _fake_torch_module() -> SimpleNamespace:
    return SimpleNamespace(
        manual_seed=lambda seed: None,
        cuda=SimpleNamespace(
            is_available=lambda: False,
            manual_seed_all=lambda seed: None,
        ),
        inference_mode=lambda: _FakeInferenceMode(),
    )


class GeneratorPromptTests(unittest.TestCase):
    def test_prompt_constants(self) -> None:
        self.assertIn("trợ lý hỏi đáp pháp luật tiếng Việt", SYSTEM_PROMPT)
        self.assertIn("### Ngữ cảnh:", RAG_TEMPLATE)
        self.assertIn("### Câu hỏi:", RAG_TEMPLATE)
        self.assertIn("### Trả lời:", RAG_TEMPLATE)
        # Bỏ mọi yêu cầu ngắn gọn / tóm tắt / súc tích
        self.assertNotIn("ngắn gọn", RAG_TEMPLATE.lower())
        self.assertNotIn("súc tích", RAG_TEMPLATE.lower())
        self.assertNotIn("chỉ đưa ra kết luận", RAG_TEMPLATE.lower())
        self.assertNotIn("concise", RAG_TEMPLATE.lower())
        self.assertNotIn("brief", RAG_TEMPLATE.lower())
        # Đảm bảo có các quy tắc trích xuất nguyên văn đầy đủ
        self.assertIn("Không được tóm tắt hoặc lược bỏ", RAG_TEMPLATE)
        self.assertIn("trả lời đầy đủ các mục liên quan", RAG_TEMPLATE)
        self.assertIn("Giữ nguyên số điều, khoản, điểm", RAG_TEMPLATE)
        self.assertIn("Ưu tiên sao chép nguyên văn nội dung trả lời từ CONTEXT", RAG_TEMPLATE)
        self.assertIn('Không mở đầu bằng "Dựa trên ngữ cảnh được cung cấp"', RAG_TEMPLATE)

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


class ViQwenRAGGeneratorTests(unittest.TestCase):
    @staticmethod
    def _make_generator(
        tokenizer: _RecordingTokenizer,
        model: _RecordingModel,
        *,
        max_new_tokens: int = 64,
        max_input_tokens: int = 4096,
    ) -> ViQwenRAGGenerator:
        generator = ViQwenRAGGenerator(
            max_new_tokens=max_new_tokens,
            max_input_tokens=max_input_tokens,
        )
        generator._tokenizer = tokenizer
        generator._model = model
        return generator

    def test_generate_budgets_context_without_truncating_chat_envelope(self) -> None:
        tokenizer = _RecordingTokenizer()
        model = _RecordingModel(max_position_embeddings=2500)
        generator = self._make_generator(
            tokenizer,
            model,
            max_new_tokens=50,
            max_input_tokens=5000,
        )
        context = (
            "[1] TOP_CONTEXT\n"
            + ("A" * 40)
            + "\n\n[2] LOW_CONTEXT\n"
            + ("B" * 1600)
            + "LOW_TAIL_MARKER"
        )

        with patch.dict(sys.modules, {"torch": _fake_torch_module()}):
            answer = generator.generate(context=context, question="QUESTION_MARKER")

        self.assertEqual(answer, "model answer")
        self.assertIsNotNone(model.seen_prompt)
        prompt = str(model.seen_prompt)
        self.assertTrue(prompt.startswith("<|im_start|>system\n"))
        self.assertIn(SYSTEM_PROMPT, prompt)
        self.assertIn("[1] TOP_CONTEXT", prompt)
        self.assertIn("QUESTION_MARKER", prompt)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))
        self.assertNotIn("LOW_TAIL_MARKER", prompt)
        self.assertTrue(tokenizer.add_special_tokens_calls)
        self.assertEqual(set(tokenizer.add_special_tokens_calls), {False})
        self.assertIsNotNone(model.generate_kwargs)
        generate_kwargs = model.generate_kwargs or {}
        effective_max_new_tokens = generate_kwargs.get("max_new_tokens", generator.max_new_tokens)
        input_length = generate_kwargs["input_ids"].shape[1]
        self.assertLessEqual(input_length + effective_max_new_tokens, 2500)

    def test_generate_rejects_output_budget_at_model_context(self) -> None:
        tokenizer = _RecordingTokenizer()
        model = _RecordingModel(max_position_embeddings=64)
        generator = self._make_generator(
            tokenizer,
            model,
            max_new_tokens=64,
            max_input_tokens=256,
        )

        with (
            patch.dict(sys.modules, {"torch": _fake_torch_module()}),
            self.assertRaisesRegex(ValueError, r"max_new_tokens.*context window"),
        ):
            generator.generate(context="context", question="question")
        self.assertIsNone(model.generate_kwargs)

    def test_generate_rejects_budget_too_small_for_prompt_envelope(self) -> None:
        tokenizer = _RecordingTokenizer()
        model = _RecordingModel()
        generator = self._make_generator(
            tokenizer,
            model,
            max_new_tokens=64,
            max_input_tokens=100,
        )

        with (
            patch.dict(sys.modules, {"torch": _fake_torch_module()}),
            self.assertRaisesRegex(ValueError, r"max_input_tokens quá nhỏ"),
        ):
            generator.generate(context="context", question="question")
        self.assertIsNone(model.generate_kwargs)

    def test_generate_preserves_model_generation_eos_and_pad_ids(self) -> None:
        expected_eos = [151645, 151643]
        expected_pad = 151643
        tokenizer = _RecordingTokenizer()
        model = _RecordingModel(
            eos_token_id=expected_eos,
            pad_token_id=expected_pad,
        )
        generator = self._make_generator(tokenizer, model)

        with patch.dict(sys.modules, {"torch": _fake_torch_module()}):
            generator.generate(context="context", question="question")

        self.assertIsNotNone(model.generate_kwargs)
        generate_kwargs = model.generate_kwargs or {}
        effective_eos = generate_kwargs.get("eos_token_id", model.generation_config.eos_token_id)
        effective_pad = generate_kwargs.get("pad_token_id", model.generation_config.pad_token_id)
        self.assertEqual(effective_eos, expected_eos)
        self.assertEqual(effective_pad, expected_pad)

    def test_generate_falls_back_to_raw_chatml_without_chat_template(self) -> None:
        tokenizer = _MissingChatTemplateTokenizer()
        model = _RecordingModel()
        generator = self._make_generator(tokenizer, model)
        context = "Ngữ cảnh luật"
        question = "Ai có thẩm quyền?"

        with patch.dict(sys.modules, {"torch": _fake_torch_module()}):
            answer = generator.generate(context=context, question=question)

        self.assertEqual(answer, "model answer")
        self.assertEqual(model.seen_prompt, format_raw_qwen_prompt(context, question))

    def test_generate_falls_back_when_template_mapping_has_no_default(self) -> None:
        tokenizer = _MissingDefaultChatTemplateTokenizer()
        model = _RecordingModel()
        generator = self._make_generator(tokenizer, model)

        with patch.dict(sys.modules, {"torch": _fake_torch_module()}):
            generator.generate(context="context", question="question")

        self.assertEqual(model.seen_prompt, format_raw_qwen_prompt("context", "question"))


class RAGPipelineTests(unittest.TestCase):
    @staticmethod
    def _raw_answer() -> str:
        return MockSearchIndex().search_contexts("", top_k=1)[0]["text"]

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


    def test_pipeline_falls_back_for_unusable_generator_outputs(self) -> None:
        unusable_outputs = [
            ("empty", None),
            ("too_short", "Có."),
            ("refusal", "Không đủ thông tin trong ngữ cảnh."),
            ("refusal", "Xin lỗi, nhưng tôi không thể trả lời câu hỏi này."),
        ]

        for expected_reason, output in unusable_outputs:
            with self.subTest(reason=expected_reason):
                class UnusableGenerator:
                    def generate(self, context: str, question: str) -> str | None:
                        return output

                pipeline = LegalQABaseline(
                    index=MockSearchIndex(),  # type: ignore[arg-type]
                    generator=UnusableGenerator(),
                )
                pred = pipeline.predict_one("Mức phạt?", mode="rag")

                self.assertEqual(pred.answer, self._raw_answer())
                self.assertEqual(pred.route, "rag_output_fallback")
                self.assertEqual(pred.evidence["fallback_reason"], expected_reason)

    def test_pipeline_falls_back_when_generator_hits_token_limit(self) -> None:
        class TokenLimitedGenerator:
            def generate(self, context: str, question: str) -> str:
                raise GenerationTokenLimitReached(509, 512)

        pipeline = LegalQABaseline(
            index=MockSearchIndex(),  # type: ignore[arg-type]
            generator=TokenLimitedGenerator(),
        )
        pred = pipeline.predict_one("Mức phạt?", mode="rag")

        self.assertEqual(pred.answer, self._raw_answer())
        self.assertEqual(pred.route, "rag_token_limit_fallback")
        self.assertEqual(pred.evidence["generated_tokens"], 509)
        self.assertEqual(pred.evidence["max_new_tokens"], 512)

    def test_refusal_phrase_inside_valid_legal_answer_is_not_rejected(self) -> None:
        answer = (
            "Theo Điều 12, trường hợp hồ sơ không có thông tin về người nộp "
            "thì cơ quan tiếp nhận yêu cầu bổ sung hồ sơ."
        )
        pipeline = LegalQABaseline(
            index=MockSearchIndex(),  # type: ignore[arg-type]
            generator=MockGenerator(),
        )

        self.assertIsNone(pipeline._invalid_generation_reason(answer))


if __name__ == "__main__":
    unittest.main()
