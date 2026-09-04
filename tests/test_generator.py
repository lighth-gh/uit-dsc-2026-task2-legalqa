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
    build_generation_question,
    build_user_prompt,
    format_raw_qwen_prompt,
)
from legalqa_baseline.pipeline import LegalQABaseline, Prediction, prediction_audit_record
from legalqa_baseline.text import is_refusal_answer


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


class FixedScoreReranker:
    def __init__(self, score: float = 3.0) -> None:
        self.score = score

    def rerank(
        self,
        question: str,
        candidates: list[dict[str, Any]],
        top_k: int = 3,
        max_length: int = 1024,
    ) -> list[dict[str, Any]]:
        output = []
        for candidate in candidates[:top_k]:
            item = dict(candidate)
            item["rerank_score"] = self.score
            output.append(item)
        return output


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


class _NearTokenLimitModel(_RecordingModel):
    def generate(self, **kwargs: Any) -> list[list[int]]:
        self.generate_kwargs = dict(kwargs)
        values = kwargs["input_ids"].values
        generated_count = int(kwargs["max_new_tokens"]) - 4
        return [values + [ord("x")] * generated_count]


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
        self.assertIn("YÊU CẦU:", prompt)
        self.assertIn("Không rút gọn danh sách hoặc thủ tục", prompt)
        self.assertTrue(prompt.endswith("### Trả lời:"))

    def test_generation_question_keeps_real_question_and_strict_requirements(self) -> None:
        question = build_generation_question("Mức thuế là bao nhiêu?")
        self.assertTrue(question.startswith("Mức thuế là bao nhiêu?"))
        self.assertIn("mốc thời gian", question)
        self.assertIn("Không nói thiếu thông tin", question)
        self.assertIn("Không tự giả định dữ kiện không được nêu", question)
        self.assertIn("hỏi cá nhân thì không chép thêm phần tổ chức", question)
        self.assertIn("Phân biệt các khái niệm gần nhau", question)

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

    def test_generate_treats_last_four_budget_tokens_as_limit_hit(self) -> None:
        tokenizer = _RecordingTokenizer()
        model = _NearTokenLimitModel()
        generator = self._make_generator(
            tokenizer,
            model,
            max_new_tokens=32,
        )

        with (
            patch.dict(sys.modules, {"torch": _fake_torch_module()}),
            self.assertRaises(GenerationTokenLimitReached),
        ):
            generator.generate(context="context", question="question")

        self.assertEqual(generator.last_generation_stats["generated_tokens"], 28)
        self.assertTrue(generator.last_generation_stats["hit_token_limit"])

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
        self.assertEqual(pred.route, "generated_512")
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
        self.assertEqual(pred.route, "generated_512")


    def test_unusable_generation_does_not_dump_low_confidence_raw_text(self) -> None:
        unusable_outputs = [
            ("empty", None),
            ("too_short", "Có."),
            ("refusal", "Không đủ thông tin trong ngữ cảnh."),
            ("refusal", "Xin lỗi, nhưng tôi không thể trả lời câu hỏi này."),
            (
                "refusal",
                "Dựa trên các ngữ cảnh được cung cấp, không có thông tin cụ thể "
                "để trả lời câu hỏi này.",
            ),
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

                self.assertNotEqual(pred.answer, self._raw_answer())
                self.assertEqual(pred.route, "recovery_exhausted")
                self.assertEqual(pred.evidence["fallback_reason"], expected_reason)
                self.assertFalse(pred.evidence["raw_fallback_allowed"])

    def test_non_list_token_limit_is_not_retried_or_dumped_to_raw(self) -> None:
        class TokenLimitedGenerator:
            def __init__(self) -> None:
                self.calls = 0

            def generate(self, context: str, question: str) -> str:
                self.calls += 1
                raise GenerationTokenLimitReached(509, 512)

        generator = TokenLimitedGenerator()
        pipeline = LegalQABaseline(
            index=MockSearchIndex(),  # type: ignore[arg-type]
            generator=generator,
        )
        pred = pipeline.predict_one("Mức phạt?", mode="rag")

        self.assertNotEqual(pred.answer, self._raw_answer())
        self.assertEqual(pred.route, "recovery_exhausted")
        self.assertEqual(pred.evidence["generated_tokens"], 509)
        self.assertEqual(pred.evidence["max_new_tokens"], 512)
        self.assertEqual(generator.calls, 1)
        audit = prediction_audit_record("sample-1", pred)
        self.assertEqual(audit["route"], "recovery_exhausted")
        self.assertEqual(audit["generated_tokens"], 509)
        self.assertTrue(audit["hit_token_limit"])
        self.assertFalse(audit["possibly_cut"])
        self.assertTrue(audit["initial_hit_token_limit"])
        self.assertEqual(audit["generation_attempts"], 1)
        self.assertEqual(audit["top_document_id"], "100")

    def test_list_token_limit_retries_once_with_768_and_keeps_generation(self) -> None:
        class RetryGenerator:
            max_new_tokens = 512

            def __init__(self) -> None:
                self.budgets: list[int | None] = []
                self.contexts: list[str] = []
                self.questions: list[str] = []
                self.last_generation_stats: dict[str, Any] = {}

            def generate(
                self,
                context: str,
                question: str,
                *,
                max_new_tokens: int | None = None,
            ) -> str:
                self.budgets.append(max_new_tokens)
                self.contexts.append(context)
                self.questions.append(question)
                if len(self.budgets) == 1:
                    raise GenerationTokenLimitReached(509, 512)
                self.last_generation_stats = {
                    "generated_tokens": 620,
                    "max_new_tokens": max_new_tokens,
                    "hit_token_limit": False,
                }
                return (
                    "Danh sách hồ sơ gồm đơn đề nghị, giấy tờ pháp lý và "
                    "tài liệu chứng minh đáp ứng đầy đủ điều kiện."
                )

        generator = RetryGenerator()
        pipeline = LegalQABaseline(
            index=MockSearchIndex(),  # type: ignore[arg-type]
            generator=generator,
            reranker=FixedScoreReranker(),
        )
        pred = pipeline.predict_one(
            "Hãy phân tích và liệt kê danh sách hồ sơ cần nộp.",
            mode="rag",
        )

        self.assertEqual(pred.route, "generated_retry_768")
        self.assertEqual(generator.budgets, [None, 768])
        self.assertNotEqual(generator.questions[0], generator.questions[1])
        self.assertIn("không quá 520 từ", generator.questions[1])
        self.assertLessEqual(len(generator.contexts[1]), len(generator.contexts[0]))
        self.assertEqual(pred.evidence["generation_attempts"], 2)
        self.assertTrue(pred.evidence["initial_hit_token_limit"])
        self.assertFalse(pred.evidence["hit_token_limit"])
        self.assertEqual(len(pred.evidence["generation_attempt_seconds"]), 2)
        self.assertIsNotNone(pred.evidence["initial_generation_seconds"])
        self.assertIsNotNone(pred.evidence["retry_generation_seconds"])
        audit = prediction_audit_record("retry", pred)
        self.assertEqual(len(audit["generation_attempt_seconds"]), 2)
        self.assertIsNotNone(audit["initial_generation_seconds"])
        self.assertIsNotNone(audit["retry_generation_seconds"])
        self.assertIn("Danh sách hồ sơ", pred.answer)

    def test_token_limit_retry_uses_strong_focused_extractive_if_retry_fails(self) -> None:
        class StrongArticleIndex:
            def search_contexts(
                self,
                question: str,
                top_k: int = 50,
            ) -> list[dict[str, Any]]:
                return [{
                    "context_id": "article-12",
                    "chunk_no": 0,
                    "name": "Nghị định quy định thủ tục",
                    "link": "https://example.com/article-12",
                    "text": (
                        "Điều 12. Quy trình xử lý gồm ba bước. Bước một, cơ quan tiếp nhận "
                        "kiểm tra hồ sơ và ghi nhận thời điểm nhận. Bước hai, người có thẩm "
                        "quyền thẩm định tài liệu theo quy định. Bước ba, cơ quan ban hành "
                        "quyết định và gửi kết quả cho người đề nghị trong thời hạn luật định."
                    ),
                    "bm25_score": -20.0,
                }]

            def search_train(
                self,
                question: str,
                top_k: int = 5,
                exclude_id: str | None = None,
            ) -> list[dict[str, Any]]:
                return []

        class AlwaysTokenLimited:
            max_new_tokens = 512

            def __init__(self) -> None:
                self.budgets: list[int | None] = []
                self.last_generation_stats: dict[str, Any] = {}

            def generate(
                self,
                context: str,
                question: str,
                *,
                max_new_tokens: int | None = None,
            ) -> str:
                self.budgets.append(max_new_tokens)
                budget = max_new_tokens or self.max_new_tokens
                raise GenerationTokenLimitReached(budget - 1, budget)

        generator = AlwaysTokenLimited()
        pipeline = LegalQABaseline(
            index=StrongArticleIndex(),  # type: ignore[arg-type]
            generator=generator,
            reranker=FixedScoreReranker(score=3.0),
            enable_long_answer_extractive=False,
        )
        pred = pipeline.predict_one(
            "Theo Điều 12, quy trình xử lý được thực hiện theo mấy bước?",
            mode="rag",
        )

        self.assertEqual(generator.budgets, [None, 768])
        self.assertEqual(pred.route, "extractive_fallback")
        self.assertEqual(pred.evidence["generation_attempts"], 2)
        self.assertEqual(
            pred.evidence["recovery_strategy"],
            "token_limit_focused_extractive",
        )
        self.assertIn("Bước ba", pred.answer)
        self.assertFalse(pred.evidence["says_no_information"])

    def test_token_retry_refusal_transitions_to_refusal_recovery_without_more_tokens(self) -> None:
        class StrongListIndex:
            def search_contexts(
                self,
                question: str,
                top_k: int = 50,
            ) -> list[dict[str, Any]]:
                return [{
                    "context_id": "article-5",
                    "chunk_no": 0,
                    "name": "Nghị định xử phạt đất đai",
                    "link": "https://example.com/article-5",
                    "text": (
                        "Điều 5. Các hình thức xử phạt gồm cảnh cáo và phạt tiền. "
                        "Hình thức xử phạt bổ sung gồm tịch thu tang vật, phương tiện "
                        "vi phạm và đình chỉ hoạt động có thời hạn theo quy định."
                    ),
                    "bm25_score": -20.0,
                }]

            def search_train(
                self,
                question: str,
                top_k: int = 5,
                exclude_id: str | None = None,
            ) -> list[dict[str, Any]]:
                return []

        class TokenThenRefusalThenAnswer:
            max_new_tokens = 512

            def __init__(self) -> None:
                self.budgets: list[int | None] = []
                self.last_generation_stats: dict[str, Any] = {}

            def generate(
                self,
                context: str,
                question: str,
                *,
                max_new_tokens: int | None = None,
            ) -> str:
                self.budgets.append(max_new_tokens)
                if len(self.budgets) == 1:
                    raise GenerationTokenLimitReached(509, 512)
                if len(self.budgets) == 2:
                    return "Không đủ thông tin trong ngữ cảnh."
                return (
                    "Các hình thức xử phạt gồm cảnh cáo, phạt tiền, tịch thu tang vật "
                    "và đình chỉ hoạt động có thời hạn theo quy định."
                )

        generator = TokenThenRefusalThenAnswer()
        pipeline = LegalQABaseline(
            index=StrongListIndex(),  # type: ignore[arg-type]
            generator=generator,
            reranker=FixedScoreReranker(score=3.0),
            enable_long_answer_extractive=False,
        )
        pred = pipeline.predict_one(
            "Theo Điều 5, các hình thức xử phạt gồm những gì?",
            mode="rag",
        )

        self.assertEqual(pred.route, "generated_refusal_recovery")
        self.assertEqual(generator.budgets, [None, 768, None])
        self.assertEqual(pred.evidence["recovery_strategy"], "focused_context")
        self.assertTrue(pred.evidence["initial_hit_token_limit"])
        self.assertFalse(pred.evidence["hit_token_limit"])

    def test_refusal_retries_different_candidate_without_larger_budget(self) -> None:
        class RefusalThenAnswerGenerator:
            def __init__(self) -> None:
                self.calls: list[tuple[str, int | None]] = []

            def generate(
                self,
                context: str,
                question: str,
                *,
                max_new_tokens: int | None = None,
            ) -> str:
                self.calls.append((context, max_new_tokens))
                if len(self.calls) == 1:
                    return "Không đủ thông tin trong ngữ cảnh."
                return (
                    "Khoản 2 Điều 5 xác định cơ quan có thẩm quyền xử phạt "
                    "theo phạm vi nhiệm vụ được giao."
                )

        generator = RefusalThenAnswerGenerator()
        pipeline = LegalQABaseline(
            index=MockSearchIndex(),  # type: ignore[arg-type]
            generator=generator,
            reranker=FixedScoreReranker(),
        )
        pred = pipeline.predict_one(
            "Hãy giải thích cơ quan có thẩm quyền xử phạt.",
            mode="rag",
        )

        self.assertEqual(pred.route, "generated_refusal_recovery")
        self.assertEqual(len(generator.calls), 2)
        self.assertEqual([budget for _, budget in generator.calls], [None, None])
        self.assertNotEqual(generator.calls[0][0], generator.calls[1][0])
        self.assertEqual(pred.evidence["recovery_strategy"], "alternate_candidate")

    def test_refusal_uses_original_strong_top_one_focused_extractive(self) -> None:
        question = (
            "Thời hiệu khiếu nại thông báo không kháng nghị theo thủ tục tái thẩm "
            "đối với bản án hình sự không đủ căn cứ, điều kiện kháng nghị là bao lâu?"
        )

        class StrongRetrievalIndex:
            def search_contexts(
                self,
                query: str,
                top_k: int = 50,
            ) -> list[dict[str, Any]]:
                return [
                    {
                        "context_id": "102434",
                        "chunk_no": 306,
                        "name": "Bộ luật Tố tụng hình sự",
                        "link": "https://example.com/102434",
                        "text": (
                            "Điều 399. Thời hiệu khiếu nại thông báo không kháng nghị "
                            "theo thủ tục tái thẩm là 15 ngày kể từ ngày nhận được thông báo. "
                            "Người có quyền khiếu nại gửi đơn đến cơ quan có thẩm quyền để "
                            "được xem xét theo quy định của pháp luật tố tụng hình sự."
                        ),
                        "bm25_score": -20.0,
                    },
                    {
                        "context_id": "wrong",
                        "chunk_no": 1,
                        "name": "Văn bản khác",
                        "link": "https://example.com/wrong",
                        "text": (
                            "Nội dung khác quy định về thời hạn xử lý công việc và trách "
                            "nhiệm chung của cơ quan có thẩm quyền trong một thủ tục khác."
                        ),
                        "bm25_score": -10.0,
                    },
                ]

            def search_train(
                self,
                question: str,
                top_k: int = 5,
                exclude_id: str | None = None,
            ) -> list[dict[str, Any]]:
                return []

        class AlwaysRefuses:
            def __init__(self) -> None:
                self.calls = 0

            def generate(self, context: str, question: str) -> str:
                self.calls += 1
                return "Không đủ thông tin trong ngữ cảnh."

        generator = AlwaysRefuses()
        pipeline = LegalQABaseline(
            index=StrongRetrievalIndex(),  # type: ignore[arg-type]
            generator=generator,
            reranker=FixedScoreReranker(score=3.0),
            enable_long_answer_extractive=False,
        )
        pred = pipeline.predict_one(question, mode="rag")

        self.assertEqual(generator.calls, 3)
        self.assertEqual(pred.route, "extractive_fallback")
        self.assertIn("15 ngày", pred.answer)
        self.assertEqual(
            pred.evidence["recovery_strategy"],
            "refusal_focused_extractive",
        )
        self.assertEqual(pred.evidence["top_contexts"][0]["context_id"], "102434")
        self.assertGreaterEqual(pred.evidence["raw_reranker_score"], 2.0)
        self.assertFalse(pred.evidence["says_no_information"])
        self.assertEqual(len(pred.evidence["generation_attempt_seconds"]), 3)

    def test_low_raw_refusal_uses_only_decisive_exact_focused_evidence(self) -> None:
        question = "Theo Điều 399, thời hiệu khiếu nại thông báo không kháng nghị là bao lâu?"

        class ExactArticleIndex:
            def search_contexts(
                self,
                query: str,
                top_k: int = 50,
            ) -> list[dict[str, Any]]:
                return [{
                    "context_id": "102434",
                    "chunk_no": 306,
                    "name": "Bộ luật Tố tụng hình sự",
                    "link": "https://example.com/102434",
                    "text": (
                        "Điều 399. Thời hiệu khiếu nại thông báo không kháng nghị theo thủ tục "
                        "tái thẩm là 15 ngày kể từ ngày nhận được thông báo. Người khiếu nại "
                        "gửi đơn đến cơ quan có thẩm quyền để được xem xét theo quy định của "
                        "pháp luật tố tụng hình sự."
                    ),
                    "bm25_score": -20.0,
                }]

            def search_train(
                self,
                question: str,
                top_k: int = 5,
                exclude_id: str | None = None,
            ) -> list[dict[str, Any]]:
                return []

        class AlwaysRefuses:
            def __init__(self) -> None:
                self.questions: list[str] = []

            def generate(self, context: str, question: str) -> str:
                self.questions.append(question)
                return "Không đủ thông tin trong ngữ cảnh."

        generator = AlwaysRefuses()
        pipeline = LegalQABaseline(
            index=ExactArticleIndex(),  # type: ignore[arg-type]
            generator=generator,
            reranker=FixedScoreReranker(score=-1.0),
            enable_long_answer_extractive=False,
        )
        pred = pipeline.predict_one(question, mode="rag")

        self.assertEqual(pred.route, "extractive_fallback")
        self.assertIn("15 ngày", pred.answer)
        self.assertEqual(
            pred.evidence["recovery_strategy"],
            "refusal_decisive_focused_extractive",
        )
        self.assertLess(pred.evidence["raw_reranker_score"], 2.0)
        self.assertEqual(len(generator.questions), 2)
        self.assertIn("Ngữ cảnh đã được thu gọn", generator.questions[1])

    def test_yes_no_refusal_moves_from_strong_focus_to_alternate_candidate(self) -> None:
        question = "Theo Điều 12, người đang trả nợ tiền sử dụng đất có bị cấm chia di sản không?"

        class YesNoRecoveryIndex:
            def search_contexts(
                self,
                query: str,
                top_k: int = 50,
            ) -> list[dict[str, Any]]:
                return [
                    {
                        "context_id": "primary",
                        "chunk_no": 12,
                        "name": "Luật Đất đai",
                        "link": "https://example.com/primary",
                        "text": (
                            "Điều 12 quy định về quyền và nghĩa vụ của người sử dụng đất "
                            "trong thời gian còn nợ tiền sử dụng đất. Người sử dụng đất "
                            "thực hiện các quyền theo điều kiện và phạm vi luật định."
                        ),
                        "bm25_score": -20.0,
                    },
                    {
                        "context_id": "alternate",
                        "chunk_no": 5,
                        "name": "Nghị định hướng dẫn Luật Đất đai",
                        "link": "https://example.com/alternate",
                        "text": (
                            "Quyền phân chia di sản là quyền sử dụng đất được thực hiện "
                            "khi đáp ứng nghĩa vụ tài chính. Việc còn trả nợ tiền sử dụng "
                            "đất không tự động làm mất quyền, nhưng phải tuân thủ điều kiện."
                        ),
                        "bm25_score": -18.0,
                    },
                ]

            def search_train(
                self,
                question: str,
                top_k: int = 5,
                exclude_id: str | None = None,
            ) -> list[dict[str, Any]]:
                return []

        class RefuseTwiceThenAnswer:
            def __init__(self) -> None:
                self.contexts: list[str] = []

            def generate(self, context: str, question: str) -> str:
                self.contexts.append(context)
                if len(self.contexts) < 3:
                    return "Không đủ thông tin trong ngữ cảnh."
                return (
                    "Người sử dụng đất không tự động bị cấm phân chia di sản, "
                    "nhưng phải hoàn thành hoặc tuân thủ nghĩa vụ tài chính theo quy định."
                )

        generator = RefuseTwiceThenAnswer()
        pipeline = LegalQABaseline(
            index=YesNoRecoveryIndex(),  # type: ignore[arg-type]
            generator=generator,
            reranker=FixedScoreReranker(score=3.0),
            enable_long_answer_extractive=False,
        )
        pred = pipeline.predict_one(question, mode="rag")

        self.assertEqual(len(generator.contexts), 3)
        self.assertEqual(pred.route, "generated_refusal_recovery")
        self.assertEqual(pred.evidence["recovery_strategy"], "alternate_candidate")
        self.assertIn("không tự động bị cấm", pred.answer)
        self.assertFalse(pred.evidence["raw_fallback_allowed"])

    def test_refusal_can_use_guarded_knn_at_point_nine(self) -> None:
        class RefusalGenerator:
            def generate(self, context: str, question: str) -> str:
                return "Không đủ thông tin trong ngữ cảnh."

        pipeline = LegalQABaseline(
            index=MockSearchIndex(),  # type: ignore[arg-type]
            generator=RefusalGenerator(),
            reranker=FixedScoreReranker(),
            guarded_knn_threshold=0.90,
        )
        pred = pipeline.predict_one("Mức phạt vi phạm hành chính?", mode="rag")

        self.assertEqual(pred.route, "knn_guarded_refusal")
        self.assertGreaterEqual(pred.confidence, 0.90)
        self.assertEqual(pred.evidence["recovery_strategy"], "guarded_knn")

    def test_low_raw_score_uses_guarded_knn_before_generation(self) -> None:
        class MustNotGenerate:
            def generate(self, context: str, question: str) -> str:
                raise AssertionError("low-confidence exact KNN must bypass generation")

        pipeline = LegalQABaseline(
            index=MockSearchIndex(),  # type: ignore[arg-type]
            generator=MustNotGenerate(),
            guarded_knn_threshold=0.90,
        )
        pred = pipeline.predict_one("Mức phạt vi phạm hành chính?", mode="rag")

        self.assertEqual(pred.route, "knn_guarded_low_confidence")
        self.assertEqual(pred.evidence["generation_attempts"], 0)
        self.assertEqual(pred.evidence["recovery_strategy"], "guarded_knn")

    def test_low_raw_score_runs_controlled_requery_before_generation(self) -> None:
        question = "Mức phạt doanh nghiệp là bao nhiêu?"

        class RequeryIndex:
            def search_contexts(self, query: str, top_k: int = 50) -> list[dict[str, Any]]:
                if query == question:
                    return [{
                        "context_id": "low",
                        "chunk_no": 0,
                        "name": "Văn bản nhiễu",
                        "link": "https://example.com/low",
                        "text": "Nội dung không xác định đúng mức phạt doanh nghiệp.",
                        "bm25_score": -5.0,
                    }]
                return [{
                    "context_id": "recovered",
                    "chunk_no": 4,
                    "name": "Nghị định xử phạt",
                    "link": "https://example.com/recovered",
                    "text": "Điều 8 quy định rõ mức phạt áp dụng đối với doanh nghiệp vi phạm.",
                    "bm25_score": -9.0,
                }]

            def search_train(
                self,
                question: str,
                top_k: int = 5,
                exclude_id: str | None = None,
            ) -> list[dict[str, Any]]:
                return []

        class ScoreByContextReranker:
            def rerank(
                self,
                question: str,
                candidates: list[dict[str, Any]],
                top_k: int = 3,
                max_length: int = 1024,
            ) -> list[dict[str, Any]]:
                output = []
                for candidate in candidates[:top_k]:
                    item = dict(candidate)
                    item["rerank_score"] = 3.0 if item["context_id"] == "recovered" else 1.0
                    output.append(item)
                return output

        pipeline = LegalQABaseline(
            index=RequeryIndex(),  # type: ignore[arg-type]
            generator=MockGenerator(),
            reranker=ScoreByContextReranker(),
        )
        pred = pipeline.predict_one(question, mode="rag")

        self.assertEqual(pred.route, "generated_512")
        self.assertEqual(pred.evidence["top_contexts"][0]["context_id"], "recovered")
        recovery = pred.evidence["retrieval_trace"]["recovery"]
        self.assertEqual(recovery["trigger"], "raw_reranker_score_below_2")
        self.assertEqual(recovery["status"], "ok")

    def test_controlled_requery_preserves_prrs_alias(self) -> None:
        question = (
            "Phương pháp ELISA dùng để chẩn đoán hội chứng rối loạn sinh sản và "
            "hô hấp ở lợn có bao nhiêu bước thực hiện?"
        )

        class AliasRequeryIndex:
            def __init__(self) -> None:
                self.queries: list[str] = []

            def search_contexts(self, query: str, top_k: int = 50) -> list[dict[str, Any]]:
                self.queries.append(query)
                if query == question:
                    return [{
                        "context_id": "low",
                        "chunk_no": 0,
                        "name": "Phương pháp xét nghiệm khác",
                        "link": "https://example.com/low",
                        "text": "Nội dung xét nghiệm không liên quan đến PRRS.",
                        "bm25_score": -5.0,
                    }]
                self.assert_alias(query)
                return [{
                    "context_id": "prrs",
                    "chunk_no": 11,
                    "name": "Tiêu chuẩn chẩn đoán PRRS",
                    "link": "https://example.com/prrs",
                    "text": "Phụ lục D. Phương pháp ELISA phát hiện kháng thể PRRS gồm các bước.",
                    "bm25_score": -20.0,
                }]

            @staticmethod
            def assert_alias(query: str) -> None:
                if "prrs" not in query.casefold():
                    raise AssertionError(f"controlled alias missing from requery: {query}")

            def search_train(
                self,
                question: str,
                top_k: int = 5,
                exclude_id: str | None = None,
            ) -> list[dict[str, Any]]:
                return []

        class ScoreByContextReranker:
            def rerank(
                self,
                question: str,
                candidates: list[dict[str, Any]],
                top_k: int = 3,
                max_length: int = 1024,
            ) -> list[dict[str, Any]]:
                output = []
                for candidate in candidates[:top_k]:
                    item = dict(candidate)
                    item["rerank_score"] = 3.0 if item["context_id"] == "prrs" else 1.0
                    output.append(item)
                return output

        index = AliasRequeryIndex()
        pipeline = LegalQABaseline(
            index=index,  # type: ignore[arg-type]
            generator=MockGenerator(),
            reranker=ScoreByContextReranker(),
        )
        pred = pipeline.predict_one(question, mode="rag")

        self.assertEqual(pred.evidence["top_contexts"][0]["context_id"], "prrs")
        self.assertGreaterEqual(len(index.queries), 2)
        self.assertIn("prrs", index.queries[1].casefold())

    def test_guarded_knn_rejects_different_intent(self) -> None:
        class IntentMismatchIndex(MockSearchIndex):
            def search_train(
                self,
                question: str,
                top_k: int = 5,
                exclude_id: str | None = None,
            ) -> list[dict[str, Any]]:
                return [{
                    "sample_id": "other",
                    "question": "Cơ quan nào có thẩm quyền xử phạt?",
                    "answer": "Cơ quan cấp huyện có thẩm quyền.",
                    "bm25_score": -1.0,
                }]

        pipeline = LegalQABaseline(
            index=IntentMismatchIndex(),  # type: ignore[arg-type]
            generator=MockGenerator(),
        )

        self.assertIsNone(
            pipeline._guarded_knn("Mức phạt là bao nhiêu?", exclude_id=None)
        )

    def test_pipeline_cleans_boilerplate_without_truncating_answer(self) -> None:
        class BoilerplateGenerator:
            def generate(self, context: str, question: str) -> str:
                return (
                    "Dựa trên các ngữ cảnh được cung cấp: Điều 12 quy định mức phạt "
                    "từ 2 triệu đến 5 triệu đồng."
                )

        pipeline = LegalQABaseline(
            index=MockSearchIndex(),  # type: ignore[arg-type]
            generator=BoilerplateGenerator(),
        )
        pred = pipeline.predict_one("Mức phạt?", mode="rag")

        self.assertEqual(pred.route, "generated_512")
        self.assertTrue(pred.answer.startswith("Điều 12"))
        self.assertIn("5 triệu đồng.", pred.answer)

    def test_refusal_detector_checks_only_first_two_sentences(self) -> None:
        pipeline = LegalQABaseline(
            index=MockSearchIndex(),  # type: ignore[arg-type]
            generator=MockGenerator(),
        )
        refusals = (
            "Dựa trên các ngữ cảnh được cung cấp, không có thông tin cụ thể để kết luận.",
            "Đã kiểm tra dữ liệu. Các ngữ cảnh không đề cập nội dung được hỏi.",
            "Theo dữ liệu hiện có, không thể trả lời chính xác câu hỏi này.",
            "Kết quả tra cứu như sau. Không tìm thấy thông tin phù hợp.",
        )
        for answer in refusals:
            with self.subTest(answer=answer):
                self.assertEqual(pipeline._invalid_generation_reason(answer), "refusal")

        third_sentence_only = (
            "Điều 12 quy định rõ chủ thể áp dụng. "
            "Mức phạt được xác định theo từng hành vi cụ thể. "
            "Không tìm thấy thông tin là một trạng thái của hệ thống lưu trữ."
        )
        self.assertIsNone(pipeline._invalid_generation_reason(third_sentence_only))

    def test_id_86293_verbatim_answer_uses_shared_refusal_detector(self) -> None:
        # Nguyên văn answer của ID 86293 trong submission đã ghi nhận lỗi.
        answer = "Không đủ thông tin trong ngữ cảnh."
        pipeline = LegalQABaseline(
            index=MockSearchIndex(),  # type: ignore[arg-type]
            generator=MockGenerator(),
        )

        self.assertTrue(is_refusal_answer(answer))
        self.assertEqual(pipeline._invalid_generation_reason(answer), "refusal")

        audit = prediction_audit_record(
            "86293",
            Prediction(answer=answer, route="generated_512", confidence=0.0, evidence={}),
        )
        self.assertTrue(audit["says_no_information"])

    def test_pipeline_falls_back_for_possibly_cut_answer(self) -> None:
        class CutGenerator:
            def generate(self, context: str, question: str) -> str:
                return "Hồ sơ phải được kiểm tra theo trình tự sau:"

        pipeline = LegalQABaseline(
            index=MockSearchIndex(),  # type: ignore[arg-type]
            generator=CutGenerator(),
        )
        pred = pipeline.predict_one("Quy định xử lý thế nào?", mode="rag")

        self.assertEqual(pred.route, "recovery_exhausted")
        self.assertEqual(pred.evidence["fallback_reason"], "possibly_cut")

    def test_pipeline_honors_generator_token_limit_metadata(self) -> None:
        class MetadataLimitedGenerator:
            last_generation_stats = {
                "generated_tokens": 504,
                "max_new_tokens": 512,
                "hit_token_limit": True,
            }

            def generate(self, context: str, question: str) -> str:
                return "Một câu trả lời vẫn có dấu kết thúc nhưng đã dùng hết ngân sách."

        pipeline = LegalQABaseline(
            index=MockSearchIndex(),  # type: ignore[arg-type]
            generator=MetadataLimitedGenerator(),
        )
        pred = pipeline.predict_one("Mức phạt?", mode="rag")

        self.assertEqual(pred.route, "recovery_exhausted")
        self.assertEqual(pred.evidence["fallback_reason"], "token_limit")
        self.assertTrue(pred.evidence["hit_token_limit"])

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
