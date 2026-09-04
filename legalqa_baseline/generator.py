from __future__ import annotations

import sys
from typing import Any

from .hardware import recommended_cuda_dtype


class GenerationTokenLimitReached(RuntimeError):
    """Raised when generation likely stopped because ``max_new_tokens`` was exhausted."""

    def __init__(
        self,
        generated_tokens: int,
        max_new_tokens: int,
        partial_answer: str | None = None,
    ) -> None:
        self.generated_tokens = generated_tokens
        self.max_new_tokens = max_new_tokens
        # Keep the decoded text for diagnostics/recovery checks. Callers must
        # still validate completeness and grounding before using it.
        self.partial_answer = str(partial_answer or "").strip()
        super().__init__(
            "Generator reached its token limit "
            f"({generated_tokens}/{max_new_tokens} generated tokens)"
        )


SYSTEM_PROMPT = """
Bạn là một trợ lý hỏi đáp pháp luật tiếng Việt.
Nhiệm vụ của bạn là trả lời câu hỏi dựa hoàn toàn trên các ngữ cảnh pháp luật được cung cấp.
Hãy trả lời chính xác, trung thực, đầy đủ và không tự bổ sung kiến thức ngoài ngữ cảnh.
""".strip()


RAG_TEMPLATE = """
Hãy thực hiện các yêu cầu sau:

- Chỉ sử dụng thông tin có trong các ngữ cảnh được cung cấp.
- Các ngữ cảnh có thể chứa thông tin không liên quan hoặc gây nhiễu; hãy bỏ qua chúng.
- Nếu nhiều ngữ cảnh cùng chứa thông tin cần thiết, hãy tổng hợp chúng để trả lời đầy đủ.
- Không suy đoán, không bổ sung kiến thức pháp luật từ bên ngoài.
- Nếu ngữ cảnh không chứa đủ thông tin để trả lời câu hỏi, hãy trả lời: "Không đủ thông tin trong ngữ cảnh."
- Không được tóm tắt hoặc lược bỏ thông tin cần thiết.
- Nếu câu hỏi yêu cầu nội dung điều luật, danh sách, hồ sơ, biểu mẫu, điều kiện hoặc các trường hợp thì phải trả lời đầy đủ các mục liên quan.
- Giữ nguyên số điều, khoản, điểm, mức tiền, thời hạn, ngày tháng và tên văn bản.
- Ưu tiên sao chép nguyên văn nội dung trả lời từ CONTEXT.
- Không mở đầu bằng "Dựa trên ngữ cảnh được cung cấp".

### Ngữ cảnh:
{context}

### Câu hỏi:
{question}

### Trả lời:
""".strip()


QUESTION_REQUIREMENTS = """
YÊU CẦU:
- Trả lời trực tiếp câu hỏi.
- Bám đúng hành vi pháp lý, chủ thể và phạm vi trong câu hỏi; không thay bằng một vấn đề pháp lý gần nghĩa.
- Không tự giả định dữ kiện không được nêu (giới tính, tuổi, tình trạng hôn nhân, năng lực hành vi hoặc vi phạm).
- Với câu hỏi có/không, chỉ dùng điều kiện hoặc trường hợp cấm áp dụng trực tiếp cho sự kiện được hỏi.
- Chỉ trả lời đúng chủ thể: hỏi cá nhân thì không chép thêm phần tổ chức, và ngược lại.
- Phân biệt các khái niệm gần nhau như thời hiệu khiếu nại, thời hạn kháng nghị và thời hạn giải quyết.
- Giữ đầy đủ các bước, điều kiện, hồ sơ, biểu mẫu, mốc thời gian, ngoại lệ, mức tiền và căn cứ pháp lý liên quan trong CONTEXT.
- Không rút gọn danh sách hoặc thủ tục thành kết luận chung.
- Chỉ chép các mục trực tiếp trả lời câu hỏi; bỏ phần trước hoặc sau thuộc chủ đề khác.
- Không lặp lại nội dung và kết thúc ngay sau mục liên quan cuối cùng.
- Không mở đầu bằng "Dựa trên ngữ cảnh được cung cấp".
- Không nói thiếu thông tin nếu CONTEXT có nội dung trả lời.
""".strip()


def build_generation_question(question: str) -> str:
    """Nối yêu cầu chống rút gọn trực tiếp vào câu hỏi thật."""
    base_question = str(question or "").strip()
    return f"{base_question}\n\n{QUESTION_REQUIREMENTS}".strip()


def build_user_prompt(context: str, question: str) -> str:
    """Tạo user prompt từ context và question theo đúng khuôn mẫu RAG_TEMPLATE."""
    return RAG_TEMPLATE.format(
        context=context.strip(),
        question=build_generation_question(question),
    )


def build_chat_messages(context: str, question: str) -> list[dict[str, str]]:
    """Tạo danh sách tin nhắn chuẩn chat cho model Qwen2."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(context=context, question=question)},
    ]


def format_raw_qwen_prompt(context: str, question: str) -> str:
    """Định dạng prompt thủ công theo chuẩn ChatML của Qwen2 khi không dùng tokenizer chat template."""
    user_content = build_user_prompt(context=context, question=question)
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


class ViQwenRAGGenerator:
    """Generator sử dụng mô hình AITeamVN/Vi-Qwen2-1.5B-RAG để sinh câu trả lời pháp lý."""

    def __init__(
        self,
        model_name_or_path: str = "AITeamVN/Vi-Qwen2-1.5B-RAG",
        device: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.9,
        max_input_tokens: int = 7168,
        seed: int = 2026,
        repetition_penalty: float = 1.05,
    ) -> None:
        if max_new_tokens <= 0 or max_input_tokens <= 0:
            raise ValueError("max_new_tokens and max_input_tokens must be greater than 0")
        if temperature < 0:
            raise ValueError("temperature must not be negative")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if repetition_penalty <= 0.0:
            raise ValueError("repetition_penalty must be greater than 0")
        self.model_name_or_path = model_name_or_path
        self.device_setting = device
        self.torch_dtype_setting = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_input_tokens = max_input_tokens
        self.seed = seed
        self.repetition_penalty = repetition_penalty

        self._tokenizer: Any = None
        self._model: Any = None
        self.last_generation_stats: dict[str, Any] = {}

    def _load_model(self) -> None:
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Chưa cài đặt PyTorch hoặc Transformers. Vui lòng chạy:\n"
                "pip install -r requirements-generator.txt"
            ) from exc

        print(f"[Generator] Đang tải tokenizer & model: {self.model_name_or_path}...", file=sys.stderr)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=False,
        )

        if self.torch_dtype_setting == "bfloat16":
            dtype = torch.bfloat16
        elif self.torch_dtype_setting == "float16":
            dtype = torch.float16
        else:
            wants_cuda = self.device_setting in ("auto", "cuda") and torch.cuda.is_available()
            if wants_cuda:
                dtype = recommended_cuda_dtype(torch)
            else:
                dtype = torch.float32

        if self.device_setting == "auto":
            device_map = "auto"
        elif self.device_setting == "cuda":
            device_map = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_map = self.device_setting

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=False,
        )
        self._model.eval()
        generation_config = getattr(self._model, "generation_config", None)
        if generation_config is not None and self.temperature <= 0.0:
            # Một số snapshot lưu sampling flags dù chạy greedy, khiến
            # Transformers cảnh báo temperature/top_p/top_k bị ignored.
            generation_config.do_sample = False
            generation_config.temperature = None
            generation_config.top_p = None
            generation_config.top_k = None
        device_map_used = getattr(self._model, "hf_device_map", None)
        if isinstance(device_map_used, dict):
            mapped_devices = sorted({str(value) for value in device_map_used.values()})
            placement = ", ".join(mapped_devices)
        else:
            placement = str(self._model.device)
        print(f"[Generator] Tải model thành công trên device(s): {placement}", file=sys.stderr)

    def _render_prompt(self, context: str, question: str) -> str:
        """Render a Qwen chat prompt, with a raw ChatML fallback."""
        messages = build_chat_messages(context=context, question=question)
        apply_chat_template = getattr(self._tokenizer, "apply_chat_template", None)
        chat_template = getattr(self._tokenizer, "chat_template", None)
        has_default_template = not isinstance(chat_template, dict) or "default" in chat_template
        if callable(apply_chat_template) and chat_template and has_default_template:
            return apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return format_raw_qwen_prompt(context=context, question=question)

    def _tokenize_prompt(self, prompt: str) -> dict[str, Any]:
        """Tokenize an already formatted chat prompt without adding tokens twice."""
        return self._tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )

    @staticmethod
    def _input_length(inputs: dict[str, Any]) -> int:
        return int(inputs["input_ids"].shape[-1])

    @staticmethod
    def _safe_context_prefix(context: str, max_characters: int) -> str:
        """Cut prompt context at a paragraph, sentence, or word boundary."""
        stripped = context.strip()
        if max_characters >= len(stripped):
            return stripped
        if max_characters <= 0:
            return ""

        prefix = stripped[:max_characters].rstrip()
        structured_boundaries = [
            prefix.rfind("\n"),
            prefix.rfind(". "),
            prefix.rfind("? "),
            prefix.rfind("! "),
            prefix.rfind("… "),
        ]
        structured_end = max(structured_boundaries)
        if structured_end > 0 and len(prefix) - structured_end <= 256:
            if prefix[structured_end] in ".?!…":
                structured_end += 1
            return prefix[:structured_end].rstrip()

        word_end = max(prefix.rfind(" "), prefix.rfind("\t"), prefix.rfind("\n"))
        if word_end > 0:
            return prefix[:word_end].rstrip()
        return prefix

    def _prepare_inputs(
        self,
        context: str,
        question: str,
        prompt_limit: int,
    ) -> dict[str, Any]:
        """Fit only the context to the prompt budget, preserving chat framing and question."""
        prompt = self._render_prompt(context=context, question=question)
        inputs = self._tokenize_prompt(prompt)
        if self._input_length(inputs) <= prompt_limit:
            return inputs

        empty_context_prompt = self._render_prompt(context="", question=question)
        empty_context_inputs = self._tokenize_prompt(empty_context_prompt)
        empty_context_length = self._input_length(empty_context_inputs)
        if empty_context_length > prompt_limit:
            raise ValueError(
                "max_input_tokens quá nhỏ để chứa system prompt và question; "
                "hãy tăng max_input_tokens hoặc giảm max_new_tokens"
            )

        stripped_context = context.strip()
        low = 0
        high = len(stripped_context)
        best_inputs = empty_context_inputs

        # Keep a valid Unicode prefix containing the highest-ranked chunks. Using
        # text boundaries avoids replacement characters from decoding a partial
        # byte-level BPE sequence.
        while low <= high:
            keep_characters = (low + high) // 2
            shortened_context = self._safe_context_prefix(
                stripped_context,
                keep_characters,
            )
            candidate_prompt = self._render_prompt(
                context=shortened_context,
                question=question,
            )
            candidate_inputs = self._tokenize_prompt(candidate_prompt)
            if self._input_length(candidate_inputs) <= prompt_limit:
                best_inputs = candidate_inputs
                low = keep_characters + 1
            else:
                high = keep_characters - 1
        return best_inputs

    def _generation_token_ids(self) -> tuple[Any, Any]:
        """Preserve the checkpoint's multi-EOS and padding configuration."""
        generation_config = getattr(self._model, "generation_config", None)
        eos_token_id = getattr(generation_config, "eos_token_id", None)
        pad_token_id = getattr(generation_config, "pad_token_id", None)

        if eos_token_id is None:
            eos_token_id = getattr(self._tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self._tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            tokenizer_eos = getattr(self._tokenizer, "eos_token_id", None)
            if isinstance(tokenizer_eos, int):
                pad_token_id = tokenizer_eos
        return eos_token_id, pad_token_id

    def generate(
        self,
        context: str,
        question: str,
        *,
        max_input_tokens: int | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        """Sinh câu trả lời cho một cặp (context, question)."""
        self._load_model()
        import torch

        self.last_generation_stats = {}

        effective_max_input_tokens = (
            self.max_input_tokens if max_input_tokens is None else max_input_tokens
        )
        effective_max_new_tokens = (
            self.max_new_tokens if max_new_tokens is None else max_new_tokens
        )
        if effective_max_input_tokens <= 0 or effective_max_new_tokens <= 0:
            raise ValueError("generation token budgets must be greater than 0")

        model_context = int(
            getattr(self._model.config, "max_position_embeddings", self.max_input_tokens)
        )
        if effective_max_new_tokens >= model_context:
            raise ValueError(
                "max_new_tokens phải nhỏ hơn context window của model "
                f"({model_context})"
            )
        prompt_limit = min(
            effective_max_input_tokens,
            model_context - effective_max_new_tokens,
        )
        inputs = self._prepare_inputs(
            context=context,
            question=question,
            prompt_limit=prompt_limit,
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": effective_max_new_tokens,
            "repetition_penalty": self.repetition_penalty,
        }
        eos_token_id, pad_token_id = self._generation_token_ids()
        if eos_token_id is not None:
            generate_kwargs["eos_token_id"] = eos_token_id
        if pad_token_id is not None:
            generate_kwargs["pad_token_id"] = pad_token_id

        if self.temperature > 0.0:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = self.temperature
            generate_kwargs["top_p"] = self.top_p
        else:
            generate_kwargs["do_sample"] = False

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        with torch.inference_mode():
            outputs = self._model.generate(**inputs, **generate_kwargs)

        input_len = int(inputs["input_ids"].shape[1])
        output_shape = getattr(outputs, "shape", None)
        if output_shape is not None:
            output_len = int(output_shape[1])
        else:
            # Keep lightweight test doubles and compatible generate() wrappers usable.
            output_len = len(outputs[0])
        generated_tokens = output_len - input_len
        hit_token_limit = generated_tokens >= effective_max_new_tokens - 4
        response_tokens = outputs[0][input_len:]
        response_text = self._tokenizer.decode(
            response_tokens,
            skip_special_tokens=True,
        ).strip()
        self.last_generation_stats = {
            "input_tokens": input_len,
            "generated_tokens": generated_tokens,
            "max_new_tokens": effective_max_new_tokens,
            "hit_token_limit": hit_token_limit,
            "partial_answer_available": bool(response_text) if hit_token_limit else False,
        }
        if hit_token_limit:
            raise GenerationTokenLimitReached(
                generated_tokens=generated_tokens,
                max_new_tokens=effective_max_new_tokens,
                partial_answer=response_text,
            )

        return response_text
