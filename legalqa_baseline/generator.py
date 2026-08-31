from __future__ import annotations

import sys
from typing import Any

from .hardware import recommended_cuda_dtype


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


def build_user_prompt(context: str, question: str) -> str:
    """Tạo user prompt từ context và question theo đúng khuôn mẫu RAG_TEMPLATE."""
    return RAG_TEMPLATE.format(context=context.strip(), question=question.strip())


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
    ) -> None:
        if max_new_tokens <= 0 or max_input_tokens <= 0:
            raise ValueError("max_new_tokens and max_input_tokens must be greater than 0")
        if temperature < 0:
            raise ValueError("temperature must not be negative")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        self.model_name_or_path = model_name_or_path
        self.device_setting = device
        self.torch_dtype_setting = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_input_tokens = max_input_tokens
        self.seed = seed

        self._tokenizer: Any = None
        self._model: Any = None

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
            shortened_context = stripped_context[:keep_characters].rstrip()
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

    def generate(self, context: str, question: str) -> str:
        """Sinh câu trả lời cho một cặp (context, question)."""
        self._load_model()
        import torch

        model_context = int(
            getattr(self._model.config, "max_position_embeddings", self.max_input_tokens)
        )
        if self.max_new_tokens >= model_context:
            raise ValueError(
                "max_new_tokens phải nhỏ hơn context window của model "
                f"({model_context})"
            )
        prompt_limit = min(
            self.max_input_tokens,
            model_context - self.max_new_tokens,
        )
        inputs = self._prepare_inputs(
            context=context,
            question=question,
            prompt_limit=prompt_limit,
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
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

        input_len = inputs["input_ids"].shape[1]
        response_tokens = outputs[0][input_len:]
        response_text = self._tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
        if not response_text:
            raise RuntimeError("Generator trả về answer rỗng")
        return response_text
