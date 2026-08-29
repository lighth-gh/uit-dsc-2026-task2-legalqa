from __future__ import annotations

import sys
from typing import Any

SYSTEM_PROMPT = """
Bạn là một trợ lý hỏi đáp pháp luật tiếng Việt.
Nhiệm vụ của bạn là trả lời câu hỏi dựa hoàn toàn trên các ngữ cảnh pháp luật được cung cấp.
Hãy trả lời chính xác, trung thực và không tự bổ sung kiến thức ngoài ngữ cảnh.
""".strip()


RAG_TEMPLATE = """
Hãy thực hiện các yêu cầu sau:

- Chỉ sử dụng thông tin có trong các ngữ cảnh được cung cấp.
- Các ngữ cảnh có thể chứa thông tin không liên quan hoặc gây nhiễu; hãy bỏ qua chúng.
- Nếu nhiều ngữ cảnh cùng chứa thông tin cần thiết, hãy tổng hợp chúng để trả lời đầy đủ.
- Không suy đoán, không bổ sung kiến thức pháp luật từ bên ngoài.
- Nếu ngữ cảnh không chứa đủ thông tin để trả lời câu hỏi, hãy trả lời: "Không đủ thông tin trong ngữ cảnh."
- Trả lời trực tiếp câu hỏi, chính xác và ngắn gọn.

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
        temperature: float = 0.1,
        top_p: float = 0.9,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.device_setting = device
        self.torch_dtype_setting = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

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
            trust_remote_code=True,
        )

        if self.torch_dtype_setting == "bfloat16":
            dtype = torch.bfloat16
        elif self.torch_dtype_setting == "float16":
            dtype = torch.float16
        else:
            dtype = "auto"

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
            trust_remote_code=True,
        )
        self._model.eval()
        print(f"[Generator] Tải model thành công trên device: {self._model.device}", file=sys.stderr)

    def generate(self, context: str, question: str) -> str:
        """Sinh câu trả lời cho một cặp (context, question)."""
        self._load_model()
        import torch

        messages = build_chat_messages(context=context, question=question)
        if hasattr(self._tokenizer, "apply_chat_template"):
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = format_raw_qwen_prompt(context=context, question=question)

        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self._tokenizer.eos_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }

        if self.temperature > 0.0:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = self.temperature
            generate_kwargs["top_p"] = self.top_p
        else:
            generate_kwargs["do_sample"] = False

        with torch.inference_mode():
            outputs = self._model.generate(**inputs, **generate_kwargs)

        input_len = inputs["input_ids"].shape[1]
        response_tokens = outputs[0][input_len:]
        response_text = self._tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
        return response_text
