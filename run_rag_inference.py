#!/usr/bin/env python3
"""
UIT DSC 2026 - Task 2 LegalQA: RAG Generator Inference Script
Sử dụng mô hình AITeamVN/Vi-Qwen2-1.5B-RAG kết hợp BM25 / Context Retrieval.
Chạy được trên cả GPU cục bộ, Google Colab (T4/A100) và Kaggle.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from legalqa_baseline.generator import SYSTEM_PROMPT, RAG_TEMPLATE, ViQwenRAGGenerator
from legalqa_baseline.pipeline import LegalQABaseline
from legalqa_baseline.storage import SearchIndex, load_qa, write_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chạy RAG generation với Vi-Qwen2-1.5B-RAG cho UIT DSC 2026"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/public-official.json",
        help="Đường dẫn file câu hỏi (public-official.json hoặc train.json)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="artifacts/legalqa.sqlite",
        help="Đường dẫn SQLite FTS5 database",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/submission_rag.json",
        help="Đường dẫn file kết quả submission",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="AITeamVN/Vi-Qwen2-1.5B-RAG",
        help="HuggingFace model ID hoặc local path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Thiết bị chạy model",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        choices=["auto", "bfloat16", "float16"],
        help="Kiểu dữ liệu float",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="Số lượng candidate chunks truy xuất từ BM25",
    )
    parser.add_argument(
        "--context-top-k",
        type=int,
        default=3,
        help="Số lượng context chunks đưa vào prompt LLM",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Số token tối đa sinh ra",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Nhiệt độ sampling",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="rag",
        choices=["rag", "hybrid_rag"],
        help="Chế độ chạy (rag thuần hoặc hybrid_rag kết hợp KNN train)",
    )
    parser.add_argument(
        "--knn-threshold",
        type=float,
        default=0.72,
        help="Ngưỡng tương đồng KNN",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Giới hạn số mẫu xử lý (0 = toàn bộ)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_path = Path(args.input)
    db_path = Path(args.db)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Lỗi: Không tìm thấy tệp đầu vào: {input_path}", file=sys.stderr)
        return 1

    if not db_path.exists():
        print(
            f"Lỗi: Không tìm thấy database: {db_path}.\n"
            f"Vui lòng chạy `python -m legalqa_baseline build-index` trước.",
            file=sys.stderr,
        )
        return 1

    print("=" * 60)
    print("UIT DSC 2026 - Task 2 LegalQA: RAG Generator Pipeline")
    print(f"• Input: {input_path}")
    print(f"• Database: {db_path}")
    print(f"• Model: {args.model_name}")
    print(f"• Mode: {args.mode}")
    print(f"• Device: {args.device}")
    print(f"• Context chunks: {args.context_top_k} / {args.top_k}")
    print("=" * 60)

    data = load_qa(input_path)
    sample_ids = list(data.keys())
    if args.limit > 0:
        sample_ids = sample_ids[: args.limit]

    print(f"[*] Đang tải mô hình LLM: {args.model_name}...")
    generator = ViQwenRAGGenerator(
        model_name_or_path=args.model_name,
        device=args.device,
        torch_dtype=args.torch_dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    predictions: dict[str, str] = {}
    routes: dict[str, int] = {}
    started = time.time()

    with SearchIndex(db_path) as index:
        pipeline = LegalQABaseline(
            index=index,
            top_k=args.top_k,
            knn_threshold=args.knn_threshold,
            generator=generator,
            context_top_k=args.context_top_k,
        )

        total = len(sample_ids)
        for idx, sample_id in enumerate(sample_ids, start=1):
            item = data[sample_id]
            question = item["question"]
            pred = pipeline.predict_one(question, mode=args.mode)
            predictions[sample_id] = pred.answer
            routes[pred.route] = routes.get(pred.route, 0) + 1

            if idx % 10 == 0 or idx == total:
                elapsed = time.time() - started
                avg_time = elapsed / idx
                print(
                    f"[{idx:,}/{total:,}] | Đã qua: {elapsed:.1f}s | "
                    f"Tốc độ: {avg_time:.2f}s/câu | Routes: {routes}",
                    flush=True,
                )

    write_predictions(output_path, predictions)
    total_elapsed = time.time() - started
    print("=" * 60)
    print(f"[✓] Hoàn thành sinh câu trả lời cho {len(predictions):,} câu hỏi.")
    print(f"[✓] Tổng thời gian: {total_elapsed:.2f} giây.")
    print(f"[✓] File kết quả: {output_path.resolve()}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
