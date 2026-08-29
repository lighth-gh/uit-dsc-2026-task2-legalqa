from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from .metrics import aggregate_scores
from .pipeline import LegalQABaseline
from .storage import SearchIndex, build_index, load_qa, write_predictions


VALID_MODES = ["extractive", "knn", "hybrid", "rag", "hybrid_rag"]


def _add_pipeline_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--max-answer-words", type=int, default=520)
    parser.add_argument("--knn-threshold", type=float, default=0.72)
    parser.add_argument("--context-top-k", type=int, default=3, help="Số chunk luật đưa vào ngữ cảnh LLM")
    parser.add_argument(
        "--generator-model",
        type=str,
        default="AITeamVN/Vi-Qwen2-1.5B-RAG",
        help="Tên mô hình LLM HuggingFace hoặc đường dẫn cục bộ",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Thiết bị chạy LLM",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        choices=["auto", "bfloat16", "float16"],
        help="Kiểu dữ liệu trọng số",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Số token tối đa sinh ra cho câu trả lời",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Nhiệt độ lấy mẫu (0.0: greedy search)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling",
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m legalqa_baseline",
        description="Baseline BM25/FTS5 & RAG LLM cho UIT DSC 2026 LegalQA",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-index", help="Lập chỉ mục corpus và train")
    build.add_argument("--contexts", required=True)
    build.add_argument("--train", required=True)
    build.add_argument("--db", required=True)
    build.add_argument("--max-chunk-words", type=int, default=620)
    build.add_argument("--overlap-words", type=int, default=100)
    build.add_argument("--force", action="store_true")

    predict = subparsers.add_parser("predict", help="Sinh tệp submission")
    predict.add_argument("--input", required=True)
    predict.add_argument("--db", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument(
        "--mode", choices=VALID_MODES, default="hybrid"
    )
    _add_pipeline_args(predict)

    validate = subparsers.add_parser("validate", help="Leave-one-out validation")
    validate.add_argument("--train", required=True)
    validate.add_argument("--db", required=True)
    validate.add_argument("--output", required=True)
    validate.add_argument("--modes", default="extractive,knn,hybrid")
    validate.add_argument("--limit", type=int, default=300)
    validate.add_argument("--seed", type=int, default=2026)
    _add_pipeline_args(validate)

    inspect = subparsers.add_parser("inspect", help="Xem evidence của một câu hỏi")
    inspect.add_argument("--db", required=True)
    inspect.add_argument("--question", required=True)
    inspect.add_argument(
        "--mode", choices=VALID_MODES, default="hybrid"
    )
    _add_pipeline_args(inspect)

    score = subparsers.add_parser("score", help="Metric nhẹ, không cần NLTK")
    score.add_argument("--reference", required=True)
    score.add_argument("--prediction", required=True)
    return parser


def _pipeline(args: argparse.Namespace, index: SearchIndex, need_generator: bool = False) -> LegalQABaseline:
    generator = None
    if need_generator:
        from .generator import ViQwenRAGGenerator
        generator = ViQwenRAGGenerator(
            model_name_or_path=getattr(args, "generator_model", "AITeamVN/Vi-Qwen2-1.5B-RAG"),
            device=getattr(args, "device", "auto"),
            torch_dtype=getattr(args, "torch_dtype", "auto"),
            max_new_tokens=getattr(args, "max_new_tokens", 512),
            temperature=getattr(args, "temperature", 0.1),
            top_p=getattr(args, "top_p", 0.9),
        )

    return LegalQABaseline(
        index=index,
        top_k=args.top_k,
        max_answer_words=args.max_answer_words,
        knn_threshold=args.knn_threshold,
        generator=generator,
        context_top_k=getattr(args, "context_top_k", 3),
    )


def command_build(args: argparse.Namespace) -> int:
    stats = build_index(
        contexts_path=args.contexts,
        train_path=args.train,
        db_path=args.db,
        max_chunk_words=args.max_chunk_words,
        overlap_words=args.overlap_words,
        force=args.force,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


def command_predict(args: argparse.Namespace) -> int:
    data = load_qa(args.input)
    predictions: dict[str, str] = {}
    routes: dict[str, int] = {}
    started = time.time()
    need_generator = args.mode in ("rag", "hybrid_rag")
    with SearchIndex(args.db) as index:
        pipeline = _pipeline(args, index, need_generator=need_generator)
        for number, (sample_id, item) in enumerate(data.items(), start=1):
            result = pipeline.predict_one(item["question"], mode=args.mode)
            predictions[sample_id] = result.answer
            routes[result.route] = routes.get(result.route, 0) + 1
            if number % 100 == 0:
                print(f"[predict] {number:,}/{len(data):,}", file=sys.stderr, flush=True)
    write_predictions(args.output, predictions)
    summary = {
        "samples": len(predictions),
        "mode": args.mode,
        "routes": routes,
        "elapsed_seconds": round(time.time() - started, 2),
        "output": str(Path(args.output).resolve()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def command_validate(args: argparse.Namespace) -> int:
    data = load_qa(args.train)
    sample_ids = list(data)
    rng = random.Random(args.seed)
    rng.shuffle(sample_ids)
    if args.limit > 0:
        sample_ids = sample_ids[: args.limit]
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    invalid = set(modes) - set(VALID_MODES)
    if invalid:
        raise ValueError(f"Mode không hợp lệ: {sorted(invalid)}")

    report: dict[str, object] = {
        "config": {
            "samples": len(sample_ids),
            "seed": args.seed,
            "top_k": args.top_k,
            "max_answer_words": args.max_answer_words,
            "knn_threshold": args.knn_threshold,
            "context_top_k": args.context_top_k,
            "generator_model": args.generator_model,
        },
        "results": {},
    }
    need_generator = any(m in ("rag", "hybrid_rag") for m in modes)
    with SearchIndex(args.db) as index:
        pipeline = _pipeline(args, index, need_generator=need_generator)
        for mode in modes:
            predictions: list[str] = []
            references: list[str] = []
            routes: dict[str, int] = {}
            for number, sample_id in enumerate(sample_ids, start=1):
                item = data[sample_id]
                result = pipeline.predict_one(
                    item["question"], mode=mode, exclude_id=sample_id
                )
                predictions.append(result.answer)
                references.append(str(item.get("answer") or ""))
                routes[result.route] = routes.get(result.route, 0) + 1
                if number % 100 == 0:
                    print(
                        f"[validate:{mode}] {number:,}/{len(sample_ids):,}",
                        file=sys.stderr,
                        flush=True,
                    )
            scores = aggregate_scores(predictions, references)
            scores["routes"] = routes
            report["results"][mode] = scores  # type: ignore[index]

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def command_inspect(args: argparse.Namespace) -> int:
    need_generator = args.mode in ("rag", "hybrid_rag")
    with SearchIndex(args.db) as index:
        result = _pipeline(args, index, need_generator=need_generator).predict_one(
            args.question, mode=args.mode
        )
    print(
        json.dumps(
            {
                "route": result.route,
                "confidence": result.confidence,
                "evidence": result.evidence,
                "answer": result.answer,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def command_score(args: argparse.Namespace) -> int:
    reference = load_qa(args.reference)
    with Path(args.prediction).open("r", encoding="utf-8") as handle:
        prediction_data = json.load(handle)
    if set(reference) != set(prediction_data):
        missing = set(reference) - set(prediction_data)
        extra = set(prediction_data) - set(reference)
        raise ValueError(f"ID lệch: missing={len(missing)}, extra={len(extra)}")
    predictions = [str(prediction_data[key]["answer"]) for key in reference]
    references = [str(reference[key].get("answer") or "") for key in reference]
    print(json.dumps(aggregate_scores(predictions, references), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = make_parser().parse_args(argv)
    commands = {
        "build-index": command_build,
        "predict": command_predict,
        "validate": command_validate,
        "inspect": command_inspect,
        "score": command_score,
    }
    try:
        return commands[args.command](args)
    except (ValueError, FileNotFoundError, FileExistsError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

