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
    parser.add_argument("--bm25-top-k", type=int, default=50, help="Số candidate truy xuất bằng BM25")
    parser.add_argument("--dense-top-k", type=int, default=50, help="Số candidate truy xuất bằng Dense FAISS")
    parser.add_argument("--rrf-k", type=int, default=60, help="Hệ số RRF fusion k (mặc định 60)")
    parser.add_argument("--rrf-top-k", type=int, default=12, help="Số candidate sau khi RRF fusion")
    parser.add_argument("--rerank-top-k", type=int, default=3, help="Số chunk chọn lọc sau khi qua Reranker")
    parser.add_argument(
        "--dense-index",
        type=str,
        default=None,
        help="Đường dẫn đến file FAISS / NumPy Dense Index (.faiss / .npy)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="AITeamVN/Vietnamese_Embedding_v2",
        help="Tên mô hình Embedding tiếng Việt (HuggingFace ID hoặc path)",
    )
    parser.add_argument(
        "--reranker-model",
        type=str,
        default="AITeamVN/Vietnamese_Reranker",
        help="Tên mô hình Reranker tiếng Việt (HuggingFace ID hoặc path)",
    )
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
        help="Thiết bị chạy model",
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
        description="Baseline BM25, Dense FAISS, RRF & RAG LLM cho UIT DSC 2026 LegalQA",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-index", help="Lập chỉ mục BM25 FTS5 corpus và train")
    build.add_argument("--contexts", required=True)
    build.add_argument("--train", required=True)
    build.add_argument("--db", required=True)
    build.add_argument("--max-chunk-words", type=int, default=620)
    build.add_argument("--overlap-words", type=int, default=100)
    build.add_argument("--force", action="store_true")

    build_dense = subparsers.add_parser("build-dense-index", help="Lập chỉ mục Dense Vector FAISS với Vietnamese_Embedding_v2")
    build_dense.add_argument("--contexts", required=True)
    build_dense.add_argument("--dense-index", required=True)
    build_dense.add_argument(
        "--embedding-model",
        type=str,
        default="AITeamVN/Vietnamese_Embedding_v2",
        help="Tên mô hình Embedding tiếng Việt",
    )
    build_dense.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    build_dense.add_argument("--batch-size", type=int, default=64)
    build_dense.add_argument("--max-chunk-words", type=int, default=620)
    build_dense.add_argument("--overlap-words", type=int, default=100)
    build_dense.add_argument("--force", action="store_true")

    predict = subparsers.add_parser("predict", help="Sinh tệp submission")
    predict.add_argument("--input", required=True)
    predict.add_argument("--db", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument(
        "--mode", choices=VALID_MODES, default="hybrid"
    )
    predict.add_argument("--resume", action="store_true", help="Tiếp tục từ checkpoint đã lưu nếu có")
    predict.add_argument("--checkpoint-interval", type=int, default=10, help="Số mẫu lưu checkpoint định kỳ")
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
    dense_index = None
    embedding_model = None
    reranker = None

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

        dense_index_path = getattr(args, "dense_index", None)
        if dense_index_path:
            p = Path(dense_index_path)
            meta_p = p.with_suffix(".meta.json") if p.suffix in (".faiss", ".npy") else p.with_name(f"{p.stem}.meta.json")
            if p.exists() or meta_p.exists():
                try:
                    from .dense import DenseVectorIndex, VietnameseEmbeddingModel
                    dense_index = DenseVectorIndex.load(p)
                    embedding_model = VietnameseEmbeddingModel(
                        model_name_or_path=getattr(args, "embedding_model", "AITeamVN/Vietnamese_Embedding_v2"),
                        device=getattr(args, "device", "auto"),
                    )
                    print(f"[pipeline] Đã nạp Dense Index: {len(dense_index.metadata):,} chunks", file=sys.stderr)
                except Exception as exc:
                    print(f"[pipeline] Cảnh báo không nạp được dense index ({exc}), fallback BM25.", file=sys.stderr)

        reranker_name = getattr(args, "reranker_model", None)
        if reranker_name:
            try:
                from .reranker import VietnameseReranker
                reranker = VietnameseReranker(
                    model_name_or_path=reranker_name,
                    device=getattr(args, "device", "auto"),
                )
                print(f"[pipeline] Đã nạp Vietnamese_Reranker: {reranker_name}", file=sys.stderr)
            except Exception as exc:
                print(f"[pipeline] Cảnh báo không nạp được reranker ({exc}), fallback RRF order.", file=sys.stderr)

    return LegalQABaseline(
        index=index,
        top_k=args.top_k,
        max_answer_words=args.max_answer_words,
        knn_threshold=args.knn_threshold,
        generator=generator,
        context_top_k=getattr(args, "context_top_k", 3),
        dense_index=dense_index,
        embedding_model=embedding_model,
        reranker=reranker,
        bm25_top_k=getattr(args, "bm25_top_k", 50),
        dense_top_k=getattr(args, "dense_top_k", 50),
        rrf_k=getattr(args, "rrf_k", 60),
        rrf_top_k=getattr(args, "rrf_top_k", 12),
        rerank_top_k=getattr(args, "rerank_top_k", 3),
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


def command_build_dense_index(args: argparse.Namespace) -> int:
    from .dense import build_dense_index
    stats = build_dense_index(
        contexts_path=args.contexts,
        output_index_path=args.dense_index,
        embedding_model_name=args.embedding_model,
        device=args.device,
        batch_size=args.batch_size,
        max_chunk_words=args.max_chunk_words,
        overlap_words=args.overlap_words,
        force=args.force,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


def command_predict(args: argparse.Namespace) -> int:
    data = load_qa(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path.with_suffix(".checkpoint.json")

    predictions: dict[str, str] = {}
    routes: dict[str, int] = {}

    if getattr(args, "resume", False) and checkpoint_path.exists():
        try:
            with checkpoint_path.open("r", encoding="utf-8") as f:
                saved = json.load(f)
                predictions = {str(k): str(v.get("answer", "")) for k, v in saved.items()}
                print(
                    f"[predict] Đã nạp lại checkpoint: {len(predictions):,}/{len(data):,} câu đã xong.",
                    file=sys.stderr,
                    flush=True,
                )
        except Exception:
            predictions = {}

    started = time.time()
    need_generator = args.mode in ("rag", "hybrid_rag")
    total = len(data)
    interval = getattr(args, "checkpoint_interval", 10)

    use_tqdm = False
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    with SearchIndex(args.db) as index:
        pipeline = _pipeline(args, index, need_generator=need_generator)
        items = list(data.items())
        pbar = tqdm(items, desc=f"Predicting [{args.mode}]", total=total, unit="câu") if use_tqdm else None

        processed_count = 0
        for number, (sample_id, item) in enumerate(items, start=1):
            if sample_id in predictions:
                if pbar:
                    pbar.update(1)
                continue

            t0 = time.time()
            result = pipeline.predict_one(item["question"], mode=args.mode)
            predictions[sample_id] = result.answer
            routes[result.route] = routes.get(result.route, 0) + 1
            processed_count += 1
            sample_time = time.time() - t0

            # Lưu checkpoint định kỳ
            if processed_count % interval == 0 or number == total:
                write_predictions(checkpoint_path, predictions)

            if pbar:
                pbar.set_postfix({
                    "routes": str(routes),
                    "last": f"{sample_time:.2f}s",
                })
                pbar.update(1)
            else:
                if number <= 5 or number % 5 == 0 or number == total:
                    elapsed = time.time() - started
                    avg_time = elapsed / max(1, processed_count)
                    remaining = total - number
                    eta = remaining * avg_time
                    percent = (number / total) * 100
                    print(
                        f"[predict:{args.mode}] {number:,}/{total:,} ({percent:.1f}%) | "
                        f"Tốc độ: {avg_time:.2f}s/câu | ETA: {eta:.0f}s | "
                        f"Routes: {routes}",
                        file=sys.stderr,
                        flush=True,
                    )

        if pbar:
            pbar.close()

    write_predictions(output_path, predictions)
    if checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
        except Exception:
            pass

    summary = {
        "samples": len(predictions),
        "mode": args.mode,
        "routes": routes,
        "elapsed_seconds": round(time.time() - started, 2),
        "output": str(output_path.resolve()),
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
        "build-dense-index": command_build_dense_index,
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

