from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from .metrics import aggregate_official_scores, aggregate_scores
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
    parser.add_argument("--rrf-top-k", type=int, default=50, help="Số candidate sau khi RRF fusion")
    parser.add_argument("--rerank-top-k", type=int, default=3, help="Số chunk chọn lọc sau khi qua Reranker")
    parser.add_argument("--dense-query-max-length", type=int, default=256)
    parser.add_argument("--reranker-max-length", type=int, default=2304)
    parser.add_argument(
        "--allow-retrieval-fallback",
        action="store_true",
        help="Cho phép tiếp tục bằng BM25/RRF khi Dense hoặc Reranker lỗi",
    )
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
        default=0.0,
        help="Nhiệt độ lấy mẫu (0.0: greedy search)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling",
    )
    parser.add_argument("--max-input-tokens", type=int, default=7168)
    parser.add_argument("--generation-seed", type=int, default=2026)


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
    build_dense.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch tổng cho dense encoder; 8 là mức an toàn cho 2x Tesla T4",
    )
    build_dense.add_argument("--max-chunk-words", type=int, default=620)
    build_dense.add_argument("--overlap-words", type=int, default=100)
    build_dense.add_argument("--embedding-max-length", type=int, default=2048)
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
    validate.add_argument(
        "--official-metrics",
        action="store_true",
        help="TÃ­nh thÃªm METEOR/ROUGE-L theo scoring program cá»§a BTC",
    )
    _add_pipeline_args(validate)

    retrieval_eval = subparsers.add_parser(
        "evaluate-retrieval",
        help="TÃ­nh pseudo Recall@K cho BM25, Dense, RRF vÃ  Reranker",
    )
    retrieval_eval.add_argument("--train", required=True)
    retrieval_eval.add_argument("--db", required=True)
    retrieval_eval.add_argument("--output", required=True)
    retrieval_eval.add_argument("--limit", type=int, default=100)
    retrieval_eval.add_argument("--seed", type=int, default=2026)
    retrieval_eval.add_argument("--ks", default="1,3,5")
    retrieval_eval.add_argument("--dense-index", default=None)
    retrieval_eval.add_argument(
        "--embedding-model",
        default="AITeamVN/Vietnamese_Embedding_v2",
    )
    retrieval_eval.add_argument(
        "--reranker-model",
        default="AITeamVN/Vietnamese_Reranker",
    )
    retrieval_eval.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    retrieval_eval.add_argument("--bm25-top-k", type=int, default=50)
    retrieval_eval.add_argument("--dense-top-k", type=int, default=50)
    retrieval_eval.add_argument("--rrf-k", type=int, default=60)
    retrieval_eval.add_argument("--rrf-top-k", type=int, default=50)
    retrieval_eval.add_argument("--dense-query-max-length", type=int, default=256)
    retrieval_eval.add_argument("--reranker-max-length", type=int, default=2304)
    retrieval_eval.add_argument("--gold-candidate-k", type=int, default=100)
    retrieval_eval.add_argument("--gold-max-chunks", type=int, default=5)
    retrieval_eval.add_argument("--gold-min-score", type=float, default=0.20)
    retrieval_eval.add_argument("--gold-relative-score", type=float, default=0.85)
    retrieval_eval.add_argument("--gold-min-answer-tokens", type=int, default=8)

    inspect = subparsers.add_parser("inspect", help="Xem evidence của một câu hỏi")
    inspect.add_argument("--db", required=True)
    inspect.add_argument("--question", required=True)
    inspect.add_argument(
        "--mode", choices=VALID_MODES, default="hybrid"
    )
    _add_pipeline_args(inspect)

    score = subparsers.add_parser("score", help="Tính toàn bộ các metrics tương đồng câu trả lời")
    score.add_argument("--reference", required=True)
    score.add_argument("--prediction", required=True)
    score.add_argument(
        "--official-metrics",
        action="store_true",
        help="Tính thêm METEOR/ROUGE-L chính thức theo scoring program của BTC",
    )
    return parser


def _pipeline(args: argparse.Namespace, index: SearchIndex, need_generator: bool = False) -> LegalQABaseline:
    generator = None
    dense_index = None
    embedding_model = None
    reranker = None
    positive_values = {
        "top_k": args.top_k,
        "max_answer_words": args.max_answer_words,
        "context_top_k": getattr(args, "context_top_k", 3),
        "bm25_top_k": getattr(args, "bm25_top_k", 50),
        "dense_top_k": getattr(args, "dense_top_k", 50),
        "rrf_k": getattr(args, "rrf_k", 60),
        "rrf_top_k": getattr(args, "rrf_top_k", 50),
        "rerank_top_k": getattr(args, "rerank_top_k", 3),
        "dense_query_max_length": getattr(args, "dense_query_max_length", 256),
        "reranker_max_length": getattr(args, "reranker_max_length", 2304),
    }
    invalid = {name: value for name, value in positive_values.items() if value <= 0}
    if invalid:
        raise ValueError(f"Các tham số phải lớn hơn 0: {invalid}")
    if positive_values["context_top_k"] > positive_values["rerank_top_k"]:
        raise ValueError("context_top_k không được lớn hơn rerank_top_k")
    if positive_values["rerank_top_k"] > positive_values["rrf_top_k"]:
        raise ValueError("rerank_top_k không được lớn hơn rrf_top_k")
    if not 0.0 <= args.knn_threshold <= 1.0:
        raise ValueError("knn_threshold phải nằm trong [0, 1]")
    allow_fallback = bool(getattr(args, "allow_retrieval_fallback", False))

    if need_generator:
        if getattr(args, "max_new_tokens", 512) <= 0:
            raise ValueError("max_new_tokens phải lớn hơn 0")
        if getattr(args, "max_input_tokens", 7168) <= 0:
            raise ValueError("max_input_tokens phải lớn hơn 0")
        if getattr(args, "temperature", 0.0) < 0:
            raise ValueError("temperature không được âm")
        if not 0.0 < getattr(args, "top_p", 0.9) <= 1.0:
            raise ValueError("top_p phải nằm trong (0, 1]")
        from .generator import ViQwenRAGGenerator
        generator = ViQwenRAGGenerator(
            model_name_or_path=getattr(args, "generator_model", "AITeamVN/Vi-Qwen2-1.5B-RAG"),
            device=getattr(args, "device", "auto"),
            torch_dtype=getattr(args, "torch_dtype", "auto"),
            max_new_tokens=getattr(args, "max_new_tokens", 512),
            temperature=getattr(args, "temperature", 0.0),
            top_p=getattr(args, "top_p", 0.9),
            max_input_tokens=getattr(args, "max_input_tokens", 7168),
            seed=getattr(args, "generation_seed", 2026),
        )

        dense_index_path = getattr(args, "dense_index", None)
        if dense_index_path:
            p = Path(dense_index_path)
            meta_p = p.with_suffix(".meta.json") if p.suffix in (".faiss", ".npy") else p.with_name(f"{p.stem}.meta.json")
            if not (p.exists() or meta_p.exists()):
                message = f"Không tìm thấy Dense index/metadata tại {p}"
                if allow_fallback:
                    print(f"[pipeline] {message}; fallback BM25.", file=sys.stderr)
                else:
                    raise FileNotFoundError(message)
            else:
                try:
                    from .dense import DenseVectorIndex, VietnameseEmbeddingModel
                    embedding_name = getattr(
                        args, "embedding_model", "AITeamVN/Vietnamese_Embedding_v2"
                    )
                    dense_index = DenseVectorIndex.load(
                        p,
                        expected_model_name=embedding_name,
                    )
                    dense_index.validate_against_bm25(index.metadata())
                    embedding_model = VietnameseEmbeddingModel(
                        model_name_or_path=embedding_name,
                        device=getattr(args, "device", "auto"),
                    )
                    print(f"[pipeline] Đã nạp Dense Index: {len(dense_index.metadata):,} chunks", file=sys.stderr)
                except Exception as exc:
                    if not allow_fallback:
                        raise
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
        rrf_top_k=getattr(args, "rrf_top_k", 50),
        rerank_top_k=getattr(args, "rerank_top_k", 3),
        dense_query_max_length=getattr(args, "dense_query_max_length", 256),
        reranker_max_length=getattr(args, "reranker_max_length", 2304),
        allow_retrieval_fallback=allow_fallback,
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
        embedding_max_length=args.embedding_max_length,
        force=args.force,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


def _load_checkpoint(
    checkpoint_path: Path,
    valid_ids: set[str],
) -> tuple[dict[str, str], dict[str, int]]:
    with checkpoint_path.open("r", encoding="utf-8") as f:
        saved = json.load(f)
    routes: dict[str, int] = {}
    if isinstance(saved, dict) and saved.get("schema_version") == 1:
        raw_predictions = saved.get("predictions", {})
        raw_routes = saved.get("routes", {})
        if not isinstance(raw_predictions, dict) or not isinstance(raw_routes, dict):
            raise ValueError("Checkpoint schema không hợp lệ")
        routes = {
            str(route): int(count)
            for route, count in raw_routes.items()
            if int(count) >= 0
        }
    elif isinstance(saved, dict):
        # Tương thích checkpoint cũ có cùng schema với submission.
        raw_predictions = saved
    else:
        raise ValueError("Checkpoint phải là JSON object")

    predictions: dict[str, str] = {}
    for sample_id, item in raw_predictions.items():
        key = str(sample_id)
        if key not in valid_ids or not isinstance(item, dict):
            continue
        answer = item.get("answer")
        if isinstance(answer, str) and answer.strip():
            predictions[key] = answer
    if routes and sum(routes.values()) != len(predictions):
        routes = {}
    if predictions and not routes:
        routes["resumed_unknown"] = len(predictions)
    return predictions, routes


def _write_checkpoint(
    checkpoint_path: Path,
    predictions: dict[str, str],
    routes: dict[str, int],
) -> None:
    payload = {
        "schema_version": 1,
        "predictions": {
            str(key): {"answer": str(answer)} for key, answer in predictions.items()
        },
        "routes": routes,
    }
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    tmp_path.replace(checkpoint_path)


def command_predict(args: argparse.Namespace) -> int:
    data = load_qa(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path.with_suffix(".checkpoint.json")

    predictions: dict[str, str] = {}
    routes: dict[str, int] = {}

    if getattr(args, "resume", False) and checkpoint_path.exists():
        try:
            predictions, routes = _load_checkpoint(checkpoint_path, set(data))
            print(
                f"[predict] Đã nạp lại checkpoint: {len(predictions):,}/{len(data):,} câu đã xong.",
                file=sys.stderr,
                flush=True,
            )
        except Exception as exc:
            print(f"[predict] Bỏ checkpoint không hợp lệ: {exc}", file=sys.stderr)
            predictions = {}
            routes = {}

    started = time.time()
    need_generator = args.mode in ("rag", "hybrid_rag")
    total = len(data)
    interval = getattr(args, "checkpoint_interval", 10)
    if interval <= 0:
        raise ValueError("checkpoint_interval phải lớn hơn 0")

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
                _write_checkpoint(checkpoint_path, predictions, routes)

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
        if processed_count:
            _write_checkpoint(checkpoint_path, predictions, routes)

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
        "retrieval": {
            "dense_active": pipeline.dense_index is not None,
            "reranker_active": pipeline.reranker is not None,
            "bm25_top_k": pipeline.bm25_top_k,
            "dense_top_k": pipeline.dense_top_k,
            "rrf_top_k": pipeline.rrf_top_k,
            "rerank_top_k": pipeline.rerank_top_k,
            "context_top_k": pipeline.context_top_k,
        },
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
            if getattr(args, "official_metrics", False):
                scores.update(aggregate_official_scores(predictions, references))
            scores["routes"] = routes
            report["results"][mode] = scores  # type: ignore[index]

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def command_evaluate_retrieval(args: argparse.Namespace) -> int:
    from .retrieval_eval import evaluate_retrieval

    data = load_qa(args.train)
    try:
        ks = tuple(int(value.strip()) for value in args.ks.split(",") if value.strip())
    except ValueError as exc:
        raise ValueError("--ks pháº£i lÃ  danh sÃ¡ch sá»‘ nguyÃªn, vÃ­ dá»¥ 1,3,5") from exc

    dense_index = None
    embedding_model = None
    reranker = None
    with SearchIndex(args.db) as index:
        if args.dense_index:
            from .dense import DenseVectorIndex, VietnameseEmbeddingModel

            dense_index = DenseVectorIndex.load(
                args.dense_index,
                expected_model_name=args.embedding_model,
            )
            dense_index.validate_against_bm25(index.metadata())
            embedding_model = VietnameseEmbeddingModel(
                model_name_or_path=args.embedding_model,
                device=args.device,
            )
        if args.reranker_model:
            from .reranker import VietnameseReranker

            reranker = VietnameseReranker(
                model_name_or_path=args.reranker_model,
                device=args.device,
            )

        report = evaluate_retrieval(
            data,
            index,
            dense_index=dense_index,
            embedding_model=embedding_model,
            reranker=reranker,
            limit=args.limit,
            seed=args.seed,
            ks=ks,
            bm25_top_k=args.bm25_top_k,
            dense_top_k=args.dense_top_k,
            rrf_k=args.rrf_k,
            rrf_top_k=args.rrf_top_k,
            dense_query_max_length=args.dense_query_max_length,
            reranker_max_length=args.reranker_max_length,
            gold_candidate_k=args.gold_candidate_k,
            gold_max_chunks=args.gold_max_chunks,
            gold_min_score=args.gold_min_score,
            gold_relative_score=args.gold_relative_score,
            gold_min_answer_tokens=args.gold_min_answer_tokens,
        )

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = target.with_suffix(target.suffix + ".tmp")
    with tmp_target.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    tmp_target.replace(target)
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
    scores = aggregate_scores(predictions, references)
    if getattr(args, "official_metrics", False):
        scores.update(aggregate_official_scores(predictions, references))
    print(json.dumps(scores, ensure_ascii=False, indent=2))
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
        "evaluate-retrieval": command_evaluate_retrieval,
        "inspect": command_inspect,
        "score": command_score,
    }
    try:
        return commands[args.command](args)
    except (
        ValueError,
        FileNotFoundError,
        FileExistsError,
        ImportError,
        RuntimeError,
        json.JSONDecodeError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
