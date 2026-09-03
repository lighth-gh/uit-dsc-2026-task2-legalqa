from __future__ import annotations

import argparse
import json
import random
import sqlite3
import statistics
import sys
import time
from pathlib import Path

from .baseline_lock import (
    aggregate_retrieval_items,
    file_sha256,
    load_locked_split,
    load_regression_samples,
    make_baseline_manifest,
    score_answer,
    score_retrieval_trace,
    write_locked_manifest,
)
from .metrics import aggregate_official_scores, aggregate_scores, official_scores
from .pipeline import LegalQABaseline, prediction_audit_record
from .storage import SearchIndex, build_index, load_qa, write_predictions
from .text import clean_answer


VALID_MODES = ["extractive", "knn", "hybrid", "rag", "hybrid_rag"]


def _add_pipeline_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--max-answer-words", type=int, default=520)
    parser.add_argument(
        "--max-long-answer-words",
        type=int,
        default=640,
        help="Giới hạn số từ cho đáp án extractive dài",
    )
    parser.add_argument("--knn-threshold", type=float, default=0.72)
    parser.add_argument("--context-top-k", type=int, default=3, help="Số chunk luật đưa vào ngữ cảnh LLM")
    parser.add_argument("--bm25-top-k", type=int, default=50, help="Số candidate truy xuất bằng BM25")
    parser.add_argument("--dense-top-k", type=int, default=50, help="Số candidate truy xuất bằng Dense FAISS")
    parser.add_argument("--rrf-k", type=int, default=60, help="Hệ số RRF fusion k (mặc định 60)")
    parser.add_argument("--rrf-top-k", type=int, default=50, help="Số candidate sau khi RRF fusion")
    parser.add_argument(
        "--reranker-candidate-k",
        type=int,
        default=20,
        help="Số candidate đầu RRF thực sự đưa qua cross-encoder",
    )
    parser.add_argument("--rerank-top-k", type=int, default=3, help="Số chunk chọn lọc sau khi qua Reranker")
    parser.add_argument("--dense-query-max-length", type=int, default=256)
    parser.add_argument("--reranker-max-length", type=int, default=1024)
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
        "--embedding-revision",
        type=str,
        default=None,
        help="Commit/revision Hugging Face cố định của embedding model",
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
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--min-llm-answer-tokens", type=int, default=8)
    parser.add_argument("--token-limit-retry-tokens", type=int, default=768, choices=[768, 1024])
    parser.add_argument("--guarded-knn-threshold", type=float, default=0.90)
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
    build_dense.add_argument("--embedding-revision", type=str, default=None)
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
    build_dense.add_argument(
        "--resume",
        action="store_true",
        help="Tiếp tục dense build từ các part checkpoint đã lưu",
    )
    build_dense.add_argument(
        "--checkpoint-chunks",
        type=int,
        default=4096,
        help="Số chunk embedding giữa hai checkpoint dense build",
    )
    build_dense.add_argument("--force", action="store_true")

    predict = subparsers.add_parser("predict", help="Sinh tệp submission")
    predict.add_argument("--input", required=True)
    predict.add_argument("--db", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument(
        "--audit-output",
        default=None,
        help="JSONL audit từng ID; mặc định là <output>.audit.jsonl",
    )
    predict.add_argument(
        "--mode", choices=VALID_MODES, default="hybrid_rag"
    )
    predict.add_argument("--resume", action="store_true", help="Tiếp tục từ checkpoint đã lưu nếu có")
    predict.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1,
        help="Số mẫu giữa hai lần ghi checkpoint + partial submission",
    )
    _add_pipeline_args(predict)

    diagnose_retrieval = subparsers.add_parser(
        "diagnose-retrieval",
        help="Chạy BM25 + Dense + RRF + Reranker, không chạy generator",
    )
    diagnose_retrieval.add_argument("--input", required=True)
    diagnose_retrieval.add_argument("--db", required=True)
    diagnose_retrieval.add_argument("--output", required=True)
    _add_pipeline_args(diagnose_retrieval)

    freeze = subparsers.add_parser(
        "freeze-baseline",
        help="Khóa validation_100, validation_300 và regression IDs vào manifest",
    )
    freeze.add_argument("--train", required=True)
    freeze.add_argument("--public", default=None)
    freeze.add_argument("--output", required=True)
    freeze.add_argument("--seed", type=int, default=2026)

    validate = subparsers.add_parser("validate", help="Leave-one-out validation")
    validate.add_argument("--train", required=True)
    validate.add_argument("--db", required=True)
    validate.add_argument("--output", required=True)
    validate.add_argument("--modes", default="extractive,knn,hybrid")
    validate.add_argument("--limit", type=int, default=300)
    validate.add_argument("--seed", type=int, default=2026)
    validate.add_argument(
        "--split-manifest",
        default=None,
        help="Manifest tạo bởi freeze-baseline; khi có, --limit/--seed không chọn lại ID",
    )
    validate.add_argument(
        "--split-name",
        default="validation_100",
        choices=["validation_100", "validation_300"],
    )
    validate.add_argument(
        "--regression-input",
        default=None,
        help="public-official.json dùng chạy các regression ID đã khóa (không có answer gold)",
    )
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
    retrieval_eval.add_argument("--split-manifest", default=None)
    retrieval_eval.add_argument(
        "--split-name",
        default="validation_100",
        choices=["validation_100", "validation_300"],
    )
    retrieval_eval.add_argument("--ks", default="1,3,5")
    retrieval_eval.add_argument("--dense-index", default=None)
    retrieval_eval.add_argument(
        "--embedding-model",
        default="AITeamVN/Vietnamese_Embedding_v2",
    )
    retrieval_eval.add_argument("--embedding-revision", default=None)
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
    retrieval_eval.add_argument("--reranker-max-length", type=int, default=1024)
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
        "reranker_candidate_k": getattr(args, "reranker_candidate_k", 20),
        "rerank_top_k": getattr(args, "rerank_top_k", 3),
        "dense_query_max_length": getattr(args, "dense_query_max_length", 256),
        "reranker_max_length": getattr(args, "reranker_max_length", 1024),
        "max_long_answer_words": getattr(args, "max_long_answer_words", 640),
        "token_limit_retry_tokens": getattr(args, "token_limit_retry_tokens", 768),
    }
    invalid = {name: value for name, value in positive_values.items() if value <= 0}
    if invalid:
        raise ValueError(f"Các tham số phải lớn hơn 0: {invalid}")
    if not (
        positive_values["context_top_k"]
        <= positive_values["rerank_top_k"]
        <= positive_values["reranker_candidate_k"]
        <= positive_values["rrf_top_k"]
    ):
        raise ValueError(
            "Cần context_top_k <= rerank_top_k <= "
            "reranker_candidate_k <= rrf_top_k"
        )
    if not 0.0 <= args.knn_threshold <= 1.0:
        raise ValueError("knn_threshold phải nằm trong [0, 1]")
    if not 0.90 <= getattr(args, "guarded_knn_threshold", 0.90) <= 1.0:
        raise ValueError("guarded_knn_threshold phải nằm trong [0.90, 1]")
    allow_fallback = bool(getattr(args, "allow_retrieval_fallback", False))

    if need_generator:
        if getattr(args, "max_new_tokens", 512) <= 0:
            raise ValueError("max_new_tokens phải lớn hơn 0")
        if getattr(args, "max_input_tokens", 7168) <= 0:
            raise ValueError("max_input_tokens phải lớn hơn 0")
        if getattr(args, "repetition_penalty", 1.05) <= 0:
            raise ValueError("repetition_penalty phải lớn hơn 0")
        if getattr(args, "min_llm_answer_tokens", 8) <= 0:
            raise ValueError("min_llm_answer_tokens phải lớn hơn 0")
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
            repetition_penalty=getattr(args, "repetition_penalty", 1.05),
        )

        dense_index_path = getattr(args, "dense_index", None)
        if dense_index_path:
            p = Path(dense_index_path)
            meta_p = p.with_suffix(".meta.json") if p.suffix in (".faiss", ".npy") else p.with_name(f"{p.stem}.meta.json")
            if not (p.exists() or meta_p.exists()):
                message = f"Không tìm thấy Dense index/metadata tại {p}"
                if allow_fallback:
                    dense_index = None
                    embedding_model = None
                    print(f"[pipeline] {message}; fallback BM25.", file=sys.stderr)
                else:
                    raise FileNotFoundError(message)
            else:
                try:
                    from .dense import DenseVectorIndex, VietnameseEmbeddingModel
                    embedding_name = getattr(
                        args, "embedding_model", "AITeamVN/Vietnamese_Embedding_v2"
                    )
                    embedding_revision = getattr(args, "embedding_revision", None)
                    dense_index = DenseVectorIndex.load(
                        p,
                        expected_model_name=embedding_name,
                        expected_model_revision=embedding_revision,
                    )
                    dense_index.validate_against_bm25(index.metadata())
                    embedding_model = VietnameseEmbeddingModel(
                        model_name_or_path=embedding_name,
                        revision=embedding_revision,
                        device=getattr(args, "device", "auto"),
                    )
                    print(f"[pipeline] Đã nạp Dense Index: {len(dense_index.metadata):,} chunks", file=sys.stderr)
                except Exception as exc:
                    if not allow_fallback:
                        raise
                    dense_index = None
                    embedding_model = None
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
                if not allow_fallback:
                    raise
                reranker = None
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
        reranker_candidate_k=getattr(args, "reranker_candidate_k", 20),
        rerank_top_k=getattr(args, "rerank_top_k", 3),
        dense_query_max_length=getattr(args, "dense_query_max_length", 256),
        reranker_max_length=getattr(args, "reranker_max_length", 1024),
        allow_retrieval_fallback=allow_fallback,
        enable_long_answer_extractive=getattr(args, "enable_long_answer_extractive", True),
        max_long_answer_words=getattr(args, "max_long_answer_words", 640),
        min_llm_answer_tokens=getattr(args, "min_llm_answer_tokens", 8),
        token_limit_retry_tokens=getattr(args, "token_limit_retry_tokens", 768),
        guarded_knn_threshold=getattr(args, "guarded_knn_threshold", 0.90),
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
        embedding_model_revision=args.embedding_revision,
        device=args.device,
        batch_size=args.batch_size,
        max_chunk_words=args.max_chunk_words,
        overlap_words=args.overlap_words,
        embedding_max_length=args.embedding_max_length,
        force=args.force,
        resume=args.resume,
        checkpoint_chunks=args.checkpoint_chunks,
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
            try:
                predictions[key] = clean_answer(answer)
            except ValueError:
                # A cleaned-empty checkpoint entry is regenerated normally.
                continue
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


def _write_prediction_progress(
    checkpoint_path: Path,
    output_path: Path,
    predictions: dict[str, str],
    routes: dict[str, int],
) -> None:
    """Persist resumable state and a readable partial submission atomically."""
    _write_checkpoint(checkpoint_path, predictions, routes)
    write_predictions(output_path, predictions)


def _append_audit_record(audit_path: Path, record: dict[str, object]) -> None:
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def command_predict(args: argparse.Namespace) -> int:
    data = load_qa(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path.with_suffix(".checkpoint.json")
    audit_path = (
        Path(args.audit_output)
        if getattr(args, "audit_output", None)
        else output_path.with_suffix(".audit.jsonl")
    )

    predictions: dict[str, str] = {}
    routes: dict[str, int] = {}

    if getattr(args, "resume", False):
        resume_path = checkpoint_path if checkpoint_path.exists() else output_path
        if resume_path.exists():
            try:
                predictions, routes = _load_checkpoint(resume_path, set(data))
                print(
                    f"[predict] Đã nạp lại {resume_path.name}: "
                    f"{len(predictions):,}/{len(data):,} câu đã xong.",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception as exc:
                print(f"[predict] Bỏ tiến độ không hợp lệ: {exc}", file=sys.stderr)
                predictions = {}
                routes = {}

    if not getattr(args, "resume", False) or not audit_path.exists():
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text("", encoding="utf-8")

    started = time.time()
    need_generator = args.mode in ("rag", "hybrid_rag")
    total = len(data)
    interval = getattr(args, "checkpoint_interval", 1)
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
            _append_audit_record(
                audit_path,
                prediction_audit_record(sample_id, result),
            )
            processed_count += 1
            sample_time = time.time() - t0
            stage_seconds = result.evidence.get("stage_seconds", {})
            if not isinstance(stage_seconds, dict):
                stage_seconds = {}

            # Lưu checkpoint định kỳ
            if processed_count % interval == 0 or number == total:
                _write_prediction_progress(
                    checkpoint_path,
                    output_path,
                    predictions,
                    routes,
                )

            if pbar:
                pbar.set_postfix({
                    "routes": str(routes),
                    "last": f"{sample_time:.2f}s",
                    "bm25": f"{float(stage_seconds.get('bm25', 0.0)):.2f}s",
                    "dense": f"{float(stage_seconds.get('dense', 0.0)):.2f}s",
                    "rerank": f"{float(stage_seconds.get('reranker', 0.0)):.2f}s",
                    "gen": f"{float(stage_seconds.get('generation', 0.0)):.2f}s",
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
                        f"BM25: {float(stage_seconds.get('bm25', 0.0)):.2f}s | "
                        f"Dense: {float(stage_seconds.get('dense', 0.0)):.2f}s | "
                        f"Rerank: {float(stage_seconds.get('reranker', 0.0)):.2f}s | "
                        f"Gen: {float(stage_seconds.get('generation', 0.0)):.2f}s | "
                        f"Routes: {routes}",
                        file=sys.stderr,
                        flush=True,
                    )

        if pbar:
            pbar.close()
        if processed_count:
            _write_prediction_progress(
                checkpoint_path,
                output_path,
                predictions,
                routes,
            )

    _write_prediction_progress(
        checkpoint_path,
        output_path,
        predictions,
        routes,
    )

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
            "reranker_candidate_k": pipeline.reranker_candidate_k,
            "rerank_top_k": pipeline.rerank_top_k,
            "reranker_max_length": pipeline.reranker_max_length,
            "context_top_k": pipeline.context_top_k,
        },
        "elapsed_seconds": round(time.time() - started, 2),
        "output": str(output_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "audit": str(audit_path.resolve()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def command_diagnose_retrieval(args: argparse.Namespace) -> int:
    """Persist retrieval ranks/timing while guaranteeing no LLM generation call."""
    data = load_qa(args.input)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    records: list[dict[str, object]] = []

    # need_generator=True also loads the Dense/Reranker stack in the shared
    # pipeline factory. ViQwenRAGGenerator is lazy; clearing it here additionally
    # guarantees retrieve_only() cannot retain or invoke a generator instance.
    with SearchIndex(args.db) as index:
        pipeline = _pipeline(args, index, need_generator=True)
        pipeline.generator = None
        for number, (sample_id, item) in enumerate(data.items(), start=1):
            diagnostic = pipeline.retrieve_only(item["question"])
            records.append({"id": str(sample_id), **diagnostic})
            if number <= 5 or number % 5 == 0 or number == len(data):
                total_seconds = diagnostic.get("stage_seconds", {}).get("total", 0.0)
                print(
                    f"[diagnose-retrieval] {number:,}/{len(data):,} | "
                    f"last={float(total_seconds):.2f}s",
                    file=sys.stderr,
                    flush=True,
                )

    stage_names = ("bm25", "dense", "fusion", "reranker", "total")
    stage_medians: dict[str, float] = {}
    for stage_name in stage_names:
        values = [
            float(record.get("stage_seconds", {}).get(stage_name, 0.0))
            for record in records
        ]
        stage_medians[stage_name] = round(statistics.median(values), 4) if values else 0.0
    payload = {
        "summary": {
            "samples": len(records),
            "generator_called": False,
            "median_stage_seconds": stage_medians,
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "output": str(target.resolve()),
        },
        "items": records,
    }
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0


def command_freeze_baseline(args: argparse.Namespace) -> int:
    train = load_qa(args.train)
    public = load_qa(args.public) if args.public else None
    manifest = make_baseline_manifest(
        train,
        args.train,
        public=public,
        public_path=args.public,
        seed=args.seed,
    )
    write_locked_manifest(args.output, manifest)
    summary = {
        "output": str(Path(args.output).resolve()),
        "manifest_sha256": file_sha256(args.output),
        "validation_100": len(manifest["splits"]["validation_100"]),
        "validation_300": len(manifest["splits"]["validation_300"]),
        "regression": len(manifest["regression_ids"]),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def command_validate(args: argparse.Namespace) -> int:
    if args.limit <= 0:
        raise ValueError("limit phải lớn hơn 0")
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    if not modes:
        raise ValueError("--modes không được rỗng, phải chứa ít nhất một mode hợp lệ")
    invalid = set(modes) - set(VALID_MODES)
    if invalid:
        raise ValueError(f"Mode không hợp lệ: {sorted(invalid)}")

    data = load_qa(args.train)
    manifest: dict[str, object] | None = None
    if args.split_manifest:
        sample_ids, manifest = load_locked_split(
            args.split_manifest,
            args.split_name,
            data,
            args.train,
        )
        split_source = "locked_manifest"
    else:
        sample_ids = list(data)
        rng = random.Random(args.seed)
        rng.shuffle(sample_ids)
        sample_ids = sample_ids[: args.limit]
        split_source = "legacy_seeded_shuffle"

    regression_data: dict[str, dict[str, object]] = {}
    if args.regression_input:
        if manifest is None:
            raise ValueError("--regression-input yêu cầu --split-manifest")
        public = load_qa(args.regression_input)
        regression_data = load_regression_samples(
            manifest,
            public,
            args.regression_input,
        )

    report: dict[str, object] = {
        "config": {
            "samples": len(sample_ids),
            "seed": manifest.get("seed") if manifest else args.seed,
            "split_source": split_source,
            "split_name": args.split_name if manifest else None,
            "split_manifest": str(Path(args.split_manifest).resolve()) if args.split_manifest else None,
            "split_manifest_sha256": file_sha256(args.split_manifest) if args.split_manifest else None,
            "sample_ids": sample_ids,
            "top_k": args.top_k,
            "max_answer_words": args.max_answer_words,
            "max_long_answer_words": args.max_long_answer_words,
            "knn_threshold": args.knn_threshold,
            "guarded_knn_threshold": args.guarded_knn_threshold,
            "context_top_k": args.context_top_k,
            "generator_model": args.generator_model,
            "max_new_tokens": args.max_new_tokens,
            "token_limit_retry_tokens": args.token_limit_retry_tokens,
        },
        "results": {},
        "regression": {},
    }
    need_generator = any(m in ("rag", "hybrid_rag") for m in modes)
    if args.split_manifest and need_generator:
        if not args.dense_index:
            raise ValueError(
                "Locked RAG baseline yêu cầu --dense-index hiện có; không được chạy BM25-only"
            )
        if args.allow_retrieval_fallback:
            raise ValueError(
                "Locked RAG baseline không cho phép --allow-retrieval-fallback vì sẽ đổi cấu hình retrieval"
            )
    with SearchIndex(args.db) as index:
        pipeline = _pipeline(args, index, need_generator=need_generator)
        if args.split_manifest and need_generator and pipeline.dense_index is None:
            raise RuntimeError("Dense index không active trong locked RAG baseline")
        report["config"]["bm25_index"] = index.metadata()  # type: ignore[index]
        report["config"]["dense_index"] = (  # type: ignore[index]
            dict(pipeline.dense_index.manifest)
            if pipeline.dense_index is not None
            else None
        )
        for mode in modes:
            predictions: list[str] = []
            references: list[str] = []
            routes: dict[str, int] = {}
            items: list[dict[str, object]] = []
            for number, sample_id in enumerate(sample_ids, start=1):
                item = data[sample_id]
                result = pipeline.predict_one(
                    item["question"], mode=mode, exclude_id=sample_id
                )
                predictions.append(result.answer)
                references.append(str(item.get("answer") or ""))
                routes[result.route] = routes.get(result.route, 0) + 1
                reference = str(item.get("answer") or "")
                answer_metrics = score_answer(result.answer, reference)
                if getattr(args, "official_metrics", False):
                    answer_metrics.update(official_scores(result.answer, reference))
                audit = prediction_audit_record(sample_id, result)
                retrieval = score_retrieval_trace(
                    index,
                    reference,
                    audit.get("retrieval_trace", {}),
                )
                items.append(
                    {
                        "id": sample_id,
                        "question": item["question"],
                        "prediction": result.answer,
                        "reference": reference,
                        "route": result.route,
                        "length": {
                            "prediction_words": answer_metrics["prediction_words"],
                            "reference_words": answer_metrics["reference_words"],
                            "length_ratio": answer_metrics["length_ratio"],
                            "context_words": audit["context_words"],
                            "generated_tokens": audit["generated_tokens"],
                        },
                        "metrics": answer_metrics,
                        "retrieval": retrieval,
                        "audit": {
                            key: value
                            for key, value in audit.items()
                            if key != "retrieval_trace"
                        },
                    }
                )
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
            scores["retrieval"] = aggregate_retrieval_items(items)
            quality_total = max(1, len(items))
            token_limit_count = sum(
                bool(item.get("audit", {}).get("hit_token_limit", False))
                for item in items
                if isinstance(item.get("audit"), dict)
            )
            refusal_count = sum(
                bool(item.get("audit", {}).get("says_no_information", False))
                for item in items
                if isinstance(item.get("audit"), dict)
            )
            fallback_count = sum(item.get("route") == "extractive_fallback" for item in items)
            scores["routing_quality"] = {
                "token_limit": {
                    "count": token_limit_count,
                    "rate": token_limit_count / quality_total,
                    "target": 0.05,
                    "passed": token_limit_count / quality_total < 0.05,
                },
                "refusal": {
                    "count": refusal_count,
                    "rate": refusal_count / quality_total,
                    "target": 0.02,
                    "passed": refusal_count / quality_total < 0.02,
                },
                "extractive_fallback": {
                    "count": fallback_count,
                    "rate": fallback_count / quality_total,
                    "target": 0.10,
                    "passed": fallback_count / quality_total < 0.10,
                },
            }
            scores["items"] = items
            report["results"][mode] = scores  # type: ignore[index]

            if regression_data:
                regression_items: list[dict[str, object]] = []
                regression_routes: dict[str, int] = {}
                for sample_id, item in regression_data.items():
                    result = pipeline.predict_one(
                        item["question"],
                        mode=mode,
                        exclude_id=sample_id,
                    )
                    regression_routes[result.route] = regression_routes.get(result.route, 0) + 1
                    audit = prediction_audit_record(sample_id, result)
                    regression_items.append(
                        {
                            "id": sample_id,
                            "question": item["question"],
                            "prediction": result.answer,
                            "route": result.route,
                            "length": {
                                "prediction_words": audit["answer_words"],
                                "reference_words": None,
                                "length_ratio": None,
                                "context_words": audit["context_words"],
                                "generated_tokens": audit["generated_tokens"],
                            },
                            "metrics": {
                                "competition_meteor": None,
                                "competition_rougeL": None,
                                "reason": "public-official.json không có reference answer",
                            },
                            "retrieval": {
                                "pseudo_gold_available": False,
                                "pseudo_gold": [],
                                "stages": {},
                                "trace": audit.get("retrieval_trace", {}),
                            },
                            "audit": {
                                key: value
                                for key, value in audit.items()
                                if key != "retrieval_trace"
                            },
                        }
                    )
                report["regression"][mode] = {  # type: ignore[index]
                    "samples": len(regression_items),
                    "routes": regression_routes,
                    "items": regression_items,
                }

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    temporary.replace(target)
    console_report = {
        "config": {
            key: value
            for key, value in report["config"].items()  # type: ignore[union-attr]
            if key != "sample_ids"
        },
        "results": {
            mode: {key: value for key, value in values.items() if key != "items"}
            for mode, values in report["results"].items()  # type: ignore[union-attr]
        },
        "regression": {
            mode: {key: value for key, value in values.items() if key != "items"}
            for mode, values in report["regression"].items()  # type: ignore[union-attr]
        },
        "output": str(target.resolve()),
    }
    print(json.dumps(console_report, ensure_ascii=False, indent=2))
    return 0


def command_evaluate_retrieval(args: argparse.Namespace) -> int:
    from .retrieval_eval import evaluate_retrieval

    data = load_qa(args.train)
    split_manifest = None
    if args.split_manifest:
        sample_ids, split_manifest = load_locked_split(
            args.split_manifest,
            args.split_name,
            data,
            args.train,
        )
        data = {sample_id: data[sample_id] for sample_id in sample_ids}
    raw_ks = [value.strip() for value in str(args.ks).split(",") if value.strip()]
    if not raw_ks:
        raise ValueError("--ks không được rỗng, ví dụ: 1,3,5")
    try:
        ks = tuple(int(value) for value in raw_ks)
    except ValueError as exc:
        raise ValueError(f"--ks phải là danh sách số nguyên dương phân tách bởi dấu phẩy: {exc}") from exc
    if any(k <= 0 for k in ks):
        raise ValueError(f"--ks chỉ chấp nhận các số nguyên dương: {ks}")

    positive_values = {
        "limit": args.limit,
        "bm25_top_k": args.bm25_top_k,
        "dense_top_k": args.dense_top_k,
        "rrf_k": args.rrf_k,
        "rrf_top_k": args.rrf_top_k,
        "dense_query_max_length": args.dense_query_max_length,
        "reranker_max_length": args.reranker_max_length,
        "gold_candidate_k": args.gold_candidate_k,
        "gold_max_chunks": args.gold_max_chunks,
        "gold_min_answer_tokens": args.gold_min_answer_tokens,
    }
    invalid = {name: value for name, value in positive_values.items() if value <= 0}
    if invalid:
        raise ValueError(f"Các tham số phải lớn hơn 0: {invalid}")

    if not 0.0 <= args.gold_min_score <= 1.0:
        raise ValueError("gold_min_score phải nằm trong [0, 1]")
    if not 0.0 <= args.gold_relative_score <= 1.0:
        raise ValueError("gold_relative_score phải nằm trong [0, 1]")

    dense_index = None
    embedding_model = None
    reranker = None
    with SearchIndex(args.db) as index:
        if args.dense_index:
            from .dense import DenseVectorIndex, VietnameseEmbeddingModel

            dense_index = DenseVectorIndex.load(
                args.dense_index,
                expected_model_name=args.embedding_model,
                expected_model_revision=args.embedding_revision,
            )
            dense_index.validate_against_bm25(index.metadata())
            embedding_model = VietnameseEmbeddingModel(
                model_name_or_path=args.embedding_model,
                revision=args.embedding_revision,
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
            limit=len(data) if split_manifest else args.limit,
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
            diagnostic_samples=len(data) if split_manifest else 5,
        )

    if split_manifest:
        report["config"].update(
            {
                "split_source": "locked_manifest",
                "split_name": args.split_name,
                "split_manifest": str(Path(args.split_manifest).resolve()),
                "split_manifest_sha256": file_sha256(args.split_manifest),
                "sample_ids": list(data),
            }
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
    if not isinstance(prediction_data, dict):
        raise ValueError("Prediction JSON phải là một dictionary/object ánh xạ id -> {'answer': ...}")
    if set(reference) != set(prediction_data):
        missing = set(reference) - set(prediction_data)
        extra = set(prediction_data) - set(reference)
        raise ValueError(f"ID lệch: missing={len(missing)}, extra={len(extra)}")
    predictions: list[str] = []
    for key in reference:
        item = prediction_data[key]
        if not isinstance(item, dict):
            raise ValueError(f"Prediction cho sample '{key}' phải là một dictionary/object")
        if "answer" not in item:
            raise ValueError(f"Prediction cho sample '{key}' thiếu trường 'answer'")
        answer = item["answer"]
        if not isinstance(answer, str):
            raise ValueError(
                f"Trường 'answer' của sample '{key}' phải là chuỗi (string), nhận được {type(answer).__name__}"
            )
        predictions.append(answer)
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
        "diagnose-retrieval": command_diagnose_retrieval,
        "freeze-baseline": command_freeze_baseline,
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
        sqlite3.Error,
        OSError,
        KeyError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
