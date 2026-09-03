from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .metrics import (
    bleu_score,
    exact_match,
    meteor_exact,
    rouge_l_f1,
    token_precision_recall_f1,
)
from .retrieval_eval import build_pseudo_gold, compute_stage_ranking_metrics


BASELINE_SPLIT_SCHEMA_VERSION = 1
DEFAULT_BASELINE_SEED = 2026
REGRESSION_IDS = (
    "80189",
    "34235",
    "31969",
    "123257",
    "35853",
    "154891",
    "117399",
    "108017",
    "18645",
    "129215",
    "129859",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_baseline_manifest(
    train: dict[str, dict[str, Any]],
    train_path: str | Path,
    *,
    public: dict[str, dict[str, Any]] | None = None,
    public_path: str | Path | None = None,
    seed: int = DEFAULT_BASELINE_SEED,
) -> dict[str, Any]:
    """Create the immutable ID manifest used by every baseline run."""
    if len(train) < 300:
        raise ValueError(f"Cần ít nhất 300 câu train để khóa baseline, chỉ có {len(train)}")

    ids = sorted(str(sample_id) for sample_id in train)
    random.Random(seed).shuffle(ids)
    validation_300 = ids[:300]
    manifest: dict[str, Any] = {
        "schema_version": BASELINE_SPLIT_SCHEMA_VERSION,
        "seed": int(seed),
        "train_sha256": file_sha256(train_path),
        "train_samples": len(train),
        "splits": {
            "validation_100": validation_300[:100],
            "validation_300": validation_300,
        },
        "regression_ids": list(REGRESSION_IDS),
    }
    if public is not None:
        missing = [sample_id for sample_id in REGRESSION_IDS if sample_id not in public]
        if missing:
            raise ValueError(f"Regression ID không có trong public data: {missing}")
        if public_path is None:
            raise ValueError("public_path là bắt buộc khi truyền public data")
        manifest["public_sha256"] = file_sha256(public_path)
        manifest["public_samples"] = len(public)
    return manifest


def write_locked_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    """Write once. An existing lock may only be reused when byte-equivalent as JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        with target.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != manifest:
            raise FileExistsError(
                f"Baseline đã khóa tại {target}; từ chối ghi đè manifest khác"
            )
        return
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(target)


def load_locked_split(
    manifest_path: str | Path,
    split_name: str,
    train: dict[str, dict[str, Any]],
    train_path: str | Path,
) -> tuple[list[str], dict[str, Any]]:
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("Baseline manifest phải là JSON object")
    if manifest.get("schema_version") != BASELINE_SPLIT_SCHEMA_VERSION:
        raise ValueError("Baseline manifest schema không được hỗ trợ")
    if manifest.get("train_sha256") != file_sha256(train_path):
        raise ValueError("train.json đã thay đổi so với baseline manifest đã khóa")
    if manifest.get("train_samples") != len(train):
        raise ValueError("Số mẫu train không khớp baseline manifest")
    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("Baseline manifest thiếu splits")
    validation_100 = splits.get("validation_100")
    validation_300 = splits.get("validation_300")
    if not isinstance(validation_100, list) or len(validation_100) != 100:
        raise ValueError("validation_100 phải chứa đúng 100 ID")
    if not isinstance(validation_300, list) or len(validation_300) != 300:
        raise ValueError("validation_300 phải chứa đúng 300 ID")
    if validation_300[:100] != validation_100:
        raise ValueError("validation_100 phải là prefix của validation_300")
    if manifest.get("regression_ids") != list(REGRESSION_IDS):
        raise ValueError("regression_ids đã lệch khỏi tập lỗi Giai đoạn 0")
    if split_name not in splits:
        raise ValueError(f"Không tìm thấy split {split_name!r} trong baseline manifest")
    sample_ids = splits[split_name]
    if not isinstance(sample_ids, list) or not sample_ids:
        raise ValueError(f"Split {split_name!r} phải là danh sách ID không rỗng")
    normalized = [str(sample_id) for sample_id in sample_ids]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"Split {split_name!r} chứa ID trùng")
    missing = [sample_id for sample_id in normalized if sample_id not in train]
    if missing:
        raise ValueError(f"Split {split_name!r} có ID không tồn tại trong train: {missing[:10]}")
    return normalized, manifest


def load_regression_samples(
    manifest: dict[str, Any],
    public: dict[str, dict[str, Any]],
    public_path: str | Path,
) -> dict[str, dict[str, Any]]:
    expected_hash = manifest.get("public_sha256")
    if expected_hash and expected_hash != file_sha256(public_path):
        raise ValueError("public-official.json đã thay đổi so với baseline manifest đã khóa")
    raw_ids = manifest.get("regression_ids")
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ValueError("Baseline manifest không có regression_ids")
    ids = [str(sample_id) for sample_id in raw_ids]
    missing = [sample_id for sample_id in ids if sample_id not in public]
    if missing:
        raise ValueError(f"Regression ID không có trong public data: {missing}")
    return {sample_id: public[sample_id] for sample_id in ids}


def score_answer(prediction: str, reference: str) -> dict[str, float | int]:
    precision, recall, f1 = token_precision_recall_f1(prediction, reference)
    prediction_words = len(str(prediction).split())
    reference_words = len(str(reference).split())
    return {
        "meteor_exact_approx": meteor_exact(prediction, reference),
        "rougeL": rouge_l_f1(prediction, reference),
        "answer_token_f1": f1,
        "answer_token_precision": precision,
        "answer_token_recall": recall,
        "exact_match": exact_match(prediction, reference),
        "bleu_1": bleu_score(prediction, reference, max_n=1, weights=(1.0,)),
        "bleu_2": bleu_score(prediction, reference, max_n=2, weights=(0.5, 0.5)),
        "bleu_4": bleu_score(prediction, reference, max_n=4),
        "prediction_words": prediction_words,
        "reference_words": reference_words,
        "length_ratio": prediction_words / reference_words if reference_words else 1.0,
    }


def score_retrieval_trace(
    index: Any,
    reference: str,
    trace: dict[str, Any],
    *,
    ks: Iterable[int] = (1, 3, 5),
) -> dict[str, Any]:
    """Score the exact candidates used for one generated answer."""
    if not trace:
        return {
            "pseudo_gold_available": False,
            "pseudo_gold": [],
            "stages": {},
            "trace": {},
        }
    gold, gold_audit = build_pseudo_gold(index, reference)
    stage_metrics: dict[str, dict[str, float]] = {}
    for stage_name in ("bm25", "dense", "rrf", "reranker_pool", "reranker_top"):
        stage = trace.get(stage_name)
        if not isinstance(stage, dict):
            continue
        candidates = stage.get("candidates")
        if not isinstance(candidates, list):
            continue
        if gold:
            stage_metrics[stage_name] = compute_stage_ranking_metrics(
                [item for item in candidates if isinstance(item, dict)], gold, ks
            )
    return {
        "pseudo_gold_available": bool(gold),
        "pseudo_gold": gold_audit,
        "stages": stage_metrics,
        "trace": trace,
    }


def aggregate_retrieval_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, Counter[str]] = {}
    counts: Counter[str] = Counter()
    pseudo_gold_count = 0
    for item in items:
        retrieval = item.get("retrieval")
        if not isinstance(retrieval, dict) or not retrieval.get("pseudo_gold_available"):
            continue
        pseudo_gold_count += 1
        stages = retrieval.get("stages")
        if not isinstance(stages, dict):
            continue
        for stage_name, metrics in stages.items():
            if not isinstance(metrics, dict):
                continue
            counts[stage_name] += 1
            totals.setdefault(stage_name, Counter()).update(
                {key: float(value) for key, value in metrics.items()}
            )
    return {
        "metric_type": "answer_derived_pseudo_retrieval",
        "warning": (
            "Train has no gold context IDs. These are answer-derived pseudo-gold "
            "metrics, not human-annotated retrieval recall."
        ),
        "samples_with_pseudo_gold": pseudo_gold_count,
        "pseudo_gold_coverage": pseudo_gold_count / len(items) if items else 0.0,
        "metrics": {
            stage: {
                key: value / counts[stage]
                for key, value in stage_totals.items()
            }
            for stage, stage_totals in totals.items()
            if counts[stage]
        },
    }
