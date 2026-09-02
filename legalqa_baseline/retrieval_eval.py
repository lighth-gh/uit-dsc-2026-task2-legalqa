from __future__ import annotations

import math
import random
import sys
from collections import Counter
from typing import Any, Iterable

from .pipeline import _apply_legal_signal_boost, reciprocal_rank_fusion
from .text import expand_retrieval_query, tokenize


ChunkKey = tuple[str, int]


def chunk_key(item: dict[str, Any]) -> ChunkKey:
    return str(item.get("context_id", "")), int(item.get("chunk_no", 0))


def _ngrams(tokens: list[str], size: int) -> set[tuple[str, ...]]:
    if size <= 0 or len(tokens) < size:
        return set()
    return set(zip(*(tokens[offset:] for offset in range(size))))


def answer_evidence_score(answer: str, passage: str, ngram_size: int = 5) -> float:
    """Score how much of an answer is supported by a corpus chunk.

    This is only used to construct pseudo-gold labels when the dataset has no
    annotated context/chunk IDs. It combines multiset token recall with phrase
    recall so a bag of coincidental legal terms is not treated as strong gold.
    """
    answer_tokens = tokenize(answer)
    passage_tokens = tokenize(passage)
    if not answer_tokens or not passage_tokens:
        return 0.0

    shared = sum((Counter(answer_tokens) & Counter(passage_tokens)).values())
    token_recall = shared / len(answer_tokens)

    size = min(max(1, ngram_size), len(answer_tokens), len(passage_tokens))
    answer_ngrams = _ngrams(answer_tokens, size)
    passage_ngrams = _ngrams(passage_tokens, size)
    phrase_recall = (
        len(answer_ngrams & passage_ngrams) / len(answer_ngrams)
        if answer_ngrams
        else 0.0
    )
    return 0.65 * token_recall + 0.35 * phrase_recall


def build_pseudo_gold(
    index: Any,
    answer: str,
    candidate_k: int = 100,
    max_gold_chunks: int = 5,
    min_score: float = 0.20,
    relative_score: float = 0.85,
    min_answer_tokens: int = 8,
) -> tuple[set[ChunkKey], list[dict[str, Any]]]:
    """Find answer-supported chunks and return their keys plus audit metadata."""
    if len(tokenize(answer)) < min_answer_tokens:
        return set(), []
    candidates = index.search_contexts(answer, top_k=candidate_k)
    scored: list[tuple[float, dict[str, Any]]] = []
    for candidate in candidates:
        score = answer_evidence_score(answer, str(candidate.get("text") or ""))
        scored.append((score, candidate))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    if not scored or scored[0][0] < min_score:
        return set(), []

    cutoff = max(min_score, scored[0][0] * relative_score)
    selected = scored[:max_gold_chunks]
    selected = [(score, item) for score, item in selected if score >= cutoff]
    audit = [
        {
            "context_id": chunk_key(item)[0],
            "chunk_no": chunk_key(item)[1],
            "score": round(float(score), 6),
            "name": item.get("name"),
        }
        for score, item in selected
    ]
    return {chunk_key(item) for score, item in selected}, audit


def compute_stage_ranking_metrics(
    results: list[dict[str, Any]],
    gold: set[ChunkKey],
    ks: Iterable[int],
) -> dict[str, float]:
    """Tính các chỉ số truy xuất đa dạng: Recall@K, Hit@K, MRR@K, NDCG@K, MAP@K, Precision@K."""
    ranked = [chunk_key(item) for item in results]
    metrics: dict[str, float] = {}
    ks_sorted = sorted(set(int(k) for k in ks))

    # MRR tổng thể trên toàn danh sách kết quả
    mrr_overall = 0.0
    for rank_idx, key in enumerate(ranked, start=1):
        if key in gold:
            mrr_overall = 1.0 / rank_idx
            break
    metrics["mrr"] = mrr_overall

    for k in ks_sorted:
        top_k_keys = ranked[:k]
        intersect = gold.intersection(top_k_keys)
        hit = 1.0 if intersect else 0.0
        metrics[f"recall@{k}"] = hit
        metrics[f"hit@{k}"] = hit
        metrics[f"gold_recall@{k}"] = len(intersect) / len(gold) if gold else 0.0
        metrics[f"precision@{k}"] = len(intersect) / k if k > 0 else 0.0

        # MRR@k
        rr_k = 0.0
        for rank_idx, key in enumerate(top_k_keys, start=1):
            if key in gold:
                rr_k = 1.0 / rank_idx
                break
        metrics[f"mrr@{k}"] = rr_k

        # NDCG@k
        dcg = 0.0
        for rank_idx, key in enumerate(top_k_keys, start=1):
            if key in gold:
                dcg += 1.0 / math.log2(rank_idx + 1)
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
        metrics[f"ndcg@{k}"] = (dcg / idcg) if idcg > 0.0 else 0.0

        # MAP@k / Average Precision@k
        hits_count = 0
        sum_precisions = 0.0
        for rank_idx, key in enumerate(top_k_keys, start=1):
            if key in gold:
                hits_count += 1
                sum_precisions += hits_count / rank_idx
        denom = min(len(gold), k)
        metrics[f"map@{k}"] = (sum_precisions / denom) if denom > 0 else 0.0

    return metrics


def evaluate_retrieval(
    samples: dict[str, dict[str, Any]],
    index: Any,
    *,
    dense_index: Any | None = None,
    embedding_model: Any | None = None,
    reranker: Any | None = None,
    limit: int = 100,
    seed: int = 2026,
    ks: tuple[int, ...] = (1, 3, 5),
    bm25_top_k: int = 50,
    dense_top_k: int = 50,
    rrf_k: int = 60,
    rrf_top_k: int = 50,
    dense_query_max_length: int = 256,
    reranker_max_length: int = 1024,
    gold_candidate_k: int = 100,
    gold_max_chunks: int = 5,
    gold_min_score: float = 0.20,
    gold_relative_score: float = 0.85,
    gold_min_answer_tokens: int = 8,
    diagnostic_samples: int = 5,
) -> dict[str, Any]:
    """Evaluate retrieval stages against answer-derived pseudo-gold chunks."""
    ks = tuple(sorted(set(int(k) for k in ks)))
    if not ks or ks[0] <= 0:
        raise ValueError("ks must contain positive integers")
    if (dense_index is None) != (embedding_model is None):
        raise ValueError("dense_index and embedding_model must be provided together")
    if not 0.0 <= gold_min_score <= 1.0:
        raise ValueError("gold_min_score must be in [0, 1]")
    if not 0.0 < gold_relative_score <= 1.0:
        raise ValueError("gold_relative_score must be in (0, 1]")

    sample_ids = list(samples)
    random.Random(seed).shuffle(sample_ids)
    if limit > 0:
        sample_ids = sample_ids[:limit]

    stages = ["bm25"]
    if dense_index is not None:
        stages.extend(["dense", "rrf"])
    if reranker is not None:
        stages.append("reranker")

    # Accumulate metrics across all evaluated queries
    stage_metric_totals: dict[str, Counter[str]] = {stage: Counter() for stage in stages}
    evaluated = 0
    skipped = 0
    diagnostics: list[dict[str, Any]] = []
    max_k = max(ks)

    for number, sample_id in enumerate(sample_ids, start=1):
        item = samples[sample_id]
        question = str(item.get("question") or "")
        answer = str(item.get("answer") or "")
        gold, gold_audit = build_pseudo_gold(
            index,
            answer,
            candidate_k=gold_candidate_k,
            max_gold_chunks=gold_max_chunks,
            min_score=gold_min_score,
            relative_score=gold_relative_score,
            min_answer_tokens=gold_min_answer_tokens,
        )
        if not gold:
            skipped += 1
            continue

        bm25 = index.search_contexts(question, top_k=max(bm25_top_k, max_k))
        stage_results: dict[str, list[dict[str, Any]]] = {"bm25": bm25}

        dense: list[dict[str, Any]] = []
        if dense_index is not None and embedding_model is not None:
            normalize = getattr(dense_index, "similarity", "cosine") == "cosine"
            query_vector = embedding_model.encode(
                [expand_retrieval_query(question)],
                max_length=dense_query_max_length,
                normalize_embeddings=normalize,
            )[0]
            dense = dense_index.search(query_vector, top_k=max(dense_top_k, max_k))
            fused = reciprocal_rank_fusion(
                bm25,
                dense,
                rrf_k=rrf_k,
                top_k=max(rrf_top_k, max_k),
            )
            fused = _apply_legal_signal_boost(question, fused)
            stage_results["dense"] = dense
            stage_results["rrf"] = fused

        if reranker is not None:
            rerank_pool = stage_results.get("rrf", bm25[:rrf_top_k])
            stage_results["reranker"] = reranker.rerank(
                question,
                rerank_pool,
                top_k=max_k,
                max_length=reranker_max_length,
            )

        evaluated += 1
        sample_metrics: dict[str, dict[str, float]] = {}
        for stage, results in stage_results.items():
            stage_m = compute_stage_ranking_metrics(results, gold, ks)
            sample_metrics[stage] = stage_m
            for m_key, m_val in stage_m.items():
                stage_metric_totals[stage][m_key] += m_val

        if len(diagnostics) < diagnostic_samples:
            diagnostics.append(
                {
                    "sample_id": sample_id,
                    "question": question,
                    "answer_preview": " ".join(answer.split()[:80]),
                    "pseudo_gold": gold_audit,
                    "metrics": sample_metrics,
                }
            )
        if number % 10 == 0 or number == len(sample_ids):
            print(
                f"[retrieval-eval] processed={number}/{len(sample_ids)} "
                f"evaluated={evaluated} skipped={skipped}",
                file=sys.stderr,
                flush=True,
            )

    aggregated_metrics: dict[str, dict[str, float]] = {}
    for stage in stages:
        aggregated_metrics[stage] = {
            m_key: (val / evaluated if evaluated else 0.0)
            for m_key, val in stage_metric_totals[stage].items()
        }

    return {
        "metric_type": "pseudo_retrieval_recall",
        "warning": (
            "Train has no gold context IDs. Labels are answer-derived pseudo-gold; "
            "do not report these values as human-annotated retrieval recall."
        ),
        "config": {
            "requested_samples": len(sample_ids),
            "seed": seed,
            "ks": list(ks),
            "gold_candidate_k": gold_candidate_k,
            "gold_max_chunks": gold_max_chunks,
            "gold_min_score": gold_min_score,
            "gold_relative_score": gold_relative_score,
            "gold_min_answer_tokens": gold_min_answer_tokens,
            "bm25_top_k": bm25_top_k,
            "dense_top_k": dense_top_k,
            "rrf_k": rrf_k,
            "rrf_top_k": rrf_top_k,
        },
        "samples_evaluated": evaluated,
        "samples_without_pseudo_gold": skipped,
        "pseudo_gold_coverage": evaluated / len(sample_ids) if sample_ids else 0.0,
        "metrics": aggregated_metrics,
        "diagnostics": diagnostics,
    }
