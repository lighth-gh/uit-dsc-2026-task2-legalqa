from __future__ import annotations

import bisect
import math
import re
from collections import Counter, defaultdict, deque
from typing import Any


def _tokens(text: str) -> list[str]:
    """Khớp chương trình BTC: không word-segment tiếng Việt, chỉ gọi str.split()."""
    return str(text).split()


def _rouge_tokens(text: str) -> list[str]:
    """Mô phỏng DefaultTokenizer trong rouge_score BTC (chỉ [a-z0-9])."""
    lowered = str(text).lower()
    return [token for token in re.sub(r"[^a-z0-9]+", " ", lowered).split() if token]


def _lcs_length(left: list[str], right: list[str]) -> int:
    """Hunt-Szymanski LCS: nhanh hơn ma trận O(n*m) với câu trả lời dài."""
    positions: dict[str, list[int]] = defaultdict(list)
    for index, token in enumerate(right):
        positions[token].append(index)
    tails: list[int] = []
    for token in left:
        for position in reversed(positions.get(token, [])):
            location = bisect.bisect_left(tails, position)
            if location == len(tails):
                tails.append(position)
            else:
                tails[location] = position
    return len(tails)


def rouge_l_f1(prediction: str, reference: str) -> float:
    """Tính ROUGE-L F1 mô phỏng tokenizer của BTC."""
    pred = _rouge_tokens(prediction)
    ref = _rouge_tokens(reference)
    if not pred or not ref:
        return 0.0
    lcs = _lcs_length(pred, ref)
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def meteor_exact(prediction: str, reference: str) -> float:
    """METEOR exact-token gần với BTC; không dùng stem/WordNet của tiếng Anh.

    Dùng để kiểm tra nhanh không cần tải NLTK. Khi báo điểm chính thức cục bộ,
    hãy dùng `score-official` để chạy đúng package/WordNet như scoring program.
    """
    pred = _tokens(prediction)
    ref = _tokens(reference)
    if not pred or not ref:
        return 0.0

    ref_positions: dict[str, deque[int]] = defaultdict(deque)
    for index, token in enumerate(ref):
        ref_positions[token].append(index)
    alignment: list[tuple[int, int]] = []
    for pred_index, token in enumerate(pred):
        if ref_positions[token]:
            alignment.append((pred_index, ref_positions[token].popleft()))
    matches = len(alignment)
    if matches == 0:
        return 0.0

    precision = matches / len(pred)
    recall = matches / len(ref)
    fmean = (precision * recall) / (0.9 * precision + 0.1 * recall)
    chunks = 1
    for current, previous in zip(alignment[1:], alignment[:-1]):
        if current[0] != previous[0] + 1 or current[1] != previous[1] + 1:
            chunks += 1
    penalty = 0.5 * (chunks / matches) ** 3
    return (1.0 - penalty) * fmean


def token_precision_recall_f1(prediction: str, reference: str) -> tuple[float, float, float]:
    """Tính precision, recall, f1 trên multiset token của câu trả lời."""
    pred = _tokens(prediction)
    ref = _tokens(reference)
    if not pred or not ref:
        return 0.0, 0.0, 0.0
    matches = sum((Counter(pred) & Counter(ref)).values())
    precision = matches / len(pred)
    recall = matches / len(ref)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def answer_token_f1(prediction: str, reference: str) -> float:
    """Multiset token F1 for a transparent answer-similarity diagnostic."""
    return token_precision_recall_f1(prediction, reference)[2]


def exact_match(prediction: str, reference: str) -> float:
    """Exact string match sau khi chuẩn hóa khoảng trắng và chữ thường."""
    return 1.0 if " ".join(str(prediction).split()).lower() == " ".join(str(reference).split()).lower() else 0.0


def _get_ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu_score(
    prediction: str,
    reference: str,
    max_n: int = 4,
    weights: tuple[float, ...] | None = None,
) -> float:
    """Sentence BLEU độc lập với brevity penalty và smoothing."""
    pred_tokens = _tokens(prediction)
    ref_tokens = _tokens(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    if weights is None:
        weights = tuple(1.0 / max_n for _ in range(max_n))
    else:
        max_n = len(weights)

    c = len(pred_tokens)
    r = len(ref_tokens)
    if c == 0:
        return 0.0
    bp = 1.0 if c > r else math.exp(1.0 - r / c)

    log_precisions: list[float] = []
    for n in range(1, max_n + 1):
        pred_ngrams = _get_ngrams(pred_tokens, n)
        ref_ngrams = _get_ngrams(ref_tokens, n)
        total_pred = sum(pred_ngrams.values())
        if total_pred == 0:
            return 0.0
        clipped = sum(
            min(count, ref_ngrams[ngram]) for ngram, count in pred_ngrams.items()
        )
        if clipped == 0:
            precision = 1.0 / (2.0 * total_pred)
        else:
            precision = clipped / total_pred
        log_precisions.append(math.log(precision))

    return bp * math.exp(sum(w * lp for w, lp in zip(weights, log_precisions)))


def aggregate_scores(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Tính toán toàn bộ bộ chỉ số tương đồng câu trả lời chuẩn."""
    if len(predictions) != len(references) or not predictions:
        raise ValueError("Prediction/reference phải cùng số lượng và không rỗng")

    meteor_list: list[float] = []
    rouge_list: list[float] = []
    f1_list: list[float] = []
    prec_list: list[float] = []
    rec_list: list[float] = []
    em_list: list[float] = []
    bleu1_list: list[float] = []
    bleu2_list: list[float] = []
    bleu4_list: list[float] = []
    pred_lens: list[int] = []
    ref_lens: list[int] = []

    for p, r in zip(predictions, references):
        p_toks = _tokens(p)
        r_toks = _tokens(r)
        pred_lens.append(len(p_toks))
        ref_lens.append(len(r_toks))

        prec, rec, f1 = token_precision_recall_f1(p, r)
        prec_list.append(prec)
        rec_list.append(rec)
        f1_list.append(f1)

        meteor_list.append(meteor_exact(p, r))
        rouge_list.append(rouge_l_f1(p, r))
        em_list.append(exact_match(p, r))
        bleu1_list.append(bleu_score(p, r, max_n=1, weights=(1.0,)))
        bleu2_list.append(bleu_score(p, r, max_n=2, weights=(0.5, 0.5)))
        bleu4_list.append(bleu_score(p, r, max_n=4))

    total = len(predictions)
    avg_pred_len = sum(pred_lens) / total
    avg_ref_len = sum(ref_lens) / total
    length_ratio = avg_pred_len / avg_ref_len if avg_ref_len > 0 else 1.0

    return {
        "meteor_exact_approx": sum(meteor_list) / total,
        "rougeL": sum(rouge_list) / total,
        "answer_token_f1": sum(f1_list) / total,
        "answer_token_precision": sum(prec_list) / total,
        "answer_token_recall": sum(rec_list) / total,
        "exact_match": sum(em_list) / total,
        "bleu_1": sum(bleu1_list) / total,
        "bleu_2": sum(bleu2_list) / total,
        "bleu_4": sum(bleu4_list) / total,
        "avg_prediction_words": avg_pred_len,
        "avg_reference_words": avg_ref_len,
        "length_ratio": length_ratio,
        "samples": total,
    }


def aggregate_official_scores(
    predictions: list[str], references: list[str]
) -> dict[str, float]:
    """Chạy đúng 2 metric từ scoring program BTC trên Codabench (NLTK METEOR + ROUGE-L)."""
    if len(predictions) != len(references) or not predictions:
        raise ValueError("Prediction/reference must have equal non-zero lengths")
    try:
        import numpy as np
        from nltk.translate.meteor_score import meteor_score
        from rouge_score import rouge_scorer
    except ImportError as exc:
        raise ImportError("Install requirements-metrics.txt for official metrics") from exc

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    try:
        meteor_result = np.array(
            [
                meteor_score([reference.split()], prediction.split())
                for prediction, reference in zip(predictions, references)
            ]
        ).mean()
    except LookupError as exc:
        raise RuntimeError(
            "NLTK WordNet is missing; run: python -m nltk.downloader wordnet omw-1.4"
        ) from exc
    rouge_result = np.array(
        [
            rouge.score(reference, prediction)["rougeL"].fmeasure
            for prediction, reference in zip(predictions, references)
        ]
    ).mean()
    return {
        "competition_meteor": float(meteor_result),
        "competition_rougeL": float(rouge_result),
    }
