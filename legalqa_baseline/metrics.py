from __future__ import annotations

import bisect
import math
import re
from collections import Counter, defaultdict, deque


def _tokens(text: str) -> list[str]:
    # Khớp chương trình BTC: không word-segment tiếng Việt, chỉ gọi str.split().
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


def aggregate_scores(predictions: list[str], references: list[str]) -> dict[str, float]:
    if len(predictions) != len(references) or not predictions:
        raise ValueError("Prediction/reference phải cùng số lượng và không rỗng")
    meteor = [meteor_exact(p, r) for p, r in zip(predictions, references)]
    rouge = [rouge_l_f1(p, r) for p, r in zip(predictions, references)]
    return {
        "meteor_exact_approx": sum(meteor) / len(meteor),
        "rougeL": sum(rouge) / len(rouge),
        "samples": len(predictions),
    }
