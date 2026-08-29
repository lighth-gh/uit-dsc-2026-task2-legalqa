"""Chạy đúng công thức của scoring program BTC trên một tập có answer.

Cài dependencies trước: pip install -r requirements-metrics.txt
Lần chạy đầu NLTK có thể cần tải wordnet và omw-1.4.
"""

from __future__ import annotations

import argparse
import json

import nltk
import numpy as np
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def answers(data: dict) -> dict[str, str]:
    output = {}
    for key, value in data.items():
        if isinstance(value, dict):
            output[str(key)] = str(value.get("answer") or "")
        else:
            output[str(key)] = str(value)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--download-nltk", action="store_true")
    args = parser.parse_args()
    if args.download_nltk:
        nltk.download("wordnet")
        nltk.download("omw-1.4")

    truth = answers(load(args.reference))
    prediction = answers(load(args.prediction))
    if set(truth) != set(prediction):
        raise ValueError("Samples in predict not match with reference")

    ids = list(prediction)
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge_result = np.array(
        [rouge.score(truth[key], prediction[key])["rougeL"].fmeasure for key in ids]
    ).mean()
    meteor_result = np.array(
        [
            meteor_score([truth[key].split()], prediction[key].split())
            for key in ids
        ]
    ).mean()
    print(json.dumps({"rouge": float(rouge_result), "meteor": float(meteor_result)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

