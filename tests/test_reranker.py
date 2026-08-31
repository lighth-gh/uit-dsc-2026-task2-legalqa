from __future__ import annotations

import copy
import math
import sys
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from legalqa_baseline.reranker import VietnameseReranker


class _FakeEncoding(dict[str, Any]):
    def __init__(self, batch_size: int) -> None:
        super().__init__(input_ids=[object()] * batch_size)
        self.device: object | None = None

    def to(self, device: object) -> _FakeEncoding:
        self.device = device
        return self


class _RecordingTokenizer:
    def __init__(self, *, model_max_length: int = 2304) -> None:
        self.model_max_length = model_max_length
        self.calls: list[dict[str, Any]] = []
        self.encodings: list[_FakeEncoding] = []

    def __call__(self, pairs: list[list[str]], **kwargs: Any) -> _FakeEncoding:
        self.calls.append({"pairs": copy.deepcopy(pairs), **kwargs})
        encoding = _FakeEncoding(len(pairs))
        self.encodings.append(encoding)
        return encoding


def _shape(values: Any) -> tuple[int, ...]:
    if not isinstance(values, list):
        return ()
    if not values:
        return (0,)
    child_shape = _shape(values[0])
    if any(_shape(item) != child_shape for item in values):
        raise ValueError("Fake tensor values must have a rectangular shape")
    return (len(values), *child_shape)


def _flatten(values: Any) -> list[float]:
    if isinstance(values, list):
        flattened: list[float] = []
        for item in values:
            flattened.extend(_flatten(item))
        return flattened
    return [float(values)]


class _FakeTruthTensor:
    def __init__(self, value: bool) -> None:
        self.value = value

    def all(self) -> _FakeTruthTensor:
        return self

    def item(self) -> bool:
        return self.value

    def __bool__(self) -> bool:
        return self.value


class _FakeLogits:
    def __init__(self, values: Any) -> None:
        self.values = copy.deepcopy(values)

    @property
    def shape(self) -> tuple[int, ...]:
        return _shape(self.values)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def dim(self) -> int:
        return self.ndim

    def numel(self) -> int:
        return len(_flatten(self.values))

    def squeeze(self, dim: int | None = None) -> _FakeLogits:
        if dim is None:
            values = self.values
            while isinstance(values, list) and len(values) == 1:
                values = values[0]
            return _FakeLogits(values)
        if dim == 1 and self.ndim == 2 and self.shape[1] == 1:
            return _FakeLogits([row[0] for row in self.values])
        return self

    def reshape(self, *shape: int) -> _FakeLogits:
        if shape in {(-1,), (self.numel(),)}:
            return _FakeLogits(_flatten(self.values))
        raise AssertionError(f"Unsupported fake reshape: {shape}")

    view = reshape

    def float(self) -> _FakeLogits:
        return self

    def detach(self) -> _FakeLogits:
        return self

    def cpu(self) -> _FakeLogits:
        return self

    def tolist(self) -> Any:
        return copy.deepcopy(self.values)

    def __getitem__(self, key: Any) -> _FakeLogits:
        if (
            isinstance(key, tuple)
            and len(key) == 2
            and isinstance(key[0], slice)
            and key[1] == 0
        ):
            return _FakeLogits([row[0] for row in self.values[key[0]]])
        return _FakeLogits(self.values[key])


class _QueuedModel:
    def __init__(
        self,
        outputs: list[Any],
        *,
        max_position_embeddings: int = 8194,
        model_type: str = "",
        pad_token_id: int | None = None,
    ) -> None:
        self.outputs = [copy.deepcopy(output) for output in outputs]
        self.config = SimpleNamespace(
            max_position_embeddings=max_position_embeddings,
            num_labels=1,
            model_type=model_type,
            pad_token_id=pad_token_id,
        )
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(dict(kwargs))
        if not self.outputs:
            raise AssertionError("Fake model was called more times than expected")
        return SimpleNamespace(logits=_FakeLogits(self.outputs.pop(0)))


def _fake_torch_module() -> SimpleNamespace:
    def isfinite(tensor: _FakeLogits) -> _FakeTruthTensor:
        values_are_finite = all(
            math.isfinite(value) for value in _flatten(tensor.values)
        )
        return _FakeTruthTensor(values_are_finite)

    return SimpleNamespace(
        inference_mode=nullcontext,
        no_grad=nullcontext,
        isfinite=isfinite,
    )


class VietnameseRerankerTests(unittest.TestCase):
    @staticmethod
    def _make_reranker(
        outputs: list[Any],
        *,
        batch_size: int = 2,
        tokenizer_max_length: int = 2304,
        model_max_length: int = 8194,
    ) -> tuple[VietnameseReranker, _RecordingTokenizer, _QueuedModel]:
        reranker = VietnameseReranker(batch_size=batch_size)
        tokenizer = _RecordingTokenizer(model_max_length=tokenizer_max_length)
        model = _QueuedModel(
            outputs,
            max_position_embeddings=model_max_length,
        )
        reranker._tokenizer = tokenizer
        reranker._model = model
        reranker._device = "cpu"
        return reranker, tokenizer, model

    def test_batches_sorts_and_does_not_mutate_candidates(self) -> None:
        candidates = [
            {"context_id": "a", "name": "Law A", "text": "Text A"},
            {"context_id": "b", "name": "Law B", "text": "Text B"},
            {"context_id": "c", "name": "", "text": "Text C"},
            {"context_id": "d", "name": "Law D", "text": "Text D"},
            {"context_id": "e", "name": "Law E", "text": "Text E"},
        ]
        original = copy.deepcopy(candidates)
        reranker, tokenizer, model = self._make_reranker(
            [
                [[0.1], [0.9]],
                [0.2, 0.7],
                [[0.3]],
            ]
        )

        with patch.dict(sys.modules, {"torch": _fake_torch_module()}):
            result = reranker.rerank(
                "Question",
                candidates,
                top_k=3,
                max_length=128,
            )

        self.assertEqual([item["context_id"] for item in result], ["b", "d", "e"])
        self.assertEqual([item["rerank_score"] for item in result], [0.9, 0.7, 0.3])
        self.assertEqual(candidates, original)
        self.assertTrue(
            all(
                result_item is not source
                for result_item in result
                for source in candidates
            )
        )
        self.assertEqual(len(tokenizer.calls), 3)
        self.assertEqual(len(model.calls), 3)
        self.assertEqual(tokenizer.calls[0]["pairs"][0], ["Question", "Law A: Text A"])
        self.assertEqual(tokenizer.calls[1]["pairs"][0], ["Question", "Text C"])
        self.assertTrue(all(call["max_length"] == 128 for call in tokenizer.calls))
        self.assertTrue(all(encoding.device == "cpu" for encoding in tokenizer.encodings))

    def test_validates_arguments_before_empty_candidate_shortcut(self) -> None:
        reranker = VietnameseReranker()
        for kwargs in ({"top_k": 0}, {"max_length": 0}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    reranker.rerank("Question", [], **kwargs)

    def test_rejects_max_length_above_effective_model_limit(self) -> None:
        reranker, tokenizer, model = self._make_reranker(
            [[[0.5]]],
            batch_size=1,
            tokenizer_max_length=8,
            model_max_length=10,
        )
        candidate = [{"context_id": "a", "name": "Law", "text": "Text"}]

        with (
            patch.dict(sys.modules, {"torch": _fake_torch_module()}),
            self.assertRaisesRegex(ValueError, "max_length"),
        ):
            reranker.rerank("Question", candidate, max_length=9)

        self.assertEqual(tokenizer.calls, [])
        self.assertEqual(model.calls, [])

    def test_xlm_roberta_position_offset_limits_sequence_length(self) -> None:
        reranker, _, model = self._make_reranker(
            [[[0.5]]],
            batch_size=1,
            tokenizer_max_length=10**30,
            model_max_length=8194,
        )
        model.config.model_type = "xlm-roberta"
        model.config.pad_token_id = 1

        self.assertEqual(reranker._maximum_sequence_length(), 8192)

        model.config.model_type = "roberta"
        model.config.pad_token_id = 0
        model.config.max_position_embeddings = 513
        self.assertEqual(reranker._maximum_sequence_length(), 512)

    def test_rejects_multi_label_or_invalid_rank_logits(self) -> None:
        invalid_outputs = (
            [[0.1, 0.9], [0.8, 0.2]],
            [[[0.1]], [[0.2]]],
        )
        candidates = [
            {"context_id": "a", "text": "A"},
            {"context_id": "b", "text": "B"},
        ]

        for output in invalid_outputs:
            with self.subTest(shape=_shape(output)):
                reranker, _, _ = self._make_reranker([output])
                with (
                    patch.dict(sys.modules, {"torch": _fake_torch_module()}),
                    self.assertRaisesRegex(ValueError, "logits|relevance|shape"),
                ):
                    reranker.rerank("Question", candidates)

    def test_rejects_non_finite_scores(self) -> None:
        candidate = [{"context_id": "a", "text": "A"}]
        for score in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(score=score):
                reranker, _, _ = self._make_reranker([[[score]]], batch_size=1)
                with (
                    patch.dict(sys.modules, {"torch": _fake_torch_module()}),
                    self.assertRaisesRegex(ValueError, "finite|NaN|infinity"),
                ):
                    reranker.rerank("Question", candidate)

    def test_rejects_score_count_mismatch_per_batch(self) -> None:
        candidates = [
            {"context_id": "a", "text": "A"},
            {"context_id": "b", "text": "B"},
        ]
        reranker, _, _ = self._make_reranker([[[0.5]]])

        with (
            patch.dict(sys.modules, {"torch": _fake_torch_module()}),
            self.assertRaisesRegex(ValueError, "score|batch|candidate"),
        ):
            reranker.rerank("Question", candidates)


if __name__ == "__main__":
    unittest.main()
