"""Tests for CausalityDetectionPipeline (no real model required)."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest


def _make_logits(values: list[float]):
    """Return a minimal tensor-like object whose softmax/argmax behave correctly.

    Mirrors the usage in postprocess:
        probs = model_outputs.logits[0].softmax(-1)
        label_id = probs.argmax().item()          # → int
        score = probs[label_id].item()            # → float
    """

    class _FloatScalar:
        def __init__(self, val):
            self._val = val

        def item(self):
            return self._val

    class _IntScalar:
        def __init__(self, val):
            self._val = val

        def item(self):
            return self._val

    class _ProbRow:
        def __init__(self, data):
            self._data = data

        def softmax(self, _dim):
            return self

        def argmax(self):
            return _IntScalar(self._data.index(max(self._data)))

        def __getitem__(self, idx):
            return _FloatScalar(self._data[idx])

    class _Logits:
        def __init__(self, row):
            self._row = row

        def __getitem__(self, idx):
            return self._row

    return _Logits(_ProbRow(values))


@pytest.fixture(autouse=True)
def stub_transformers():
    """Inject a minimal transformers stub so the module can be imported."""
    transformers = types.ModuleType("transformers")

    class _Pipeline:
        pass

    transformers.Pipeline = _Pipeline
    transformers.AutoModelForSequenceClassification = MagicMock()
    transformers.AutoModelForTokenClassification = MagicMock()

    # transformers 4.57+ no longer re-exports PIPELINE_REGISTRY from the top level -- it must be
    # reachable as a real submodule (transformers.pipelines), not just a flat attribute, since
    # causalatee.integrations.huggingface does `from transformers.pipelines import PIPELINE_REGISTRY`.
    transformers_pipelines = types.ModuleType("transformers.pipelines")
    transformers_pipelines.PIPELINE_REGISTRY = MagicMock()
    transformers.pipelines = transformers_pipelines

    with patch.dict(sys.modules, {"transformers": transformers, "transformers.pipelines": transformers_pipelines}):
        # Remove any cached causalatee.integrations modules so they re-import cleanly.
        for key in list(sys.modules):
            if key.startswith("causalatee.integrations"):
                del sys.modules[key]
        yield transformers


class TestCausalityDetectionPipeline:
    def _make_pipeline(self, label: str, score: float = 0.9):
        from causalatee.integrations.huggingface._detection import CausalityDetectionPipeline

        pipe = CausalityDetectionPipeline.__new__(CausalityDetectionPipeline)
        pipe.tokenizer = MagicMock(return_value={"input_ids": MagicMock()})

        logit_values = [score, 1 - score] if label == "CAUSAL" else [1 - score, score]
        model = MagicMock()
        model.config.id2label = {0: "CAUSAL", 1: "UNCAUSAL"}
        outputs = MagicMock()
        outputs.logits = _make_logits(logit_values)
        model.return_value = outputs
        pipe.model = model
        return pipe

    def test_preprocess_calls_tokenizer(self):
        pipe = self._make_pipeline("CAUSAL")
        pipe.preprocess("The fire caused the smoke.")
        pipe.tokenizer.assert_called_once_with(
            "The fire caused the smoke.",
            return_tensors="pt",
            truncation=True,
            padding=True,
        )

    def test_postprocess_causal_label(self):
        pipe = self._make_pipeline("CAUSAL", score=0.95)
        outputs = MagicMock()
        outputs.logits = _make_logits([0.95, 0.05])
        result = pipe.postprocess(outputs)
        assert result["label"] == "CAUSAL"
        assert 0.0 < result["score"] <= 1.0

    def test_postprocess_uncausal_label(self):
        pipe = self._make_pipeline("UNCAUSAL", score=0.8)
        outputs = MagicMock()
        outputs.logits = _make_logits([0.2, 0.8])
        result = pipe.postprocess(outputs)
        assert result["label"] == "UNCAUSAL"

    def test_sanitize_parameters_returns_three_empty_dicts(self):
        from causalatee.integrations.huggingface._detection import CausalityDetectionPipeline

        pipe = CausalityDetectionPipeline.__new__(CausalityDetectionPipeline)
        assert pipe._sanitize_parameters() == ({}, {}, {})
