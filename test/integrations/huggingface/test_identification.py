"""Tests for CausalityIdentificationPipeline (no real model required)."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest


def _make_logits(probs: list[float]):
    """Return outputs.logits[0] whose softmax/argmax behave correctly.

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

    class _Row:
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

    return _Logits(_Row(probs))


@pytest.fixture(autouse=True)
def stub_transformers():
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
        for key in list(sys.modules):
            if key.startswith("causalatee.integrations"):
                del sys.modules[key]
        yield transformers


class TestCausalityIdentificationPipeline:
    def _make_pipeline(self, id2label: dict[int, str] | None = None):
        from causalatee.integrations.huggingface._identification import CausalityIdentificationPipeline

        if id2label is None:
            id2label = {0: "CAUSAL", 1: "NO-REL"}

        pipe = CausalityIdentificationPipeline.__new__(CausalityIdentificationPipeline)
        pipe.tokenizer = MagicMock(return_value={"input_ids": MagicMock()})
        model = MagicMock()
        model.config.id2label = id2label
        pipe.model = model
        return pipe

    def test_postprocess_returns_highest_prob_label(self):
        pipe = self._make_pipeline()
        outputs = MagicMock()
        outputs.logits = _make_logits([0.9, 0.1])
        result = pipe.postprocess(outputs)
        assert result["relation"] == "CAUSAL"
        assert abs(result["score"] - 0.9) < 1e-6

    def test_postprocess_no_rel(self):
        pipe = self._make_pipeline()
        outputs = MagicMock()
        outputs.logits = _make_logits([0.2, 0.8])
        result = pipe.postprocess(outputs)
        assert result["relation"] == "NO-REL"
        assert abs(result["score"] - 0.8) < 1e-6

    def test_preprocess_calls_tokenizer(self):
        pipe = self._make_pipeline()
        pipe.preprocess("<e1>fire</e1> caused <e2>smoke</e2>")
        pipe.tokenizer.assert_called_once_with(
            "<e1>fire</e1> caused <e2>smoke</e2>",
            return_tensors="pt",
            truncation=True,
            padding=True,
        )

    def test_sanitize_parameters_returns_three_empty_dicts(self):
        from causalatee.integrations.huggingface._identification import CausalityIdentificationPipeline

        pipe = CausalityIdentificationPipeline.__new__(CausalityIdentificationPipeline)
        assert pipe._sanitize_parameters() == ({}, {}, {})
