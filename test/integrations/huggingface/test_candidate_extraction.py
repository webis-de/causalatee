"""Tests for CausalCandidateExtractionPipeline (no real model required)."""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest


def _make_token_logits(label_sequence: list[str], id2label: dict[int, str]):
    """Return a batch-logits stub that matches the postprocess access pattern.

    postprocess does:
        logits = model_outputs["logits"][0]   # (seq_len, num_labels)
        label_ids = logits.argmax(-1).tolist()
    So model_outputs["logits"][0] must have argmax(-1).tolist() → list[int].
    """
    label2id = {v: k for k, v in id2label.items()}
    ids = [label2id[label] for label in label_sequence]

    class _IdList:
        def __init__(self, data):
            self._data = data

        def tolist(self):
            return self._data

    class _SeqLogits:
        """Represents the (seq_len, num_labels) slice after [0]."""

        def __init__(self, label_ids):
            self._ids = label_ids

        def argmax(self, _dim):
            return _IdList(self._ids)

    class _BatchLogits:
        def __init__(self, label_ids):
            self._seq = _SeqLogits(label_ids)

        def __getitem__(self, _idx):
            return self._seq

    return _BatchLogits(ids)


def _make_offsets(pairs: list[tuple[int, int]]):
    class _OffsetTensor:
        def __init__(self, data):
            self._data = data

        def __getitem__(self, idx):
            return self

        def tolist(self):
            return self._data

    return _OffsetTensor(pairs)


@pytest.fixture(autouse=True)
def stub_transformers():
    transformers = types.ModuleType("transformers")

    class _Pipeline:
        pass

    transformers.Pipeline = _Pipeline
    transformers.PIPELINE_REGISTRY = MagicMock()
    transformers.AutoModelForSequenceClassification = MagicMock()
    transformers.AutoModelForTokenClassification = MagicMock()

    with patch.dict(sys.modules, {"transformers": transformers}):
        for key in list(sys.modules):
            if key.startswith("causalatee.integrations"):
                del sys.modules[key]
        yield transformers


class TestCausalCandidateExtractionPipeline:
    _id2label = {0: "O", 1: "B-CAUSE", 2: "I-CAUSE", 3: "B-EFFECT", 4: "I-EFFECT"}

    def _make_pipeline(self):
        from causalatee.integrations.huggingface._candidate_extraction import CausalCandidateExtractionPipeline

        pipe = CausalCandidateExtractionPipeline.__new__(CausalCandidateExtractionPipeline)
        pipe.tokenizer = MagicMock()
        model = MagicMock()
        model.config.id2label = self._id2label
        pipe.model = model
        return pipe

    def _run_postprocess(self, pipe, labels: list[str], offsets: list[tuple[int, int]]):
        logits = _make_token_logits(labels, self._id2label)
        offsets_tensor = _make_offsets(offsets)
        return pipe.postprocess({"logits": logits, "offset_mapping": offsets_tensor})

    def test_single_cause_span(self):
        pipe = self._make_pipeline()
        # Special token, then B-CAUSE (0-5), I-CAUSE (6-10), O, special token
        labels = ["O", "B-CAUSE", "I-CAUSE", "O", "O"]
        offsets = [(0, 0), (0, 5), (6, 10), (11, 15), (0, 0)]
        spans = self._run_postprocess(pipe, labels, offsets)
        assert len(spans) == 1
        assert spans[0] == {"start": 0, "end": 10, "entity": "CAUSE"}

    def test_cause_and_effect_spans(self):
        pipe = self._make_pipeline()
        labels = ["O", "B-CAUSE", "O", "B-EFFECT", "I-EFFECT", "O"]
        offsets = [(0, 0), (0, 4), (5, 8), (9, 13), (14, 20), (0, 0)]
        spans = self._run_postprocess(pipe, labels, offsets)
        assert len(spans) == 2
        entities = {s["entity"] for s in spans}
        assert entities == {"CAUSE", "EFFECT"}
        effect = next(s for s in spans if s["entity"] == "EFFECT")
        assert effect["start"] == 9
        assert effect["end"] == 20

    def test_no_spans_when_all_o(self):
        pipe = self._make_pipeline()
        labels = ["O", "O", "O", "O"]
        offsets = [(0, 0), (0, 3), (4, 7), (0, 0)]
        spans = self._run_postprocess(pipe, labels, offsets)
        assert spans == []

    def test_preprocess_passes_offset_mapping(self):
        pipe = self._make_pipeline()
        pipe.tokenizer.return_value = {"input_ids": MagicMock(), "offset_mapping": MagicMock()}
        pipe.preprocess("hello")
        pipe.tokenizer.assert_called_once_with(
            "hello",
            return_tensors="pt",
            truncation=True,
            return_offsets_mapping=True,
        )

    def test_forward_pops_offset_mapping(self):
        pipe = self._make_pipeline()
        offset = MagicMock()
        inputs = {"input_ids": MagicMock(), "offset_mapping": offset}
        pipe.model.return_value = MagicMock()
        result = pipe._forward(inputs)
        assert "offset_mapping" not in inputs  # popped
        assert result["offset_mapping"] is offset

    def test_sanitize_parameters_returns_three_empty_dicts(self):
        from causalatee.integrations.huggingface._candidate_extraction import CausalCandidateExtractionPipeline

        pipe = CausalCandidateExtractionPipeline.__new__(CausalCandidateExtractionPipeline)
        assert pipe._sanitize_parameters() == ({}, {}, {})
