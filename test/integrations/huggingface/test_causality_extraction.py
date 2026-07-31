"""Tests for CausalityExtractionPipeline (end-to-end chain, no real model)."""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest


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


def _make_extraction_pipeline(
    detection_label: str = "CAUSAL",
    candidate_spans: list[dict] | None = None,
    identification_result: dict | None = None,
):
    from causalatee.integrations.huggingface._causality_extraction import CausalityExtractionPipeline

    if candidate_spans is None:
        candidate_spans = [
            {"start": 4, "end": 9, "entity": "CAUSE"},
            {"start": 17, "end": 25, "entity": "EFFECT"},
        ]
    if identification_result is None:
        identification_result = {"relation": "CAUSAL", "score": 0.95}

    detection = MagicMock(return_value={"label": detection_label})
    candidate_extraction = MagicMock(return_value=candidate_spans)
    identification = MagicMock(return_value=identification_result)

    return CausalityExtractionPipeline(detection, candidate_extraction, identification)


class TestCausalityExtractionPipeline:
    def test_returns_empty_when_uncausal(self):
        pipe = _make_extraction_pipeline(detection_label="uncausal")
        result = pipe("Some random sentence without causality.")
        assert result == []

    def test_returns_empty_when_fewer_than_two_spans(self):
        pipe = _make_extraction_pipeline(
            detection_label="CAUSAL",
            candidate_spans=[{"start": 0, "end": 4, "entity": "CAUSE"}],
        )
        result = pipe("Fire.")
        assert result == []

    def test_returns_empty_when_no_spans(self):
        pipe = _make_extraction_pipeline(detection_label="CAUSAL", candidate_spans=[])
        result = pipe("Nothing here.")
        assert result == []

    def test_skips_no_rel_pairs(self):
        pipe = _make_extraction_pipeline(
            identification_result={"relation": "no-rel", "score": 0.9}
        )
        result = pipe("The storm caused flooding.")
        assert result == []

    def test_returns_relation_for_causal_pair(self):
        text = "The storm caused flooding."
        # storm = 4:9, flooding = 17:25
        pipe = _make_extraction_pipeline(
            candidate_spans=[
                {"start": 4, "end": 9, "entity": "CAUSE"},
                {"start": 17, "end": 25, "entity": "EFFECT"},
            ],
            identification_result={"relation": "CAUSAL", "score": 0.95},
        )
        result = pipe(text)
        assert len(result) == 1
        assert result[0]["e1"] == text[4:9]
        assert result[0]["e2"] == text[17:25]
        assert result[0]["relation"] == "CAUSAL"
        assert abs(result[0]["score"] - 0.95) < 1e-6

    def test_produces_all_pairs_from_n_spans(self):
        # 3 spans → 3 pairs; identification always returns causal
        spans = [
            {"start": 0, "end": 3, "entity": "CAUSE"},
            {"start": 4, "end": 7, "entity": "EFFECT"},
            {"start": 8, "end": 11, "entity": "CAUSE"},
        ]
        pipe = _make_extraction_pipeline(
            candidate_spans=spans,
            identification_result={"relation": "CAUSAL", "score": 0.8},
        )
        result = pipe("abc def ghi")
        assert len(result) == 3  # C(3,2) pairs

    def test_identification_receives_marked_text(self):
        text = "Fire smoke damage."
        spans = [
            {"start": 0, "end": 4, "entity": "CAUSE"},
            {"start": 5, "end": 10, "entity": "EFFECT"},
        ]
        pipe = _make_extraction_pipeline(
            candidate_spans=spans,
            identification_result={"relation": "CAUSAL", "score": 0.9},
        )
        pipe(text)
        called_with = pipe._identification.call_args[0][0]
        assert "<e1>" in called_with
        assert "<e2>" in called_with


class TestInsertMarkers:
    def test_markers_inserted_correctly(self):
        from causalatee.integrations.huggingface._causality_extraction import _insert_markers

        text = "Fire caused smoke."
        # F(0)i(1)r(2)e(3) (4)c(5)a(6)u(7)s(8)e(9)d(10) (11)s(12)m(13)o(14)k(15)e(16).(17)
        span1 = {"start": 0, "end": 4}   # Fire
        span2 = {"start": 12, "end": 17}  # smoke
        result = _insert_markers(text, span1, span2)
        assert result == "<e1>Fire</e1> caused <e2>smoke</e2>."

    def test_markers_with_reversed_span_order(self):
        from causalatee.integrations.huggingface._causality_extraction import _insert_markers

        text = "Flooding followed the storm."
        # Flooding=0:8, storm=22:27
        span1 = {"start": 0, "end": 8}    # Flooding
        span2 = {"start": 22, "end": 27}  # storm
        result = _insert_markers(text, span1, span2)
        assert "<e1>Flooding</e1>" in result
        assert "<e2>storm</e2>" in result
