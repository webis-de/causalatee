"""Tests for causalatee.nlp._sdp — shortest dependency path extraction."""
from __future__ import annotations

import pytest
import spacy

from causalatee.nlp._sdp import format_sdp, shortest_dependency_path, span_head_token

_nlp = spacy.load("en_core_web_sm")

_SVO_TEXT = "The storm caused significant flooding."
# Verified programmatically (str.index), not hand-counted:
_SPAN_THE_STORM = (0, 9)     # "The storm"
_SPAN_STORM = (4, 9)         # "storm"
_SPAN_CAUSED = (10, 16)      # "caused"
_SPAN_SIG_FLOODING = (17, 37)  # "significant flooding"
_SPAN_FLOODING = (29, 37)    # "flooding"

_CROSS_SENT_TEXT = "The storm hit land. Flooding followed everywhere."
_SPAN_STORM_2 = (4, 9)       # "storm"
_SPAN_FLOODING_2 = (20, 28)  # "Flooding"


class TestSpanHeadToken:
    def test_exact_boundary(self):
        doc = _nlp(_SVO_TEXT)
        assert span_head_token(doc, _SPAN_STORM).text == "storm"

    def test_multi_token_span_returns_syntactic_root(self):
        doc = _nlp(_SVO_TEXT)
        # "significant" is an amod of "flooding" -> "flooding" is the root.
        assert span_head_token(doc, _SPAN_SIG_FLOODING).text == "flooding"

    def test_misaligned_span_expands_to_nearest_token(self):
        doc = _nlp(_SVO_TEXT)
        # (5, 8) falls inside "storm" (4-9) without matching its boundaries.
        assert span_head_token(doc, (5, 8)).text == "storm"

    def test_out_of_range_span_raises(self):
        doc = _nlp("Short.")
        with pytest.raises(ValueError):
            span_head_token(doc, (100, 110))


class TestShortestDependencyPath:
    def test_simple_svo_sentence(self):
        doc = _nlp(_SVO_TEXT)
        path = shortest_dependency_path(doc, _SPAN_THE_STORM, _SPAN_FLOODING)
        assert format_sdp(path) == "storm --nsubj↑--> caused --dobj↓--> flooding"

    def test_path_passes_through_causal_verb(self):
        # The SDP's defining property: it should surface the causal predicate
        # connecting the two entities, per Bunescu & Mooney (2005).
        doc = _nlp(_SVO_TEXT)
        path = shortest_dependency_path(doc, _SPAN_THE_STORM, _SPAN_FLOODING)
        path_tokens = [path[0].from_token.text] + [step.to_token.text for step in path]
        assert "caused" in path_tokens

    def test_identical_span_returns_empty_path(self):
        doc = _nlp(_SVO_TEXT)
        path = shortest_dependency_path(doc, _SPAN_THE_STORM, _SPAN_THE_STORM)
        assert path == []

    def test_different_sentences_returns_none(self):
        doc = _nlp(_CROSS_SENT_TEXT)
        path = shortest_dependency_path(doc, _SPAN_STORM_2, _SPAN_FLOODING_2)
        assert path is None


class TestFormatSdp:
    def test_empty_path(self):
        assert format_sdp([]) == ""

    def test_two_hop_path(self):
        doc = _nlp(_SVO_TEXT)
        path = shortest_dependency_path(doc, _SPAN_THE_STORM, _SPAN_FLOODING)
        formatted = format_sdp(path)
        assert formatted.startswith("storm")
        assert formatted.endswith("flooding")
        assert "↑" in formatted and "↓" in formatted
