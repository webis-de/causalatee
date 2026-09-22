"""Tests for causalatee.nlp._baselines — LexiconDetector and SDPIdentifier."""

from __future__ import annotations

from causalatee.data.constants import Relation
from causalatee.nlp._baselines import LexiconDetector, SDPIdentifier

from .conftest import load_spacy_model

_nlp = load_spacy_model("en_core_web_sm")


class TestLexiconDetector:
    def test_causal_sentence(self):
        detector = LexiconDetector(_nlp)
        assert detector("The storm caused significant flooding.") == {"label": "causal", "score": 1.0}

    def test_noncausal_sentence(self):
        detector = LexiconDetector(_nlp)
        assert detector("The cat sat on the mat.") == {"label": "uncausal", "score": 1.0}

    def test_negated_connective_still_counts_as_causal(self):
        # Detection's ClassLabel.Causal covers both causal AND countercausal sentences -- only
        # identification distinguishes them via Relation. See ClassLabel's own docstring.
        detector = LexiconDetector(_nlp)
        assert detector("The vaccine did not cause autism.") == {"label": "causal", "score": 1.0}

    def test_batch_call_preserves_order(self):
        detector = LexiconDetector(_nlp)
        results = detector(
            [
                "The storm caused significant flooding.",
                "The cat sat on the mat.",
                "Poor sleep is a major factor in fatigue.",
            ]
        )
        assert [r["label"] for r in results] == ["causal", "uncausal", "causal"]

    def test_default_constructor_loads_its_own_spacy_model(self):
        # No nlp passed -- exercises the lazy spacy.load default, not just the injected-model path
        # every other test in this file uses for speed.
        detector = LexiconDetector()
        assert detector("The storm caused significant flooding.")["label"] == "causal"


class TestSDPIdentifier:
    def test_causal_pair(self):
        identifier = SDPIdentifier(_nlp)
        text = "<e1>The storm</e1> caused <e2>significant flooding</e2>."
        assert identifier(text) == {"relationship": "Causal", "score": 1.0}

    def test_negated_connective_on_path_is_countercausal(self):
        identifier = SDPIdentifier(_nlp)
        text = "<e1>The vaccine</e1> did not cause <e2>autism</e2>."
        assert identifier(text) == {"relationship": "Countercausal", "score": 1.0}

    def test_no_connective_on_path_is_norelation(self):
        identifier = SDPIdentifier(_nlp)
        text = "<e1>The cat</e1> sat on <e2>the mat</e2>."
        assert identifier(text) == {"relationship": "NoRelation", "score": 1.0}

    def test_different_sentences_is_norelation(self):
        # shortest_dependency_path returns None across sentence boundaries -- exercises that
        # branch specifically, not just "no connective found".
        identifier = SDPIdentifier(_nlp)
        text = "<e1>The storm</e1> hit land. <e2>Flooding</e2> followed everywhere."
        assert identifier(text) == {"relationship": "NoRelation", "score": 1.0}

    def test_batch_call_preserves_order(self):
        identifier = SDPIdentifier(_nlp)
        results = identifier(
            [
                "<e1>The storm</e1> caused <e2>significant flooding</e2>.",
                "<e1>The cat</e1> sat on <e2>the mat</e2>.",
            ]
        )
        assert [r["relationship"] for r in results] == ["Causal", "NoRelation"]

    def test_only_first_segment_of_a_discontinuous_mention_is_used(self):
        # e1 opens/closes twice (a discontinuous mention); only its first segment feeds the SDP.
        identifier = SDPIdentifier(_nlp)
        text = "<e1>The storm</e1> caused <e2>flooding</e2>, <e1>the storm</e1> did."
        result = identifier(text)
        assert result["relationship"] in {"Causal", "NoRelation"}  # doesn't crash on the second <e1>

    def test_predict_shared_head_token_uses_empty_path_fallback(self):
        # e1 and e2 are the SAME span ("triggered"), so shortest_dependency_path returns [] (the
        # "both spans share one head token" case) rather than a real path -- exercises the
        # fallback to the single shared head token, which itself IS a causal connective.
        identifier = SDPIdentifier(_nlp)
        doc = _nlp("The storm triggered flooding.")
        span = (10, 19)  # "triggered"
        assert doc.text[span[0] : span[1]] == "triggered"
        assert identifier._predict(doc, span, span) == Relation.Causal

    def test_connective_hit_falls_back_to_lemma_when_no_match_covers_token(self):
        # A token whose lemma IS a causal verb but has no ConnectiveMatch of its own (e.g. dropped
        # by overlap resolution elsewhere in the sentence) is still accepted as a signal, just
        # with no negation information available.
        class _FakeToken:
            idx = 10
            text = "triggered"
            lemma_ = "trigger"

        matched, negated = SDPIdentifier._connective_hit(_FakeToken(), connectives=[])
        assert matched is True
        assert negated is False

    def test_unalignable_span_is_norelation(self):
        # span_head_token raises ValueError for a span outside the doc's character range;
        # shortest_dependency_path propagates it, and _predict must not let it escape.
        identifier = SDPIdentifier(_nlp)
        doc = _nlp("Short.")
        assert identifier._predict(doc, (100, 110), (0, 5)) == Relation.NoRelation

    def test_connective_hit_no_match_and_non_causal_lemma(self):
        class _FakeToken:
            idx = 0
            text = "cat"
            lemma_ = "cat"

        matched, negated = SDPIdentifier._connective_hit(_FakeToken(), connectives=[])
        assert matched is False
        assert negated is False

    def test_default_constructor_loads_its_own_spacy_model(self):
        identifier = SDPIdentifier()
        text = "<e1>The storm</e1> caused <e2>significant flooding</e2>."
        assert identifier(text)["relationship"] == "Causal"
