"""Tests for causalatee.nlp._connectives — causal discourse connective matching."""
from __future__ import annotations

import spacy

from causalatee.nlp._connectives import find_causal_connectives

_nlp = spacy.load("en_core_web_sm")


class TestFindCausalConnectives:
    def test_causal_verb(self):
        doc = _nlp("The storm caused significant flooding.")
        matches = find_causal_connectives(doc)
        assert [(m.text, m.category) for m in matches] == [("caused", "verb")]

    def test_cue_phrase(self):
        doc = _nlp("Flooding occurred because of the storm.")
        matches = find_causal_connectives(doc)
        assert [(m.text, m.category) for m in matches] == [("because of", "cue_phrase")]

    def test_adverbial(self):
        doc = _nlp("The mining of metals will therefore continue.")
        matches = find_causal_connectives(doc)
        assert [(m.text, m.category) for m in matches] == [("therefore", "adverbial")]

    def test_multi_word_verb_phrase(self):
        # "outcomes" also legitimately matches the noun category (both are
        # real, independent causal-lexicon hits) — assert containment, not
        # an exact full-list match.
        doc = _nlp("Increased funding led to better outcomes.")
        matches = [(m.text, m.category) for m in find_causal_connectives(doc)]
        assert ("led to", "verb") in matches

    def test_lemma_matching_covers_inflections_without_listing_them(self):
        # "cause" is listed once in CAUSAL_CONNECTIVES; lemma matching should
        # still catch every inflected surface form.
        for text, expected in [
            ("The storm causes flooding.", "causes"),
            ("The storm caused flooding.", "caused"),
            ("The storm is causing flooding.", "causing"),
        ]:
            doc = _nlp(text)
            matches = find_causal_connectives(doc)
            assert [(m.text, m.category) for m in matches] == [(expected, "verb")]

    def test_lemma_matching_covers_phrasal_verb_inflections(self):
        # "lead to" is listed once; "leads to"/"led to"/"leading to" should
        # all resolve to it via lemma matching.
        for text, expected in [
            ("This leads to problems.", "leads to"),
            ("This led to problems.", "led to"),
            ("This is leading to problems.", "leading to"),
        ]:
            doc = _nlp(text)
            matches = find_causal_connectives(doc)
            assert [(m.text, m.category) for m in matches] == [(expected, "verb")]

    def test_causal_noun(self):
        # "cause" is deliberately verb-only (see CAUSAL_CONNECTIVES comment
        # on the "noun" category) — use an unambiguous causal noun instead.
        doc = _nlp("Poor sleep is a major factor in fatigue.")
        matches = find_causal_connectives(doc)
        assert ("factor", "noun") in [(m.text, m.category) for m in matches]

    def test_cause_is_verb_only_not_noun(self):
        doc = _nlp("Poor sleep is a major cause of fatigue.")
        matches = [(m.text, m.category) for m in find_causal_connectives(doc)]
        assert ("cause", "verb") in matches
        assert ("cause", "noun") not in matches

    def test_no_false_positive_on_substring(self):
        # "cause" must not spuriously match inside an unrelated word like
        # "because" being tokenized differently, or "causeway", etc. — the
        # matcher operates over spaCy's own tokens, not raw substrings.
        doc = _nlp("They walked along the causeway at dusk.")
        assert find_causal_connectives(doc) == []

    def test_no_match_in_unrelated_sentence(self):
        doc = _nlp("The cat sat on the mat.")
        assert find_causal_connectives(doc) == []

    def test_overlapping_candidates_resolve_to_longest(self):
        # "As", "As a result", and "As a result of" all match the same
        # opening text; only the longest should survive.
        doc = _nlp("As a result of the storm, flooding occurred.")
        matches = find_causal_connectives(doc)
        assert len(matches) == 1
        assert matches[0].text == "As a result of"
        assert matches[0].start == 0

    def test_because_prefix_of_because_of_resolves_to_longer(self):
        doc = _nlp("Flooding occurred because of the storm.")
        matches = find_causal_connectives(doc)
        # Only "because of" should be reported, not also standalone "because".
        assert len(matches) == 1
        assert matches[0].text == "because of"

    def test_matches_are_sorted_by_position(self):
        doc = _nlp("Because of the storm, flooding occurred, and it therefore caused damage.")
        matches = find_causal_connectives(doc)
        starts = [m.start for m in matches]
        assert starts == sorted(starts)

    def test_match_offsets_align_with_source_text(self):
        text = "The storm caused significant flooding."
        doc = _nlp(text)
        matches = find_causal_connectives(doc)
        assert len(matches) == 1
        m = matches[0]
        assert text[m.start:m.end] == m.text

    def test_unnegated_connective_reports_negated_false(self):
        doc = _nlp("The storm caused significant flooding.")
        matches = find_causal_connectives(doc)
        assert matches[0].negated is False

    def test_verbal_negation_via_neg_dependency(self):
        # "not" attaches to "cause" with dep_="neg" -- the standard
        # Universal Dependencies analysis of verbal negation.
        doc = _nlp("The storm did not cause significant flooding.")
        matches = find_causal_connectives(doc)
        assert [(m.text, m.negated) for m in matches] == [("cause", True)]

    def test_contracted_negation(self):
        doc = _nlp("The vaccine doesn't cause autism.")
        matches = find_causal_connectives(doc)
        assert [(m.text, m.negated) for m in matches] == [("cause", True)]

    def test_never_negates_a_noun_connective(self):
        doc = _nlp("The policy was never a factor in the decline.")
        matches = find_causal_connectives(doc)
        assert [(m.text, m.negated) for m in matches] == [("factor", True)]

    def test_determiner_negation_not_labeled_neg_by_ud(self):
        # "No" is parsed as a plain `det` of "factor", not dep_="neg" --
        # exercises the cue-lemma fallback, not the neg-dependency check.
        doc = _nlp("No factor was found to cause the delay.")
        matches = {m.text: m.negated for m in find_causal_connectives(doc)}
        assert matches["factor"] is True

    def test_lexical_negation_cue_verb(self):
        # "fail" is itself a negation-cue lemma (its own meaning negates
        # the verb it governs), not a "neg"-dependency case at all.
        doc = _nlp("The company failed to cause a meaningful improvement.")
        matches = find_causal_connectives(doc)
        assert [(m.text, m.negated) for m in matches] == [("cause", True)]

    def test_negation_on_multi_word_verb_phrase(self):
        doc = _nlp("This did not lead to a resolution.")
        matches = find_causal_connectives(doc)
        assert [(m.text, m.negated) for m in matches] == [("lead to", True)]

    def test_positive_and_negative_examples_side_by_side(self):
        # Same connective, minimal edit distance -- isolates negation as
        # the only variable.
        pos = find_causal_connectives(_nlp("The drug caused the reaction."))
        neg = find_causal_connectives(_nlp("The drug did not cause the reaction."))
        assert [m.negated for m in pos] == [False]
        assert [m.negated for m in neg] == [True]
