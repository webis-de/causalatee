"""Tests for causalatee.evaluation.spans._dicematched (Liu, Mitamura & Hovy 2015)."""

from __future__ import annotations

import pytest

from causalatee.evaluation.spans import (
    dataset_dice_matched_span_scores,
    dice_matched_span_metrics,
    dice_matched_span_scores,
)


class TestDiceMatchedSpanScores:
    """Per-instance scoring — dice_matched_span_scores(truths, predictions)."""

    def test_perfect_match(self):
        scores = dice_matched_span_scores([(0, 10)], [(0, 10)])
        assert scores == pytest.approx({"precision": 1.0, "recall": 1.0, "f1": 1.0})

    def test_partial_overlap(self):
        # gold=(0,10), pred=(0,5): dice = 2*5 / (10+5) = 2/3
        scores = dice_matched_span_scores([(0, 10)], [(0, 5)])
        assert scores["precision"] == pytest.approx(2 / 3)
        assert scores["recall"] == pytest.approx(2 / 3)

    def test_no_overlap(self):
        scores = dice_matched_span_scores([(0, 5)], [(10, 15)])
        assert scores == pytest.approx({"precision": 0.0, "recall": 0.0, "f1": 0.0})

    def test_no_double_crediting_against_overlapping_truths(self):
        """One predicted span can only be matched to ONE gold span, even if it overlaps both.

        This is the concrete advantage over the granularity-discounted metric for nested/overlapping
        gold spans (e.g. SCITE's embedded-cause spans): a single prediction matching two overlapping
        truths does not get credited toward both.
        """
        # pred (0,10) overlaps both truths; dice((0,10),(0,10))=1.0 > dice((0,10),(5,15))=0.5,
        # so the Hungarian assignment takes the better (0,10)-(0,10) pair and leaves (5,15) unmatched.
        scores = dice_matched_span_scores([(0, 10), (5, 15)], [(0, 10)])
        assert scores["precision"] == pytest.approx(1.0)  # the one prediction fully matched
        assert scores["recall"] == pytest.approx(0.5)  # only 1 of 2 gold spans got credit

    def test_empty_truths_and_predictions_is_trivial_perfect_match(self):
        """Correctly predicting 'nothing here' is a perfect match — unlike
        granularity_discounted_span_scores(), which currently scores this case as 0.0
        (see TestGranularityDiscountedSpanScores.test_empty_both)."""
        scores = dice_matched_span_scores([], [])
        assert scores == pytest.approx({"precision": 1.0, "recall": 1.0, "f1": 1.0})

    def test_empty_truths_only_is_a_false_positive(self):
        scores = dice_matched_span_scores([], [(0, 5)])
        assert scores == pytest.approx({"precision": 0.0, "recall": 0.0, "f1": 0.0})

    def test_empty_predictions_only_is_a_false_negative(self):
        scores = dice_matched_span_scores([(0, 5)], [])
        assert scores == pytest.approx({"precision": 0.0, "recall": 0.0, "f1": 0.0})

    def test_degenerate_span_raises_by_default(self):
        with pytest.raises(ValueError, match="positive length"):
            dice_matched_span_scores([(5, 5)], [(0, 10)])

    def test_degenerate_span_scores_zero_overlap_when_configured(self):
        scores = dice_matched_span_scores([(5, 5)], [(0, 10)], on_degenerate_span="zero")
        assert scores == pytest.approx({"precision": 0.0, "recall": 0.0, "f1": 0.0})


class TestDatasetDiceMatchedSpanScores:
    """Macro-averaged dataset aggregate — every instance weighted equally, mirroring
    dataset_granularity_discounted_span_scores()."""

    def test_macro_average(self):
        # Same structure as TestDatasetGranularityDiscountedSpanScores.test_macro_average, for a
        # direct side-by-side comparison against f1_gran's aggregation.
        result = dataset_dice_matched_span_scores(
            [[(0, 5)], [(0, 5)]],
            [[(0, 5)], [(10, 15)]],
        )
        assert result["f1"] == pytest.approx(0.5)
        assert result["precision"] == pytest.approx(0.5)

    def test_empty_dataset(self):
        # Zero INSTANCES (no data at all) -- distinct from a single instance with no spans, which
        # scores 1.0/1.0/1.0 (see test_empty_truths_and_predictions_is_trivial_perfect_match above).
        result = dataset_dice_matched_span_scores([], [])
        assert result == pytest.approx({"precision": 0.0, "recall": 0.0, "f1": 0.0})

    def test_degenerate_span_propagates_configuration(self):
        with pytest.raises(ValueError, match="positive length"):
            dataset_dice_matched_span_scores([[(5, 5)]], [[(0, 10)]])
        result = dataset_dice_matched_span_scores([[(5, 5)]], [[(0, 10)]], on_degenerate_span="zero")
        assert result == pytest.approx({"precision": 0.0, "recall": 0.0, "f1": 0.0})


class TestDiceMatchedSpanMetrics:
    """Corpus-pooled (micro) aggregate — dice_matched_span_metrics(predictions, references),
    matching Liu et al.'s own protocol. Genuinely different from the macro aggregation above
    whenever instances have unequal span counts."""

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            dice_matched_span_metrics([[(0, 5)]], [[(0, 5)], [(0, 5)]])

    def test_empty_corpus_is_trivial_perfect_match(self):
        result = dice_matched_span_metrics([], [])
        assert result == pytest.approx({"precision": 1.0, "recall": 1.0, "f1": 1.0})

    def test_micro_and_macro_diverge_on_unequal_span_counts(self):
        # Instance A: 1 gold, 1 pred, perfect match -> per-instance f1 = 1.0.
        # Instance B: 1 gold, 10 unrelated preds -> per-instance f1 = 0.0.
        # Macro (mean over instances) treats A and B equally -> f1 = 0.5.
        # Micro (pooled) weights by span count -> B's 10 false-positive predictions dominate.
        references = [[(0, 10)], [(100, 110)]]
        predictions = [[(0, 10)], [(200 + 10 * i, 205 + 10 * i) for i in range(10)]]

        macro = dataset_dice_matched_span_scores(references, predictions)
        micro = dice_matched_span_metrics(predictions, references)

        assert macro["f1"] == pytest.approx(0.5)
        assert micro["precision"] == pytest.approx(1 / 11)
        assert micro["recall"] == pytest.approx(0.5)
        assert micro["f1"] != pytest.approx(macro["f1"])

    def test_degenerate_span_propagates_configuration(self):
        with pytest.raises(ValueError, match="positive length"):
            dice_matched_span_metrics([[(0, 10)]], [[(5, 5)]])
        result = dice_matched_span_metrics([[(0, 10)]], [[(5, 5)]], on_degenerate_span="zero")
        assert result == pytest.approx({"precision": 0.0, "recall": 0.0, "f1": 0.0})
