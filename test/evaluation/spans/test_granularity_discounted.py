"""Tests for causalatee.evaluation.spans._granularity_discounted (Potthast et al. 2014)."""

from __future__ import annotations

import math

import pytest

import causalatee.evaluation.spans._granularity_discounted as _gd_module
from causalatee.evaluation.spans import (
    dataset_granularity_discounted_span_scores,
    granularity_discounted_span_scores,
)
from causalatee.evaluation.spans._granularity_discounted import (
    overlap,
    span_granularity,
    span_iou,
    span_precision,
    span_recall,
)


@pytest.fixture(autouse=True)
def _reset_empty_truths_warning_flag():
    _gd_module._warned_empty_truths = False
    yield
    _gd_module._warned_empty_truths = False


class TestOverlap:
    def test_full_overlap(self):
        assert overlap((0, 10), (0, 10)) == 1.0

    def test_no_overlap(self):
        assert overlap((0, 5), (5, 10)) == 0.0

    def test_partial_overlap(self):
        # (0,10) overlapped by (5,15): overlap = 5/10
        assert overlap((0, 10), (5, 15)) == pytest.approx(0.5)

    def test_b_contained_in_a(self):
        # (0,10) overlapped by (2,4): overlap = 2/10
        assert overlap((0, 10), (2, 4)) == pytest.approx(0.2)

    def test_a_contained_in_b(self):
        # (3,5) overlapped by (0,10): overlap = 2/2 = 1.0
        assert overlap((3, 5), (0, 10)) == pytest.approx(1.0)

    def test_empty_a_returns_zero(self):
        assert overlap((5, 5), (0, 10)) == 0.0

    def test_inverted_a_returns_zero(self):
        assert overlap((10, 5), (0, 20)) == 0.0


class TestPrecisionRecall:
    def test_perfect_prediction(self):
        spans = [(0, 5), (10, 15)]
        assert span_precision(spans, spans) == pytest.approx(1.0)
        assert span_recall(spans, spans) == pytest.approx(1.0)

    def test_no_prediction(self):
        assert span_precision([(0, 5)], []) == pytest.approx(0.0)
        assert span_recall([(0, 5)], []) == pytest.approx(0.0)

    def test_no_truths(self):
        assert span_precision([], [(0, 5)]) == pytest.approx(0.0)
        assert span_recall([], [(0, 5)]) == pytest.approx(0.0)

    def test_partial_overlap(self):
        # truth = (0,10), prediction = (0,5): precision = 0.5/1=0.5, recall = 0.5
        # wait: precision = overlap(pred=(0,5), truths) = overlap((0,5),(0,10)) = 5/5=1.0
        # recall = overlap(truth=(0,10), preds) = overlap((0,10),(0,5)) = 5/10=0.5
        assert span_precision([(0, 10)], [(0, 5)]) == pytest.approx(1.0)
        assert span_recall([(0, 10)], [(0, 5)]) == pytest.approx(0.5)


class TestGranularity:
    def test_single_matching_prediction(self):
        assert span_granularity([(0, 10)], [(0, 10)]) == pytest.approx(1.0)

    def test_two_predictions_matching_same_truth(self):
        # Both (0,5) and (5,10) overlap (0,10) → granularity = 2.0
        assert span_granularity([(0, 10)], [(0, 5), (5, 10)]) == pytest.approx(2.0)

    def test_no_overlap_returns_zero(self):
        assert span_granularity([(0, 5)], [(10, 15)]) == pytest.approx(0.0)


class TestIoU:
    def test_perfect(self):
        assert span_iou([(0, 5)], [(0, 5)]) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert span_iou([(0, 5)], [(5, 10)]) == pytest.approx(0.0)

    def test_both_empty(self):
        assert span_iou([], []) == pytest.approx(0.0)

    def test_partial(self):
        # truth {0,1,2,3,4}, pred {2,3,4,5,6}; intersection=3, union=7
        assert span_iou([(0, 5)], [(2, 7)]) == pytest.approx(3 / 7)


class TestGranularityDiscountedSpanScores:
    def test_f1_gran_equals_f1_when_granularity_zero(self):
        # No predictions match the truth → granularity = 0 → f1_gran = f1
        scores = granularity_discounted_span_scores([(0, 5)], [(20, 25)])
        assert scores["f1_gran"] == pytest.approx(scores["f1"])

    def test_f1_gran_penalised_when_granularity_gt_one(self):
        # Two fragments match one truth span → f1_gran < f1
        scores = granularity_discounted_span_scores([(0, 10)], [(0, 5), (5, 10)])
        assert scores["granularity"] > 1.0
        assert scores["f1_gran"] < scores["f1"]

    def test_f1_gran_formula(self):
        # With g > 0: f1_gran = f1 / log2(1 + g)
        scores = granularity_discounted_span_scores([(0, 10)], [(0, 5), (5, 10)])
        g = scores["granularity"]
        assert scores["f1_gran"] == pytest.approx(scores["f1"] / math.log2(1 + g))

    def test_empty_both(self):
        scores = granularity_discounted_span_scores([], [])
        assert all(v == pytest.approx(0.0) for v in scores.values())

    def test_empty_truths_only(self):
        scores = granularity_discounted_span_scores([], [(0, 5)])
        assert all(v == pytest.approx(0.0) for v in scores.values())

    def test_empty_predictions_only(self):
        scores = granularity_discounted_span_scores([(0, 5)], [])
        assert all(v == pytest.approx(0.0) for v in scores.values())

    def test_empty_truths_warns_once(self, caplog):
        with caplog.at_level("WARNING", logger="causalatee.evaluation.spans._granularity_discounted"):
            granularity_discounted_span_scores([], [(0, 5)])
            granularity_discounted_span_scores([], [(0, 5)])
            granularity_discounted_span_scores([], [])
        warnings = [r for r in caplog.records if "empty gold spans" in r.message]
        assert len(warnings) == 1

    def test_empty_predictions_does_not_warn(self, caplog):
        with caplog.at_level("WARNING", logger="causalatee.evaluation.spans._granularity_discounted"):
            granularity_discounted_span_scores([(0, 5)], [])
        assert not any("empty gold spans" in r.message for r in caplog.records)


class TestDatasetGranularityDiscountedSpanScores:
    def test_macro_average(self):
        # Instance 0: perfect match → all 1.0
        # Instance 1: no match → all 0.0
        # Macro avg → all 0.5
        result = dataset_granularity_discounted_span_scores(
            [[(0, 5)], [(0, 5)]],
            [[(0, 5)], [(10, 15)]],
        )
        assert result["f1"] == pytest.approx(0.5)
        assert result["precision"] == pytest.approx(0.5)

    def test_empty_dataset(self):
        result = dataset_granularity_discounted_span_scores([], [])
        assert all(v == 0.0 for v in result.values())
