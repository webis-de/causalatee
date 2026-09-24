"""Tests for causalatee.evaluation.spans._utils.bio_to_spans."""

from __future__ import annotations

from causalatee.evaluation.spans import bio_to_spans


class TestBioToSpans:
    _id2label = {0: "O", 1: "B-CAUSE", 2: "I-CAUSE", 3: "B-EFFECT", 4: "I-EFFECT"}

    def test_single_span(self):
        label_ids = [-100, 1, 2, 0, -100]
        offsets = [(0, 0), (0, 5), (6, 10), (11, 15), (0, 0)]
        spans = bio_to_spans(label_ids, offsets, self._id2label)
        assert spans == [(0, 10)]

    def test_two_spans(self):
        label_ids = [-100, 1, 0, 3, 4, -100]
        offsets = [(0, 0), (0, 4), (5, 8), (9, 13), (14, 20), (0, 0)]
        spans = bio_to_spans(label_ids, offsets, self._id2label)
        assert (0, 4) in spans
        assert (9, 20) in spans

    def test_all_o(self):
        label_ids = [-100, 0, 0, -100]
        offsets = [(0, 0), (0, 3), (4, 7), (0, 0)]
        assert bio_to_spans(label_ids, offsets, self._id2label) == []

    def test_b_without_i(self):
        label_ids = [-100, 1, 3, -100]
        offsets = [(0, 0), (0, 4), (5, 9), (0, 0)]
        spans = bio_to_spans(label_ids, offsets, self._id2label)
        assert (0, 4) in spans
        assert (5, 9) in spans

    def test_ignored_id_acts_as_boundary(self):
        label_ids = [1, -100, 2]
        offsets = [(0, 3), (3, 5), (5, 8)]
        spans = bio_to_spans(label_ids, offsets, self._id2label)
        # -100 in the middle breaks the span
        assert (0, 3) in spans
        assert len(spans) == 1  # I- without preceding B- is dropped
