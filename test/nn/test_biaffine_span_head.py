"""Tests for causalatee.nn._biaffine_span_head — the backbone-agnostic span-grid read-out head."""

from __future__ import annotations

import torch

from causalatee.nn._biaffine_span_head import Biaffine, BiaffineSpanHead


class TestBiaffine:
    def test_output_shape(self):
        biaffine = Biaffine(in_dim=8)
        x = torch.randn(2, 5, 8)
        y = torch.randn(2, 7, 8)
        scores = biaffine(x, y)
        assert scores.shape == (2, 5, 7)


class TestBiaffineSpanHeadForward:
    def test_output_shape_matches_seq_len(self):
        head = BiaffineSpanHead(hidden_size=16, ffn_dim=8)
        hidden_states = torch.randn(3, 10, 16)
        grid = head(hidden_states)
        assert grid.shape == (3, 10, 10)

    def test_is_backbone_agnostic(self):
        # Only hidden_size need match the encoder; arbitrary values work.
        for hidden_size in (768, 1024, 4096):
            head = BiaffineSpanHead(hidden_size=hidden_size, ffn_dim=8)
            hidden_states = torch.randn(1, 6, hidden_size)
            grid = head(hidden_states)
            assert grid.shape == (1, 6, 6)


class TestGridTargets:
    def test_single_span(self):
        # span 1 occupies tokens 2-4
        span_ids = torch.tensor([[0, 0, 1, 1, 1, 0]])
        target = BiaffineSpanHead.grid_targets(span_ids)
        assert target.shape == (1, 6, 6)
        assert target[0, 2, 4] == 1.0
        assert target.sum() == 1.0

    def test_multiple_spans(self):
        span_ids = torch.tensor([[0, 1, 1, 0, 2, 0]])
        target = BiaffineSpanHead.grid_targets(span_ids)
        assert target[0, 1, 2] == 1.0
        assert target[0, 4, 4] == 1.0
        assert target.sum() == 2.0

    def test_no_spans(self):
        span_ids = torch.zeros(1, 5, dtype=torch.long)
        target = BiaffineSpanHead.grid_targets(span_ids)
        assert target.sum() == 0.0


class TestValidCellMask:
    def test_excludes_lower_triangle(self):
        real = torch.tensor([[True, True, True]])
        mask = BiaffineSpanHead.valid_cell_mask(real)
        assert mask[0, 0, 2]
        assert not mask[0, 2, 0]

    def test_excludes_non_real_tokens(self):
        real = torch.tensor([[False, True, True]])
        mask = BiaffineSpanHead.valid_cell_mask(real)
        assert not mask[0, 0, 1]
        assert mask[0, 1, 2]


class TestDecodeSpansFromGrid:
    def _offsets(self, spans: list[tuple[int, int]]) -> torch.Tensor:
        return torch.tensor(spans, dtype=torch.long)

    def test_single_hit(self):
        scores = torch.full((3, 3), -10.0)
        scores[0, 2] = 10.0
        offsets = self._offsets([(0, 4), (5, 9), (10, 14)])
        spans = BiaffineSpanHead.decode_spans_from_grid(scores, offsets)
        assert spans == [(0, 14)]

    def test_nms_suppresses_overlapping_lower_score(self):
        scores = torch.full((4, 4), -10.0)
        scores[0, 2] = 5.0  # best, char span (0, 14)
        scores[0, 3] = 3.0  # overlaps -> suppressed
        scores[1, 2] = 2.0  # overlaps -> suppressed
        scores[3, 3] = 4.0  # disjoint -> kept
        offsets = self._offsets([(0, 4), (5, 9), (10, 14), (15, 19)])
        spans = BiaffineSpanHead.decode_spans_from_grid(scores, offsets, threshold=0.5)
        assert spans == [(0, 14), (15, 19)]

    def test_allow_overlap_keeps_every_hit(self):
        scores = torch.full((3, 3), -10.0)
        scores[0, 2] = 5.0
        scores[0, 1] = 4.0
        offsets = self._offsets([(0, 4), (5, 9), (10, 14)])
        spans = BiaffineSpanHead.decode_spans_from_grid(scores, offsets, allow_overlap=True)
        assert sorted(spans) == [(0, 9), (0, 14)]

    def test_ignores_special_and_padding_tokens(self):
        scores = torch.full((3, 3), 10.0)  # everything above threshold
        offsets = self._offsets([(0, 0), (5, 9), (0, 0)])  # only middle token is real
        spans = BiaffineSpanHead.decode_spans_from_grid(scores, offsets)
        assert spans == [(5, 9)]
