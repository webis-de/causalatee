"""Biaffine span-grid read-out head for causal event span extraction.

NER-as-parsing formulation (Yu et al. 2020, ACL): score every (start, end)
token pair with a biaffine classifier over an L x L grid,

    grid[i, j] = 1  iff  a span starts at token i and ends at token j

Each cell is an independent binary prediction, so overlapping and nested
spans are supported natively (unlike BIO tagging). Decoding is a threshold
over the upper triangle plus optional greedy non-max suppression -- no BIO
transition constraints, no gap-tolerance heuristics.

``BiaffineSpanHead`` is intentionally backbone-agnostic: it consumes generic
``(batch, seq_len, hidden_size)`` hidden states and has no dependency on any
specific encoder. Attach it to the last hidden state of any pretrained
transformer (BERT, RoBERTa, DeBERTa, ...) and fine-tune end to end -- the
head itself carries no pretrained weights and must be trained (or fine-tuned)
on top of whichever backbone it is attached to; the projections it learns
are specific to that backbone's hidden space and do not transfer to a
different one without retraining.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class Biaffine(nn.Module):
    """Biaffine scorer: ``s(i, j) = x_i^T U y_j`` with bias terms via appended ones.

    Parameters
    ----------
    in_dim : int
        Dimensionality of the two projected input representations.
    """

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.U = nn.Parameter(torch.empty(in_dim + 1, in_dim + 1))
        nn.init.xavier_uniform_(self.U)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Score every pair of positions between ``x`` and ``y``.

        Parameters
        ----------
        x : Tensor
            Shape ``(batch, len_x, in_dim)``.
        y : Tensor
            Shape ``(batch, len_y, in_dim)``.

        Returns
        -------
        Tensor
            Shape ``(batch, len_x, len_y)`` pairwise scores.
        """
        # Separate ones per side: x and y may have different sequence
        # lengths (e.g. scoring across two distinct sequences), so a single
        # ones tensor sized for one side cannot be reused for the other.
        # BiaffineSpanHead always calls this with x and y of equal length
        # (both derived from the same hidden states), so this had no
        # observable effect there -- caught by a more general unit test.
        x = torch.cat([x, x.new_ones(*x.shape[:-1], 1)], dim=-1)
        y = torch.cat([y, y.new_ones(*y.shape[:-1], 1)], dim=-1)
        return torch.einsum("bxi,ij,byj->bxy", x, self.U, y)


class BiaffineSpanHead(nn.Module):
    """Scores every (start, end) token pair for being an event span.

    Backbone-agnostic read-out layer: takes the final hidden states of any
    transformer encoder and produces an ``(L, L)`` span grid. Only
    ``hidden_size`` needs to match the backbone; everything else about the
    head is independent of which encoder produced the hidden states.

    Parameters
    ----------
    hidden_size : int
        Dimensionality of the backbone's hidden states (e.g. 1024 for
        roberta-large, 768 for bert-base).
    ffn_dim : int, default=256
        Dimensionality of the start/end projections feeding the biaffine
        scorer.
    dropout : float, default=0.2
        Dropout probability applied after each projection.

    Examples
    --------
    Attach to any HuggingFace encoder and fine-tune end to end::

        from transformers import AutoModel
        from causalatee.nn import BiaffineSpanHead

        backbone = AutoModel.from_pretrained("bert-base-uncased")
        head = BiaffineSpanHead(backbone.config.hidden_size)

        hidden_states = backbone(input_ids, attention_mask=attention_mask).last_hidden_state
        grid = head(hidden_states)  # (batch, seq_len, seq_len) raw logits
    """

    def __init__(self, hidden_size: int, ffn_dim: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        self.start_mlp = nn.Sequential(
            nn.Linear(hidden_size, ffn_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.end_mlp = nn.Sequential(
            nn.Linear(hidden_size, ffn_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.biaffine = Biaffine(ffn_dim)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Compute the span grid.

        Parameters
        ----------
        hidden_states : Tensor
            Shape ``(batch, seq_len, hidden_size)`` -- the backbone's final
            hidden states (or any intermediate layer with the same
            dimensionality).

        Returns
        -------
        Tensor
            Shape ``(batch, seq_len, seq_len)`` raw (pre-sigmoid) span
            logits. Cell ``[b, i, j]`` is the score for "a span starts at
            token i and ends at token j" in batch element ``b``.
        """
        return self.biaffine(self.start_mlp(hidden_states), self.end_mlp(hidden_states))

    @staticmethod
    def grid_targets(span_ids: Tensor) -> Tensor:
        """Build the gold (start, end) grid from per-token span identities.

        Parameters
        ----------
        span_ids : Tensor
            Shape ``(batch, seq_len)``, integer. ``0`` marks background
            tokens; a positive value ``k`` marks all tokens belonging to the
            ``k``-th gold span within that batch element.

        Returns
        -------
        Tensor
            Shape ``(batch, seq_len, seq_len)`` float grid with a ``1.0`` at
            each gold span's ``(start, end)`` cell, ``0.0`` elsewhere.

        Notes
        -----
        ``span_ids`` can only encode non-overlapping gold spans (one id per
        token). Overlapping gold annotations need span boundaries supplied
        directly; the head itself supports overlapping spans at inference
        time regardless of how training targets were constructed.
        """
        batch_size, seq_len = span_ids.shape
        target = span_ids.new_zeros((batch_size, seq_len, seq_len), dtype=torch.float)
        for b in range(batch_size):
            sids = span_ids[b]
            for k in sids.unique().tolist():
                if k == 0:
                    continue
                idx = (sids == k).nonzero(as_tuple=True)[0]
                target[b, idx[0], idx[-1]] = 1.0
        return target

    @staticmethod
    def valid_cell_mask(real_token_mask: Tensor) -> Tensor:
        """Upper-triangle cells where both start and end are real tokens.

        Parameters
        ----------
        real_token_mask : Tensor
            Shape ``(batch, seq_len)``, boolean. ``True`` for non-special,
            non-padding tokens.

        Returns
        -------
        Tensor
            Shape ``(batch, seq_len, seq_len)`` boolean mask selecting cells
            usable for loss computation (``start <= end``, both real tokens).
        """
        pair = real_token_mask.unsqueeze(2) & real_token_mask.unsqueeze(1)
        seq_len = real_token_mask.shape[1]
        triu = torch.ones(seq_len, seq_len, dtype=torch.bool, device=real_token_mask.device).triu()
        return pair & triu

    @staticmethod
    def decode_spans_from_grid(
        scores: Tensor,
        offset_mapping: Tensor,
        threshold: float = 0.5,
        allow_overlap: bool = False,
    ) -> list[tuple[int, int]]:
        """Decode character-level spans from a predicted grid.

        Parameters
        ----------
        scores : Tensor
            Shape ``(seq_len, seq_len)`` raw logits for a single example
            (as returned by :meth:`forward`, indexed to drop the batch dim).
        offset_mapping : Tensor
            Shape ``(seq_len, 2)`` character offsets per token, as returned
            by a HuggingFace fast tokenizer with ``return_offsets_mapping``.
            Special/padding tokens are expected to carry offset ``(0, 0)``.
        threshold : float, default=0.5
            Sigmoid probability threshold for accepting a cell.
        allow_overlap : bool, default=False
            If ``False`` (default), applies greedy non-max suppression:
            candidates are ranked by score and a candidate is kept only if
            it does not character-overlap an already-accepted span. This
            resolves the common case of several adjacent/off-by-one cells
            firing for the same true span into one clean prediction. Set to
            ``True`` to keep every above-threshold cell instead -- needed
            when spans in the target domain genuinely overlap or nest.

        Returns
        -------
        list[tuple[int, int]]
            Predicted spans as ``(char_start, char_end)`` tuples.
        """
        offsets = offset_mapping
        real = (offsets[:, 0] != 0) | (offsets[:, 1] != 0)
        valid = (real.unsqueeze(1) & real.unsqueeze(0)).triu()
        probs = torch.sigmoid(scores)
        hits = ((probs >= threshold) & valid).nonzero(as_tuple=False)
        candidates = [
            (probs[i, j].item(), int(offsets[i, 0]), int(offsets[j, 1]))
            for i, j in hits.tolist()
        ]
        if allow_overlap:
            return [(s, e) for _, s, e in candidates]

        candidates.sort(key=lambda c: c[0], reverse=True)
        accepted: list[tuple[int, int]] = []
        for _, s, e in candidates:
            if not any(s < ae and e > as_ for as_, ae in accepted):
                accepted.append((s, e))
        return sorted(accepted)
