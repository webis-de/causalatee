from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from scipy.optimize import linear_sum_assignment

from ._utils import _mean

OnDegenerateSpan = Literal["raise", "zero"]


def _dice(a: tuple[int, int], b: tuple[int, int], *, on_degenerate_span: OnDegenerateSpan) -> float:
    """Dice = 2|A∩B| / (|A|+|B|), Liu et al. §2.2."""
    a_len = a[1] - a[0]
    b_len = b[1] - b[0]

    if a_len <= 0 or b_len <= 0:
        if on_degenerate_span == "raise":
            raise ValueError("Spans must have positive length")
        return 0.0

    overlap = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return 2.0 * overlap / (a_len + b_len)


def _instance_soft_tp(
    pred_spans: Sequence[tuple[int, int]],
    gold_spans: Sequence[tuple[int, int]],
    *,
    on_degenerate_span: OnDegenerateSpan,
) -> float:
    """Sum of Dice scores under the optimal one-to-one (Hungarian) matching for one instance."""
    if not pred_spans or not gold_spans:
        return 0.0

    scores = np.asarray(
        [[_dice(pred, gold, on_degenerate_span=on_degenerate_span) for gold in gold_spans] for pred in pred_spans]
    )
    pred_idx, gold_idx = linear_sum_assignment(-scores)
    return float(scores[pred_idx, gold_idx].sum())


def dice_matched_span_scores(
    truths: Sequence[tuple[int, int]],
    predictions: Sequence[tuple[int, int]],
    *,
    on_degenerate_span: OnDegenerateSpan = "raise",
) -> dict[str, float]:
    """Dice-matched precision/recall/F1 for a single instance (Liu, Mitamura & Hovy 2015, §§2.2, 2.4).

    https://aclanthology.org/W15-0807/

    Liu et al.: Dice overlap + greedy mention mapping.
    Here: Dice overlap + optimal one-to-one Hungarian matching — avoids one predicted span being
    credited against more than one gold span (and vice versa), which matters for overlapping/nested
    gold spans (e.g. SCITE's embedded-cause spans).

    An instance with no gold spans and no predicted spans is a trivial perfect match (1.0/1.0/1.0),
    unlike :func:`causalatee.evaluation.spans.granularity_discounted_span_scores`, which currently
    scores that case as 0.0 (see that function's own empty-truths warning).

    Parameters
    ----------
    on_degenerate_span:
        How to handle a zero/negative-length span reaching the Dice computation (can arise from a
        biaffine span-grid decode landing on a degenerate token offset, or from malformed gold data —
        BIO-decoded spans via :func:`causalatee.evaluation.spans.bio_to_spans` cannot produce these).
        ``"raise"`` (default) fails loudly; ``"zero"`` scores the degenerate span as 0 overlap against
        everything, matching :func:`causalatee.evaluation.spans.overlap`'s convention for the
        granularity-discounted metric.
    """
    n_pred = len(predictions)
    n_gold = len(truths)

    if n_pred == 0 and n_gold == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    soft_tp = _instance_soft_tp(predictions, truths, on_degenerate_span=on_degenerate_span)

    precision = soft_tp / n_pred if n_pred else 0.0
    recall = soft_tp / n_gold if n_gold else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def dataset_dice_matched_span_scores(
    all_truths: Sequence[Sequence[tuple[int, int]]],
    all_predictions: Sequence[Sequence[tuple[int, int]]],
    *,
    on_degenerate_span: OnDegenerateSpan = "raise",
) -> dict[str, float]:
    """Macro-average of :func:`dice_matched_span_scores` over a collection of instances.

    Every instance is weighted equally, matching
    :func:`causalatee.evaluation.spans.dataset_granularity_discounted_span_scores`'s aggregation —
    use this (not :func:`dice_matched_span_metrics`) for a like-for-like comparison against
    ``f1_gran``. See :func:`dice_matched_span_metrics` for the corpus-pooled (micro) alternative.
    """
    per_instance = [
        dice_matched_span_scores(t, p, on_degenerate_span=on_degenerate_span)
        for t, p in zip(all_truths, all_predictions)
    ]
    if not per_instance:
        return {k: 0.0 for k in ("precision", "recall", "f1")}
    keys = per_instance[0].keys()
    return {k: _mean([s[k] for s in per_instance]) for k in keys}


def dice_matched_span_metrics(
    predictions: Sequence[Sequence[tuple[int, int]]],
    references: Sequence[Sequence[tuple[int, int]]],
    *,
    on_degenerate_span: OnDegenerateSpan = "raise",
) -> dict[str, float]:
    """Character-level partial event-span F1, pooled over the whole corpus (Liu, Mitamura & Hovy 2015).

    https://aclanthology.org/W15-0807/, §§2.2, 2.4.

    Unlike :func:`dataset_dice_matched_span_scores`, this is *micro*-averaged: true positives,
    predicted-span count, and gold-span count are summed across every instance before computing one
    P/R/F1 — matching Liu et al.'s own protocol, and giving larger-span-count instances proportionally
    more weight than smaller ones (as opposed to the macro convention used everywhere else in this
    project, where every instance counts equally).

    Parameters
    ----------
    on_degenerate_span:
        See :func:`dice_matched_span_scores`.
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have equal length")

    soft_tp = 0.0
    n_pred = sum(len(spans) for spans in predictions)
    n_gold = sum(len(spans) for spans in references)

    for pred_spans, gold_spans in zip(predictions, references):
        soft_tp += _instance_soft_tp(pred_spans, gold_spans, on_degenerate_span=on_degenerate_span)

    if n_pred == 0 and n_gold == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    # P = TP / N_pred; R = TP / N_gold, Liu et al. §2.4.
    precision = soft_tp / n_pred if n_pred else 0.0
    recall = soft_tp / n_gold if n_gold else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}
