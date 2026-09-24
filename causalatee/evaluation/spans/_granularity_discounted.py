from __future__ import annotations

import logging
import math

from ._utils import _mean

logger = logging.getLogger(__name__)

_warned_empty_truths = False


def _warn_empty_truths_once() -> None:
    global _warned_empty_truths
    if _warned_empty_truths:
        return
    _warned_empty_truths = True
    logger.warning(
        "granularity_discounted_span_scores() called with empty gold spans — precision/recall/f1 for this "
        "instance will be 0.0. Further warnings of this kind will not be shown."
    )


def overlap(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Portion of interval *a* covered by interval *b* (character-level)."""
    if a[0] >= a[1]:
        return 0.0
    o = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return o / (a[1] - a[0])


def _max_span_score(a: tuple[int, int], bs: list[tuple[int, int]]) -> tuple[float, int]:
    scores = [overlap(a, b) for b in bs]
    return max(scores, default=0.0), sum(s > 0 for s in scores)


def span_precision(truths: list[tuple[int, int]], predictions: list[tuple[int, int]]) -> float:
    """Average best overlap each *predicted* span has with any truth span."""
    return _mean([_max_span_score(p, truths)[0] for p in predictions])


def span_recall(truths: list[tuple[int, int]], predictions: list[tuple[int, int]]) -> float:
    """Average best overlap each *truth* span has with any predicted span."""
    return _mean([_max_span_score(t, predictions)[0] for t in truths])


def span_granularity(truths: list[tuple[int, int]], predictions: list[tuple[int, int]]) -> float:
    """Average number of predicted spans matching each truth span (Potthast et al. 2014).

    Values > 1 indicate over-fragmented predictions and penalise the F1 score.
    """
    return _mean([m for _, m in (_max_span_score(t, predictions) for t in truths) if m > 0])


def span_iou(truths: list[tuple[int, int]], predictions: list[tuple[int, int]]) -> float:
    """Set-level character intersection-over-union."""
    set_a = {x for a in truths for x in range(a[0], a[1])}
    set_b = {x for b in predictions for x in range(b[0], b[1])}
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0


def granularity_discounted_span_scores(
    truths: list[tuple[int, int]],
    predictions: list[tuple[int, int]],
) -> dict[str, float]:
    """All granularity-discounted span metrics for a single instance (Potthast et al. 2014).

    Returns precision, recall, f1, granularity, f1_gran (granularity-penalised F1),
    and intersection_over_union — matching the Touché evaluator output.
    """
    if not truths:
        _warn_empty_truths_once()

    p = span_precision(truths, predictions)
    r = span_recall(truths, predictions)
    g = span_granularity(truths, predictions)
    iou = span_iou(truths, predictions)

    f1 = 2 * p * r / (p + r) if (p > 0 and r > 0) else 0.0
    f1_gran = f1 / math.log2(1 + g) if g > 0 else f1

    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "granularity": g,
        "f1_gran": f1_gran,
        "intersection_over_union": iou,
    }


def dataset_granularity_discounted_span_scores(
    all_truths: list[list[tuple[int, int]]],
    all_predictions: list[list[tuple[int, int]]],
) -> dict[str, float]:
    """Macro-average of :func:`granularity_discounted_span_scores` over a collection of instances."""
    per_instance = [granularity_discounted_span_scores(t, p) for t, p in zip(all_truths, all_predictions)]
    if not per_instance:
        return {k: 0.0 for k in ("precision", "recall", "f1", "granularity", "f1_gran", "intersection_over_union")}
    keys = per_instance[0].keys()
    return {k: _mean([s[k] for s in per_instance]) for k in keys}
