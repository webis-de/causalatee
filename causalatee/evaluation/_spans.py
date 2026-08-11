from __future__ import annotations

import math
from collections.abc import Sequence


def _mean(values: list[float], default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


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


def span_scores(
    truths: list[tuple[int, int]],
    predictions: list[tuple[int, int]],
) -> dict[str, float]:
    """All span metrics for a single instance.

    Returns precision, recall, f1, granularity, f1_gran (granularity-penalised F1),
    and intersection_over_union — matching the Touché evaluator output.
    """
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


def dataset_span_scores(
    all_truths: list[list[tuple[int, int]]],
    all_predictions: list[list[tuple[int, int]]],
) -> dict[str, float]:
    """Macro-average of :func:`span_scores` over a collection of instances."""
    per_instance = [span_scores(t, p) for t, p in zip(all_truths, all_predictions)]
    if not per_instance:
        return {k: 0.0 for k in ("precision", "recall", "f1", "granularity", "f1_gran", "intersection_over_union")}
    keys = per_instance[0].keys()
    return {k: _mean([s[k] for s in per_instance]) for k in keys}


def bio_to_spans(
    label_ids: Sequence[int],
    offset_mapping: Sequence[tuple[int, int]],
    id2label: dict[int, str],
    *,
    ignored_id: int = -100,
) -> list[tuple[int, int]]:
    """Decode a BIO label-id sequence into character-level span tuples.

    Special tokens (offset ``(0, 0)``) and tokens with ``label_id == ignored_id``
    are treated as span boundaries.  Recognises labels of the form ``B-*``/ ``I-*``
    (entity type is ignored; all non-O spans are collected).
    """
    spans: list[tuple[int, int]] = []
    current: dict | None = None

    for label_id, (char_start, char_end) in zip(label_ids, offset_mapping):
        is_special = char_start == char_end
        if is_special or label_id == ignored_id:
            if current is not None:
                spans.append((current["start"], current["end"]))
                current = None
            continue

        label = id2label.get(label_id, "O")

        if label.startswith("B-"):
            if current is not None:
                spans.append((current["start"], current["end"]))
            current = {"start": char_start, "end": char_end}
        elif label.startswith("I-") and current is not None:
            current["end"] = char_end
        else:
            if current is not None:
                spans.append((current["start"], current["end"]))
                current = None

    if current is not None:
        spans.append((current["start"], current["end"]))

    return spans
