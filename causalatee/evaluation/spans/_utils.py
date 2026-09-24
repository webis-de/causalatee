from __future__ import annotations

from collections.abc import Sequence


def _mean(values: list[float], default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


# TODO: for the future: define bilou_to_spans and make bio_to_spans do this:
# 1) assert that id2label only contains "B-*" and "I-*" labels (no "L-*" and "U-*")
# 2) call bilou_to_spans


def bio_to_spans(
    label_ids: Sequence[int],
    offset_mapping: Sequence[tuple[int, int]],
    id2label: dict[int, str],
    *,
    ignored_id: int = -100,
) -> list[tuple[int, int]]:
    """Decode a BIO label-id sequence into character-level span tuples.

    Special tokens (offset ``(0, 0)``) and tokens with ``label_id == ignored_id`` are treated as span boundaries.
    Recognizes labels of the form ``B-*``/ ``I-*`` (entity type is ignored; all non-O spans are collected).
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
