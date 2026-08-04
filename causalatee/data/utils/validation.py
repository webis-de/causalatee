"""Structural sanity checks over one converted dataset split, independent of which converter produced it.

Meant to run right after a ``conversion_script.py`` (or the shared ``FormatConverter``
base it may use) builds a split's ``DataFrame`` and before it's written to parquet, so a defect in the source data
becomes visible immediately at conversion time instead of being silently inherited by every downstream user of the
parquet file.

Identification-only for now -- add task-specific checks here as new issue
classes turn up; nothing about [`verify_dataset`][causalatee.data.utils.verify_dataset]'s signature needs to
change for that.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from causalatee.data.constants import Task

from .markers import parse_entity_markers


def _check_relation_pairs(relations_col: Sequence[Sequence[Mapping]]) -> list[str]:
    """Flag every (first, second) pair with more than one relation record.

    Two flavours, both worth surfacing but for different reasons:

    * Conflicting values (e.g. 0 and 1 for the same pair) -- a genuine upstream annotation contradiction. There's no
      principled way to silently pick a winner here, so this is reported rather than resolved.
    * Agreeing duplicates (the same value recorded twice) -- harmless in effect (the label is unambiguous either
      way) but still redundant data worth cleaning up at the source.
    """
    errors: list[str] = []
    for row_idx, relations in enumerate(relations_col):
        by_pair: dict[tuple[str, str], list[int]] = defaultdict(list)
        for r in relations or []:
            by_pair[(r["first"], r["second"])].append(int(r["relationship"]))
        for (first, second), values in by_pair.items():
            if len(values) < 2:
                continue
            if len(set(values)) > 1:
                errors.append(
                    f"row {row_idx}: conflicting relationship values for pair "
                    f"({first!r}, {second!r}): {sorted(set(values))} -- same entity "
                    f"pair recorded with disagreeing labels"
                )
            else:
                errors.append(
                    f"row {row_idx}: duplicate (agreeing) relation record for pair "
                    f"({first!r}, {second!r}) -- relationship={values[0]} listed "
                    f"{len(values)} times"
                )
    return errors


def _check_entity_markup(text_col: Sequence[str], relations_col: Sequence[Sequence[Mapping]]) -> list[str]:
    """Flag a relation whose entity id has no parseable ``<eN>...</eN>`` span.

    Corrupted or truncated markup can name an entity id in ``relations`` that
    [`parse_entity_markers`][causalatee.data.utils.parse_entity_markers] never recovers a span for -- this would
    otherwise KeyError deep in a downstream consumer's span lookup instead of failing here with a clear cause.
    """
    errors: list[str] = []
    for row_idx, (text, relations) in enumerate(zip(text_col, relations_col)):
        _, segments = parse_entity_markers(text)
        for r in relations or []:
            for eid in (r["first"], r["second"]):
                if eid not in segments:
                    errors.append(
                        f"row {row_idx}: relation references entity id {eid!r} with "
                        f"no parseable <{eid}>...</{eid}> marker span in text"
                    )
    return errors


def verify_dataset(batch: Mapping[str, Sequence], task: Task) -> list[str]:
    """Run every applicable structural check over one converted split.

    ``batch`` is column-oriented (``batch["text"]``, ``batch["relations"]``, ...) -- same convention as the
    ``batch`` module's ``Dataset.map``-ready functions, so a ``pandas.DataFrame`` works directly by column name right
    before ``.to_parquet()``, with no conversion needed. Returns a flat list of human-readable error descriptions
    (one entry per issue found), or ``[]`` if the split is clean.
    """
    if task != Task.CausalityIdentification or "relations" not in batch:
        return []
    errors: list[str] = []
    errors.extend(_check_relation_pairs(batch["relations"]))
    errors.extend(_check_entity_markup(batch["text"], batch["relations"]))
    return errors
