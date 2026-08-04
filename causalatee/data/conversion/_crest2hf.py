# Convert CREST (https://github.com/phosseini/CREST) to HF format

import math
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from ..constants import ClassLabel, Relation, Task
from ..utils import insert_entity_markers
from ._converter import FormatConverter


class CRESTSource(IntEnum):
    SemEval2007Task4 = 1
    SemEval2010Task8 = 2
    EventCausality = 3
    CausalTimeBank = 4
    EventStoryLine = 5
    CaTeRS = 6
    BECauSE = 7
    COPA = 8
    PDTB3 = 9
    BioCause = 10
    TCR = 11
    ADE = 12
    SemEval2020Task5 = 13


class _CRESTSplit(IntEnum):
    Train = 0
    Dev = 1
    Test = 2


def _str2SplitId(name: str) -> _CRESTSplit:
    return {"train": _CRESTSplit.Train, "dev": _CRESTSplit.Dev, "test": _CRESTSplit.Test}[name]


def _parse_idx(idx_str: str) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    """Parse the idx field and return (span1_offsets, span2_offsets) as (start, end) tuples.

    The idx format is: "span1 S:E\\nspan2 S:E\\nsignal S:E"
    where S:E are character offsets into the context field.  Signal and either
    span can be absent (no colon-separated range present).
    """
    spans: dict[str, tuple[int, int] | None] = {}
    for line in idx_str.strip().split("\n"):
        parts = line.strip().split(" ")
        key = parts[0]
        spans[key] = tuple(int(x) for x in parts[1].split(":")) if len(parts) > 1 and ":" in parts[1] else None  # type: ignore[assignment]
    return spans.get("span1"), spans.get("span2")


class CREST2HF(FormatConverter):
    def __init__(self, source: Path, target: Path, prefix: str, filters: Optional[dict[str, Any]] = None):
        super().__init__(target)
        self._prefix = prefix
        self._source = source
        self._filters = filters or {}

    def _load_base_df(self, split: str) -> pd.DataFrame:
        """Load, filter, and return the CREST DataFrame for the given split.

        Only rows with a non-null ann_file are kept; these are the rows that can
        be grouped into sentences for all three tasks.  Sources without ann_file
        (SemEval 2007/2010, ADE, SemEval 2020) are therefore excluded unless
        explicitly selected and handled by a subclass.
        """
        df = pd.read_excel(self._source)
        df["split"] = df["split"].apply(lambda x: int(x) if not math.isnan(x) else None)
        for col, val in {**self._filters, "split": _str2SplitId(split)}.items():
            df = df[df[col] == val]
        return df[df["ann_file"].notna()]

    def _convert(self, task: Task, split: str) -> pd.DataFrame:
        converter: dict[Task, Callable[[str], pd.DataFrame]] = {
            Task.CausalityDetection: self._convert_causality_detection,
            Task.CausalCandidateExtraction: self._convert_causal_candidate_extraction,
            Task.CausalityIdentification: self._convert_causality_identification,
        }
        return converter[task](split)

    def _convert_causality_detection(self, split: str) -> pd.DataFrame:
        df = self._load_base_df(split)
        df = df.groupby(by="ann_file")
        df = df.agg({"context": "first", "split": "first", "ann_file": "first", "label": "max"})
        df["text"] = df["context"]
        df["index"] = df.apply(lambda row: f"{self._prefix}_{row['ann_file']}", axis=1)
        df["label"] = df["label"].apply(lambda lbl: ClassLabel.Causal if lbl == 1 else ClassLabel.Uncausal)
        return df[["index", "text", "label"]].set_index("index", verify_integrity=True)

    def _convert_causal_candidate_extraction(self, split: str) -> pd.DataFrame:
        df = self._load_base_df(split)
        rows = []
        for ann_file, group in df.groupby("ann_file"):
            context: str = group["context"].iloc[0]
            seen: set[tuple[int, int]] = set()
            entity_spans: list[list[int]] = []
            for _, row in group[group["label"] == 1].iterrows():
                span1, span2 = _parse_idx(row["idx"])
                for span in (span1, span2):
                    if span is not None and span not in seen:
                        seen.add(span)
                        entity_spans.append(list(span))
            rows.append({"index": f"{self._prefix}_{ann_file}", "text": context, "entity": entity_spans})
        assert bool(rows)
        return pd.DataFrame(rows).set_index("index")

    def _convert_causality_identification(self, split: str) -> pd.DataFrame:
        df = self._load_base_df(split)
        rows = []
        for ann_file, group in df.groupby("ann_file"):
            context: str = group["context"].iloc[0]
            causal = group[group["label"] == 1]

            # First pass: assign stable entity IDs to all unique spans across all causal pairs.
            span_to_id: dict[tuple[int, int], int] = {}
            pairs: list[tuple[tuple[int, int], tuple[int, int], int]] = []
            for _, row in causal.iterrows():
                span1, span2 = _parse_idx(row["idx"])
                if span1 is None or span2 is None:
                    continue
                for span in (span1, span2):
                    if span not in span_to_id:
                        span_to_id[span] = len(span_to_id)
                pairs.append((span1, span2, int(row["direction"])))

            # Drop spans with garbled/out-of-bounds offsets (seen in practice on
            # CaTeRS: some CREST idx entries reference positions past the end of
            # `context`, or start>=end) before attempting to tag anything.
            # Spans that merely CROSS another span (partial overlap, neither
            # containing the other) do NOT need to be dropped: the <eN>/</eN>
            # markers below are matched by entity id when parsed back
            # (causalatee.data.utils), not by XML-style nesting depth, so
            # crossing spans round-trip correctly regardless (see
            # causalatee/data/utils/markers.py's module docstring).
            usable_spans = {span: eid for span, eid in span_to_id.items() if 0 <= span[0] < span[1] <= len(context)}
            dropped_invalid = set(span_to_id) - set(usable_spans)

            if dropped_invalid:
                print(
                    f"WARNING: {self._prefix}_{ann_file}: dropped {len(dropped_invalid)} out-of-bounds/invalid span(s)."
                )

            # Build relations list (direction 0: span1=cause; direction 1: span1=effect).
            relations: list[dict] = []
            for span1, span2, direction in pairs:
                if span1 not in usable_spans or span2 not in usable_spans:
                    continue
                cause_span, effect_span = (span2, span1) if direction == 1 else (span1, span2)
                relations.append(
                    {
                        "relationship": Relation.Causal,
                        "first": f"e{usable_spans[cause_span] + 1}",
                        "second": f"e{usable_spans[effect_span] + 1}",
                    }
                )

            segments_by_eid = {f"e{eid + 1}": [span] for span, eid in usable_spans.items()}
            text = insert_entity_markers(context, segments_by_eid) if segments_by_eid else context

            rows.append({"index": f"{self._prefix}_{ann_file}", "text": text, "relations": relations})
        if not rows:
            return pd.DataFrame(columns=["text", "relations"]).rename_axis("index")
        return pd.DataFrame(rows).set_index("index")
