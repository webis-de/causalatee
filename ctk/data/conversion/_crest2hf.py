# Convert CREST (https://github.com/phosseini/CREST) to HF format

from pathlib import Path
from typing import Callable, Any
from enum import IntEnum
import math

import pandas as pd

from ..constants import Task, ClassLabel, Relation
from ._converter import FormatConverter


class _CRESTSplit(IntEnum):
    Train = 0
    Dev = 1
    Test = 2


def _str2SplitId(name: str) -> _CRESTSplit:
    return {
        "train": _CRESTSplit.Train,
        "dev": _CRESTSplit.Dev,
        "test": _CRESTSplit.Test
    }[name]


class CREST2HF(FormatConverter):

    def __init__(self, source: Path, target: Path, prefix: str, filters: dict[str, Any] = {}):
        super().__init__(target)
        self._prefix = prefix
        self._source = source
        self._filters = filters

    def _convert(self, task: str, split: str) -> pd.DataFrame:
        converter: dict[Task, Callable[[str], pd.DataFrame]] = {
            Task.CausalityDetection: self._convert_causality_detection,
            Task.CausalCandidateExtraction: self._convert_causal_candidate_extraction,
            Task.CausalityIdentification: self._convert_causality_identification,
        }
        return converter.get(task)(split)

    def _convert_causality_detection(self, split: str) -> pd.DataFrame:
        df = pd.read_excel(self._source)
        df["split"] = df["split"].apply(lambda x: int(x) if not math.isnan(x) else None)
        for col, val in {**self._filters, "split": _str2SplitId(split)}.items():
            df = df[df[col] == val]
        assert len(df) != 0
        df = df[["context", "idx", "label", "split", "original_id", "ann_file", "direction"]]
        df = df.groupby(by="ann_file")
        assert (df[["context", "split", "ann_file"]].nunique() == 1).all().all()
        df = df.agg({"context": "first", "split": "first", "ann_file": "first", "label": "max"})

        df["text"] = df["context"]
        df["index"] = df.apply(lambda row: f"{self._prefix}_{row['ann_file']}", axis=1)
        df["label"] = df["label"].apply(lambda l: ClassLabel.Causal if l == 1 else ClassLabel.Uncausal)
        return df[["index", "text", "label"]].set_index("index", verify_integrity=True)

    def _convert_causal_candidate_extraction(self, split: str) -> pd.DataFrame:
        raise NotImplementedError

    def _convert_causality_identification(self, split: str) -> pd.DataFrame:
        raise NotImplementedError
