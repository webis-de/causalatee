from pathlib import Path

import pandas as pd

from causalatee.data.constants import Task
from causalatee.data.conversion import CREST2HF

_COLUMNS = ["split", "source", "ann_file", "context", "idx", "label", "direction"]


def _write_source_xlsx(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "crest.xlsx"
    pd.DataFrame(rows, columns=_COLUMNS).to_excel(path, index=False)
    return path


def _row(ann_file: str, context: str, idx: str, direction: int = 0) -> dict:
    return {
        "split": 0,
        "source": 1,
        "ann_file": ann_file,
        "context": context,
        "idx": idx,
        "label": 1,
        "direction": direction,
    }


def test_candidate_extraction_drops_out_of_bounds_spans(tmp_path):
    # "doc" is 12 chars long ("hello world!"); the second row's span2 (50:55) is out of bounds -- the kind of
    # CREST idx/context misalignment confirmed on CaTeRS and EventCausality (see notes.txt).
    context = "hello world!"
    rows = [
        _row("doc", context, "span1 0:5\nspan2 6:11\nsignal None"),
        _row("doc", context, "span1 0:5\nspan2 50:55\nsignal None"),
    ]
    source = _write_source_xlsx(tmp_path, rows)
    converter = CREST2HF(source, tmp_path, prefix="t")

    df = converter._convert(Task.CausalCandidateExtraction, "train")

    entities = [tuple(e) for e in df.loc["t_doc", "entity"]]
    assert (50, 55) not in entities
    assert (0, 5) in entities
    assert (6, 11) in entities


def test_candidate_extraction_keeps_in_bounds_spans_only(tmp_path):
    context = "hello world!"
    rows = [_row("doc", context, "span1 0:5\nspan2 6:11\nsignal None")]
    source = _write_source_xlsx(tmp_path, rows)
    converter = CREST2HF(source, tmp_path, prefix="t")

    df = converter._convert(Task.CausalCandidateExtraction, "train")

    entities = [tuple(e) for e in df.loc["t_doc", "entity"]]
    assert sorted(entities) == [(0, 5), (6, 11)]
