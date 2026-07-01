from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import Pipeline


class CausalityExtractionPipeline:
    """End-to-end causality extraction pipeline.

    Chains the three subtask pipelines — detection, causal candidate
    extraction, and identification — to extract structured causal relations
    from raw text::

        pipe = CausalityExtractionPipeline(
            detection=pipeline("causality-detection", model="..."),
            candidate_extraction=pipeline("causal-candidate-extraction", model="..."),
            identification=pipeline("causality-identification", model="..."),
        )
        pipe("The storm caused significant flooding.")
        # [{"e1": "The storm", "e2": "significant flooding", "relation": "causal", "score": 0.97}]
    """

    def __init__(
        self,
        detection: Pipeline,
        candidate_extraction: Pipeline,
        identification: Pipeline,
    ) -> None:
        self._detection = detection
        self._candidate_extraction = candidate_extraction
        self._identification = identification

    def __call__(self, text: str) -> list[dict]:
        if self._detection(text)["label"].lower() == "uncausal":
            return []

        spans = self._candidate_extraction(text)
        if len(spans) < 2:
            return []

        results = []
        for i in range(len(spans)):
            for j in range(i + 1, len(spans)):
                marked = _insert_markers(text, spans[i], spans[j])
                rel = self._identification(marked)
                if rel["relation"].lower() == "no-rel":
                    continue
                results.append(
                    {
                        "e1": text[spans[i]["start"] : spans[i]["end"]],
                        "e2": text[spans[j]["start"] : spans[j]["end"]],
                        "relation": rel["relation"],
                        "score": rel["score"],
                    }
                )
        return results


def _insert_markers(text: str, span1: dict, span2: dict) -> str:
    # Insert right-to-left so earlier offsets stay valid.
    pairs = sorted(
        [(span1["start"], span1["end"], 1), (span2["start"], span2["end"], 2)],
        key=lambda x: x[0],
        reverse=True,
    )
    for start, end, eid in pairs:
        text = f"{text[:start]}<e{eid}>{text[start:end]}</e{eid}>{text[end:]}"
    return text
