"""``Dataset.map(fn, batched=True, remove_columns=[...])``-ready adapters, so callers never have to hand-roll the loop
over ``.map()``'s batch dicts.

[`identification_batch_to_sentences`][causalatee.data.utils.identification_batch_to_sentences],
[`extraction_batch_to_sentences`][causalatee.data.utils.extraction_batch_to_sentences],
[`identification_batch_to_detection_sentences`][causalatee.data.utils.identification_batch_to_detection_sentences]
re-split a whole-document/whole-section dataset to sentence level
(one whole-document batch in, one larger per-sentence batch out) -- thin loops around the `splitting` module's
single-example functions, kept here instead of hand-rolled per-project so "re-split this dataset into sentences" is
always a one-line ``.map()`` call:

Example:
    ```python
    dataset["train"] = dataset["train"].map(
        lambda batch: identification_batch_to_sentences(batch, sentence_ranges_fn),
        batched=True,
    )
    ```

[`identification_batch_to_detection`][causalatee.data.utils.identification_batch_to_detection] /
[`identification_batch_to_extraction`][causalatee.data.utils.identification_batch_to_extraction] instead derive a
detection/extraction table at the SAME granularity (no sentence splitting at all) -- for datasets that only ship an
identification table (e.g. CTB, ESL) but need a detection or extraction table too.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from causalatee.data.constants import Relation

from .markers import Span, parse_entity_markers
from .splitting import split_document_by_sentence, split_extraction_to_sentences, split_identification_to_sentences


def _filter_causal_relations(relations: Sequence[Mapping]) -> list:
    """Filters the relation list for those that are causal; i.e., those that have not set ``relationship`` to 0
    (``Relation.NoRelation``).

    Some converters explicitly record a ``relationship: 0`` (NoRelation) entry for every non-causal candidate pair
    rather than omitting it, so "relations list is non-empty" is NOT a valid causal check for those. Other
    converters never list an absent pair at all, so filtering here is a no-op for them. Handles ``relationship``
    arriving as either the ``Relation`` IntEnum or a plain int (parquet round-trips it as a numpy/plain int, not the
    enum).
    """
    return [r for r in relations if int(r["relationship"]) != int(Relation.NoRelation)]


def identification_batch_to_sentences(
    batch: Mapping[str, Sequence],
    sentence_ranges_fn: Callable[[str], Sequence[Span]],
) -> dict[str, list]:
    """Batched wrapper around
    [`split_identification_to_sentences`][causalatee.data.utils.split_identification_to_sentences].

    ``sentence_ranges_fn(clean_text) -> [(start, end), ...]`` supplies sentence boundaries for the marker-stripped
    text (any segmenter -- spaCy, nltk, a regex splitter).
    """
    out_text: list[str] = []
    out_relations: list[list] = []
    for text, relations in zip(batch["text"], batch["relations"]):
        clean_text, _ = parse_entity_markers(text)
        rows, _ = split_identification_to_sentences(text, relations, sentence_ranges_fn(clean_text))
        out_text.extend(row["text"] for row in rows)
        out_relations.extend(row["relations"] for row in rows)
    return {"text": out_text, "relations": out_relations}


def extraction_batch_to_sentences(
    batch: Mapping[str, Sequence],
    sentence_ranges_fn: Callable[[str], Sequence[Span]],
) -> dict[str, list]:
    """Batched wrapper around [`split_extraction_to_sentences`][causalatee.data.utils.split_extraction_to_sentences]."""
    out_text: list[str] = []
    out_entity: list[list] = []
    for text, entities in zip(batch["text"], batch["entity"]):
        rows, _ = split_extraction_to_sentences(text, entities, sentence_ranges_fn(text))
        out_text.extend(row["text"] for row in rows)
        out_entity.extend(row["entity"] for row in rows)
    return {"text": out_text, "entity": out_entity}


def identification_batch_to_detection_sentences(
    batch: Mapping[str, Sequence],
    sentence_ranges_fn: Callable[[str], Sequence[Span]],
) -> dict[str, list]:
    """Derive per-sentence causality-DETECTION rows (text + 0/1 label) from a whole-document IDENTIFICATION batch.

    Some corpora's own detection table is whole-document/section only (e.g. BioCause) even though their
    identification table marks every entity -- cause+effect pairs AND unpaired effect-only ones alike. This reuses
    that richer information to still produce a per-sentence label: a sentence is causal if it contains ANY marked
    entity at all, matching the common "a document is causal if it contains any causal event, span or no span"
    convention.
    """
    out_text: list[str] = []
    out_label: list[int] = []
    for text in batch["text"]:
        clean_text, entity_segments = parse_entity_markers(text)
        sent_ranges = sentence_ranges_fn(clean_text)
        groups = [(eid,) for eid in entity_segments]
        rows, _ = split_document_by_sentence(sent_ranges, entity_segments, groups)
        for (s, e), row in zip(sent_ranges, rows):
            out_text.append(clean_text[s:e])
            out_label.append(1 if row["entities"] else 0)
    return {"text": out_text, "label": out_label}


def identification_batch_to_detection(batch: Mapping[str, Sequence]) -> dict[str, list]:
    """Derive a causality-DETECTION batch (text + 0/1 label) from an IDENTIFICATION batch, at the SAME granularity (no
    sentence splitting).

    A row is causal iff it has at least one ACTUALLY CAUSAL relation (see ``_filter_causal_relations`` -- a
    merely non-empty ``relations`` list is NOT enough, since some converters explicitly list NoRelation entries for
    non-causal pairs rather than omitting them). This matches how every converter in this project derives a
    detection label from relation presence (e.g. CCNC's countercausal sentences, which still count as causal for
    detection since they're about a causal relationship, just negating it). ``<eN>...</eN>`` markers are stripped
    from ``text``.

    NOT the right choice for BioCause: its own detection convention counts a document as causal if it has ANY marked
    entity, even an unpaired Effect-only one with no Cause and hence no relation -- see
    [`identification_batch_to_detection_sentences`][causalatee.data.utils.identification_batch_to_detection_sentences]
    for that case.
    """
    out_text: list[str] = []
    out_label: list[int] = []
    for text, relations in zip(batch["text"], batch["relations"]):
        clean_text, _ = parse_entity_markers(text)
        out_text.append(clean_text)
        out_label.append(1 if _filter_causal_relations(relations) else 0)
    return {"text": out_text, "label": out_label}


def identification_batch_to_extraction(batch: Mapping[str, Sequence]) -> dict[str, list]:
    """Derive a causal-candidate-EXTRACTION batch (text + entity spans) from an IDENTIFICATION batch, at the SAME
    granularity (no sentence splitting).

    Only entities that participate in at least one ACTUALLY CAUSAL relation are kept (see
    ``_filter_causal_relations``) -- matching this project's "only causal_eids" convention used by other
    converters throughout. Rows with no causal relation at all are dropped from the output entirely (nothing to
    extract), so the output can have fewer rows than the input, same as this module's sentence-splitting adapters.
    """
    out_text: list[str] = []
    out_entity: list[list] = []
    for text, relations in zip(batch["text"], batch["relations"]):
        causal = _filter_causal_relations(relations)
        if not causal:
            continue
        clean_text, segments = parse_entity_markers(text)
        involved = sorted({eid for rel in causal for eid in (rel["first"], rel["second"])})
        entity = [[x for seg in segments[eid] for x in seg] for eid in involved]
        out_text.append(clean_text)
        out_entity.append(entity)
    return {"text": out_text, "entity": out_entity}
