"""Derive an intra-sentence (per-sentence) view of a whole-document / whole-section causality dataset ("inter-sentence"
granularity -- see the ``granularity`` field on causalatee's dataset doc pages, e.g. BioCause's).

Several source corpora (BioCause, CaTeRS's brat annotations, CREST's own aggregation) annotate causal relations over a
whole document or section rather than a single sentence. This module lets callers work with those datasets at sentence
granularity instead -- useful e.g. for models with a limited context length. A per-sentence schema cannot represent a
relation whose participants straddle a sentence boundary (discourse-level causality does occur, e.g. BioCause: "X
requires Y ... [new sentence] This substitution decreases Z"), so relations that don't fit wholly inside one sentence
are dropped and counted.

Sentence boundaries are supplied by the caller (as a list of (start, end) character ranges) rather than computed here,
so this module has no dependency on any particular sentence segmenter (spaCy, nltk, a regex splitter, ...).

[`split_document_by_sentence`][causalatee.data.utils.split_document_by_sentence] is the generic core. Given sentence
ranges, a mapping of entity id -> character segments, and a list of "groups" (entity ids that must be kept together,
e.g. a relation's two participants), it assigns each group to the one sentence it fits wholly inside and drops+counts
the rest.
[`split_identification_to_sentences`][causalatee.data.utils.split_identification_to_sentences] /
[`split_extraction_to_sentences`][causalatee.data.utils.split_extraction_to_sentences] are single-example convenience
wrappers over that core, one per causalatee schema that carries spans (identification, extraction).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .markers import Span, insert_entity_markers, parse_entity_markers


def _wholly_inside(segments: Sequence[Span], start: int, end: int) -> bool:
    return all(start <= s and e <= end for s, e in segments)


def _sentence_index(pos: int, sent_ranges: Sequence[Span]) -> int | None:
    for i, (s, e) in enumerate(sent_ranges):
        if s <= pos < e:
            return i
    return None


def split_document_by_sentence(
    sent_ranges: Sequence[Span],
    entity_segments: Mapping[str, Sequence[Span]],
    groups: Sequence[Sequence[str]],
) -> tuple[list[dict], int]:
    """Assign each entity group to the one sentence it fits wholly inside.

    A ``group`` is a list of entity ids that must be kept together (e.g. the two participants of a relation, or a
    single unpaired entity). A group survives only if every entity id it references has ALL of its segments wholly
    inside the SAME sentence's character range; groups that straddle a sentence boundary, or whose entities don't
    overlap any known sentence at all, are dropped and counted rather than silently corrupted or discarded without a
    trace.

    Returns ``(rows, num_dropped)``: ``rows`` has one dict per input sentence range, in order, ``{"entities": {eid:
    [(local_start, local_end), ...]}, "group_indices": [i, ...]}`` -- entity offsets already re-based to be local to
    that sentence, and ``group_indices`` the indices into the input ``groups`` list that landed in this sentence
    (empty for sentences with no surviving group, so callers can always align rows 1:1 with ``sent_ranges``).
    """
    rows: list[dict] = [{"entities": {}, "group_indices": []} for _ in sent_ranges]
    num_dropped = 0
    for group_index, group in enumerate(groups):
        all_segments = [seg for eid in group for seg in entity_segments[eid]]
        anchor = min((s for s, _ in all_segments), default=None)
        si = _sentence_index(anchor, sent_ranges) if anchor is not None else None
        if si is None:
            num_dropped += 1
            continue
        s, e = sent_ranges[si]
        if not all(_wholly_inside(entity_segments[eid], s, e) for eid in group):
            num_dropped += 1
            continue
        for eid in group:
            if eid not in rows[si]["entities"]:
                rows[si]["entities"][eid] = [(a - s, b - s) for a, b in entity_segments[eid]]
        rows[si]["group_indices"].append(group_index)
    return rows, num_dropped


def split_identification_to_sentences(
    text: str,
    relations: Sequence[Mapping],
    sent_ranges: Sequence[Span],
) -> tuple[list[dict], int]:
    """Whole-document causality-IDENTIFICATION example -> per-sentence rows.

    ``text`` carries ``<eN>...</eN>`` entity markers (causalatee's identification schema); ``relations`` is the
    matching list of ``{"relationship", "first": eid, "second": eid}`` dicts. ``sent_ranges`` are character offsets
    into the MARKER-STRIPPED text (i.e. what [`parse_entity_markers`][causalatee.data.utils.parse_entity_markers]
    returns as ``clean_text``), not the raw
    ``text`` -- markers shift offsets, so a sentence segmenter should run on the clean text first. Returns ``(rows,
    num_dropped)`` where each row is ``{"text": ..., "relations": [...]}``, one per sentence, and ``num_dropped``
    counts relations whose two entities don't both fit inside a single sentence.
    """
    clean_text, entity_segments = parse_entity_markers(text)
    groups = [(rel["first"], rel["second"]) for rel in relations]
    split_rows, num_dropped = split_document_by_sentence(sent_ranges, entity_segments, groups)

    rows = []
    for (s, e), split_row in zip(sent_ranges, split_rows):
        local_relations = [relations[gi] for gi in split_row["group_indices"]]
        local_text = (
            insert_entity_markers(clean_text[s:e], split_row["entities"]) if split_row["entities"] else clean_text[s:e]
        )
        rows.append({"text": local_text, "relations": local_relations})
    return rows, num_dropped


def split_extraction_to_sentences(
    text: str,
    entities: Sequence[Sequence[int]],
    sent_ranges: Sequence[Span],
) -> tuple[list[dict], int]:
    """Whole-document causal-candidate-EXTRACTION example -> per-sentence rows.

    ``entities`` is causalatee's ``entity`` field: a list of flat, even-length document-level offset lists (``[s,
    e]`` contiguous, ``[s1, e1, s2, e2, ...]`` discontinuous). Returns ``(rows, num_dropped)`` where each row is
    ``{"text": ..., "entity": [[...], ...]}``, one per sentence, and ``num_dropped`` counts entities that don't fit
    wholly inside a single sentence.
    """
    entity_segments = {
        str(i): [(flat[j], flat[j + 1]) for j in range(0, len(flat), 2)] for i, flat in enumerate(entities)
    }
    groups = [(str(i),) for i in range(len(entities))]
    split_rows, num_dropped = split_document_by_sentence(sent_ranges, entity_segments, groups)

    rows = []
    for (s, e), split_row in zip(sent_ranges, split_rows):
        local_entities = [
            [x for seg in split_row["entities"][group_id] for x in seg]
            for gi in split_row["group_indices"]
            for group_id in [groups[gi][0]]
        ]
        rows.append({"text": text[s:e], "entity": local_entities})
    return rows, num_dropped
