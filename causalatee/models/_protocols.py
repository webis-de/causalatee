"""Structural interfaces (``typing.Protocol``, not ``abc.ABC``) for causality models.

One per [`causalatee.data.constants.Task`][causalatee.data.constants.Task] member, plus
[`PairwiseIdentification`][causalatee.models.PairwiseIdentification] (what nearly every trained relation classifier
actually implements) and [`Extraction`][causalatee.models.Extraction] (the end-to-end task, built by composing
Detection, CandidateExtraction, and PairwiseIdentification models via
[`compose_extraction`][causalatee.models.compose_extraction]).

These are Protocols rather than ABCs deliberately. Contrast with [`causalatee.graph.Graph`][causalatee.graph.Graph]:
every ``Graph`` implementation (``CauseNet``, ``CGFGraph``) lives inside this package, so requiring explicit subclassing
there costs nothing. Causality models are the opposite case -- usually a fine-tuned ``transformers.Pipeline``, a custom
``torch.nn.Module`` baseline, or a spaCy-based rule system defined elsewhere, which should not have to import and
subclass anything here to be recognized. A class satisfies these protocols purely by having a matching ``__call__``
signature -- exactly how every current pipeline in ``causalatee.integrations.huggingface`` is already invoked
(``self._detection(text)``, not ``self._detection.detect(text)``) -- so every one of them satisfies its protocol here
with no code changes at all. ``compose_extraction`` deliberately does NOT require its three arguments to be
``transformers.Pipeline`` instances specifically, unlike the single-model ``Pipeline`` subclasses in
``causalatee.integrations.huggingface`` (which each need ``Pipeline``'s device/tokenizer machinery for their own one
model): a rule-based detector, a custom biaffine extractor, or any other object satisfying the matching protocol works
too, since there is no single underlying checkpoint for a 3-model orchestrator to hang ``Pipeline`` machinery off of in
the first place.

Every ``__call__`` is overloaded on a single ``str`` vs. a ``list[str]``, mirroring ``transformers.Pipeline.__call__``
itself: passing a list is not just a convenience loop, it lets the underlying model batch the actual forward pass (see
``Pipeline.__call__``'s own ``batch_size``/iterator handling). Every current HF-pipeline-based implementation already
gets this for free from the base class; [`lift_pairwise_identification`][causalatee.models.lift_pairwise_identification]
and [`compose_extraction`][causalatee.models.compose_extraction] take advantage of it explicitly rather than calling the
underlying model(s) once per item.

None of these are made generic (no ``Protocol[T]`` type parameter). See the module's usage docs for why: the whole point
of the shared dataset schema (``causalatee.data.constants``) is that every model for a given task returns the SAME
shape, so a per-implementation type parameter would reintroduce the heterogeneity the schema exists to remove.
``Graph``/``Node``/``Edge`` are generic because different backends genuinely differ in representation; these protocols
are not, on purpose.
"""

from __future__ import annotations

import itertools
from typing import Protocol, TypedDict, overload, runtime_checkable

from causalatee.data.constants import Relation
from causalatee.data.utils.markers import insert_entity_markers, parse_entity_markers


class DetectionResult(TypedDict):
    label: str
    score: float


class CandidateSpan(TypedDict):
    start: int
    end: int
    entity: str


class PairwiseRelation(TypedDict):
    relation: str
    score: float


class IdentifiedRelation(TypedDict):
    relationship: Relation
    first: str
    second: str
    score: float


class ExtractedRelation(TypedDict):
    e1: str
    e2: str
    relation: str
    score: float


@runtime_checkable
class Detection(Protocol):
    """Task.CausalityDetection: classifies whether a sentence is causal at all.

    Matches ``integrations.huggingface.CausalityDetectionPipeline`` as-is (batch calling is inherited for free from
    ``transformers.Pipeline``).
    """

    @overload
    def __call__(self, text: str) -> DetectionResult: ...
    @overload
    def __call__(self, text: list[str]) -> list[DetectionResult]: ...


@runtime_checkable
class CandidateExtraction(Protocol):
    """Task.CausalCandidateExtraction: extracts candidate cause/effect spans.

    Matches ``integrations.huggingface.CausalCandidateExtractionPipeline`` as-is.
    """

    @overload
    def __call__(self, text: str) -> list[CandidateSpan]: ...
    @overload
    def __call__(self, text: list[str]) -> list[list[CandidateSpan]]: ...


@runtime_checkable
class PairwiseIdentification(Protocol):
    """Classifies the relation between exactly the two spans already marked ``<e1>``/``<e2>`` in ``text``.

    This is what every current identification model in this repo implements
    (``integrations.huggingface.CausalityIdentificationPipeline``), and what nearly every relation-classification
    model in the literature implements: one relation in, one relation out, entity ids fixed to e1/e2 by convention
    because the caller already knows which two spans it's asking about.
    """

    @overload
    def __call__(self, text: str) -> PairwiseRelation: ...
    @overload
    def __call__(self, text: list[str]) -> list[PairwiseRelation]: ...


@runtime_checkable
class Identification(Protocol):
    """Task.CausalityIdentification in full generality.

    Classifies every relation among an arbitrary number of pre-marked spans (``<e1>``...``<eN>``) in one call,
    matching the ``relations`` column of the identification dataset schema directly -- a sentence may carry more
    than two marked entities and more than one relation record. No current model in this repo implements this
    directly; see [`lift_pairwise_identification`][causalatee.models.lift_pairwise_identification] to build one from
    any ``PairwiseIdentification`` model.
    """

    @overload
    def __call__(self, text: str) -> list[IdentifiedRelation]: ...
    @overload
    def __call__(self, text: list[str]) -> list[list[IdentifiedRelation]]: ...


@runtime_checkable
class Extraction(Protocol):
    """The end-to-end task: raw, unmarked text straight to structured relations.

    No current model in this repo implements this directly; see
    [`compose_extraction`][causalatee.models.compose_extraction] to build one from a ``Detection``, a
    ``CandidateExtraction``, and a ``PairwiseIdentification`` model.
    """

    @overload
    def __call__(self, text: str) -> list[ExtractedRelation]: ...
    @overload
    def __call__(self, text: list[str]) -> list[list[ExtractedRelation]]: ...


def _relation_from_label(label: str) -> Relation:
    # Assumes the pairwise model's label strings match Relation's member names (e.g. "Causal", "Countercausal",
    # "NoRelation"), as causalatee's own training notebooks set id2label up to do. A checkpoint using different
    # label strings needs its own small translation, not this adapter.
    return Relation[label]


class _LiftedPairwiseIdentification:
    """Adapts a ``PairwiseIdentification`` model to the full ``Identification`` interface by exhaustively enumerating
    ordered pairs of marked spans and re-marking each pair as ``<e1>``/``<e2>`` for the underlying model. Reused by
    ``compose_extraction`` to turn a batch of raw-text candidate spans into identified relations, rather than
    duplicating this enumeration a second time.

    O(n^2) pairs per text for n marked entities: fine for the 2-4 entities typical of current datasets, not a
    substitute for a genuinely joint/N-ary model (e.g. a ``BiaffineSpanHead``-style read-out over all pairs at
    once). Every pair -- across every input text in one call -- is flattened into a SINGLE batched call to the
    underlying model rather than one call per pair, so a batch-capable ``PairwiseIdentification`` (any
    ``transformers.Pipeline``-based one) gets real batched throughput here, not a Python loop of single calls.
    """

    def __init__(self, model: PairwiseIdentification) -> None:
        self._model = model

    @overload
    def __call__(self, text: str) -> list[IdentifiedRelation]: ...
    @overload
    def __call__(self, text: list[str]) -> list[list[IdentifiedRelation]]: ...
    def __call__(self, text):
        texts = [text] if isinstance(text, str) else text
        results = self._identify_many(texts)
        return results[0] if isinstance(text, str) else results

    def _identify_many(self, texts: list[str]) -> list[list[IdentifiedRelation]]:
        per_text_pairs: list[list[tuple[str, str]]] = []
        pair_texts: list[str] = []
        for text in texts:
            clean_text, segments_by_eid = parse_entity_markers(text)
            pairs = list(itertools.permutations(segments_by_eid, 2))
            per_text_pairs.append(pairs)
            pair_texts.extend(
                insert_entity_markers(
                    clean_text,
                    {"e1": segments_by_eid[first], "e2": segments_by_eid[second]},
                )
                for first, second in pairs
            )

        if not pair_texts:
            return [[] for _ in texts]

        # One batched round-trip through the underlying model for every pair across every input text, not one
        # call per pair.
        flat_results = self._model(pair_texts)

        relations_per_text: list[list[IdentifiedRelation]] = []
        cursor = 0
        for pairs in per_text_pairs:
            relations: list[IdentifiedRelation] = []
            for first, second in pairs:
                result = flat_results[cursor]
                cursor += 1
                relationship = _relation_from_label(result["relation"])
                if relationship == Relation.NoRelation:
                    continue
                relations.append(
                    {
                        "relationship": relationship,
                        "first": first,
                        "second": second,
                        "score": result["score"],
                    }
                )
            relations_per_text.append(relations)
        return relations_per_text


def lift_pairwise_identification(model: PairwiseIdentification) -> Identification:
    """Lift any ``PairwiseIdentification`` model into the general ``Identification`` interface via exhaustive pair
    enumeration.

    See ``_LiftedPairwiseIdentification`` (this module) for the enumeration strategy, its batched underlying calls,
    and its O(n^2)-pairs-per-text cost caveat.
    """

    return _LiftedPairwiseIdentification(model)


def identify_candidates(
    texts: list[str],
    spans_per_text: list[list[CandidateSpan]],
    identification: Identification,
) -> list[list[ExtractedRelation]]:
    """Mark every text's candidate spans at once (``<e0>``, ``<e1>``, ... one per span, not just two) and identify
    relations among them via ``identification``, remapping the resulting entity ids back to actual span text.

    ``texts`` and ``spans_per_text`` must be the same length and pairwise-aligned (``spans_per_text[i]`` are
    ``texts[i]``'s candidate spans). A text with fewer than two spans contributes an empty relation list without
    ever reaching ``identification`` -- callers that need to gate on e.g. ``Detection`` first should pass an empty
    span list for texts that shouldn't be identified at all (see ``_ComposedExtraction`` for the causality-gating
    example).

    ``identification`` can be any ``Identification`` model, including one built by
    [`lift_pairwise_identification`][causalatee.models.lift_pairwise_identification] from a
    ``PairwiseIdentification`` model, or a genuinely joint/N-ary model. Shared by ``compose_extraction`` and
    ``causalatee.mining``'s identification stage so the marking/enumeration/remapping logic exists once, not twice.
    """

    marked_texts: list[str] = []
    marked_source_index: list[int] = []
    for i, spans in enumerate(spans_per_text):
        if len(spans) < 2:
            continue
        segments = {f"e{j}": [(span["start"], span["end"])] for j, span in enumerate(spans)}
        marked_texts.append(insert_entity_markers(texts[i], segments))
        marked_source_index.append(i)

    identified = identification(marked_texts) if marked_texts else []

    results: list[list[ExtractedRelation]] = [[] for _ in texts]
    for source_index, relations in zip(marked_source_index, identified):
        text = texts[source_index]
        spans = spans_per_text[source_index]
        extracted: list[ExtractedRelation] = []
        for rel in relations:
            first_span = spans[int(rel["first"][1:])]
            second_span = spans[int(rel["second"][1:])]
            extracted.append(
                {
                    "e1": text[first_span["start"] : first_span["end"]],
                    "e2": text[second_span["start"] : second_span["end"]],
                    "relation": rel["relationship"].name,
                    "score": rel["score"],
                }
            )
        results[source_index] = extracted
    return results


class _ComposedExtraction:
    """Composes a ``Detection``, a ``CandidateExtraction``, and a ``PairwiseIdentification`` model into the full
    ``Extraction`` interface.

    For a batch of texts: detect which are causal, extract candidate spans for every text (including non-causal ones
    -- see below), then hand every text's spans to ``identify_candidates`` (which gates out texts with fewer than
    two spans, marks the rest, and identifies relations via a lifted ``PairwiseIdentification``).

    Candidate extraction runs on every text in the batch, not just the causal ones: filtering first would need a
    second index-remapping layer for one fewer batched call, and candidate spans from a non-causal text are simply
    discarded afterwards (forced to an empty list before reaching ``identify_candidates``), never used. This trades
    a small amount of wasted compute on non-causal texts for a simpler, single-pass implementation.
    """

    def __init__(
        self,
        detection: Detection,
        candidate_extraction: CandidateExtraction,
        identification: PairwiseIdentification,
    ) -> None:
        self._detection = detection
        self._candidate_extraction = candidate_extraction
        self._identification = lift_pairwise_identification(identification)

    @overload
    def __call__(self, text: str) -> list[ExtractedRelation]: ...
    @overload
    def __call__(self, text: list[str]) -> list[list[ExtractedRelation]]: ...
    def __call__(self, text):
        texts = [text] if isinstance(text, str) else text
        results = self._extract_many(texts)
        return results[0] if isinstance(text, str) else results

    def _extract_many(self, texts: list[str]) -> list[list[ExtractedRelation]]:
        if not texts:
            return []

        detections = self._detection(texts)
        all_spans = self._candidate_extraction(texts)
        causal_spans = [
            spans if detection["label"].lower() != "uncausal" else [] for detection, spans in zip(detections, all_spans)
        ]
        return identify_candidates(texts, causal_spans, self._identification)


def compose_extraction(
    detection: Detection,
    candidate_extraction: CandidateExtraction,
    identification: PairwiseIdentification,
) -> Extraction:
    """Compose a ``Detection``, a ``CandidateExtraction``, and a ``PairwiseIdentification`` model into the end-to-end
    ``Extraction`` interface.

    Mirrors ``lift_pairwise_identification``'s shape (narrower model(s) in, a model satisfying a broader protocol
    out), but across three DIFFERENT protocols rather than widening one. Backend-agnostic like every function in
    this module: each argument can be an HF ``transformers.Pipeline``, a rule-based detector, a custom biaffine
    model, or any other object satisfying the matching protocol.

    See ``_ComposedExtraction`` (this module) for the batching and gating strategy.
    """

    return _ComposedExtraction(detection, candidate_extraction, identification)
