"""Rule-based causality baselines built on top of this package's own utilities, satisfying
``causalatee.models`` Protocols directly -- no training, no learned parameters, no calibrated
confidence (every ``score`` is ``1.0``).

Ported from a downstream reproduction project's own evaluation runners
(``conf-causality-repro/evaluation/runners/{lexicon,dependency}.py``), where this exact logic was
already written once and exercised against real datasets -- rather than leaving every user of
``find_causal_connectives``/``shortest_dependency_path`` to re-derive the same decision rule (and
its edge cases) independently.
"""

from __future__ import annotations

from typing import overload

try:
    import spacy
except ImportError as e:
    raise ImportError("causalatee.nlp requires spacy.\nInstall it with: pip install 'causalatee[baselines]'") from e

from causalatee.data.constants import Relation
from causalatee.data.utils.markers import parse_entity_markers
from causalatee.models import DetectionResult, PairwiseRelation

from ._connectives import CAUSAL_CONNECTIVES, ConnectiveMatch, find_causal_connectives
from ._sdp import shortest_dependency_path, span_head_token

_DEFAULT_SPACY_MODEL = "en_core_web_sm"

# For the "connective dropped by overlap resolution" fallback in SDPIdentifier._connective_hit --
# see that method's docstring for why a lemma match alone (with no ConnectiveMatch to read
# .negated from) is still accepted as a signal.
_CAUSAL_VERB_LEMMAS = {v.split()[0] for v in CAUSAL_CONNECTIVES["verb"]}
_CAUSAL_NOUN_LEMMAS = set(CAUSAL_CONNECTIVES["noun"])


class LexiconDetector:
    """``causalatee.models.Detection`` baseline: a sentence is ``causal`` iff it contains any
    causal connective (:func:`causalatee.nlp.find_causal_connectives`).

    Deterministic and untrained -- ``score`` is always ``1.0``, not a calibrated probability.
    causalatee's own SDP model documentation warns that a connective match is "a candidate
    signal, not proof of a causal relation" and should be used as one feature among several,
    never a standalone decision rule; this class uses it as exactly that anyway, as the cheapest
    possible floor baseline. Expect the resulting precision/recall asymmetry: high precision on
    canonical connectives, capped recall on implicit causality or "alternative lexicalizations"
    that use no fixed connective at all (see the AltLex dataset). A negated connective ("did *not*
    cause") still counts as causal here, matching ``ClassLabel.Causal``'s own definition, which
    covers both causal and countercausal sentences alike -- detection does not distinguish them,
    only identification does.
    """

    def __init__(self, nlp: spacy.language.Language | None = None) -> None:
        self._nlp = nlp or spacy.load(_DEFAULT_SPACY_MODEL)

    @overload
    def __call__(self, text: str) -> DetectionResult: ...
    @overload
    def __call__(self, text: list[str]) -> list[DetectionResult]: ...
    def __call__(self, text):
        texts = [text] if isinstance(text, str) else text
        docs = self._nlp.pipe(texts)
        results: list[DetectionResult] = [
            {"label": "causal" if find_causal_connectives(doc) else "uncausal", "score": 1.0} for doc in docs
        ]
        return results[0] if isinstance(text, str) else results


class SDPIdentifier:
    """``causalatee.models.PairwiseIdentification`` baseline: the relation between the two
    ``<e1>``/``<e2>``-marked spans in ``text`` is CAUSAL iff a causal connective lies on the
    shortest dependency path between their head tokens
    (:func:`causalatee.nlp.shortest_dependency_path`); NEGATED, it is COUNTERCAUSAL instead of
    CAUSAL; absent, it is NORELATION.

    Negation is a MODIFIER of the relation the connective signals, not a separate fact: "did *not*
    cause" is still a claim ABOUT causality, just denying it -- exactly how CCNC's own
    ``Relation.Countercausal`` label is defined (see
    :attr:`causalatee.nlp.ConnectiveMatch.negated`).

    Deterministic and untrained -- ``score`` is always ``1.0``. Only the FIRST segment of each
    marked span is used: discontinuous entity mentions have no single syntactic head token, so
    this baseline can't consume them directly -- a caller needing that would have to pick a
    representative segment itself (e.g. the longest one) before calling this.
    """

    def __init__(self, nlp: spacy.language.Language | None = None) -> None:
        self._nlp = nlp or spacy.load(_DEFAULT_SPACY_MODEL)

    @overload
    def __call__(self, text: str) -> PairwiseRelation: ...
    @overload
    def __call__(self, text: list[str]) -> list[PairwiseRelation]: ...
    def __call__(self, text):
        texts = [text] if isinstance(text, str) else text
        clean_texts: list[str] = []
        span_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for one_text in texts:
            clean_text, segments_by_eid = parse_entity_markers(one_text)
            clean_texts.append(clean_text)
            span_pairs.append((segments_by_eid["e1"][0], segments_by_eid["e2"][0]))

        docs = self._nlp.pipe(clean_texts)
        results: list[PairwiseRelation] = [
            {"relationship": self._predict(doc, e1_span, e2_span).name, "score": 1.0}
            for doc, (e1_span, e2_span) in zip(docs, span_pairs)
        ]
        return results[0] if isinstance(text, str) else results

    def _predict(self, doc, e1_span, e2_span) -> Relation:
        """The predicted ``Relation`` for one pair -- see the class docstring for the rule."""
        try:
            steps = shortest_dependency_path(doc, e1_span, e2_span)
        except ValueError:
            return Relation.NoRelation
        if steps is None:
            return Relation.NoRelation

        connectives = find_causal_connectives(doc)
        # Empty path (steps == []) means both spans share one head token -- there is nothing to
        # take the union of, so fall back to that single shared token instead.
        path_tokens = {t for step in steps for t in (step.from_token, step.to_token)} or {span_head_token(doc, e1_span)}

        matched = negated = False
        for token in path_tokens:
            token_matched, token_negated = self._connective_hit(token, connectives)
            matched = matched or token_matched
            negated = negated or token_negated
        if not matched:
            return Relation.NoRelation
        return Relation.Countercausal if negated else Relation.Causal

    @staticmethod
    def _connective_hit(token, connectives: list[ConnectiveMatch]) -> tuple[bool, bool]:
        """``(matched, negated)`` for whichever ``ConnectiveMatch`` covers ``token``, if any.

        Falls back to a plain verb/noun-lemma check (with no ``negated`` information available,
        hence ``False``) when the token's lemma IS a causal verb/noun but got no
        ``ConnectiveMatch`` of its own: ``find_causal_connectives``'s overlap resolution can drop
        a real match in favour of a longer overlapping match from another category at the same
        position, but the token itself is still a genuine signal even without its own match
        object to read ``.negated`` from. Rare in practice -- most causal-lemma tokens get their
        own accepted match -- but real when it happens.
        """
        start, end = token.idx, token.idx + len(token.text)
        for c in connectives:
            if start < c.end and c.start < end:
                return True, c.negated
        if token.lemma_.lower() in _CAUSAL_VERB_LEMMAS | _CAUSAL_NOUN_LEMMAS:
            return True, False
        return False, False
