"""Surface-form matching for causal discourse connectives.

A causal connective is a word or phrase that lexically signals a causal
relation between two spans of text (e.g. "X *causes* Y", "Y, *because* X").
Connective presence is a classical feature for causal relation
classification — used, for instance, on the shortest dependency path (see
:mod:`causalatee.nlp._sdp` and the accompanying model documentation) alongside path
length, POS tags, and dependency labels [Girju et al. 2003].

The lexicon here is a hand-curated list organized into the categories
commonly distinguished in the discourse-connective literature: causal verbs
and nouns, subordinating cue phrases, and sentence-linking adverbials
(broadly following the PDTB's ``Contingency.Cause`` connective class
[Prasad et al. 2008] and the causal verb patterns catalogued by Girju et al.
2003). It is **not** a reproduction of any single published list; treat it
as a reasonable, general-purpose starting point rather than an exhaustively
validated resource.

Important limitation: a connective match is a *candidate signal*, not proof
of a causal relation. Several entries here are highly ambiguous outside of
context — "since" and "as" are far more often temporal or comparative than
causal, and even "lead to" / "result in" can describe a purely temporal or
correlational sequence. This is exactly why the SDP literature uses the
causal lexicon as one feature among several feeding a trained classifier,
never as a standalone decision rule. Hidey & McKeown (2016) additionally
show that many real causal relations use no fixed connective at all
("alternative lexicalizations" — see the AltLex dataset), so absence of a
match is equally uninformative about the absence of a causal relation.
"""
from __future__ import annotations

from dataclasses import dataclass

try:
    from spacy.matcher import PhraseMatcher
    from spacy.tokens import Doc
    from spacy.vocab import Vocab
except ImportError as e:
    raise ImportError(
        "causalatee.nlp requires spacy.\nInstall it with: pip install 'causalatee[baselines]'"
    ) from e

CAUSAL_CONNECTIVES: dict[str, list[str]] = {
    # Matched by LEMMA: single- or multi-word causal verbs/verb phrases.
    # Lemma matching covers every inflection from one base-form entry
    # ("leads to"/"led to"/"leading to" all match "lead to"), so no
    # inflected forms need to be listed separately.
    "verb": [
        "cause", "lead to", "result in", "trigger", "induce", "bring about",
        "give rise to", "stem from", "contribute to", "produce", "generate",
        "force", "enable", "prevent", "provoke", "prompt", "necessitate",
        "compel", "determine", "arise",
    ],
    # Matched by LEMMA: causal nouns. Deliberately excludes "cause" itself
    # (also a verb lemma, above) — PhraseMatcher(attr="LEMMA") has no POS
    # awareness, so a word listed in both categories would match both
    # identically on the same token regardless of its actual grammatical
    # role, an ambiguity found via testing that a POS-blind matcher cannot
    # resolve. "reason"/"effect"/"result" etc. already cover the concept
    # without that overlap.
    "noun": [
        "reason", "result", "effect", "consequence", "outcome",
        "impact", "source", "origin", "factor",
    ],
    # Matched by LOWER surface form: closed-class subordinating cue phrases.
    # These don't meaningfully inflect, so lemma matching adds no value.
    "cue_phrase": [
        "because", "because of", "since", "as", "so that", "given that",
        "due to", "owing to", "on account of", "as a result of",
        "as a consequence of", "in view of", "thanks to",
    ],
    # Matched by LOWER: sentence-linking adverbials.
    "adverbial": [
        "therefore", "thus", "hence", "consequently", "accordingly",
        "as a result", "for this reason",
    ],
}

_LEMMA_CATEGORIES = ("verb", "noun")
_LOWER_CATEGORIES = ("cue_phrase", "adverbial")


@dataclass(frozen=True)
class ConnectiveMatch:
    """A causal connective found in text.

    Attributes
    ----------
    start : int
        Character offset of the match's first character.
    end : int
        Character offset one past the match's last character.
    text : str
        The matched surface text.
    category : str
        One of the keys of :data:`CAUSAL_CONNECTIVES`
        (``"verb"``, ``"noun"``, ``"cue_phrase"``, or ``"adverbial"``).
    negated : bool
        True if the connective itself (or its syntactic head) is under
        direct negation -- see :func:`_is_negated` for exactly what that
        checks. A negated causal connective ("did NOT cause", "was NOT a
        factor") still signals that the text is making a claim ABOUT a
        causal relationship, just denying it -- e.g. CCNC (Countercausal
        News Corpus) labels exactly this pattern as its own relation type
        (``Relation.Countercausal``) rather than "no relation", and this field
        is what lets a caller draw that same distinction instead of
        collapsing every connective match to a single causal/not-causal
        bit.
    """

    start: int
    end: int
    text: str
    category: str
    negated: bool = False


_NEGATION_CUE_LEMMAS = {"not", "never", "no", "neither", "nor", "without", "fail", "unable", "cannot"}


def _is_negated(token) -> bool:
    """True if ``token`` or its syntactic head is under direct negation.

    Checks ``token`` and ``token.head`` (its dependency governor) each two
    ways, either sufficient:

    * A ``neg``-labeled dependency child -- Universal Dependencies' own
      convention for a negation particle modifying its governing predicate
      (spaCy's standard analysis for "not"/"n't" attaching to the verb they
      negate, e.g. "did *not* cause"; the same underlying idea as PropBank's
      ``ARGM-NEG``: negation is a MODIFIER of the predicate, not a separate
      fact about it).
    * Any child (of whatever dependency label) whose lemma is in the
      closed-class cue set :data:`_NEGATION_CUE_LEMMAS` -- catches
      determiner/adverb-style negation that UD doesn't always label
      ``neg`` (e.g. "*no* factor caused..." with "no" parsed as a plain
      ``det`` of "factor"). The cue list itself follows the pre-negation
      trigger-term tradition from clinical-NLP negation detection (NegEx,
      Chapman et al. 2001; see also the ConanDoyle-neg negation-cue
      inventory) rather than being invented ad hoc.

    Heuristic, not a full negation-scope resolver (double negation, negation
    scoped over a clause boundary, or negation attached to something other
    than the connective's own head/argument are all missed) -- same
    "candidate signal, not proof" caveat as connective matching itself; see
    the module docstring.
    """
    for t in (token, token.head):
        for child in t.children:
            if child.dep_ == "neg" or child.lemma_.lower() in _NEGATION_CUE_LEMMAS:
                return True
        if t.lemma_.lower() in _NEGATION_CUE_LEMMAS:
            return True
    return False


_matcher_cache: dict[int, tuple[PhraseMatcher, PhraseMatcher]] = {}


def _get_matchers(vocab: Vocab) -> tuple[PhraseMatcher, PhraseMatcher]:
    """Build (or fetch cached) LEMMA and LOWER PhraseMatchers for the given vocab.

    PhraseMatcher respects token boundaries, so e.g. the token "cause"
    cannot spuriously match inside "because" the way a naive substring
    search would — matching is over spaCy's own tokenization, not raw
    characters. Matchers are vocab-specific in spaCy, so they're cached per
    vocab (identified by id()) instead of rebuilt on every call.
    """
    key = id(vocab)
    if key not in _matcher_cache:
        lemma_matcher = PhraseMatcher(vocab, attr="LEMMA")
        for category in _LEMMA_CATEGORIES:
            patterns = []
            for phrase in CAUSAL_CONNECTIVES[category]:
                words = phrase.split(" ")
                pattern = Doc(vocab, words=words)
                # PhraseMatcher(attr="LEMMA") reads each token's stored lemma;
                # Doc(vocab, words=...) does not run the lemmatizer, so it
                # would otherwise be empty. Every entry here is already
                # written in base/lemma form, so each token's own text IS
                # its lemma — no need to invoke the full nlp pipeline.
                for token, word in zip(pattern, words):
                    token.lemma_ = word
                patterns.append(pattern)
            lemma_matcher.add(category, patterns)

        lower_matcher = PhraseMatcher(vocab, attr="LOWER")
        for category in _LOWER_CATEGORIES:
            patterns = [Doc(vocab, words=phrase.split(" ")) for phrase in CAUSAL_CONNECTIVES[category]]
            lower_matcher.add(category, patterns)

        _matcher_cache[key] = (lemma_matcher, lower_matcher)
    return _matcher_cache[key]


def find_causal_connectives(doc: Doc) -> list[ConnectiveMatch]:
    """Find causal discourse connectives in a parsed document.

    Parameters
    ----------
    doc : spacy.tokens.Doc
        A document parsed with a pipeline that assigns lemmas (e.g.
        ``spacy.load("en_core_web_sm")``) — the ``verb``/``noun``
        categories are matched by lemma, so a tagger/lemmatizer must have
        run. A dependency parse is not required.

    Returns
    -------
    list[ConnectiveMatch]
        One non-overlapping entry per match, in document order. When
        multiple entries in :data:`CAUSAL_CONNECTIVES` overlap at the same
        position (e.g. "because" is a prefix of "because of"), only the
        longest is kept — greedy resolution, the same non-max-suppression
        idiom used by :func:`causalatee.nn.BiaffineSpanHead.decode_spans_from_grid`.
        See the module docstring for the caveat that a match is a candidate
        signal, not proof of a causal relation.

    Examples
    --------
    >>> import spacy
    >>> nlp = spacy.load("en_core_web_sm")
    >>> doc = nlp("The storm caused significant flooding.")
    >>> [(m.text, m.category) for m in find_causal_connectives(doc)]
    [('caused', 'verb')]
    """
    lemma_matcher, lower_matcher = _get_matchers(doc.vocab)
    candidates = []
    for matcher in (lemma_matcher, lower_matcher):
        for match_id, start, end in matcher(doc):
            span = doc[start:end]
            candidates.append(
                ConnectiveMatch(
                    start=span.start_char,
                    end=span.end_char,
                    text=span.text,
                    category=doc.vocab.strings[match_id],
                    # A multi-word phrase's negation can attach to any of
                    # its tokens (e.g. "lead" in "did not lead to"), so
                    # check every token in the span, not just its head.
                    negated=any(_is_negated(t) for t in span),
                )
            )

    candidates.sort(key=lambda m: m.end - m.start, reverse=True)
    accepted: list[ConnectiveMatch] = []
    for candidate in candidates:
        if not any(candidate.start < a.end and candidate.end > a.start for a in accepted):
            accepted.append(candidate)
    accepted.sort(key=lambda m: m.start)
    return accepted
