"""Shortest Dependency Path (SDP) extraction between two entity spans.

Classical, linguistically motivated technique for relation extraction: given
a sentence and two entity spans, find the shortest path connecting the two
entities' syntactic heads in the dependency parse tree and use that path as
the feature representation for a downstream classifier.

Introduced for general relation extraction by Bunescu & Mooney (2005) and
applied specifically to causal relations by Girju, Badulescu & Moldovan
(2003). See :doc:`the Biaffine/SDP model documentation
</models/sdp_causality_extraction>` for the full scientific background,
formal definition, and references.

This module implements steps 1-4 of the classical algorithm (parse, locate
heads, find the path, encode direction) — turning the extracted path into
features and applying a classifier is left to the caller, exactly as in the
literature (the SDP is a *representation*, not a full classifier).
"""
from __future__ import annotations

from dataclasses import dataclass

try:
    import networkx as nx
    from spacy.tokens import Doc, Token
except ImportError as e:
    raise ImportError(
        "causalatee.nlp requires spacy and networkx.\nInstall them with: pip install 'causalatee[baselines]'"
    ) from e

Span = tuple[int, int]


@dataclass(frozen=True)
class SDPStep:
    """One hop along a shortest dependency path.

    Attributes
    ----------
    from_token : spacy.tokens.Token
        The token this hop starts from.
    to_token : spacy.tokens.Token
        The token this hop arrives at.
    dep : str
        The dependency label of the arc traversed.
    direction : str
        ``"↑"`` if the arc points toward the root (``to_token`` is
        ``from_token``'s syntactic head), ``"↓"`` if it points away from the
        root (``to_token`` is one of ``from_token``'s dependents).
    """

    from_token: Token
    to_token: Token
    dep: str
    direction: str


def span_head_token(doc: Doc, span: Span) -> Token:
    """Return the syntactic head token of a character span.

    Parameters
    ----------
    doc : spacy.tokens.Doc
        A parsed document (``doc.has_annotation("DEP")`` must be true).
    span : tuple[int, int]
        Character offsets ``(start, end)``.

    Returns
    -------
    spacy.tokens.Token
        The root token of the span, per spaCy's dependency parse — i.e. the
        token within the span with no ancestor also inside the span.

    Notes
    -----
    Uses ``alignment_mode="expand"`` so a span whose character offsets don't
    fall exactly on token boundaries still resolves to a token, rather than
    silently returning ``None`` (spaCy's default ``char_span`` behavior).
    """
    start, end = span
    aligned = doc.char_span(start, end, alignment_mode="expand")
    if aligned is None:
        raise ValueError(f"Could not align span {span!r} to any token in the given Doc")
    return aligned.root


def shortest_dependency_path(doc: Doc, span_a: Span, span_b: Span) -> list[SDPStep] | None:
    """Find the shortest dependency path between two entity spans.

    Parameters
    ----------
    doc : spacy.tokens.Doc
        A parsed document (requires a pipeline with a dependency parser,
        e.g. ``spacy.load("en_core_web_sm")``).
    span_a : tuple[int, int]
        Character offsets of the first entity span (e.g. the cause).
    span_b : tuple[int, int]
        Character offsets of the second entity span (e.g. the effect).

    Returns
    -------
    list[SDPStep] or None
        The path from ``span_a``'s head to ``span_b``'s head as a sequence
        of hops (see :class:`SDPStep`). Empty list if both spans resolve to
        the same head token. ``None`` if no path exists in the undirected
        dependency graph — this happens when the two spans are in different
        sentences, since spaCy's dependency arcs only connect tokens within
        one sentence.

    Examples
    --------
    >>> import spacy
    >>> nlp = spacy.load("en_core_web_sm")
    >>> doc = nlp("The storm caused significant flooding.")
    >>> path = shortest_dependency_path(doc, (0, 9), (29, 37))  # "The storm", "flooding"
    >>> format_sdp(path)
    'storm --nsubj↑--> caused --dobj↓--> flooding'
    """
    head_a = span_head_token(doc, span_a)
    head_b = span_head_token(doc, span_b)

    if head_a.i == head_b.i:
        return []

    graph: nx.Graph[int] = nx.Graph()
    for token in doc:
        if token.head is not token:
            graph.add_edge(token.i, token.head.i)

    try:
        path_indices = nx.shortest_path(graph, head_a.i, head_b.i)
    except nx.NetworkXNoPath:
        return None

    steps: list[SDPStep] = []
    for prev_i, cur_i in zip(path_indices, path_indices[1:]):
        prev_tok, cur_tok = doc[prev_i], doc[cur_i]
        if cur_tok.head.i == prev_i:
            # cur_tok's head is prev_tok -> moving away from the root.
            steps.append(SDPStep(from_token=prev_tok, to_token=cur_tok, dep=cur_tok.dep_, direction="↓"))
        else:
            # prev_tok's head is cur_tok -> moving toward the root.
            steps.append(SDPStep(from_token=prev_tok, to_token=cur_tok, dep=prev_tok.dep_, direction="↑"))
    return steps


def format_sdp(steps: list[SDPStep]) -> str:
    """Render a path as a compact human-readable string.

    Parameters
    ----------
    steps : list[SDPStep]
        A path as returned by :func:`shortest_dependency_path`.

    Returns
    -------
    str
        E.g. ``"storm --nsubj↑--> caused --dobj↓--> flooding"``. Empty
        string for an empty path.
    """
    if not steps:
        return ""
    parts = [steps[0].from_token.text]
    for step in steps:
        parts.append(f"--{step.dep}{step.direction}--> {step.to_token.text}")
    return " ".join(parts)
