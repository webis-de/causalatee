"""A ``Graph`` adapter for the Lexical Cause-Effect Graph (CEG).

CEG pairs word-level cause/effect tokens mined at CausalBank scale (see github.com/eecrazy/CausalBank) with three per-
pair statistics: an occurrence count and two causal-strength scores ("necessity" and "sufficiency", from the
"Commonsense Causal Reasoning between Short Texts" methodology the CausalBank authors cite). Unlike CauseNet's ~11.6M
relations, a full CEG release runs to tens of millions of edges over a comparatively small (tens-of-thousands-sized)
lexical vocabulary -- too many edges to eagerly materialize as Python objects the way ``load_causenet`` does, but few
enough unique nodes that eager node loading is still fine. ``CauseEffectGraph`` is therefore a deliberately two-tier
adapter: eager nodes, streaming edges. See ``load_cause_effect_graph`` and the class docstring below for what that means
in practice.

Expected file format: one edge per line, tab-separated::

    cause->effect\tcount\tnecessity\tsufficiency

``count`` is an integer; ``necessity``/``sufficiency`` are floats and are NOT bounded to [0, 1] (a strength score, not a
probability). A small number of lines have a literal ``->`` embedded inside the effect token itself (e.g.
``future->cap->lock``, i.e. cause ``"future"``, effect ``"cap->lock"``) -- parsing therefore splits on the FIRST ``->``
only, never on all occurrences.

Only local, uncompressed files are supported for now. Unlike ``load_causenet`` (which accepts gzip/bzip2 and http(s)
sources), CEG has no documented compressed or remote distribution to design against yet -- add that support if/when one
surfaces, rather than guessing at it here.
"""

from __future__ import annotations

import os
import re
from collections.abc import Collection, Iterable, Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._graph import Edge, Graph, Node
else:
    try:  # Package use.
        from ._graph import Edge, Graph, Node
    except ImportError:  # Direct use with the reference files in one directory.
        from _graph import Edge, Graph, Node


CAUSE_EFFECT_GRAPH_NODE_METADATA_SCHEMA: Mapping[str, object] = {
    "type": "record",
    "name": "CauseEffectGraphNodeMetadata",
    "namespace": "causalatee.graph.cause_effect_graph",
    "fields": [],
}

CAUSE_EFFECT_GRAPH_EDGE_METADATA_SCHEMA: Mapping[str, object] = {
    "type": "record",
    "name": "CauseEffectGraphEdgeMetadata",
    "namespace": "causalatee.graph.cause_effect_graph",
    "fields": [
        {"name": "count", "type": "long"},
        {"name": "necessity", "type": "double"},
        {"name": "sufficiency", "type": "double"},
    ],
}

Source = str | os.PathLike[str]


class CauseEffectNode(Node["CauseEffectNode", "CauseEffectEdge"]):
    """A single lexical (word-level) node in the Cause-Effect Graph."""

    def __init__(self, graph: "CauseEffectGraph", token: str) -> None:
        super().__init__(graph)
        self._token = token

    @property
    def id(self) -> str:
        return self._token

    @property
    def token(self) -> str:
        """Return the lexical token this node represents."""

        return self._token

    def __repr__(self) -> str:
        return f"CauseEffectNode({self._token!r})"


class CauseEffectEdge(Edge[CauseEffectNode, "CauseEffectEdge"]):
    """A directed cause->effect edge with an occurrence count and two causal-strength scores."""

    def __init__(
        self,
        graph: "CauseEffectGraph",
        cause: CauseEffectNode,
        effect: CauseEffectNode,
        *,
        count: int,
        necessity: float,
        sufficiency: float,
    ) -> None:
        super().__init__(graph)
        self._cause = cause
        self._effect = effect
        self._count = count
        self._necessity = necessity
        self._sufficiency = sufficiency

    @property
    def source(self) -> CauseEffectNode:
        return self._cause

    @property
    def target(self) -> CauseEffectNode:
        return self._effect

    @property
    def cause(self) -> CauseEffectNode:
        """Return the cause node."""

        return self._cause

    @property
    def effect(self) -> CauseEffectNode:
        """Return the effect node."""

        return self._effect

    @property
    def count(self) -> int:
        """Return the number of occurrences supporting this pair."""

        return self._count

    @property
    def necessity(self) -> float:
        """Return the necessity causal-strength score."""

        return self._necessity

    @property
    def sufficiency(self) -> float:
        """Return the sufficiency causal-strength score."""

        return self._sufficiency

    def __repr__(self) -> str:
        return f"CauseEffectEdge({self._cause.id!r} -> {self._effect.id!r})"


_LINE_RE = re.compile(
    r"^(?P<cause>[^\t]+?)->(?P<effect>[^\t]+)"
    r"\t(?P<count>\d+)\t(?P<necessity>-?\d+(?:\.\d+)?)\t(?P<sufficiency>-?\d+(?:\.\d+)?)$"
)


def _parse_line(line: str, *, line_number: int) -> tuple[str, str, int, float, float]:
    match = _LINE_RE.match(line.rstrip("\n"))
    if match is None:
        raise ValueError(f"line {line_number}: malformed CEG line {line!r}")
    return match["cause"], match["effect"], int(match["count"]), float(match["necessity"]), float(match["sufficiency"])


class _StreamingEdges(Collection[CauseEffectEdge]):
    """Re-iterable, disk-backed view over ``CauseEffectGraph.edges``.

    Each ``iter(...)`` call re-reads the source file from the top and yields freshly constructed ``CauseEffectEdge``
    objects -- nothing is cached between iterations. That is the entire point: a 90M-line file's edges never exist
    as a Python list at once, so ``save_cgf``'s single pass over ``graph.edges`` can convert a CEG file far larger
    than available RAM, the same way its own writer is already disk-backed on the write side.

    ``__len__``/``__contains__`` are implemented for interface completeness, not for performance: both force a full
    re-scan of the file (``len`` the first time it's called, caching the result; ``in`` every time, since two
    ``CauseEffectEdge`` instances from different iterations are never ``==``-equal to each other by identity).
    Neither is called by ``save_cgf`` itself. Prefer converting to CGF once and querying the resulting ``CGFGraph``
    for anything that needs repeated traversal -- exactly the "eager/streaming source -> CGF -> fast repeated
    queries" pattern documented for ``CauseNet`` too.
    """

    def __init__(self, graph: "CauseEffectGraph") -> None:
        self._graph = graph
        self._length: int | None = None

    def __iter__(self) -> Iterator[CauseEffectEdge]:
        graph = self._graph
        limit = graph._limit
        with Path(graph.source).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if limit is not None and line_number > limit:
                    break
                cause, effect, count, necessity, sufficiency = _parse_line(line, line_number=line_number)
                yield CauseEffectEdge(
                    graph,
                    graph.get_node(cause),
                    graph.get_node(effect),
                    count=count,
                    necessity=necessity,
                    sufficiency=sufficiency,
                )

    def __len__(self) -> int:
        if self._length is None:
            self._length = sum(1 for _ in self)
        return self._length

    def __contains__(self, value: object) -> bool:
        return any(value == edge for edge in self)


class CauseEffectGraph(Graph[CauseEffectNode, CauseEffectEdge]):
    """A ``Graph`` view over a Lexical Cause-Effect Graph (CEG) text file.

    Node vocabulary is materialized eagerly in ``__init__`` (a real CEG release has on the order of 10^4-10^5 unique
    lexical tokens -- trivial to hold in memory, the same way ``CauseNet`` holds its whole node set). Edges are NOT
    materialized: ``.edges`` streams from disk on every iteration (see ``_StreamingEdges``).
    ``edges_from``/``edges_to`` are O(edge count) per call as a direct consequence -- this backend has no adjacency
    index, unlike ``CauseNet``'s in-memory dicts or ``CGFGraph``'s CSR arrays. It exists to feed ``save_cgf`` once,
    not for interactive per-node traversal; convert to CGF first if you need that.
    """

    def __init__(self, source: Source, *, limit: int | None = None) -> None:
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative or None")

        self._source = os.fspath(source)
        self._limit = limit
        self._nodes_by_id: dict[str, CauseEffectNode] = {}

        with Path(self._source).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if limit is not None and line_number > limit:
                    break
                cause, effect, *_ = _parse_line(line, line_number=line_number)
                self._get_or_create_node(cause)
                self._get_or_create_node(effect)

    @property
    def source(self) -> str:
        """Return the path this graph was loaded from."""

        return self._source

    @property
    def nodes(self) -> Collection[CauseEffectNode]:
        return self._nodes_by_id.values()

    @property
    def edges(self) -> Collection[CauseEffectEdge]:
        return _StreamingEdges(self)

    @property
    def node_metadata_schema(self) -> Mapping[str, object]:
        return CAUSE_EFFECT_GRAPH_NODE_METADATA_SCHEMA

    @property
    def edge_metadata_schema(self) -> Mapping[str, object]:
        return CAUSE_EFFECT_GRAPH_EDGE_METADATA_SCHEMA

    def get_node(self, node_id: str) -> CauseEffectNode:
        return self._nodes_by_id[node_id]

    def edges_from(self, node: CauseEffectNode) -> Iterable[CauseEffectEdge]:
        self._require_node(node)
        return (edge for edge in self.edges if edge.source.id == node.id)

    def edges_to(self, node: CauseEffectNode) -> Iterable[CauseEffectEdge]:
        self._require_node(node)
        return (edge for edge in self.edges if edge.target.id == node.id)

    def node_metadata(self, node: CauseEffectNode) -> Mapping[str, object]:
        self._require_node(node)
        return {}

    def edge_metadata(self, edge: CauseEffectEdge) -> Mapping[str, object]:
        if edge.graph is not self:
            raise ValueError("edge belongs to a different graph")
        return {"count": edge.count, "necessity": edge.necessity, "sufficiency": edge.sufficiency}

    def _get_or_create_node(self, token: str) -> CauseEffectNode:
        node = self._nodes_by_id.get(token)
        if node is None:
            node = CauseEffectNode(self, token)
            self._nodes_by_id[token] = node
        return node

    def _require_node(self, node: CauseEffectNode) -> None:
        if node.graph is not self:
            raise ValueError("node belongs to a different graph")


def load_cause_effect_graph(source: Source, *, limit: int | None = None) -> CauseEffectGraph:
    """Load a Lexical Cause-Effect Graph (CEG) file from ``source``.

    ``source`` must be a local path to the tab-separated CEG text file (see module docstring for the exact line
    format).

    Unlike ``load_causenet``, omitting ``limit`` is safe even for the complete, tens-of-millions-of-edges release:
    node vocabulary is materialized eagerly (cheap -- a real release has on the order of 10^4-10^5 unique tokens),
    but edges are streamed from disk rather than materialized as Python objects, so
    ``save_cgf(load_cause_effect_graph(path), out_path)`` converts a file far larger than available RAM.

    ``limit``, when given, caps the number of EDGE LINES read from the top of the file -- applied consistently to
    both the node vocabulary built here and every later ``.edges`` iteration -- for quick examples/tests, exactly
    like ``load_causenet``'s own ``limit``.
    """

    return CauseEffectGraph(source, limit=limit)
