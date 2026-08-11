"""A ``Pipeline.reduce`` sink that aggregates mined relations into a
``causalatee.graph.Graph``.

Two-phase by design, deliberately NOT conflated into one step: (1) every ``list[ExtractedRelation]`` batch the sink
receives while the pipeline drains is appended, unaggregated, to a temp SQLite spool -- bounded memory regardless of
corpus size, no aggregation logic here at all (mirrors the sibling ``causalgraph`` project's own ``causal_relations``
table, and the disk-backed philosophy ``causalatee.graph._cgf.py``'s own writer already uses rather than assuming things
fit in RAM). (2) An ordinary, synchronous ``GROUP BY cause, effect`` aggregation runs ONCE, lazily, the first time
``.nodes``/``.edges`` is actually accessed -- which is always after ``Pipeline.reduce`` has finished draining into the
sink, so no explicit "finalize" call is needed. Countercausal and causal mentions of the SAME (cause, effect) pair
aggregate into ONE edge with both counts as metadata (matching ``causalgraph``'s own
``leaf_edges.relation_count``/``countercausal_count`` split), not two parallel edges.

``GraphSink`` IS both the reduce sink (a plain callable) and the resulting ``causalatee.graph.Graph`` -- so ``await
Pipeline(...).reduce(graph_sink())`` returns something directly usable with ``causalatee.graph.save_cgf``, no extra
conversion step. It owns a temp directory it created itself (unlike ``CauseEffectGraph``, which only ever re-opens a
caller-owned static file) -- always close it (or use it as a context manager) once done, mirroring ``CGFGraph``'s
``close()``/``__enter__``/``__exit__`` pattern in ``causalatee.graph._cgf.py``.
"""

from __future__ import annotations

import sqlite3
import tempfile
from collections.abc import Collection, Iterable, Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from causalatee.models import ExtractedRelation

if TYPE_CHECKING:
    from causalatee.graph import Edge, Graph, Node
else:
    from causalatee.graph import Edge, Graph, Node


MINED_NODE_METADATA_SCHEMA: Mapping[str, object] = {
    "type": "record",
    "name": "MinedNodeMetadata",
    "namespace": "causalatee.mining",
    "fields": [],
}

MINED_EDGE_METADATA_SCHEMA: Mapping[str, object] = {
    "type": "record",
    "name": "MinedEdgeMetadata",
    "namespace": "causalatee.mining",
    "fields": [
        {"name": "support", "type": "long"},
        {"name": "causal_count", "type": "long"},
        {"name": "countercausal_count", "type": "long"},
        {"name": "avg_score", "type": "double"},
    ],
}


def _normalize(text: str) -> str:
    return text.strip().lower()


class MinedNode(Node["MinedNode", "MinedEdge"]):
    """A single normalized cause/effect phrase mined from the corpus."""

    def __init__(self, graph: GraphSink, phrase: str) -> None:
        super().__init__(graph)
        self._phrase = phrase

    @property
    def id(self) -> str:
        return self._phrase

    def __repr__(self) -> str:
        return f"MinedNode({self._phrase!r})"


class MinedEdge(Edge[MinedNode, "MinedEdge"]):
    """A directed cause->effect edge aggregated across every mention in the corpus."""

    def __init__(
        self,
        graph: GraphSink,
        cause: MinedNode,
        effect: MinedNode,
        *,
        support: int,
        causal_count: int,
        countercausal_count: int,
        avg_score: float,
    ) -> None:
        super().__init__(graph)
        self._cause = cause
        self._effect = effect
        self.support = support
        self.causal_count = causal_count
        self.countercausal_count = countercausal_count
        self.avg_score = avg_score

    @property
    def source(self) -> MinedNode:
        return self._cause

    @property
    def target(self) -> MinedNode:
        return self._effect

    def __repr__(self) -> str:
        return f"MinedEdge({self._cause.id!r} -> {self._effect.id!r}, support={self.support})"


class _StreamingMinedEdges(Collection[MinedEdge]):
    """Re-iterable view over the aggregated ``edges`` table -- re-queries on every ``__iter__`` call rather than
    caching, the same "streams from durable storage, nothing materialized in Python between iterations" contract
    ``CauseEffectGraph._StreamingEdges`` already establishes."""

    def __init__(self, sink: GraphSink) -> None:
        self._sink = sink

    def __iter__(self) -> Iterator[MinedEdge]:
        cursor = self._sink._connection.execute(
            "SELECT cause, effect, support, causal_count, countercausal_count, avg_score FROM edges"
        )
        for cause, effect, support, causal_count, countercausal_count, avg_score in cursor:
            yield MinedEdge(
                self._sink,
                self._sink._get_or_create_node(cause),
                self._sink._get_or_create_node(effect),
                support=support,
                causal_count=causal_count,
                countercausal_count=countercausal_count,
                avg_score=avg_score,
            )

    def __len__(self) -> int:
        (count,) = self._sink._connection.execute("SELECT COUNT(*) FROM edges").fetchone()
        return count

    def __contains__(self, value: object) -> bool:
        return any(value == edge for edge in self)


class GraphSink(Graph[MinedNode, MinedEdge]):
    """Reduce sink + resulting ``Graph``, see module docstring."""

    def __init__(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory(prefix="causalatee-mining-")
        db_path = Path(self._tempdir.name) / "mentions.sqlite3"
        self._connection = sqlite3.connect(db_path)
        self._connection.execute(
            "CREATE TABLE mentions (cause TEXT NOT NULL, effect TEXT NOT NULL, "
            "relation TEXT NOT NULL, score REAL NOT NULL)"
        )
        self._connection.commit()
        self._aggregated = False
        self._nodes_by_id: dict[str, MinedNode] = {}

    def __call__(self, relations: Iterable[ExtractedRelation]) -> None:
        """The ``Pipeline.reduce`` sink call: append every relation in this batch (e.g. one text's identified
        relations) to the spool, unaggregated."""

        rows = [
            (_normalize(rel["e1"]), _normalize(rel["e2"]), rel["relation"], rel["score"]) for rel in relations
        ]
        if rows:
            self._connection.executemany(
                "INSERT INTO mentions (cause, effect, relation, score) VALUES (?, ?, ?, ?)", rows
            )
            self._connection.commit()

    def _ensure_aggregated(self) -> None:
        if self._aggregated:
            return
        self._connection.execute(
            """
            CREATE TABLE edges AS
            SELECT cause, effect,
                   COUNT(*) AS support,
                   SUM(CASE WHEN relation = 'Causal' THEN 1 ELSE 0 END) AS causal_count,
                   SUM(CASE WHEN relation = 'Countercausal' THEN 1 ELSE 0 END) AS countercausal_count,
                   AVG(score) AS avg_score
            FROM mentions
            GROUP BY cause, effect
            """
        )
        self._connection.commit()
        for cause, effect in self._connection.execute("SELECT cause, effect FROM edges"):
            self._get_or_create_node(cause)
            self._get_or_create_node(effect)
        self._aggregated = True

    def _get_or_create_node(self, phrase: str) -> MinedNode:
        node = self._nodes_by_id.get(phrase)
        if node is None:
            node = MinedNode(self, phrase)
            self._nodes_by_id[phrase] = node
        return node

    @property
    def nodes(self) -> Collection[MinedNode]:
        self._ensure_aggregated()
        return self._nodes_by_id.values()

    @property
    def edges(self) -> Collection[MinedEdge]:
        self._ensure_aggregated()
        return _StreamingMinedEdges(self)

    @property
    def node_metadata_schema(self) -> Mapping[str, object]:
        return MINED_NODE_METADATA_SCHEMA

    @property
    def edge_metadata_schema(self) -> Mapping[str, object]:
        return MINED_EDGE_METADATA_SCHEMA

    def get_node(self, node_id: str) -> MinedNode:
        self._ensure_aggregated()
        return self._nodes_by_id[node_id]

    def edges_from(self, node: MinedNode) -> Iterable[MinedEdge]:
        return (edge for edge in self.edges if edge.source.id == node.id)

    def edges_to(self, node: MinedNode) -> Iterable[MinedEdge]:
        return (edge for edge in self.edges if edge.target.id == node.id)

    def node_metadata(self, node: MinedNode) -> Mapping[str, object]:
        return {}

    def edge_metadata(self, edge: MinedEdge) -> Mapping[str, object]:
        if edge.graph is not self:
            raise ValueError("edge belongs to a different graph")
        return {
            "support": edge.support,
            "causal_count": edge.causal_count,
            "countercausal_count": edge.countercausal_count,
            "avg_score": edge.avg_score,
        }

    def __enter__(self) -> GraphSink:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the spool database and remove its temp directory."""

        self._connection.close()
        self._tempdir.cleanup()


def graph_sink() -> GraphSink:
    """Create a ``Pipeline.reduce`` sink that aggregates every ``list[ExtractedRelation]`` batch it receives into a
    ``causalatee.graph.Graph``, by ``(cause, effect)`` pair across the whole corpus. See ``GraphSink`` for the
    aggregation/lifecycle details.
    """

    return GraphSink()
