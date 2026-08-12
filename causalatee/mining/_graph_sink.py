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

The final aggregated graph is stored via ``causalatee.graph.SQLGraph`` -- ``GraphSink`` only owns the raw-mention
spooling and the ``GROUP BY`` aggregation SQL; once aggregation runs, each resulting ``(cause, effect)`` group is
handed to ``SQLGraph.add_edge`` exactly once. This is a deliberate split: ``SQLGraph`` is a generic, reusable,
mutable graph primitive with no knowledge of mining/mentions at all, while ``GraphSink`` is the mining-specific part
(batching, spooling, aggregating) that produces one.

``GraphSink`` IS both the reduce sink (a plain callable) and the resulting ``causalatee.graph.Graph`` (by inheriting
``SQLGraph``) -- so ``await Pipeline(...).reduce(graph_sink())`` returns something directly usable with
``causalatee.graph.save_cgf``, no extra conversion step. It owns a temp directory it created itself (unlike
``CauseEffectGraph``, which only ever re-opens a caller-owned static file) -- always close it (or use it as a context
manager) once done, mirroring ``CGFGraph``'s ``close()``/``__enter__``/``__exit__`` pattern in
``causalatee.graph._cgf.py``.
"""

from __future__ import annotations

import sqlite3
import tempfile
from collections.abc import Collection, Iterable, Mapping
from pathlib import Path

from causalatee.graph import SQLGraph, SQLGraphEdge, SQLGraphNode
from causalatee.models import ExtractedRelation

# Public names for causalatee.mining's own vocabulary -- the underlying types are exactly SQLGraph's,
# since GraphSink's nodes/edges are generic (metadata-dict-based, no mining-specific fields baked into the
# class itself); see the module docstring for why the split lives here rather than in a domain subclass.
MinedNode = SQLGraphNode
MinedEdge = SQLGraphEdge

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


class GraphSink(SQLGraph):
    """Reduce sink + resulting ``Graph``, see module docstring."""

    def __init__(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory(prefix="causalatee-mining-")
        db_path = Path(self._tempdir.name) / "mentions.sqlite3"
        connection = sqlite3.connect(db_path)
        connection.execute(
            "CREATE TABLE mentions (cause TEXT NOT NULL, effect TEXT NOT NULL, "
            "relation TEXT NOT NULL, score REAL NOT NULL)"
        )
        connection.commit()
        super().__init__(connection)
        self._aggregated = False

    def __call__(self, relations: Iterable[ExtractedRelation]) -> None:
        """The ``Pipeline.reduce`` sink call: append every relation in this batch (e.g. one text's identified
        relations) to the spool, unaggregated."""

        rows = [(_normalize(rel["e1"]), _normalize(rel["e2"]), rel["relation"], rel["score"]) for rel in relations]
        if rows:
            self._connection.executemany(
                "INSERT INTO mentions (cause, effect, relation, score) VALUES (?, ?, ?, ?)", rows
            )
            self._connection.commit()

    def _ensure_aggregated(self) -> None:
        if self._aggregated:
            return
        # Set before doing the work, not after: add_edge() below calls self.get_node(), which resolves to
        # THIS class's override (Python method resolution doesn't know it's being called from inside base-class
        # code) -- so it would recurse right back into _ensure_aggregated() without this guard set first.
        self._aggregated = True
        cursor = self._connection.execute(
            """
            SELECT cause, effect,
                   COUNT(*) AS support,
                   SUM(CASE WHEN relation = 'Causal' THEN 1 ELSE 0 END) AS causal_count,
                   SUM(CASE WHEN relation = 'Countercausal' THEN 1 ELSE 0 END) AS countercausal_count,
                   AVG(score) AS avg_score
            FROM mentions
            GROUP BY cause, effect
            """
        ).fetchall()
        for cause, effect, support, causal_count, countercausal_count, avg_score in cursor:
            self.add_edge(
                cause,
                effect,
                metadata={
                    "support": support,
                    "causal_count": causal_count,
                    "countercausal_count": countercausal_count,
                    "avg_score": avg_score,
                },
            )

    @property
    def nodes(self) -> Collection[MinedNode]:
        self._ensure_aggregated()
        return super().nodes

    @property
    def edges(self) -> Collection[MinedEdge]:
        self._ensure_aggregated()
        return super().edges

    def get_node(self, node_id: str) -> MinedNode:
        self._ensure_aggregated()
        return super().get_node(node_id)

    def edges_from(self, node: MinedNode) -> Iterable[MinedEdge]:
        self._ensure_aggregated()
        return super().edges_from(node)

    def edges_to(self, node: MinedNode) -> Iterable[MinedEdge]:
        self._ensure_aggregated()
        return super().edges_to(node)

    @property
    def node_metadata_schema(self) -> Mapping[str, object]:
        return MINED_NODE_METADATA_SCHEMA

    @property
    def edge_metadata_schema(self) -> Mapping[str, object]:
        return MINED_EDGE_METADATA_SCHEMA

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
