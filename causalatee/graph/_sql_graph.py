"""A generic, mutable, SQLite-backed ``Graph`` -- ``add_node()``/``add_edge()`` as the incremental write
API, the standard ``Graph`` interface (``nodes``/``edges``/``get_node``/``edges_from``/``edges_to``/
``*_metadata``) as the read API.

Unlike this package's other implementations (``CauseNet``, ``CGFGraph``, ``CauseEffectGraph``), which are
read-only views over an existing external resource, ``SQLGraph`` has no source to load from -- it's a
graph you build yourself, one ``add_node()``/``add_edge()`` call at a time (e.g.
``causalatee.mining.GraphSink`` uses one as its storage engine, populating it once its own raw-mention
aggregation is done).

Streams from the backing SQLite connection rather than materializing anything in Python between calls --
same "streams from durable storage" contract ``causalatee.graph.CauseEffectGraph``'s streaming edges
collection already establishes. Node/edge metadata is stored generically as a JSON blob per row: this
class has no domain knowledge of what fields any particular caller cares about (unlike, e.g.,
``CauseNetEdge``'s typed ``support``/``sources`` properties) -- callers needing typed access should read
``.metadata`` or wrap ``SQLGraphNode``/``SQLGraphEdge`` in their own thin subclass.

Connection lifecycle is the caller's responsibility (open before constructing, close after -- this class
never opens or closes it itself), since a caller might reasonably share the connection with other tables
it manages independently (e.g. ``GraphSink``'s separate "mentions" spool table).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Collection, Iterable, Iterator, Mapping

from ._graph import Edge, Graph, Node


class SQLGraphNode(Node["SQLGraphNode", "SQLGraphEdge"]):
    """A node in a ``SQLGraph``. Metadata is fetched once at construction time, not re-queried on access."""

    def __init__(self, graph: SQLGraph, node_id: str, metadata: Mapping[str, object]) -> None:
        super().__init__(graph)
        self._id = node_id
        self._metadata = metadata

    @property
    def id(self) -> str:
        return self._id

    def __repr__(self) -> str:
        return f"SQLGraphNode({self._id!r})"


class SQLGraphEdge(Edge[SQLGraphNode, "SQLGraphEdge"]):
    """A directed edge in a ``SQLGraph``. Metadata is fetched once at construction time, not re-queried on
    access."""

    def __init__(
        self, graph: SQLGraph, source: SQLGraphNode, target: SQLGraphNode, metadata: Mapping[str, object]
    ) -> None:
        super().__init__(graph)
        self._source = source
        self._target = target
        self._metadata = metadata

    @property
    def source(self) -> SQLGraphNode:
        return self._source

    @property
    def target(self) -> SQLGraphNode:
        return self._target

    def __repr__(self) -> str:
        return f"SQLGraphEdge({self._source.id!r} -> {self._target.id!r})"


class _StreamingSQLNodes(Collection[SQLGraphNode]):
    """Re-iterable view over the ``nodes`` table -- re-queries on every ``__iter__`` call rather than
    caching."""

    def __init__(self, graph: SQLGraph) -> None:
        self._graph = graph

    def __iter__(self) -> Iterator[SQLGraphNode]:
        cursor = self._graph._connection.execute("SELECT id, metadata FROM nodes")
        for node_id, metadata_json in cursor:
            yield SQLGraphNode(self._graph, node_id, json.loads(metadata_json))

    def __len__(self) -> int:
        (count,) = self._graph._connection.execute("SELECT COUNT(*) FROM nodes").fetchone()
        return count

    def __contains__(self, value: object) -> bool:
        return any(value == node for node in self)


class _StreamingSQLEdges(Collection[SQLGraphEdge]):
    """Re-iterable view over the ``edges`` table, optionally filtered by source or target id -- re-queries
    on every ``__iter__`` call rather than caching, same contract as ``_StreamingSQLNodes``."""

    def __init__(self, graph: SQLGraph, *, source_id: str | None = None, target_id: str | None = None) -> None:
        self._graph = graph
        self._source_id = source_id
        self._target_id = target_id

    def _where(self) -> tuple[str, tuple[str, ...]]:
        if self._source_id is not None:
            return " WHERE source = ?", (self._source_id,)
        if self._target_id is not None:
            return " WHERE target = ?", (self._target_id,)
        return "", ()

    def __iter__(self) -> Iterator[SQLGraphEdge]:
        clause, params = self._where()
        cursor = self._graph._connection.execute(f"SELECT source, target, metadata FROM edges{clause}", params)
        for source_id, target_id, metadata_json in cursor:
            yield SQLGraphEdge(
                self._graph,
                self._graph.get_node(source_id),
                self._graph.get_node(target_id),
                json.loads(metadata_json),
            )

    def __len__(self) -> int:
        clause, params = self._where()
        (count,) = self._graph._connection.execute(f"SELECT COUNT(*) FROM edges{clause}", params).fetchone()
        return count

    def __contains__(self, value: object) -> bool:
        return any(value == edge for edge in self)


class SQLGraph(Graph[SQLGraphNode, SQLGraphEdge]):
    """A mutable graph backed by a caller-supplied SQLite connection. See module docstring."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self._connection.execute("CREATE TABLE IF NOT EXISTS nodes (id TEXT PRIMARY KEY, metadata TEXT NOT NULL)")
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS edges (source TEXT NOT NULL, target TEXT NOT NULL, metadata TEXT NOT NULL)"
        )
        self._connection.execute("CREATE INDEX IF NOT EXISTS sqlgraph_edges_source_idx ON edges (source)")
        self._connection.execute("CREATE INDEX IF NOT EXISTS sqlgraph_edges_target_idx ON edges (target)")
        self._connection.commit()

    def add_node(self, node_id: str, metadata: Mapping[str, object] | None = None) -> SQLGraphNode:
        """Add (or replace) a node. Safe to call again for an existing id -- overwrites its metadata."""

        payload = dict(metadata or {})
        self._connection.execute(
            "INSERT OR REPLACE INTO nodes (id, metadata) VALUES (?, ?)", (node_id, json.dumps(payload))
        )
        self._connection.commit()
        return SQLGraphNode(self, node_id, payload)

    def add_edge(self, source_id: str, target_id: str, metadata: Mapping[str, object] | None = None) -> SQLGraphEdge:
        """Add a directed edge, auto-creating either endpoint (with empty metadata) if it doesn't exist yet
        -- same convention networkx's ``add_edge`` uses. Does NOT deduplicate or merge with an existing
        edge between the same pair (a multigraph, in effect); callers that need one aggregated edge per
        pair must aggregate before calling this once per pair (see ``causalatee.mining.GraphSink``)."""

        for node_id in (source_id, target_id):
            self._connection.execute("INSERT OR IGNORE INTO nodes (id, metadata) VALUES (?, '{}')", (node_id,))
        payload = dict(metadata or {})
        self._connection.execute(
            "INSERT INTO edges (source, target, metadata) VALUES (?, ?, ?)",
            (source_id, target_id, json.dumps(payload)),
        )
        self._connection.commit()
        return SQLGraphEdge(self, self.get_node(source_id), self.get_node(target_id), payload)

    @property
    def nodes(self) -> Collection[SQLGraphNode]:
        return _StreamingSQLNodes(self)

    @property
    def edges(self) -> Collection[SQLGraphEdge]:
        return _StreamingSQLEdges(self)

    def get_node(self, node_id: str) -> SQLGraphNode:
        row = self._connection.execute("SELECT metadata FROM nodes WHERE id = ?", (node_id,)).fetchone()
        if row is None:
            raise KeyError(node_id)
        (metadata_json,) = row
        return SQLGraphNode(self, node_id, json.loads(metadata_json))

    def edges_from(self, node: SQLGraphNode) -> Iterable[SQLGraphEdge]:
        return _StreamingSQLEdges(self, source_id=node.id)

    def edges_to(self, node: SQLGraphNode) -> Iterable[SQLGraphEdge]:
        return _StreamingSQLEdges(self, target_id=node.id)

    def node_metadata(self, node: SQLGraphNode) -> Mapping[str, object]:
        return node._metadata

    def edge_metadata(self, edge: SQLGraphEdge) -> Mapping[str, object]:
        return edge._metadata
