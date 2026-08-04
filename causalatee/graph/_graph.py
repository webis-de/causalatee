"""Typed abstract interfaces for directed graph resources.

The recursive type parameters preserve concrete node and edge types.  For
example, ``CauseNetNode.outgoing_edges()`` is statically inferred as
``Iterable[CauseNetEdge]`` rather than merely ``Iterable[Edge]``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable, Mapping
from typing import Any, Generic, TypeVar

N = TypeVar("N", bound="Node[Any, Any]")
E = TypeVar("E", bound="Edge[Any, Any]")


class Node(Generic[N, E], ABC):
    """A lightweight node associated with a graph backend."""

    def __init__(self, graph: Graph[N, E]) -> None:
        self._graph = graph

    @property
    def graph(self) -> Graph[N, E]:
        """Return the graph that owns this node."""

        return self._graph

    @property
    @abstractmethod
    def id(self) -> str:
        """Return the node identifier, unique within its graph."""

        raise NotImplementedError

    def outgoing_edges(self: N) -> Iterable[E]:
        """Iterate over edges whose source is this node."""

        return self._graph.edges_from(self)

    def incoming_edges(self: N) -> Iterable[E]:
        """Iterate over edges whose target is this node."""

        return self._graph.edges_to(self)

    @property
    def metadata(self: N) -> Mapping[str, object]:
        """Return backend-provided metadata for this node."""

        return self._graph.node_metadata(self)


class Edge(Generic[N, E], ABC):
    """A directed edge associated with a graph backend."""

    def __init__(self, graph: Graph[N, E]) -> None:
        self._graph = graph

    @property
    def graph(self) -> Graph[N, E]:
        """Return the graph that owns this edge."""

        return self._graph

    @property
    @abstractmethod
    def source(self) -> N:
        """Return the source node."""

        raise NotImplementedError

    @property
    @abstractmethod
    def target(self) -> N:
        """Return the target node."""

        raise NotImplementedError

    @property
    def metadata(self: E) -> Mapping[str, object]:
        """Return backend-provided metadata for this edge."""

        return self._graph.edge_metadata(self)


class Graph(Generic[N, E], ABC):
    """Abstract interface for a directed graph with typed nodes and edges."""

    @property
    @abstractmethod
    def nodes(self) -> Collection[N]:
        """Return a collection view of the graph's nodes."""

        raise NotImplementedError

    @property
    @abstractmethod
    def edges(self) -> Collection[E]:
        """Return a collection view of the graph's edges."""

        raise NotImplementedError

    @abstractmethod
    def get_node(self, node_id: str) -> N:
        """Return the node identified by ``node_id``.

        Implementations should raise ``KeyError`` when no node exists.
        """

        raise NotImplementedError

    @abstractmethod
    def edges_from(self, node: N) -> Iterable[E]:
        """Iterate over edges whose source is ``node``."""

        raise NotImplementedError

    @abstractmethod
    def edges_to(self, node: N) -> Iterable[E]:
        """Iterate over edges whose target is ``node``."""

        raise NotImplementedError

    def node_metadata(self, node: N) -> Mapping[str, object]:
        """Return resource-specific node metadata."""

        return {}

    def edge_metadata(self, edge: E) -> Mapping[str, object]:
        """Return resource-specific edge metadata."""

        return {}

    @property
    def node_metadata_schema(self) -> Mapping[str, object] | str | None:
        """Return the Avro writer schema for node metadata, if defined."""

        return None

    @property
    def edge_metadata_schema(self) -> Mapping[str, object] | str | None:
        """Return the Avro writer schema for edge metadata, if defined."""

        return None
