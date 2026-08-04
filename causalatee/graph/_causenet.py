"""An eager, typed adapter for the CauseNet precision graph.

The loader accepts a local JSONL file or an HTTP(S) URL.  Gzip and bzip2
compression are detected from magic bytes.  This implementation materializes
the selected records as Python objects; use ``limit`` for examples and tests,
because loading the complete CauseNet resource may require substantial memory.
"""

from __future__ import annotations

import bz2
import gzip
import io
import json
import os
import urllib.request
from collections import defaultdict
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, TextIO, cast

if TYPE_CHECKING:
    from ._graph import Edge, Graph, Node
else:
    try:  # Package use.
        from ._graph import Edge, Graph, Node
    except ImportError:  # Direct use with the reference files in one directory.
        from _graph import Edge, Graph, Node


CAUSENET_NODE_METADATA_SCHEMA: Mapping[str, object] = {
    "type": "record",
    "name": "CauseNetNodeMetadata",
    "namespace": "causalatee.graph.causenet",
    "fields": [],
}

CAUSENET_EDGE_METADATA_SCHEMA: Mapping[str, object] = {
    "type": "record",
    "name": "CauseNetEdgeMetadata",
    "namespace": "causalatee.graph.causenet",
    "fields": [
        {
            "name": "sources",
            "type": {
                "type": "array",
                "items": {
                    "type": "record",
                    "name": "CauseNetSource",
                    "fields": [
                        {"name": "type", "type": "string"},
                        {
                            "name": "payload",
                            "type": {"type": "map", "values": "string"},
                        },
                    ],
                },
            },
        },
        {
            "name": "support",
            "type": "long",
        },
    ],
}

Source = str | os.PathLike[str]


@contextmanager
def open_jsonl_text(
    source: Source,
    *,
    encoding: str = "utf-8",
    timeout: float = 60.0,
) -> Iterator[TextIO]:
    """Open a local or remote, optionally compressed JSONL text stream."""

    source_string = os.fspath(source)

    with ExitStack() as stack:
        raw: BinaryIO
        if source_string.startswith(("http://", "https://")):
            # The scheme is checked above, so this never reaches file:/ or other
            # unexpected urllib schemes.
            raw = stack.enter_context(
                urllib.request.urlopen(source_string, timeout=timeout)  # nosec B310
            )
        else:
            raw = stack.enter_context(Path(source_string).open("rb"))

        buffered = io.BufferedReader(cast(io.RawIOBase, raw))
        signature = buffered.peek(3)[:3]

        binary_stream: BinaryIO
        if signature.startswith(b"\x1f\x8b"):
            binary_stream = cast(BinaryIO, stack.enter_context(gzip.GzipFile(fileobj=buffered, mode="rb")))
        elif signature.startswith(b"BZh"):
            binary_stream = cast(BinaryIO, stack.enter_context(bz2.BZ2File(buffered, mode="rb")))
        else:
            binary_stream = buffered

        text_stream = stack.enter_context(io.TextIOWrapper(binary_stream, encoding=encoding))
        yield text_stream


def iter_jsonl(
    source: Source,
    *,
    encoding: str = "utf-8",
    timeout: float = 60.0,
    skip_blank_lines: bool = True,
) -> Iterator[Any]:
    """Lazily decode JSON values from a local or remote JSONL resource."""

    with open_jsonl_text(source, encoding=encoding, timeout=timeout) as stream:
        for line_number, line in enumerate(stream, start=1):
            if skip_blank_lines and not line.strip():
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {source_string(source)!r}") from exc


def source_string(source: Source) -> str:
    """Return a printable representation of a local path or URL."""

    return os.fspath(source)


class CauseNetNode(Node["CauseNetNode", "CauseNetEdge"]):
    """A CauseNet concept node."""

    def __init__(self, graph: CauseNet, concept: str) -> None:
        super().__init__(graph)
        self._concept = concept

    @property
    def id(self) -> str:
        return self._concept

    @property
    def concept(self) -> str:
        """Return the CauseNet concept represented by this node."""

        return self._concept

    def __repr__(self) -> str:
        return f"CauseNetNode({self._concept!r})"


class CauseNetEdge(Edge[CauseNetNode, "CauseNetEdge"]):
    """A directed CauseNet relation from a cause to an effect."""

    def __init__(
        self,
        graph: CauseNet,
        cause: CauseNetNode,
        effect: CauseNetNode,
        *,
        sources: Sequence[Mapping[str, object]],
        support: int,
    ) -> None:
        super().__init__(graph)
        self._cause = cause
        self._effect = effect
        self._sources = tuple(sources)
        self._support = support

    @property
    def source(self) -> CauseNetNode:
        return self._cause

    @property
    def target(self) -> CauseNetNode:
        return self._effect

    @property
    def cause(self) -> CauseNetNode:
        """Return the cause node."""

        return self._cause

    @property
    def effect(self) -> CauseNetNode:
        """Return the effect node."""

        return self._effect

    @property
    def sources(self) -> tuple[Mapping[str, object], ...]:
        """Return the source records supporting this relation."""

        return self._sources

    @property
    def support(self) -> int:
        """Return the number of source records supporting this relation."""

        return self._support

    def __repr__(self) -> str:
        return f"CauseNetEdge({self._cause.id!r} -> {self._effect.id!r})"


class CauseNet(Graph[CauseNetNode, CauseNetEdge]):
    """An in-memory view of CauseNet with typed adjacency queries."""

    def __init__(
        self,
        source: Source,
        *,
        limit: int | None = None,
        timeout: float = 60.0,
    ) -> None:
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative or None")

        self._source = source_string(source)
        self._nodes_by_id: dict[str, CauseNetNode] = {}
        self._edges: list[CauseNetEdge] = []
        self._outgoing: dict[CauseNetNode, list[CauseNetEdge]] = defaultdict(list)
        self._incoming: dict[CauseNetNode, list[CauseNetEdge]] = defaultdict(list)

        for index, value in enumerate(iter_jsonl(source, timeout=timeout)):
            if limit is not None and index >= limit:
                break
            self._add_entry(value, line_number=index + 1)

    @property
    def source(self) -> str:
        """Return the path or URL from which this graph was loaded."""

        return self._source

    @property
    def nodes(self) -> Collection[CauseNetNode]:
        return self._nodes_by_id.values()

    @property
    def edges(self) -> Collection[CauseNetEdge]:
        return self._edges

    @property
    def node_metadata_schema(self) -> Mapping[str, object]:
        return CAUSENET_NODE_METADATA_SCHEMA

    @property
    def edge_metadata_schema(self) -> Mapping[str, object]:
        return CAUSENET_EDGE_METADATA_SCHEMA

    def get_node(self, node_id: str) -> CauseNetNode:
        return self._nodes_by_id[node_id]

    def edges_from(self, node: CauseNetNode) -> Iterable[CauseNetEdge]:
        self._require_node(node)
        return iter(self._outgoing.get(node, ()))

    def edges_to(self, node: CauseNetNode) -> Iterable[CauseNetEdge]:
        self._require_node(node)
        return iter(self._incoming.get(node, ()))

    def node_metadata(self, node: CauseNetNode) -> Mapping[str, object]:
        self._require_node(node)
        return {}

    def edge_metadata(self, edge: CauseNetEdge) -> Mapping[str, object]:
        if edge.graph is not self:
            raise ValueError("edge belongs to a different graph")
        return {"sources": edge.sources, "support": edge.support}

    def effects_of(self, concept: str) -> Iterable[CauseNetNode]:
        """Iterate over concepts directly caused by ``concept``."""

        for edge in self.get_node(concept).outgoing_edges():
            yield edge.effect

    def causes_of(self, concept: str) -> Iterable[CauseNetNode]:
        """Iterate over concepts that directly cause ``concept``."""

        for edge in self.get_node(concept).incoming_edges():
            yield edge.cause

    def _get_or_create_node(self, concept: str) -> CauseNetNode:
        node = self._nodes_by_id.get(concept)
        if node is None:
            node = CauseNetNode(self, concept)
            self._nodes_by_id[concept] = node
        return node

    def _require_node(self, node: CauseNetNode) -> None:
        if node.graph is not self:
            raise ValueError("node belongs to a different graph")

    def _add_entry(self, value: object, *, line_number: int) -> None:
        entry = _require_mapping(value, line_number=line_number, field="entry")
        relation = _require_mapping(
            entry.get("causal_relation"),
            line_number=line_number,
            field="causal_relation",
        )
        cause_data = _require_mapping(relation.get("cause"), line_number=line_number, field="cause")
        effect_data = _require_mapping(relation.get("effect"), line_number=line_number, field="effect")
        cause_concept = _require_string(cause_data.get("concept"), line_number=line_number, field="cause.concept")
        effect_concept = _require_string(effect_data.get("concept"), line_number=line_number, field="effect.concept")

        source_values = entry.get("sources", ())
        if not isinstance(source_values, list):
            raise ValueError(f"line {line_number}: sources must be a list")
        sources = tuple(_require_source(item, line_number=line_number) for item in source_values)
        support = entry.get("support")
        if isinstance(support, bool) or not isinstance(support, int):
            raise ValueError(f"line {line_number}: support must be an integer")

        cause = self._get_or_create_node(cause_concept)
        effect = self._get_or_create_node(effect_concept)
        edge = CauseNetEdge(
            self,
            cause,
            effect,
            sources=sources,
            support=support,
        )
        self._edges.append(edge)
        self._outgoing[cause].append(edge)
        self._incoming[effect].append(edge)


def _require_mapping(
    value: object,
    *,
    line_number: int,
    field: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"line {line_number}: {field} must be an object")
    return value


def _require_string(value: object, *, line_number: int, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"line {line_number}: {field} must be a string")
    return value


def _require_source(value: object, *, line_number: int) -> Mapping[str, object]:
    source = _require_mapping(value, line_number=line_number, field="sources[]")
    source_type = _require_string(source.get("type"), line_number=line_number, field="sources[].type")
    payload = _require_mapping(source.get("payload"), line_number=line_number, field="sources[].payload")
    if not all(isinstance(key, str) and isinstance(item, str) for key, item in payload.items()):
        raise ValueError(f"line {line_number}: sources[].payload must map strings to strings")
    return {"type": source_type, "payload": dict(payload)}


def load_causenet(
    source: Source,
    *,
    limit: int | None = None,
    timeout: float = 60.0,
) -> CauseNet:
    """Load CauseNet from ``source``.

    Omit ``limit`` to load the complete resource.  This eager implementation
    may use substantial memory for the full graph.
    """

    return CauseNet(source, limit=limit, timeout=timeout)
