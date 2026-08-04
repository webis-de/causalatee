"""Round-trip tests for the CGF reference implementation."""

from __future__ import annotations

import io
import json
import sys
from collections.abc import Collection, Iterable, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from causalatee.graph import Edge, Graph, Node, load_causenet, load_cgf, save_cgf


class ExampleNode(Node["ExampleNode", "ExampleEdge"]):
    def __init__(self, graph: ExampleGraph, node_id: str, metadata: Mapping[str, object]):
        super().__init__(graph)
        self._id = node_id
        self._metadata = metadata

    @property
    def id(self) -> str:
        return self._id


class ExampleEdge(Edge[ExampleNode, "ExampleEdge"]):
    def __init__(
        self,
        graph: ExampleGraph,
        source: ExampleNode,
        target: ExampleNode,
        metadata: Mapping[str, object],
    ) -> None:
        super().__init__(graph)
        self._source = source
        self._target = target
        self._metadata = metadata

    @property
    def source(self) -> ExampleNode:
        return self._source

    @property
    def target(self) -> ExampleNode:
        return self._target


class ExampleGraph(Graph[ExampleNode, ExampleEdge]):
    NODE_SCHEMA = {
        "type": "record",
        "name": "NodeMetadata",
        "fields": [
            {"name": "kind", "type": "string"},
            {"name": "unicode", "type": ["null", "string"], "default": None},
        ],
    }
    EDGE_SCHEMA = {
        "type": "record",
        "name": "EdgeMetadata",
        "fields": [
            {"name": "support", "type": "long"},
            {"name": "duplicate", "type": ["null", "boolean"], "default": None},
        ],
    }

    def __init__(self) -> None:
        self._nodes = {
            node_id: ExampleNode(self, node_id, metadata)
            for node_id, metadata in [
                ("rain", {"kind": "event"}),
                ("flood", {"kind": "event"}),
                ("damage", {"kind": "effect", "unicode": "âœ“"}),
            ]
        }
        self._edges = [
            ExampleEdge(
                self,
                self._nodes["rain"],
                self._nodes["flood"],
                {"support": 4},
            ),
            ExampleEdge(
                self,
                self._nodes["flood"],
                self._nodes["damage"],
                {"support": 2},
            ),
            ExampleEdge(
                self,
                self._nodes["rain"],
                self._nodes["flood"],
                {"support": 1, "duplicate": True},
            ),
        ]

    @property
    def nodes(self) -> Collection[ExampleNode]:
        return self._nodes.values()

    @property
    def edges(self) -> Collection[ExampleEdge]:
        return self._edges

    def get_node(self, node_id: str) -> ExampleNode:
        return self._nodes[node_id]

    def edges_from(self, node: ExampleNode) -> Iterable[ExampleEdge]:
        return (edge for edge in self._edges if edge.source is node)

    def edges_to(self, node: ExampleNode) -> Iterable[ExampleEdge]:
        return (edge for edge in self._edges if edge.target is node)

    def node_metadata(self, node: ExampleNode) -> Mapping[str, object]:
        return node._metadata

    def edge_metadata(self, edge: ExampleEdge) -> Mapping[str, object]:
        return edge._metadata

    @property
    def node_metadata_schema(self) -> Mapping[str, object]:
        return self.NODE_SCHEMA

    @property
    def edge_metadata_schema(self) -> Mapping[str, object]:
        return self.EDGE_SCHEMA


@contextmanager
def fake_fastavro():
    """Exercise CGF's codec integration when the optional package is absent."""

    def parse_schema(schema):
        return schema

    def schemaless_writer(stream: io.BytesIO, schema, record):
        stream.write(json.dumps(record, sort_keys=True).encode("utf-8"))

    def schemaless_reader(stream: io.BytesIO, schema):
        return json.loads(stream.read())

    previous = sys.modules.get("fastavro")
    sys.modules["fastavro"] = SimpleNamespace(
        parse_schema=parse_schema,
        schemaless_writer=schemaless_writer,
        schemaless_reader=schemaless_reader,
    )
    try:
        yield
    finally:
        if previous is None:
            del sys.modules["fastavro"]
        else:
            sys.modules["fastavro"] = previous


def test_round_trip_with_incoming_index(tmp_path: Path) -> None:
    output = tmp_path / "example.cgf"
    with fake_fastavro():
        save_cgf(ExampleGraph(), output, include_incoming=True)
        graph = load_cgf(output, validate=True)
    with fake_fastavro(), graph:
        assert [node.id for node in graph.nodes] == ["damage", "flood", "rain"]
        assert graph.get_node("damage").metadata == {
            "kind": "effect",
            "unicode": "âœ“",
        }

        outgoing = list(graph.get_node("rain").outgoing_edges())
        assert [edge.target.id for edge in outgoing] == ["flood", "flood"]
        assert [edge.metadata["support"] for edge in outgoing] == [4, 1]

        incoming = list(graph.get_node("flood").incoming_edges())
        assert [edge.source.id for edge in incoming] == ["rain", "rain"]
        assert graph.has_incoming_index


def test_round_trip_without_incoming_index(tmp_path: Path) -> None:
    output = tmp_path / "example-no-incoming.cgf"
    with fake_fastavro():
        save_cgf(ExampleGraph(), output, include_incoming=False)
        graph = load_cgf(output, validate=True)
    with fake_fastavro(), graph:
        assert not graph.has_incoming_index
        incoming = list(graph.get_node("damage").incoming_edges())
        assert [(edge.source.id, edge.target.id) for edge in incoming] == [("flood", "damage")]


def test_exact_lookup_failure(tmp_path: Path) -> None:
    output = tmp_path / "example.cgf"
    with fake_fastavro():
        save_cgf(ExampleGraph(), output)
        graph = load_cgf(output)
    with fake_fastavro(), graph:
        try:
            graph.get_node("missing")
        except KeyError:
            pass
        else:
            raise AssertionError("missing node lookup did not raise KeyError")


def test_graph_supplied_schemas_are_embedded(tmp_path: Path) -> None:
    output = tmp_path / "schemas.cgf"
    with fake_fastavro():
        save_cgf(ExampleGraph(), output)
        with load_cgf(output, validate=True) as graph:
            assert graph.node_metadata_schema["name"] == "NodeMetadata"
            assert graph.edge_metadata_schema["name"] == "EdgeMetadata"
            assert graph.get_node("rain").metadata == {"kind": "event"}


def test_avro_metadata(tmp_path: Path) -> None:
    graph = ExampleGraph()
    for node in graph._nodes.values():
        node._metadata["unicode"] = node._metadata.get("unicode")
    for edge in graph._edges:
        edge._metadata["duplicate"] = edge._metadata.get("duplicate")
        edge._metadata["note"] = f"{edge.source.id}->{edge.target.id}"

    node_schema = json.dumps(
        {
            "type": "record",
            "name": "NodeMetadata",
            "fields": [
                {"name": "kind", "type": "string"},
                {"name": "unicode", "type": ["null", "string"]},
            ],
        }
    )
    edge_schema = {
        "type": "record",
        "name": "EdgeMetadata",
        "fields": [
            {"name": "support", "type": "long"},
            {"name": "duplicate", "type": ["null", "boolean"], "default": None},
            {"name": "note", "type": "string"},
        ],
    }
    output = tmp_path / "avro.cgf"
    with fake_fastavro():
        save_cgf(
            graph,
            output,
            node_metadata_schema=node_schema,
            edge_metadata_schema=edge_schema,
        )
        with load_cgf(output, validate=True) as mapped:
            assert mapped.node_metadata_schema["name"] == "NodeMetadata"
            assert mapped.edge_metadata_schema["name"] == "EdgeMetadata"
            assert mapped.get_node("damage").metadata == {
                "kind": "effect",
                "unicode": "âœ“",
            }
            outgoing = list(mapped.get_node("rain").outgoing_edges())
            assert outgoing[0].metadata["support"] == 4
            assert outgoing[0].metadata["note"] == "rain->flood"


def test_logical_json_escape_hatch(tmp_path: Path) -> None:
    graph = ExampleGraph()
    graph._edges[0]._metadata["provenance"] = [{"type": "sentence", "rank": 2}]
    graph._edges[1]._metadata["provenance"] = []
    graph._edges[2]._metadata["provenance"] = [{"nested": [True, None]}]
    edge_schema = {
        "type": "record",
        "name": "FlexibleEdgeMetadata",
        "fields": [
            {"name": "support", "type": "long"},
            {"name": "duplicate", "type": ["null", "boolean"], "default": None},
            {
                "name": "provenance",
                "type": {"type": "string", "logicalType": "causalatee.json"},
            },
        ],
    }
    output = tmp_path / "logical-json.cgf"
    with fake_fastavro():
        save_cgf(graph, output, edge_metadata_schema=edge_schema)
        with load_cgf(output, validate=True) as mapped:
            metadata = next(iter(mapped.get_node("rain").outgoing_edges())).metadata
            assert metadata["provenance"] == [{"type": "sentence", "rank": 2}]


def test_causenet_adapter_supplies_schemas(tmp_path: Path) -> None:
    source = tmp_path / "causenet.jsonl"
    source.write_text(
        json.dumps(
            {
                "causal_relation": {
                    "cause": {"concept": "rain"},
                    "effect": {"concept": "flood"},
                },
                "sources": [{"type": "wikipedia_sentence", "payload": {"sentence": "x"}}],
                "support": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "causenet.cgf"
    with fake_fastavro():
        save_cgf(load_causenet(source), output)
        with load_cgf(output, validate=True) as mapped:
            edge = next(iter(mapped.get_node("rain").outgoing_edges()))
            assert edge.metadata["sources"][0]["type"] == "wikipedia_sentence"
            assert edge.metadata["support"] == 1


def test_streamed_writer_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.cgf"
    second = tmp_path / "second.cgf"
    with fake_fastavro():
        save_cgf(ExampleGraph(), first)
        save_cgf(ExampleGraph(), second)
    assert first.read_bytes() == second.read_bytes()


def test_streamed_writer_handles_empty_graph(tmp_path: Path) -> None:
    graph = ExampleGraph()
    graph._nodes.clear()
    graph._edges.clear()
    output = tmp_path / "empty.cgf"
    with fake_fastavro():
        save_cgf(graph, output)
        with load_cgf(output, validate=True) as mapped:
            assert len(mapped.nodes) == 0
            assert len(mapped.edges) == 0
