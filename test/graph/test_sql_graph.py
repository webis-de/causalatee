import sqlite3

import pytest

from causalatee.graph import SQLGraph


@pytest.fixture
def graph() -> SQLGraph:
    return SQLGraph(sqlite3.connect(":memory:"))


class TestAddNodeAndAddEdge:
    def test_add_node_is_retrievable(self, graph: SQLGraph):
        graph.add_node("a", metadata={"kind": "concept"})
        node = graph.get_node("a")
        assert node.id == "a"
        assert node.metadata == {"kind": "concept"}

    def test_get_node_raises_key_error_for_missing_node(self, graph: SQLGraph):
        with pytest.raises(KeyError):
            graph.get_node("missing")

    def test_add_edge_auto_creates_missing_endpoints(self, graph: SQLGraph):
        graph.add_edge("a", "b", metadata={"weight": 1})
        assert {n.id for n in graph.nodes} == {"a", "b"}
        (edge,) = list(graph.edges)
        assert edge.source.id == "a"
        assert edge.target.id == "b"
        assert edge.metadata == {"weight": 1}

    def test_add_edge_does_not_overwrite_existing_endpoint_metadata(self, graph: SQLGraph):
        graph.add_node("a", metadata={"kind": "concept"})
        graph.add_edge("a", "b")
        assert graph.get_node("a").metadata == {"kind": "concept"}

    def test_add_edge_allows_multiple_edges_between_the_same_pair(self, graph: SQLGraph):
        graph.add_edge("a", "b", metadata={"n": 1})
        graph.add_edge("a", "b", metadata={"n": 2})
        edges = list(graph.edges)
        assert len(edges) == 2
        assert {e.metadata["n"] for e in edges} == {1, 2}


class TestEdgesFromAndTo:
    def test_edges_from_filters_by_source(self, graph: SQLGraph):
        graph.add_edge("a", "b")
        graph.add_edge("a", "c")
        graph.add_edge("b", "c")
        a = graph.get_node("a")
        assert {e.target.id for e in graph.edges_from(a)} == {"b", "c"}

    def test_edges_to_filters_by_target(self, graph: SQLGraph):
        graph.add_edge("a", "c")
        graph.add_edge("b", "c")
        graph.add_edge("a", "b")
        c = graph.get_node("c")
        assert {e.source.id for e in graph.edges_to(c)} == {"a", "b"}


class TestStreamingCollections:
    def test_edges_is_re_iterable(self, graph: SQLGraph):
        graph.add_edge("a", "b")
        first_pass = list(graph.edges)
        second_pass = list(graph.edges)
        assert len(first_pass) == len(second_pass) == 1

    def test_len_matches_iteration_count(self, graph: SQLGraph):
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")
        assert len(graph.nodes) == 3
        assert len(graph.edges) == 2
