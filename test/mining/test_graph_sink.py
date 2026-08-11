"""Tests for causalatee.mining.GraphSink: the two-phase spool-then-aggregate sink, and its round-trip through
causalatee.graph.save_cgf/load_cgf."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from causalatee.graph import load_cgf, save_cgf
from causalatee.mining import Document, Pipeline, graph_sink


def run(coro):
    return asyncio.run(coro)


class TestGraphSink:
    def test_call_before_reduce_completes_does_not_aggregate_yet(self):
        sink = graph_sink()
        sink([{"e1": "a", "e2": "b", "relation": "Causal", "score": 0.9}])
        assert not sink._aggregated
        sink.close()

    def test_aggregates_lazily_on_first_nodes_access(self):
        sink = graph_sink()
        sink([{"e1": "a", "e2": "b", "relation": "Causal", "score": 0.9}])
        assert {n.id for n in sink.nodes} == {"a", "b"}
        assert sink._aggregated
        sink.close()

    def test_normalizes_case_and_whitespace_for_node_identity(self):
        sink = graph_sink()
        sink([{"e1": " Storm ", "e2": "Flooding", "relation": "Causal", "score": 0.9}])
        sink([{"e1": "storm", "e2": "FLOODING", "relation": "Causal", "score": 0.8}])
        assert {n.id for n in sink.nodes} == {"storm", "flooding"}
        (edge,) = list(sink.edges)
        assert edge.metadata == {
            "support": 2,
            "causal_count": 2,
            "countercausal_count": 0,
            "avg_score": pytest.approx(0.85),
        }
        sink.close()

    def test_causal_and_countercausal_mentions_of_same_pair_aggregate_into_one_edge(self):
        sink = graph_sink()
        sink([{"e1": "sugar", "e2": "hyperactivity", "relation": "Causal", "score": 0.6}])
        sink([{"e1": "sugar", "e2": "hyperactivity", "relation": "Countercausal", "score": 0.9}])
        edges = list(sink.edges)
        assert len(edges) == 1  # one edge, not two parallel ones
        assert edges[0].metadata["support"] == 2
        assert edges[0].metadata["causal_count"] == 1
        assert edges[0].metadata["countercausal_count"] == 1
        sink.close()

    def test_edges_is_re_iterable(self):
        sink = graph_sink()
        sink([{"e1": "a", "e2": "b", "relation": "Causal", "score": 0.9}])
        first_pass = list(sink.edges)
        second_pass = list(sink.edges)
        assert len(first_pass) == len(second_pass) == 1
        sink.close()

    def test_context_manager_closes_cleanly(self):
        with graph_sink() as sink:
            sink([{"e1": "a", "e2": "b", "relation": "Causal", "score": 0.9}])
            assert {n.id for n in sink.nodes} == {"a", "b"}
        # closing again (idempotence not required, but shouldn't be reached twice in normal use) -- just
        # confirm no exception escaped the `with` block.


class TestEndToEndMiningToCgf:
    def test_pipeline_through_graph_sink_round_trips_through_cgf(self, tmp_path: Path):
        async def source():
            yield Document(id="1", text="The storm caused flooding.")
            yield Document(id="2", text="THE STORM CAUSED FLOODING.")
            yield Document(id="3", text="Sugar does not cause hyperactivity.")

        def fake_extraction(doc):
            text = doc.text
            if "storm" in text.lower():
                return [{"e1": "storm", "e2": "flooding", "relation": "Causal", "score": 0.9}]
            return [{"e1": "sugar", "e2": "hyperactivity", "relation": "Countercausal", "score": 0.7}]

        async def go():
            return await Pipeline(source()).map(fake_extraction, concurrency=1).reduce(graph_sink())

        sink = run(go())
        with sink:
            assert {n.id for n in sink.nodes} == {"storm", "flooding", "sugar", "hyperactivity"}

            cgf_path = tmp_path / "mined.cgf"
            save_cgf(sink, cgf_path)

            with load_cgf(cgf_path, validate=True) as mapped:
                assert len(mapped.nodes) == 4
                storm = mapped.get_node("storm")
                (edge,) = list(storm.outgoing_edges())
                assert edge.target.id == "flooding"
                assert edge.metadata == {
                    "support": 2,
                    "causal_count": 2,
                    "countercausal_count": 0,
                    "avg_score": 0.9,
                }
