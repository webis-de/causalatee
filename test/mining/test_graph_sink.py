"""Tests for causalatee.mining.GraphSink: the two-phase spool-then-aggregate sink, its configurable
support/evidence provenance, and its round-trip through causalatee.graph.save_cgf/load_cgf."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from causalatee.graph import load_cgf, save_cgf
from causalatee.mining import Document, EvidencePolicy, GraphSinkError, Pipeline, PipelineItem, graph_sink


def run(coro):
    return asyncio.run(coro)


def _item(relations, metadata=None):
    return PipelineItem(relations, metadata or {})


class TestGraphSink:
    def test_call_before_reduce_completes_does_not_aggregate_yet(self):
        sink = graph_sink()
        sink(_item([{"e1": "a", "e2": "b", "relationship": "Causal", "score": 0.9}]))
        assert not sink._aggregated
        sink.close()

    def test_aggregates_lazily_on_first_nodes_access(self):
        sink = graph_sink()
        sink(_item([{"e1": "a", "e2": "b", "relationship": "Causal", "score": 0.9}]))
        assert {n.id for n in sink.nodes} == {"a", "b"}
        assert sink._aggregated
        sink.close()

    def test_normalizes_case_and_whitespace_for_node_identity(self):
        sink = graph_sink()
        sink(_item([{"e1": " Storm ", "e2": "Flooding", "relationship": "Causal", "score": 0.9}]))
        sink(_item([{"e1": "storm", "e2": "FLOODING", "relationship": "Causal", "score": 0.8}]))
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
        sink(_item([{"e1": "sugar", "e2": "hyperactivity", "relationship": "Causal", "score": 0.6}]))
        sink(_item([{"e1": "sugar", "e2": "hyperactivity", "relationship": "Countercausal", "score": 0.9}]))
        edges = list(sink.edges)
        assert len(edges) == 1  # one edge, not two parallel ones
        assert edges[0].metadata["support"] == 2
        assert edges[0].metadata["causal_count"] == 1
        assert edges[0].metadata["countercausal_count"] == 1
        sink.close()

    def test_edges_is_re_iterable(self):
        sink = graph_sink()
        sink(_item([{"e1": "a", "e2": "b", "relationship": "Causal", "score": 0.9}]))
        first_pass = list(sink.edges)
        second_pass = list(sink.edges)
        assert len(first_pass) == len(second_pass) == 1
        sink.close()

    def test_context_manager_closes_cleanly(self):
        with graph_sink() as sink:
            sink(_item([{"e1": "a", "e2": "b", "relationship": "Causal", "score": 0.9}]))
            assert {n.id for n in sink.nodes} == {"a", "b"}
        # closing again (idempotence not required, but shouldn't be reached twice in normal use) -- just
        # confirm no exception escaped the `with` block.


class TestSupportDimensions:
    def test_no_support_configured_means_no_support_by_key(self):
        sink = graph_sink()
        sink(_item([{"e1": "a", "e2": "b", "relationship": "Causal", "score": 0.9}], {"source": {"id": "doc1"}}))
        (edge,) = list(sink.edges)
        assert "support_by" not in edge.metadata
        sink.close()

    def test_distinct_dimensions_counted_over_the_complete_mention_set(self):
        sink = graph_sink(
            support={
                "sentences": lambda m: (m["source"]["id"], m["sentence_index"]),
                "documents": lambda m: m["source"]["id"],
            }
        )
        # Same document, two different sentences -- both should count for "sentences", only one for "documents".
        sink(
            _item(
                [{"e1": "a", "e2": "b", "relationship": "Causal", "score": 0.9}],
                {"source": {"id": "doc1"}, "sentence_index": 0},
            )
        )
        sink(
            _item(
                [{"e1": "a", "e2": "b", "relationship": "Causal", "score": 0.8}],
                {"source": {"id": "doc1"}, "sentence_index": 1},
            )
        )
        # A duplicate mention of the SAME sentence -- should not inflate the sentence count further.
        sink(
            _item(
                [{"e1": "a", "e2": "b", "relationship": "Causal", "score": 0.7}],
                {"source": {"id": "doc1"}, "sentence_index": 0},
            )
        )
        (edge,) = list(sink.edges)
        assert edge.metadata["support"] == 3  # the raw mention count is never capped by support_by
        assert edge.metadata["support_by"] == {"sentences": 2, "documents": 1}
        sink.close()


class TestEvidence:
    def test_no_evidence_policy_means_no_evidence_key(self):
        sink = graph_sink()
        sink(_item([{"e1": "a", "e2": "b", "relationship": "Causal", "score": 0.9}], {"sentence": "A causes B."}))
        (edge,) = list(sink.edges)
        assert "evidence" not in edge.metadata
        sink.close()

    def test_evidence_bounded_but_support_reflects_the_whole_set(self):
        sink = graph_sink(evidence=EvidencePolicy(max_items=2))
        for i in range(5):
            sink(_item([{"e1": "a", "e2": "b", "relationship": "Causal", "score": 0.9}], {"sentence": f"sentence {i}"}))
        (edge,) = list(sink.edges)
        assert edge.metadata["support"] == 5  # never capped
        assert len(edge.metadata["evidence"]) == 2  # capped, per EvidencePolicy(max_items=2)
        assert edge.metadata["evidence"] == [{"sentence": "sentence 0"}, {"sentence": "sentence 1"}]
        sink.close()

    def test_evidence_default_policy_keeps_three(self):
        assert EvidencePolicy().max_items == 3


class TestEdgeMetadataSchema:
    def test_falls_back_to_base_schema_when_no_edges_were_aggregated(self):
        sink = graph_sink()
        schema = sink.edge_metadata_schema
        assert {f["name"] for f in schema["fields"]} == {
            "support",
            "causal_count",
            "countercausal_count",
            "avg_score",
        }
        sink.close()

    def test_inferred_schema_includes_support_by_and_evidence_fields(self):
        sink = graph_sink(
            support={"documents": lambda m: m["source"]["id"]},
            evidence=EvidencePolicy(max_items=1),
        )
        sink(
            _item(
                [{"e1": "a", "e2": "b", "relationship": "Causal", "score": 0.9}],
                {"source": {"id": "doc1"}, "sentence": "A causes B."},
            )
        )
        schema = sink.edge_metadata_schema
        field_names = {f["name"] for f in schema["fields"]}
        assert field_names == {"support", "causal_count", "countercausal_count", "avg_score", "support_by", "evidence"}
        sink.close()


class TestSchemaMismatchRaises:
    """Every edge's metadata must share ONE consistent shape within a run, since edge_metadata_schema is inferred
    from just the first aggregated edge (see infer_avro_schema's own stated limitation) -- a later edge with a
    genuinely different shape must raise immediately, at aggregation time, rather than silently lose data or fail
    confusingly later inside save_cgf."""

    def test_extra_field_on_a_later_edge_raises(self):
        sink = graph_sink(evidence=EvidencePolicy(max_items=5))
        sink(_item([{"e1": "a", "e2": "b", "relationship": "Causal", "score": 0.9}], {"sentence": "A causes B."}))
        sink(
            _item(
                [{"e1": "c", "e2": "d", "relationship": "Causal", "score": 0.9}],
                {"sentence": "C causes D.", "topic": "medicine"},
            )
        )
        with pytest.raises(GraphSinkError, match=r"unexpected extra field"):
            list(sink.edges)
        sink.close()

    def test_missing_field_on_a_later_edge_raises(self):
        sink = graph_sink(evidence=EvidencePolicy(max_items=5))
        sink(
            _item(
                [{"e1": "a", "e2": "b", "relationship": "Causal", "score": 0.9}],
                {"sentence": "A causes B.", "topic": "medicine"},
            )
        )
        sink(_item([{"e1": "c", "e2": "d", "relationship": "Causal", "score": 0.9}], {"sentence": "C causes D."}))
        with pytest.raises(GraphSinkError, match=r"missing"):
            list(sink.edges)
        sink.close()

    def test_type_mismatch_on_a_later_edge_raises(self):
        sink = graph_sink(evidence=EvidencePolicy(max_items=5))
        sink(_item([{"e1": "a", "e2": "b", "relationship": "Causal", "score": 0.9}], {"confidence": 5}))
        sink(_item([{"e1": "c", "e2": "d", "relationship": "Causal", "score": 0.9}], {"confidence": "high"}))
        with pytest.raises(GraphSinkError, match=r"expected type 'long', got 'string'"):
            list(sink.edges)
        sink.close()

    def test_error_message_identifies_both_edges_and_the_field(self):
        sink = graph_sink(evidence=EvidencePolicy(max_items=5))
        sink(_item([{"e1": "a", "e2": "b", "relationship": "Causal", "score": 0.9}], {"sentence": "A causes B."}))
        sink(
            _item(
                [{"e1": "c", "e2": "d", "relationship": "Causal", "score": 0.9}],
                {"sentence": "C causes D.", "topic": "medicine"},
            )
        )
        with pytest.raises(GraphSinkError) as exc_info:
            list(sink.edges)
        message = str(exc_info.value)
        assert "'a', 'b'" in message  # names the first (schema-defining) edge
        assert "'c'" in message and "'d'" in message  # names the diverging edge
        assert "topic" in message  # names the specific field that differs
        sink.close()

    def test_retrying_after_a_schema_mismatch_reraises_consistently_not_silently_succeeds(self):
        sink = graph_sink(evidence=EvidencePolicy(max_items=5))
        sink(_item([{"e1": "a", "e2": "b", "relationship": "Causal", "score": 0.9}], {"sentence": "A causes B."}))
        sink(
            _item(
                [{"e1": "c", "e2": "d", "relationship": "Causal", "score": 0.9}],
                {"sentence": "C causes D.", "topic": "medicine"},
            )
        )
        with pytest.raises(GraphSinkError):
            list(sink.edges)
        # A second access must NOT silently treat aggregation as having succeeded -- it must keep raising.
        with pytest.raises(GraphSinkError):
            list(sink.nodes)
        with pytest.raises(GraphSinkError):
            _ = sink.edge_metadata_schema
        sink.close()

    def test_save_cgf_also_raises_via_the_same_check_not_a_silent_drop(self, tmp_path: Path):
        sink = graph_sink(evidence=EvidencePolicy(max_items=5))
        sink(_item([{"e1": "a", "e2": "b", "relationship": "Causal", "score": 0.9}], {"sentence": "A causes B."}))
        sink(
            _item(
                [{"e1": "c", "e2": "d", "relationship": "Causal", "score": 0.9}],
                {"sentence": "C causes D.", "topic": "medicine"},
            )
        )
        with pytest.raises(GraphSinkError):
            save_cgf(sink, tmp_path / "should_not_be_written.cgf")
        assert not (tmp_path / "should_not_be_written.cgf").exists()
        sink.close()

    def test_consistent_shapes_across_many_edges_do_not_raise(self):
        # Regression guard: the check itself must not be over-eager -- genuinely uniform metadata across many
        # edges (the common case) must aggregate cleanly.
        sink = graph_sink(evidence=EvidencePolicy(max_items=5))
        for i in range(5):
            sink(
                _item(
                    [{"e1": f"a{i}", "e2": f"b{i}", "relationship": "Causal", "score": 0.9}],
                    {"sentence": f"sentence {i}", "sentence_index": i},
                )
            )
        assert len(list(sink.edges)) == 5
        sink.close()


class TestEndToEndMiningToCgf:
    def test_pipeline_through_graph_sink_round_trips_through_cgf(self, tmp_path: Path):
        async def source():
            yield Document(id="1", text="The storm caused flooding.")
            yield Document(id="2", text="THE STORM CAUSED FLOODING.")
            yield Document(id="3", text="Sugar does not cause hyperactivity.")

        def fake_extraction(text):
            if "storm" in text.lower():
                return [{"e1": "storm", "e2": "flooding", "relationship": "Causal", "score": 0.9}]
            return [{"e1": "sugar", "e2": "hyperactivity", "relationship": "Countercausal", "score": 0.7}]

        async def go():
            return await Pipeline.documents(source()).map(fake_extraction, concurrency=1).reduce_items(graph_sink())

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

    def test_round_trip_preserves_support_by_and_evidence_through_cgf(self, tmp_path: Path):
        async def source():
            yield Document(id="1", text="The storm caused flooding.", metadata={"url": "https://a.example"})
            yield Document(id="2", text="THE STORM CAUSED FLOODING.", metadata={"url": "https://b.example"})

        def fake_extraction(text):
            return [{"e1": "storm", "e2": "flooding", "relationship": "Causal", "score": 0.9}]

        async def go():
            sink = graph_sink(
                support={"documents": lambda m: m["source"]["id"]},
                evidence=EvidencePolicy(max_items=5),
            )
            return await Pipeline.documents(source()).map(fake_extraction, concurrency=1).reduce_items(sink)

        sink = run(go())
        with sink:
            cgf_path = tmp_path / "mined_with_evidence.cgf"
            save_cgf(sink, cgf_path)

            with load_cgf(cgf_path, validate=True) as mapped:
                storm = mapped.get_node("storm")
                (edge,) = list(storm.outgoing_edges())
                assert edge.metadata["support"] == 2
                assert edge.metadata["support_by"] == {"documents": 2}
                assert len(edge.metadata["evidence"]) == 2
                urls = {e["source"]["url"] for e in edge.metadata["evidence"]}
                assert urls == {"https://a.example", "https://b.example"}
