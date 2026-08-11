from __future__ import annotations

from pathlib import Path

import pytest

from causalatee.graph import load_cause_effect_graph, load_cgf, save_cgf
from causalatee.graph._cause_effect_graph import _parse_line

CEG_FIXTURE = (
    "favorite->finally\t1351\t0.08\t0.09\n"
    "condensation->polymer\t221\t1.09\t1.4\n"
    "polymer->polymer\t10749\t33.17\t33.3\n"
    "future->cap->lock\t11\t0.0\t0.0\n"
)


@pytest.fixture
def ceg_path(tmp_path: Path) -> Path:
    path = tmp_path / "ceg.txt"
    path.write_text(CEG_FIXTURE, encoding="utf-8")
    return path


class TestParseLine:
    def test_splits_on_first_arrow_only(self):
        cause, effect, count, necessity, sufficiency = _parse_line(
            "future->cap->lock\t11\t0.0\t0.0\n", line_number=1
        )
        assert cause == "future"
        assert effect == "cap->lock"
        assert (count, necessity, sufficiency) == (11, 0.0, 0.0)

    def test_missing_arrow_raises(self):
        with pytest.raises(ValueError, match="malformed CEG line"):
            _parse_line("nofield\t1\t0.0\t0.0\n", line_number=1)

    def test_wrong_field_count_raises(self):
        with pytest.raises(ValueError, match="malformed CEG line"):
            _parse_line("a->b\t1\t0.0\n", line_number=1)


class TestCauseEffectGraph:
    def test_loads_nodes_eagerly_and_deduplicated(self, ceg_path: Path):
        graph = load_cause_effect_graph(ceg_path)

        node_ids = {node.id for node in graph.nodes}
        assert node_ids == {"favorite", "finally", "condensation", "polymer", "future", "cap->lock"}
        assert len(graph.nodes) == 6  # "polymer" appears on both sides, deduplicated

    def test_edges_stream_and_are_re_iterable(self, ceg_path: Path):
        graph = load_cause_effect_graph(ceg_path)

        first_pass = list(graph.edges)
        second_pass = list(graph.edges)
        assert len(first_pass) == len(second_pass) == 4
        assert [e.cause.id for e in first_pass] == [e.cause.id for e in second_pass]

    def test_edge_metadata_round_trips(self, ceg_path: Path):
        graph = load_cause_effect_graph(ceg_path)
        edge = next(e for e in graph.edges if e.cause.id == "polymer" and e.effect.id == "polymer")

        assert edge.metadata == {"count": 10749, "necessity": 33.17, "sufficiency": 33.3}

    def test_limit_bounds_both_nodes_and_edges(self, ceg_path: Path):
        graph = load_cause_effect_graph(ceg_path, limit=1)

        assert {node.id for node in graph.nodes} == {"favorite", "finally"}
        assert len(list(graph.edges)) == 1

    def test_edges_from_and_to(self, ceg_path: Path):
        graph = load_cause_effect_graph(ceg_path)
        polymer = graph.get_node("polymer")

        assert {e.effect.id for e in polymer.outgoing_edges()} == {"polymer"}
        assert {e.cause.id for e in polymer.incoming_edges()} == {"condensation", "polymer"}

    def test_negative_limit_rejected(self, ceg_path: Path):
        with pytest.raises(ValueError):
            load_cause_effect_graph(ceg_path, limit=-1)

    def test_round_trips_through_cgf(self, ceg_path: Path, tmp_path: Path):
        graph = load_cause_effect_graph(ceg_path)
        cgf_path = tmp_path / "ceg.cgf"
        save_cgf(graph, cgf_path)

        with load_cgf(cgf_path, validate=True) as mapped:
            assert len(mapped.nodes) == 6
            polymer = mapped.get_node("polymer")
            self_loop = next(e for e in polymer.outgoing_edges() if e.target.id == "polymer")
            assert self_loop.metadata == {"count": 10749, "necessity": 33.17, "sufficiency": 33.3}
