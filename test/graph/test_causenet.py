from __future__ import annotations

from causalatee.graph import load_causenet

CAUSENET_PRECISION_URL = "https://groups.uni-paderborn.de/wdqa/causenet/causality-graphs/causenet-precision.jsonl.bz2"


class TestCauseNet:
    def test_causenet(self):
        graph = load_causenet(CAUSENET_PRECISION_URL, limit=100)

        assert len(graph.nodes) > 0
        assert len(graph.edges) == 100

        edge = next(iter(graph.edges))
        assert edge.cause.concept
        assert edge.effect.concept
        assert edge.support >= 1
        assert edge.cause in graph.nodes
        assert edge in edge.cause.outgoing_edges()
