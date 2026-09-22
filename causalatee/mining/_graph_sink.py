"""A ``Pipeline.reduce_items`` sink that aggregates mined relations into a
``causalatee.graph.Graph``, retaining bounded provenance.

Two-phase by design, deliberately NOT conflated into one step: (1) every relation the sink receives while the
pipeline drains is appended, unaggregated, to a temp SQLite spool alongside its item's full metadata -- bounded
memory regardless of corpus size, no aggregation logic here at all (mirrors the sibling ``causalgraph`` project's
own ``causal_relations`` table, and the disk-backed philosophy ``causalatee.graph._cgf.py``'s own writer already
uses rather than assuming things fit in RAM). (2) An ordinary, synchronous ``GROUP BY cause, effect`` aggregation
runs ONCE, lazily, the first time ``.nodes``/``.edges`` (or ``.edge_metadata_schema``) is actually accessed --
which is always after ``Pipeline.reduce_items`` has finished draining into the sink, so no explicit "finalize" call
is needed. Countercausal and causal mentions of the SAME (cause, effect) pair aggregate into ONE edge with both
counts as metadata (matching ``causalgraph``'s own ``leaf_edges.relation_count``/``countercausal_count`` split),
not two parallel edges.

Provenance design: ``support``/``causal_count``/``countercausal_count``/``avg_score`` are always computed from the
COMPLETE mention set for an edge -- these are never capped, regardless of corpus size, since a partial count would
misrepresent the aggregate. ``support`` (the raw mention count) is one dimension "for free"; additional named
dimensions can be declared via the ``support`` constructor argument -- e.g. a WARC pipeline's distinct-URL count vs.
distinct-sentence count, a PDF pipeline's distinct-file count -- each a function from one mention's metadata to a
hashable grouping key, counted as DISTINCT keys over the whole mention set. Separately, ``evidence`` (an
``EvidencePolicy``) bounds an illustrative SAMPLE of raw mention metadata embedded per edge -- deliberately a
DIFFERENT mechanism from the support counts, since "how many mentions support this edge" and "show me a few
concrete examples" have different (and conflicting) size requirements: the former must reflect everything, the
latter must stay small enough to keep ``save_cgf``'s output bounded. Grouping-key functions for ``support`` operate
on arbitrary Python metadata, which can't be pushed into a SQL ``GROUP BY`` -- aggregation therefore deserializes
and materializes one edge's full mention list at a time (discarded before the next edge), so memory stays bounded
by the corpus's single most-mentioned edge, not by the whole corpus -- a deliberate, documented tradeoff, not an
oversight. Two possible future optimizations without changing this API: compiling common selectors to SQL/JSON
expressions, or materializing selector fields as indexed spool columns -- not pursued here unless a real benchmark
shows the per-edge case is actually a problem.

The final aggregated graph is stored via ``causalatee.graph.SQLGraph`` -- ``GraphSink`` only owns the raw-mention
spooling and the ``GROUP BY`` aggregation SQL; once aggregation runs, each resulting ``(cause, effect)`` group is
handed to ``SQLGraph.add_edge`` exactly once. This is a deliberate split: ``SQLGraph`` is a generic, reusable,
mutable graph primitive with no knowledge of mining/mentions at all, while ``GraphSink`` is the mining-specific part
(batching, spooling, aggregating) that produces one.

``GraphSink`` IS both the reduce sink (a plain callable) and the resulting ``causalatee.graph.Graph`` (by inheriting
``SQLGraph``) -- so ``await Pipeline(...).reduce_items(graph_sink())`` returns something directly usable with
``causalatee.graph.save_cgf``, no extra conversion step. It owns a temp directory it created itself (unlike
``CauseEffectGraph``, which only ever re-opens a caller-owned static file) -- always close it (or use it as a context
manager) once done, mirroring ``CGFGraph``'s ``close()``/``__enter__``/``__exit__`` pattern in
``causalatee.graph._cgf.py``.
"""

from __future__ import annotations

import itertools
import json
import sqlite3
import tempfile
from collections.abc import Callable, Collection, Hashable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from causalatee.graph import SQLGraph, SQLGraphEdge, SQLGraphNode, diff_avro_schema, infer_avro_schema
from causalatee.models import ExtractedRelation

from ._pipeline import PipelineItem

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

# Fallback edge schema for a GraphSink that never aggregated any mentions at all (an empty corpus) -- there is no
# example edge to infer a schema from in that case, but save_cgf still needs a valid schema to write against.
_FALLBACK_EDGE_METADATA_SCHEMA: Mapping[str, object] = {
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


class GraphSinkError(ValueError):
    """Raised when ``GraphSink`` can't safely aggregate its spooled mentions into a graph -- currently, only when two
    edges' metadata have genuinely different shapes (see ``GraphSink._ensure_aggregated``'s own comment)."""


@dataclass(frozen=True)
class EvidencePolicy:
    """How many raw mentions' metadata to keep as an illustrative ``evidence`` sample per aggregated edge.

    ``support``/``support_by`` counts are ALWAYS computed from the complete mention set regardless of this policy
    -- this only bounds the separate, illustrative sample embedded in edge metadata, never the counts themselves.
    """

    max_items: int = 3

    def select(self, mentions: list[Mapping[str, object]]) -> list[Mapping[str, object]]:
        """First ``max_items`` mentions, in spool order -- the simplest deterministic policy. Subclass (or swap
        in a different callable-based policy) for a different selection strategy, e.g. highest-score or a random
        sample."""

        return mentions[: self.max_items]


class GraphSink(SQLGraph):
    """Reduce sink + resulting ``Graph``, see module docstring."""

    def __init__(
        self,
        *,
        support: Mapping[str, Callable[[Mapping[str, object]], Hashable]] | None = None,
        evidence: EvidencePolicy | None = None,
    ) -> None:
        """``support`` declares named support dimensions beyond the raw mention count -- e.g.
        ``{"sentences": lambda m: (m["source"]["id"], m["sentence_index"]), "documents": lambda m:
        m["source"]["id"]}`` for a WARC corpus. Each function maps one mention's metadata to a hashable grouping
        key; the resulting edge metadata's ``support_by[name]`` is the number of DISTINCT keys seen for that edge,
        always over the complete mention set. ``evidence``, if given, bounds a separate sample of raw mention
        metadata embedded per edge as ``evidence`` -- omit it (the default) for no evidence at all."""

        self._support = dict(support) if support else {}
        self._evidence_policy = evidence

        self._tempdir = tempfile.TemporaryDirectory(prefix="causalatee-mining-")
        db_path = Path(self._tempdir.name) / "mentions.sqlite3"
        connection = sqlite3.connect(db_path)
        connection.execute(
            "CREATE TABLE mentions (cause TEXT NOT NULL, effect TEXT NOT NULL, "
            "relationship TEXT NOT NULL, score REAL NOT NULL, metadata TEXT NOT NULL)"
        )
        connection.commit()
        super().__init__(connection)
        self._aggregated = False
        self._aggregation_error: Exception | None = None
        self._edge_metadata_schema: Mapping[str, object] | None = None

    def __call__(self, item: PipelineItem[Iterable[ExtractedRelation]]) -> None:
        """The ``Pipeline.reduce_items`` sink call: append every relation in this item's value (e.g. one
        sentence's identified relations) to the spool, unaggregated, paired with the item's own metadata -- every
        relation extracted from the same item shares that same source provenance."""

        metadata_json = json.dumps(dict(item.metadata))
        # TODO: is _normalize necessary here or does it conflate semantics? E.g., would we expect a .map(normalize)
        # instead?
        rows = [
            (_normalize(rel["e1"]), _normalize(rel["e2"]), rel["relationship"], rel["score"], metadata_json)
            for rel in item.value
        ]
        if rows:
            self._connection.executemany(
                "INSERT INTO mentions (cause, effect, relationship, score, metadata) VALUES (?, ?, ?, ?, ?)", rows
            )
            self._connection.commit()

    def _ensure_aggregated(self) -> None:
        if self._aggregated:
            # Re-raise the error: without this, a GraphSinkError partway through the loop below (which already set
            # self._aggregated -- see why below) would make every SUBSEQUENT call quietly treat aggregation as having
            # succeeded, hiding that some edges were added before the error and others never were.
            if self._aggregation_error is not None:
                raise self._aggregation_error
            return
        # Set before doing the work: add_edge() below calls self.get_node(), which resolves to THIS class's override
        # (Python method resolution doesn't know it's being called from inside base-class code); so it would recurse
        # right back into _ensure_aggregated() without this guard set first.
        self._aggregated = True

        try:
            # ORDER BY cause, effect lets itertools.groupby process one edge's mentions at a time -- materialized in
            # Python only for the CURRENT group, discarded before the next -- rather than one query per (cause, effect)
            # pair. See the module docstring for why support_by/evidence can't be pushed into this same SQL aggregate
            # the way support/causal_count/countercausal_count/avg_score already are.
            cursor = self._connection.execute(
                "SELECT cause, effect, relationship, score, metadata FROM mentions ORDER BY cause, effect"
            )
            first_edge: tuple[str, str] | None = None
            for (cause, effect), grouped_rows in itertools.groupby(cursor, key=lambda row: (row[0], row[1])):
                rows = list(grouped_rows)
                support = len(rows)
                causal_count = sum(1 for row in rows if row[2] == "Causal")
                countercausal_count = sum(1 for row in rows if row[2] == "Countercausal")
                avg_score = sum(row[3] for row in rows) / support

                metadata: dict[str, object] = {
                    "support": support,
                    "causal_count": causal_count,
                    "countercausal_count": countercausal_count,
                    "avg_score": avg_score,
                }
                if self._support or self._evidence_policy is not None:
                    mentions_metadata = [json.loads(row[4]) for row in rows]
                    if self._support:
                        metadata["support_by"] = {
                            name: len({key_fn(m) for m in mentions_metadata}) for name, key_fn in self._support.items()
                        }
                    if self._evidence_policy is not None:
                        metadata["evidence"] = self._evidence_policy.select(mentions_metadata)

                # Every edge's metadata must share the SAME shape: the schema this GraphSink ultimately reports
                # (`edge_metadata_schema`) is inferred from just the FIRST edge (see infer_avro_schema's own
                # stated limitation), so a later edge with a genuinely different shape would otherwise either
                # silently lose fields (extras not in the schema are dropped by the Avro encoder) or fail
                # confusingly, far away from here, inside save_cgf. Caught here instead, immediately, at the
                # exact edge that diverges.
                this_schema = infer_avro_schema("MinedEdgeMetadata", metadata)
                if self._edge_metadata_schema is None:
                    self._edge_metadata_schema = this_schema
                    first_edge = (cause, effect)
                else:
                    differences = diff_avro_schema(self._edge_metadata_schema, this_schema)
                    if differences:
                        raise GraphSinkError(
                            f"edge {cause!r} -> {effect!r}'s metadata shape doesn't match the shape inferred "
                            f"from the first aggregated edge {first_edge!r}: " + "; ".join(differences) + ". "
                            "GraphSink requires every mention's metadata to share one consistent shape across "
                            "a run -- see GraphSink's own docstring."
                        )

                self.add_edge(cause, effect, metadata=metadata)
        except Exception as exc:
            self._aggregation_error = exc
            raise

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
        """Inferred (via ``causalatee.graph.infer_avro_schema``) from the first aggregated edge, since
        ``support_by``/``evidence``'s shape depends entirely on this instance's own ``support``/``evidence``
        configuration -- there is no static schema that could anticipate it. Falls back to the base
        support/causal_count/countercausal_count/avg_score-only schema if aggregation produced no edges at all
        (an empty corpus)."""

        self._ensure_aggregated()
        if self._edge_metadata_schema is None:
            return _FALLBACK_EDGE_METADATA_SCHEMA
        return self._edge_metadata_schema

    def __enter__(self) -> GraphSink:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the spool database and remove its temp directory."""

        self._connection.close()
        self._tempdir.cleanup()


def graph_sink(
    *,
    support: Mapping[str, Callable[[Mapping[str, object]], Hashable]] | None = None,
    evidence: EvidencePolicy | None = None,
) -> GraphSink:
    """Create a ``Pipeline.reduce_items`` sink that aggregates every relation-carrying item it receives into a
    ``causalatee.graph.Graph``, by ``(cause, effect)`` pair across the whole corpus. See ``GraphSink`` for the
    aggregation/lifecycle/provenance details.
    """

    return GraphSink(support=support, evidence=evidence)
