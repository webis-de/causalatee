"""Async document sources for causalatee.mining.

A ``Document`` is deliberately minimal: an id, its text, and a free-form metadata mapping for provenance (source URL,
score, timestamp, ...). Downstream pipeline stages carry whatever richer per-stage type they produce (spans, relations)
-- nothing forces every stage through one rigid schema, since each ``Pipeline.map``/``.flat_map`` call genuinely changes
the item type as data moves through the stages.

``DocumentSource`` is a type alias for ``AsyncIterable[Document]``, not a new ``Protocol``/ABC. Any async generator
function already satisfies it with zero ceremony -- the same "duck-typed, no inheritance required" choice
``causalatee.models`` already made, here applied to something the stdlib already has the right structural type for.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    """A single whole document flowing into a mining ``Pipeline``."""

    id: str
    text: str
    metadata: Mapping[str, object] = field(default_factory=dict)


DocumentSource = AsyncIterable[Document]
