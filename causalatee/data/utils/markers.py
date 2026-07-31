"""``<eN>...</eN>`` marker <-> ``(clean_text, segments_by_eid)`` roundtrip
used by causalatee's identification schema.

These markers only *look* like XML/HTML tags. Each ``</eN>`` is matched to its own id's most recent ``<eN>``, not to
whatever tag is "on top of the stack" -- parsing tracks one open position per entity id, not a shared depth counter. Two
entities whose spans genuinely cross (partially overlap with neither containing the other, e.g. spans ``[0, 24)`` and
``[13, 39)``) therefore round-trip correctly even though the resulting text -- something like ``<e1>demanding the<e2>
arrest of </e1>a jawan who tri</e2>ed...`` -- would be invalid XML, since XML tags must close in strict
reverse-of-opening order. A non-nested marker sequence is valid output here, not a bug.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

Span = tuple[int, int]  # (start, end) character offsets, end-exclusive

_MARKER_RE = re.compile(r"<(/?)(e\w+)>")


def parse_entity_markers(text: str) -> tuple[str, dict[str, list[Span]]]:
    """Strip ``<eN>``/``</eN>`` tags; return (clean_text, segments_by_eid).

    An id may open/close more than once -- a discontinuous mention, rendered as repeated ``<eN>...</eN>`` occurrences
    sharing the same id -- every occurrence accumulates onto that id's segment list, in left-to-right order. Inverse
    of :func:`insert_entity_markers`.

    NOT an XML parser: matching is per entity id (module docstring has the details), so ``text`` does not need to be
    well-nested -- markers for two genuinely crossing spans parse correctly.
    """
    spans: dict[str, list[Span]] = {}
    open_positions: dict[str, int] = {}
    clean: list[str] = []
    pos = 0
    for m in _MARKER_RE.finditer(text):
        clean.append(text[pos : m.start()])
        offset = sum(len(c) for c in clean)
        closing, eid = m.group(1), m.group(2)
        if closing:
            spans.setdefault(eid, []).append((open_positions.get(eid, offset), offset))
        else:
            open_positions[eid] = offset
        pos = m.end()
    clean.append(text[pos:])
    return "".join(clean), spans


def insert_entity_markers(text: str, segments_by_eid: Mapping[str, Sequence[Span]]) -> str:
    """Insert ``<eN>...</eN>`` tags around each id's segment(s).

    Left-to-right, stack-based: closes innermost-first and opens outermost (longest span) first at each boundary, so
    nested and disjoint spans come out correctly nested. Output need not be nested when spans cross -- see the
    module docstring. Inverse of :func:`parse_entity_markers`.
    """
    opens: dict[int, list[str]] = {}
    closes: dict[int, list[str]] = {}
    for eid, segments in segments_by_eid.items():
        for start, end in segments:
            opens.setdefault(start, []).append(eid)
            closes.setdefault(end, []).append(eid)

    pieces: list[str] = []
    stack: list[str] = []
    pos = 0
    for boundary in sorted(set(opens) | set(closes)):
        pieces.append(text[pos:boundary])
        pos = boundary
        for eid in reversed(stack.copy()):
            if eid in closes.get(boundary, []):
                pieces.append(f"</{eid}>")
                stack.remove(eid)
        # Open outer (longer) spans first among spans starting here, so
        # nesting stays valid when several ids open at the same boundary.
        for eid in sorted(opens.get(boundary, []), key=lambda e: -_span_length(segments_by_eid[e])):
            pieces.append(f"<{eid}>")
            stack.append(eid)
    pieces.append(text[pos:])
    return "".join(pieces)


def _span_length(segments: Sequence[Span]) -> int:
    return max(e for _, e in segments) - min(s for s, _ in segments)
