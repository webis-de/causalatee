"""CGF 1.0 streaming reader/writer: CSR topology plus Avro metadata."""

from __future__ import annotations

import bisect
import contextlib
import io
import json
import mmap
import os
import sqlite3
import struct
import tempfile
from collections import OrderedDict
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast, overload

if TYPE_CHECKING:
    from ._graph import Edge, Graph, Node
else:
    try:  # Package use.
        from ._graph import Edge, Graph, Node
    except ImportError:  # Direct use with the reference files in one directory.
        from _graph import Edge, Graph, Node


MAGIC = b"CGF\r\n\x1a\n\x00"
MAJOR_VERSION = 1
MINOR_VERSION = 0
BYTE_ORDER_LITTLE = 1
FLAG_HAS_INCOMING = 1 << 0

HEADER = struct.Struct("<8sHHBBHQQIHHQQ8x")
DIRECTORY_ENTRY = struct.Struct("<IIQQQ")
U64 = struct.Struct("<Q")

NODE_ID_OFFSETS = 1
NODE_ID_DATA = 2
NODE_METADATA_OFFSETS = 3
NODE_METADATA_DATA = 4
OUT_OFFSETS = 5
EDGE_TARGETS = 6
EDGE_METADATA_OFFSETS = 7
EDGE_METADATA_DATA = 8
IN_OFFSETS = 9
IN_SOURCES = 10
IN_EDGE_INDICES = 11
NODE_METADATA_SCHEMA = 12
EDGE_METADATA_SCHEMA = 13

REQUIRED_SECTIONS = {
    NODE_ID_OFFSETS,
    NODE_ID_DATA,
    NODE_METADATA_OFFSETS,
    NODE_METADATA_DATA,
    OUT_OFFSETS,
    EDGE_TARGETS,
    EDGE_METADATA_OFFSETS,
    EDGE_METADATA_DATA,
    NODE_METADATA_SCHEMA,
    EDGE_METADATA_SCHEMA,
}
INCOMING_SECTIONS = {IN_OFFSETS, IN_SOURCES, IN_EDGE_INDICES}

PathLike = str | os.PathLike[str]


class CGFError(ValueError):
    """Raised when a CGF file is malformed or unsupported."""


class _U64Array(Sequence[int]):
    """A zero-copy sequence view over little-endian u64 values."""

    def __init__(self, data: mmap.mmap, offset: int, count: int) -> None:
        self._data = data
        self._offset = offset
        self._count = count

    def __len__(self) -> int:
        return self._count

    @overload
    def __getitem__(self, index: int) -> int: ...

    @overload
    def __getitem__(self, index: slice) -> list[int]: ...

    def __getitem__(self, index: int | slice) -> int | list[int]:
        if isinstance(index, slice):
            start, stop, step = index.indices(self._count)
            return [self[i] for i in range(start, stop, step)]
        if index < 0:
            index += self._count
        if index < 0 or index >= self._count:
            raise IndexError(index)
        return U64.unpack_from(self._data, self._offset + index * 8)[0]


class CGFNode(Node["CGFNode", "CGFEdge"]):
    """A lightweight node handle backed by a mapped CGF file."""

    def __init__(self, graph: CGFGraph, index: int) -> None:
        super().__init__(graph)
        self._index = index

    @property
    def graph(self) -> CGFGraph:
        return cast(CGFGraph, self._graph)

    @property
    def index(self) -> int:
        return self._index

    @property
    def id(self) -> str:
        return self.graph._node_id(self._index)

    def __hash__(self) -> int:
        return hash((id(self._graph), self._index))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CGFNode) and self._graph is other._graph and self._index == other._index

    def __repr__(self) -> str:
        return f"CGFNode({self.id!r})"


class CGFEdge(Edge[CGFNode, "CGFEdge"]):
    """A lightweight edge handle backed by a mapped CGF file."""

    def __init__(
        self,
        graph: CGFGraph,
        index: int,
        *,
        source_hint: int | None = None,
    ) -> None:
        super().__init__(graph)
        self._index = index
        self._source_hint = source_hint

    @property
    def graph(self) -> CGFGraph:
        return cast(CGFGraph, self._graph)

    @property
    def index(self) -> int:
        return self._index

    @property
    def source(self) -> CGFNode:
        source_index = self._source_hint
        if source_index is None:
            source_index = self.graph._source_index_for_edge(self._index)
        return CGFNode(self.graph, source_index)

    @property
    def target(self) -> CGFNode:
        return CGFNode(self.graph, self.graph._edge_targets[self._index])

    def __hash__(self) -> int:
        return hash((id(self._graph), self._index))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CGFEdge) and self._graph is other._graph and self._index == other._index

    def __repr__(self) -> str:
        return f"CGFEdge({self.source.id!r} -> {self.target.id!r})"


T = TypeVar("T")


class _HandleCollection(Collection[T], Generic[T]):
    def __init__(self, length: int, factory: Any) -> None:
        self._length = length
        self._factory = factory

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterator[T]:
        for index in range(self._length):
            yield self._factory(index)

    def __contains__(self, value: object) -> bool:
        graph = getattr(value, "graph", None)
        index = getattr(value, "index", None)
        expected_graph = getattr(self._factory, "__self__", None)
        return graph is expected_graph and isinstance(index, int) and 0 <= index < self._length


class CGFGraph(Graph[CGFNode, CGFEdge]):
    """A read-only causal graph whose indexes and data remain memory mapped."""

    def __init__(self, path: PathLike, *, validate: bool = False) -> None:
        self._path = Path(path)
        file_handle = self._path.open("rb")
        try:
            self._mmap = mmap.mmap(file_handle.fileno(), 0, access=mmap.ACCESS_READ)
        finally:
            file_handle.close()

        try:
            self._read_header_and_sections()
            self._bind_arrays()
            if validate:
                self.validate()
        except Exception:
            self._mmap.close()
            raise

    def __enter__(self) -> CGFGraph:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def has_incoming_index(self) -> bool:
        return bool(self._flags & FLAG_HAS_INCOMING)

    @property
    def node_metadata_schema(self) -> Mapping[str, object]:
        return self._node_avro_schema

    @property
    def edge_metadata_schema(self) -> Mapping[str, object]:
        return self._edge_avro_schema

    @property
    def nodes(self) -> Collection[CGFNode]:
        return _HandleCollection(self._node_count, self._node)

    @property
    def edges(self) -> Collection[CGFEdge]:
        return _HandleCollection(self._edge_count, self._edge)

    def close(self) -> None:
        """Close the mapping. Existing node and edge handles become invalid."""

        if not self._mmap.closed:
            self._mmap.close()

    def get_node(self, node_id: str) -> CGFNode:
        query = node_id.encode("utf-8")
        low = 0
        high = self._node_count
        while low < high:
            middle = (low + high) // 2
            candidate = self._node_id_bytes(middle)
            if candidate < query:
                low = middle + 1
            else:
                high = middle
        if low >= self._node_count or self._node_id_bytes(low) != query:
            raise KeyError(node_id)
        return CGFNode(self, low)

    def edges_from(self, node: CGFNode) -> Iterable[CGFEdge]:
        self._require_node(node)
        start = self._out_offsets[node.index]
        stop = self._out_offsets[node.index + 1]
        for edge_index in range(start, stop):
            yield CGFEdge(self, edge_index, source_hint=node.index)

    def edges_to(self, node: CGFNode) -> Iterable[CGFEdge]:
        self._require_node(node)
        if self.has_incoming_index:
            start = self._in_offsets[node.index]
            stop = self._in_offsets[node.index + 1]
            for position in range(start, stop):
                yield CGFEdge(
                    self,
                    self._in_edge_indices[position],
                    source_hint=self._in_sources[position],
                )
            return

        for source_index in range(self._node_count):
            for edge in self.edges_from(CGFNode(self, source_index)):
                if edge.target.index == node.index:
                    yield edge

    def node_metadata(self, node: CGFNode) -> Mapping[str, object]:
        self._require_node(node)
        return self._metadata(
            self._node_metadata_offsets,
            self._section(NODE_METADATA_DATA),
            node.index,
            owner="node",
        )

    def edge_metadata(self, edge: CGFEdge) -> Mapping[str, object]:
        self._require_edge(edge)
        return self._metadata(
            self._edge_metadata_offsets,
            self._section(EDGE_METADATA_DATA),
            edge.index,
            owner="edge",
        )

    def validate(self) -> None:
        """Perform a full `O(N + M)` structural and metadata validation."""

        self._validate_offsets(self._node_id_offsets, self._section(NODE_ID_DATA)[1])
        self._validate_offsets(self._node_metadata_offsets, self._section(NODE_METADATA_DATA)[1])
        self._validate_offsets(self._out_offsets, self._edge_count)
        self._validate_offsets(self._edge_metadata_offsets, self._section(EDGE_METADATA_DATA)[1])

        previous: bytes | None = None
        for index in range(self._node_count):
            node_id = self._node_id_bytes(index)
            node_id.decode("utf-8")
            if previous is not None and node_id <= previous:
                raise CGFError("node IDs must be unique and strictly byte-sorted")
            previous = node_id
            self.node_metadata(CGFNode(self, index))

        for edge_index in range(self._edge_count):
            if self._edge_targets[edge_index] >= self._node_count:
                raise CGFError("edge target ordinal is out of range")
            self.edge_metadata(CGFEdge(self, edge_index))

        if self.has_incoming_index:
            self._validate_offsets(self._in_offsets, self._edge_count)
            for position in range(self._edge_count):
                if self._in_sources[position] >= self._node_count:
                    raise CGFError("incoming source ordinal is out of range")
                if self._in_edge_indices[position] >= self._edge_count:
                    raise CGFError("incoming edge ordinal is out of range")

    def _read_header_and_sections(self) -> None:
        if len(self._mmap) < HEADER.size:
            raise CGFError("file is shorter than the CGF header")
        (
            magic,
            major,
            minor,
            byte_order,
            self._flags,
            header_size,
            self._node_count,
            self._edge_count,
            section_count,
            entry_size,
            reserved,
            directory_offset,
            declared_size,
        ) = HEADER.unpack_from(self._mmap)

        if magic != MAGIC:
            raise CGFError("invalid CGF magic")
        if major != MAJOR_VERSION or minor > MINOR_VERSION:
            raise CGFError(f"unsupported CGF version {major}.{minor}")
        if byte_order != BYTE_ORDER_LITTLE:
            raise CGFError("unsupported byte order")
        if self._flags & ~FLAG_HAS_INCOMING:
            raise CGFError("unsupported header flags")
        if header_size != HEADER.size or entry_size != DIRECTORY_ENTRY.size:
            raise CGFError("unsupported header or directory entry size")
        if reserved != 0:
            raise CGFError("reserved header field is non-zero")
        if declared_size != len(self._mmap):
            raise CGFError("declared file size does not match actual size")

        directory_length = section_count * entry_size
        if directory_offset + directory_length > len(self._mmap):
            raise CGFError("section directory is out of bounds")

        self._sections: dict[int, tuple[int, int, int]] = {}
        for index in range(section_count):
            position = directory_offset + index * entry_size
            section_type, flags, offset, length, count = DIRECTORY_ENTRY.unpack_from(self._mmap, position)
            if flags != 0:
                raise CGFError(f"unsupported flags on section {section_type}")
            if section_type in self._sections:
                raise CGFError(f"duplicate section {section_type}")
            if offset % 8 or offset + length > len(self._mmap):
                raise CGFError(f"section {section_type} is misaligned or out of bounds")
            self._sections[section_type] = (offset, length, count)

        missing = REQUIRED_SECTIONS - self._sections.keys()
        if missing:
            raise CGFError(f"missing required sections: {sorted(missing)}")
        present_incoming = INCOMING_SECTIONS & self._sections.keys()
        if present_incoming and present_incoming != INCOMING_SECTIONS:
            raise CGFError("incoming index sections are incomplete")
        has_incoming_sections = present_incoming == INCOMING_SECTIONS
        if self.has_incoming_index != has_incoming_sections:
            raise CGFError("incoming flag and sections are inconsistent")

    def _bind_arrays(self) -> None:
        self._node_id_offsets = self._u64_section(NODE_ID_OFFSETS, self._node_count + 1)
        self._node_metadata_offsets = self._u64_section(NODE_METADATA_OFFSETS, self._node_count + 1)
        self._out_offsets = self._u64_section(OUT_OFFSETS, self._node_count + 1)
        self._edge_targets = self._u64_section(EDGE_TARGETS, self._edge_count)
        self._edge_metadata_offsets = self._u64_section(EDGE_METADATA_OFFSETS, self._edge_count + 1)
        if self.has_incoming_index:
            self._in_offsets = self._u64_section(IN_OFFSETS, self._node_count + 1)
            self._in_sources = self._u64_section(IN_SOURCES, self._edge_count)
            self._in_edge_indices = self._u64_section(IN_EDGE_INDICES, self._edge_count)
        self._node_avro_schema = self._read_avro_schema(NODE_METADATA_SCHEMA)
        self._edge_avro_schema = self._read_avro_schema(EDGE_METADATA_SCHEMA)
        self._parsed_avro_schemas: dict[str, Any] = {}

    def _read_avro_schema(self, section_type: int) -> Mapping[str, object]:
        offset, length, _ = self._section(section_type)
        try:
            schema = json.loads(self._mmap[offset : offset + length])
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise CGFError("invalid Avro metadata schema JSON") from exc
        if not isinstance(schema, dict):
            raise CGFError("Avro metadata schema must be a JSON object")
        return schema

    def _section(self, section_type: int) -> tuple[int, int, int]:
        return self._sections[section_type]

    def _u64_section(self, section_type: int, expected_count: int) -> _U64Array:
        offset, length, count = self._section(section_type)
        if count != expected_count or length != expected_count * 8:
            raise CGFError(f"section {section_type} has an invalid size or count")
        return _U64Array(self._mmap, offset, count)

    def _node(self, index: int) -> CGFNode:
        return CGFNode(self, index)

    def _edge(self, index: int) -> CGFEdge:
        return CGFEdge(self, index)

    def _node_id_bytes(self, index: int) -> bytes:
        data_offset, data_length, _ = self._section(NODE_ID_DATA)
        start = self._node_id_offsets[index]
        stop = self._node_id_offsets[index + 1]
        if start > stop or stop > data_length:
            raise CGFError("invalid node ID offsets")
        return self._mmap[data_offset + start : data_offset + stop]

    def _node_id(self, index: int) -> str:
        return self._node_id_bytes(index).decode("utf-8")

    def _metadata(
        self,
        offsets: Sequence[int],
        data_section: tuple[int, int, int],
        index: int,
        *,
        owner: str,
    ) -> Mapping[str, object]:
        data_offset, data_length, _ = data_section
        start = offsets[index]
        stop = offsets[index + 1]
        if start > stop or stop > data_length:
            raise CGFError("invalid metadata offsets")
        payload = self._mmap[data_offset + start : data_offset + stop]
        value = self._decode_avro_metadata(owner, payload)
        if not isinstance(value, dict):
            raise CGFError("metadata record must decode to a mapping")
        return value

    def _decode_avro_metadata(self, owner: str, payload: bytes) -> Mapping[str, object]:
        fastavro = _require_fastavro()
        schema = self._node_avro_schema if owner == "node" else self._edge_avro_schema
        parsed = self._parsed_avro_schemas.get(owner)
        if parsed is None:
            parsed = fastavro.parse_schema(dict(schema))
            self._parsed_avro_schemas[owner] = parsed
        try:
            value = fastavro.schemaless_reader(io.BytesIO(payload), parsed)
            return _transform_logical_json(value, schema, decode=True)
        except Exception as exc:
            raise CGFError(f"invalid Avro {owner} metadata record") from exc

    def _source_index_for_edge(self, edge_index: int) -> int:
        if edge_index < 0 or edge_index >= self._edge_count:
            raise IndexError(edge_index)
        return bisect.bisect_right(self._out_offsets, edge_index) - 1

    def _require_node(self, node: CGFNode) -> None:
        if node.graph is not self:
            raise ValueError("node belongs to a different graph")

    def _require_edge(self, edge: CGFEdge) -> None:
        if edge.graph is not self:
            raise ValueError("edge belongs to a different graph")

    @staticmethod
    def _validate_offsets(offsets: Sequence[int], expected_final: int) -> None:
        previous = 0
        for index, value in enumerate(offsets):
            if index == 0 and value != 0:
                raise CGFError("offset array must start at zero")
            if value < previous:
                raise CGFError("offset array is not monotonic")
            previous = value
        if previous != expected_final:
            raise CGFError("offset array has an invalid final value")


def load_cgf(path: PathLike, *, validate: bool = False) -> CGFGraph:
    """Memory-map and return a read-only CGF graph."""

    return CGFGraph(path, validate=validate)


def save_cgf(
    graph: Graph[Any, Any],
    path: PathLike,
    *,
    include_incoming: bool = True,
    node_metadata_schema: Mapping[str, object] | str | None = None,
    edge_metadata_schema: Mapping[str, object] | str | None = None,
) -> None:
    """Serialize ``graph`` using disk-backed sorting and streamed sections.

    Memory use is bounded independently of metadata and edge count. SQLite is
    used as a disposable external-sort spool; final CGF sections are written to
    separate temporary streams and assembled atomically. The only graph-side
    state retained by this function is whatever the supplied graph itself owns.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    schemas, schema_payloads, parsed_schemas = _prepare_writer_schemas(
        graph,
        node_metadata_schema=node_metadata_schema,
        edge_metadata_schema=edge_metadata_schema,
    )

    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.cgf-", dir=destination.parent) as temporary_directory:
        staging = Path(temporary_directory)
        database = sqlite3.connect(staging / "sort.sqlite3")
        try:
            _configure_spool(database)
            database.executescript(
                """
                CREATE TABLE nodes (
                    id BLOB PRIMARY KEY,
                    metadata BLOB NOT NULL,
                    ordinal INTEGER UNIQUE
                ) WITHOUT ROWID;
                CREATE TABLE edges (
                    source INTEGER NOT NULL,
                    target INTEGER NOT NULL,
                    input_order INTEGER NOT NULL,
                    metadata BLOB NOT NULL,
                    edge_index INTEGER
                );
                """
            )

            node_count = _spool_nodes(database, graph, parsed_schemas["node"], schemas["node"])
            sections = _emit_node_sections(database, staging, node_count, schema_payloads["node"])
            edge_count = _spool_edges(database, graph, parsed_schemas["edge"], schemas["edge"])
            sections.extend(
                _emit_outgoing_sections(
                    database,
                    staging,
                    node_count=node_count,
                    edge_count=edge_count,
                    edge_schema_payload=schema_payloads["edge"],
                )
            )

            flags = 0
            if include_incoming:
                flags |= FLAG_HAS_INCOMING
                sections.extend(
                    _emit_incoming_sections(
                        database,
                        staging,
                        node_count=node_count,
                        edge_count=edge_count,
                    )
                )
        finally:
            database.close()

        _write_staged_file(
            destination,
            sections,
            flags=flags,
            node_count=node_count,
            edge_count=edge_count,
        )


def _prepare_writer_schemas(
    graph: Graph[Any, Any],
    *,
    node_metadata_schema: Mapping[str, object] | str | None,
    edge_metadata_schema: Mapping[str, object] | str | None,
) -> tuple[dict[str, dict[str, object]], dict[str, bytes], dict[str, Any]]:
    supplied = {"node": node_metadata_schema, "edge": edge_metadata_schema}
    inherited = {
        "node": graph.node_metadata_schema,
        "edge": graph.edge_metadata_schema,
    }
    schemas: dict[str, dict[str, object]] = {}
    payloads: dict[str, bytes] = {}
    parsed: dict[str, Any] = {}
    fastavro = _require_fastavro()
    for owner in ("node", "edge"):
        selected = supplied[owner] or inherited[owner]
        if selected is None:
            raise ValueError(
                f"no {owner} metadata schema: define graph.{owner}_metadata_schema or pass {owner}_metadata_schema=..."
            )
        schema, payload = _normalize_avro_schema(selected)
        try:
            parsed[owner] = fastavro.parse_schema(schema)
        except Exception as exc:
            raise ValueError(f"invalid {owner} Avro metadata schema") from exc
        schemas[owner] = schema
        payloads[owner] = payload
    return schemas, payloads, parsed


def _configure_spool(database: sqlite3.Connection) -> None:
    database.execute("PRAGMA journal_mode=OFF")
    database.execute("PRAGMA synchronous=OFF")
    database.execute("PRAGMA temp_store=FILE")
    database.execute("PRAGMA cache_size=-32768")


def _spool_nodes(
    database: sqlite3.Connection,
    graph: Graph[Any, Any],
    parsed_schema: Any,
    schema: Mapping[str, object],
) -> int:
    count = 0
    try:
        with database:
            for node in graph.nodes:
                node_id = node.id.encode("utf-8")
                metadata = _encode_avro_metadata(node.metadata, parsed_schema, schema, owner="node")
                database.execute(
                    "INSERT INTO nodes(id, metadata) VALUES (?, ?)",
                    (node_id, metadata),
                )
                count += 1
    except sqlite3.IntegrityError as exc:
        raise ValueError("duplicate node ID") from exc
    return count


def _spool_edges(
    database: sqlite3.Connection,
    graph: Graph[Any, Any],
    parsed_schema: Any,
    schema: Mapping[str, object],
) -> int:
    lookup = database.cursor()
    insert = database.cursor()
    ordinal_cache: OrderedDict[bytes, int] = OrderedDict()

    def resolve(node_id: str) -> int:
        encoded = node_id.encode("utf-8")
        ordinal = ordinal_cache.get(encoded)
        if ordinal is not None:
            ordinal_cache.move_to_end(encoded)
            return ordinal
        row = lookup.execute("SELECT ordinal FROM nodes WHERE id = ?", (encoded,)).fetchone()
        if row is None:
            raise ValueError("edge endpoint is absent from graph.nodes")
        ordinal = row[0]
        ordinal_cache[encoded] = ordinal
        if len(ordinal_cache) > 65_536:
            ordinal_cache.popitem(last=False)
        return ordinal

    count = 0
    with database:
        for input_order, edge in enumerate(graph.edges):
            source = resolve(edge.source.id)
            target = resolve(edge.target.id)
            metadata = _encode_avro_metadata(edge.metadata, parsed_schema, schema, owner="edge")
            insert.execute(
                "INSERT INTO edges(source, target, input_order, metadata) VALUES (?, ?, ?, ?)",
                (source, target, input_order, metadata),
            )
            count += 1
    return count


def _section_path(staging: Path, section_type: int) -> Path:
    return staging / f"section-{section_type:02d}.bin"


def _emit_node_sections(
    database: sqlite3.Connection,
    staging: Path,
    node_count: int,
    node_schema_payload: bytes,
) -> list[tuple[int, Path, int]]:
    id_offsets_path = _section_path(staging, NODE_ID_OFFSETS)
    id_data_path = _section_path(staging, NODE_ID_DATA)
    metadata_offsets_path = _section_path(staging, NODE_METADATA_OFFSETS)
    metadata_data_path = _section_path(staging, NODE_METADATA_DATA)
    schema_path = _section_path(staging, NODE_METADATA_SCHEMA)
    schema_path.write_bytes(node_schema_payload)

    id_position = 0
    metadata_position = 0
    updates: list[tuple[int, bytes]] = []
    with (
        id_offsets_path.open("wb") as id_offsets,
        id_data_path.open("wb") as id_data,
        metadata_offsets_path.open("wb") as metadata_offsets,
        metadata_data_path.open("wb") as metadata_data,
    ):
        _write_u64(id_offsets, 0)
        _write_u64(metadata_offsets, 0)
        cursor = database.execute("SELECT id, metadata FROM nodes ORDER BY id")
        for ordinal, (node_id, metadata) in enumerate(cursor):
            id_data.write(node_id)
            id_position += len(node_id)
            _write_u64(id_offsets, id_position)
            metadata_data.write(metadata)
            metadata_position += len(metadata)
            _write_u64(metadata_offsets, metadata_position)
            updates.append((ordinal, node_id))
            if len(updates) >= 10_000:
                database.executemany("UPDATE nodes SET ordinal = ? WHERE id = ?", updates)
                updates.clear()
        if updates:
            database.executemany("UPDATE nodes SET ordinal = ? WHERE id = ?", updates)
    database.commit()
    return [
        (NODE_ID_OFFSETS, id_offsets_path, node_count + 1),
        (NODE_ID_DATA, id_data_path, node_count),
        (NODE_METADATA_OFFSETS, metadata_offsets_path, node_count + 1),
        (NODE_METADATA_DATA, metadata_data_path, node_count),
        (NODE_METADATA_SCHEMA, schema_path, 1),
    ]


def _emit_outgoing_sections(
    database: sqlite3.Connection,
    staging: Path,
    *,
    node_count: int,
    edge_count: int,
    edge_schema_payload: bytes,
) -> list[tuple[int, Path, int]]:
    out_offsets_path = _section_path(staging, OUT_OFFSETS)
    targets_path = _section_path(staging, EDGE_TARGETS)
    metadata_offsets_path = _section_path(staging, EDGE_METADATA_OFFSETS)
    metadata_data_path = _section_path(staging, EDGE_METADATA_DATA)
    schema_path = _section_path(staging, EDGE_METADATA_SCHEMA)
    schema_path.write_bytes(edge_schema_payload)

    current_node = 0
    emitted = 0
    metadata_position = 0
    updates: list[tuple[int, int]] = []
    with (
        out_offsets_path.open("wb") as out_offsets,
        targets_path.open("wb") as targets,
        metadata_offsets_path.open("wb") as metadata_offsets,
        metadata_data_path.open("wb") as metadata_data,
    ):
        _write_u64(out_offsets, 0)
        _write_u64(metadata_offsets, 0)
        cursor = database.execute(
            "SELECT rowid, source, target, metadata FROM edges ORDER BY source, target, input_order"
        )
        for rowid, source, target, metadata in cursor:
            while current_node < source:
                _write_u64(out_offsets, emitted)
                current_node += 1
            _write_u64(targets, target)
            metadata_data.write(metadata)
            metadata_position += len(metadata)
            _write_u64(metadata_offsets, metadata_position)
            updates.append((emitted, rowid))
            emitted += 1
            if len(updates) >= 10_000:
                database.executemany("UPDATE edges SET edge_index = ? WHERE rowid = ?", updates)
                updates.clear()
        while current_node < node_count:
            _write_u64(out_offsets, emitted)
            current_node += 1
        if updates:
            database.executemany("UPDATE edges SET edge_index = ? WHERE rowid = ?", updates)
    database.commit()
    if emitted != edge_count:
        raise RuntimeError("edge spool count changed during output")
    return [
        (OUT_OFFSETS, out_offsets_path, node_count + 1),
        (EDGE_TARGETS, targets_path, edge_count),
        (EDGE_METADATA_OFFSETS, metadata_offsets_path, edge_count + 1),
        (EDGE_METADATA_DATA, metadata_data_path, edge_count),
        (EDGE_METADATA_SCHEMA, schema_path, 1),
    ]


def _emit_incoming_sections(
    database: sqlite3.Connection,
    staging: Path,
    *,
    node_count: int,
    edge_count: int,
) -> list[tuple[int, Path, int]]:
    offsets_path = _section_path(staging, IN_OFFSETS)
    sources_path = _section_path(staging, IN_SOURCES)
    indices_path = _section_path(staging, IN_EDGE_INDICES)
    current_node = 0
    emitted = 0
    with (
        offsets_path.open("wb") as offsets,
        sources_path.open("wb") as sources,
        indices_path.open("wb") as indices,
    ):
        _write_u64(offsets, 0)
        cursor = database.execute("SELECT target, source, edge_index FROM edges ORDER BY target, source, edge_index")
        for target, source, edge_index in cursor:
            while current_node < target:
                _write_u64(offsets, emitted)
                current_node += 1
            _write_u64(sources, source)
            _write_u64(indices, edge_index)
            emitted += 1
        while current_node < node_count:
            _write_u64(offsets, emitted)
            current_node += 1
    if emitted != edge_count:
        raise RuntimeError("edge spool count changed during incoming output")
    return [
        (IN_OFFSETS, offsets_path, node_count + 1),
        (IN_SOURCES, sources_path, edge_count),
        (IN_EDGE_INDICES, indices_path, edge_count),
    ]


def _require_fastavro() -> Any:
    try:
        import fastavro
    except ImportError as exc:
        raise RuntimeError("Avro metadata requires the optional 'fastavro' package") from exc
    return fastavro


def _normalize_avro_schema(
    schema: Mapping[str, object] | str,
) -> tuple[dict[str, object], bytes]:
    if isinstance(schema, str):
        try:
            decoded = json.loads(schema)
        except json.JSONDecodeError as exc:
            raise ValueError("Avro schema string is not valid JSON") from exc
        if not isinstance(decoded, dict):
            raise ValueError("Avro metadata schema must be a JSON object")
        # Preserve a caller-supplied schema string exactly apart from UTF-8 encoding.
        return decoded, schema.encode("utf-8")
    try:
        payload = json.dumps(
            dict(schema),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Avro metadata schema is not JSON-serializable") from exc
    # Parse a fresh tree because some Avro implementations annotate schemas.
    return json.loads(payload), payload


def _encode_avro_metadata(
    metadata: Mapping[str, object],
    parsed_schema: Any,
    schema: Mapping[str, object],
    *,
    owner: str,
) -> bytes:
    fastavro = _require_fastavro()
    stream = io.BytesIO()
    try:
        record = _transform_logical_json(dict(metadata), schema, decode=False)
        fastavro.schemaless_writer(stream, parsed_schema, record)
    except Exception as exc:
        raise ValueError(f"cannot encode {owner} metadata with its Avro schema") from exc
    return stream.getvalue()


def _transform_logical_json(value: Any, schema: Any, *, decode: bool) -> Any:
    """Encode or restore values marked with the ``causalatee.json`` logical type."""

    if isinstance(schema, list):
        if value is None:
            return None
        logical_branch = next(
            (
                branch
                for branch in schema
                if isinstance(branch, dict) and branch.get("logicalType") == "causalatee.json"
            ),
            None,
        )
        if logical_branch is not None:
            return _transform_logical_json(value, logical_branch, decode=decode)
        for branch in schema:
            if branch != "null":
                return _transform_logical_json(value, branch, decode=decode)
        return value
    if not isinstance(schema, dict):
        return value
    if schema.get("logicalType") == "causalatee.json":
        if decode:
            try:
                return json.loads(value)
            except (TypeError, json.JSONDecodeError) as exc:
                raise CGFError("invalid causalatee.json logical value") from exc
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    schema_type = schema.get("type")
    if isinstance(schema_type, (dict, list)):
        return _transform_logical_json(value, schema_type, decode=decode)
    if schema_type == "record" and isinstance(value, Mapping):
        fields = {field["name"]: field["type"] for field in schema.get("fields", [])}
        return {
            key: _transform_logical_json(item, fields[key], decode=decode) if key in fields else item
            for key, item in value.items()
        }
    if schema_type == "array" and isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_transform_logical_json(item, schema.get("items"), decode=decode) for item in value]
    if schema_type == "map" and isinstance(value, Mapping):
        return {key: _transform_logical_json(item, schema.get("values"), decode=decode) for key, item in value.items()}
    return value


def _align(value: int, alignment: int = 8) -> int:
    return (value + alignment - 1) // alignment * alignment


def _write_u64(stream: Any, value: int) -> None:
    stream.write(U64.pack(value))


def _write_staged_file(
    destination: Path,
    sections: list[tuple[int, Path, int]],
    *,
    flags: int,
    node_count: int,
    edge_count: int,
) -> None:
    directory_offset = HEADER.size
    payload_offset = _align(directory_offset + len(sections) * DIRECTORY_ENTRY.size)
    directory: list[tuple[int, int, int, int, int]] = []
    current = payload_offset
    for section_type, payload_path, count in sections:
        current = _align(current)
        length = payload_path.stat().st_size
        directory.append((section_type, 0, current, length, count))
        current += length
    file_size = current

    header = HEADER.pack(
        MAGIC,
        MAJOR_VERSION,
        MINOR_VERSION,
        BYTE_ORDER_LITTLE,
        flags,
        HEADER.size,
        node_count,
        edge_count,
        len(sections),
        DIRECTORY_ENTRY.size,
        0,
        directory_offset,
        file_size,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(header)
            for entry in directory:
                stream.write(DIRECTORY_ENTRY.pack(*entry))
            _write_padding(stream, payload_offset)
            for (_, payload_path, _), (_, _, offset, _, _) in zip(sections, directory):
                _write_padding(stream, offset)
                with payload_path.open("rb") as payload:
                    while chunk := payload.read(1024 * 1024):
                        stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        if temporary_name is not None:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary_name)
        raise


def _write_padding(stream: Any, target_offset: int) -> None:
    missing = target_offset - stream.tell()
    if missing < 0:
        raise RuntimeError("writer advanced beyond the planned section offset")
    if missing:
        stream.write(b"\x00" * missing)
