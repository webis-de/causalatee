# Causal Graph Format (CGF) v1.0

## Status and goals

CGF is a compact, memory-mappable format for directed causal graphs. It optimizes the operations needed by Causalatee: iterate nodes, resolve an exact node ID, iterate outgoing edges, optionally iterate incoming edges, and retrieve node or edge metadata without loading the graph into Python objects.

CGF 1.0 uses one metadata representation: independently encoded Avro binary records with an embedded writer schema for nodes and another for edges. There is no JSON metadata codec or codec-selection flag. This keeps the contract small while supporting typed, evolvable metadata.

The reference writer is deterministic and disk-backed. It uses a disposable SQLite spool for external ordering and separate temporary streams for final sections, so metadata and edge descriptors are not accumulated in RAM.

## Design summary

- Node IDs are unique UTF-8 strings, sorted by their encoded bytes. Exact lookup uses binary search; text search belongs in a separate index.
- Nodes are addressed internally by dense zero-based ordinals.
- Outgoing topology is compressed sparse row (CSR): `OUT_OFFSETS` plus `EDGE_TARGETS`.
- An optional reverse CSR index enables fast incoming traversal.
- Variable-length IDs and metadata use offset tables, preserving constant-time random access and direct memory mapping.
- Node and edge metadata are schemaless Avro binary records, each delimited by the corresponding offset table.
- The node and edge Avro writer schemas are stored once per file.
- All structural integers are fixed-width little-endian `u64`. VarInts are intentionally excluded from mapped arrays because they prevent constant-time indexing and add branch-heavy decoding. Avro may use its own compact integer encoding inside metadata records.

## Primitive rules

- Byte order: little-endian.
- Alignment: every section payload begins at an 8-byte boundary.
- Strings: UTF-8 without a terminator.
- Offsets: byte offsets relative to the start of the paired data section.
- Ordinals and edge indices: unsigned 64-bit integers.
- Unknown flags, required-section omissions, duplicate section types, invalid ranges, and non-zero reserved fields are errors.

## File header

The fixed 64-byte header uses this layout:

| Offset | Size | Field | Meaning |
|---:|---:|---|---|
| 0 | 8 | magic | `CGF\r\n\x1a\n\x00` |
| 8 | 2 | major | `1` |
| 10 | 2 | minor | `0` |
| 12 | 1 | byte order | `1` for little-endian |
| 13 | 1 | flags | See below |
| 14 | 2 | header size | `64` |
| 16 | 8 | node count | `N` |
| 24 | 8 | edge count | `M` |
| 32 | 4 | section count | Number of directory entries |
| 36 | 2 | entry size | `32` |
| 38 | 2 | reserved | Zero |
| 40 | 8 | directory offset | Normally `64` |
| 48 | 8 | declared file size | Exact byte length |
| 56 | 8 | reserved | Zero padding |

Flag bit 0 is `HAS_INCOMING`. Bits 1â€“7 are reserved and must be zero.

## Section directory

Each 32-byte entry is `(type:u32, flags:u32, offset:u64, length:u64, count:u64)`. Section flags are zero in version 1.0. Section ranges must be aligned, in bounds, and non-overlapping.

| Type | Name | Required | Contents |
|---:|---|---|---|
| 1 | `NODE_ID_OFFSETS` | Yes | `N + 1` `u64` byte offsets |
| 2 | `NODE_ID_DATA` | Yes | Concatenated UTF-8 node IDs |
| 3 | `NODE_METADATA_OFFSETS` | Yes | `N + 1` `u64` offsets |
| 4 | `NODE_METADATA_DATA` | Yes | Concatenated Avro node records |
| 5 | `OUT_OFFSETS` | Yes | `N + 1` CSR offsets |
| 6 | `EDGE_TARGETS` | Yes | `M` target ordinals |
| 7 | `EDGE_METADATA_OFFSETS` | Yes | `M + 1` offsets |
| 8 | `EDGE_METADATA_DATA` | Yes | Concatenated Avro edge records |
| 9 | `IN_OFFSETS` | With flag | `N + 1` reverse-CSR offsets |
| 10 | `IN_SOURCES` | With flag | `M` source ordinals |
| 11 | `IN_EDGE_INDICES` | With flag | `M` outgoing-edge indices |
| 12 | `NODE_METADATA_SCHEMA` | Yes | One UTF-8 Avro schema JSON document |
| 13 | `EDGE_METADATA_SCHEMA` | Yes | One UTF-8 Avro schema JSON document |

Offset arrays start at zero, are monotonic, and end at the paired data length (or at `M` for CSR arrays). Empty records are valid when the schema permits them.

## Ordering and topology

Nodes are sorted by UTF-8 ID bytes. Edges are sorted stably by `(source ordinal, target ordinal, input order)`. For node `u`, outgoing edge indices are:

```python
range(OUT_OFFSETS[u], OUT_OFFSETS[u + 1])
```

`EDGE_TARGETS[e]` is the target ordinal. A source ordinal can be recovered by binary-searching `OUT_OFFSETS`, though an iterator should retain the source as a handle hint.

When present, incoming entries are sorted by `(target ordinal, source ordinal, outgoing edge index)`. `IN_EDGE_INDICES` identifies the canonical outgoing edge and therefore its metadata record; metadata is never duplicated.

## Avro metadata

Each schema section contains a complete Avro writer schema as UTF-8 JSON. Each metadata data section concatenates independently encoded Avro binary records. CGF does not use the Avro Object Container File format: there are no block headers, sync markers, or repeated schemas because CGF's own offsets and schema sections provide framing.

Graph adapters should expose:

```python
@property
def node_metadata_schema(self) -> Mapping[str, object] | str | None: ...

@property
def edge_metadata_schema(self) -> Mapping[str, object] | str | None: ...
```

`save_cgf` resolves each schema independently: an explicit function argument wins, otherwise the graph property is used, and absence is an error. This lets `save_cgf(load_cgf(path), other_path)` retain the embedded schemas.

Avro records should use native Avro types for stable, frequently accessed fields. Nullable or optional values should be represented with Avro unions and defaults. Schema names and namespaces should be stable and resource-specific.

### Irregular JSON-shaped values

Some source resources contain irregular nested values that do not justify a large union schema. CGF defines the custom Avro logical type `causalatee.json` on an Avro `string`:

```json
{"type": "string", "logicalType": "causalatee.json"}
```

The writer converts the Python value to canonical compact JSON; the Causalatee reader restores the JSON value. Generic Avro implementations that do not recognize the logical type safely see an ordinary string. This is an escape hatch, not the default representation: native Avro fields remain smaller, more explicit, and easier to validate.

For CauseNet, node metadata is an empty record. Edge metadata is fully native Avro: `sources` is an array of records containing a source `type` and a `map<string>` payload, while `support` is a `long` containing the number of supporting source records. CauseNet's documented provenance payload values are strings, so this preserves the source objects without repeating JSON syntax. A later adapter version could replace the payload map with a union of source-specific records if benchmarks justify the added schema and conversion complexity.

## Python API

Normal usage relies on schemas supplied by the graph adapter:

```python
from causalatee.graph import load_causenet, load_cgf, save_cgf

graph = load_causenet("causenet-precision.jsonl.bz2", limit=10_000)
save_cgf(graph, "causenet.cgf")

with load_cgf("causenet.cgf", validate=True) as mapped:
    rain = mapped.get_node("rain")
    print(mapped.node_metadata_schema)
    for edge in rain.outgoing_edges():
        print(edge.target.id, edge.metadata)
```

Callers can override either adapter schema:

```python
save_cgf(
    graph,
    "graph.cgf",
    node_metadata_schema=node_schema,
    edge_metadata_schema=edge_schema,
    include_incoming=True,
)
```

The reference implementation uses `fastavro` for schema parsing and schemaless record I/O. Opening and traversing topology is memory mapped; `fastavro` is needed when writing or decoding/validating metadata.

## Writer architecture

`save_cgf` is a multi-stage, bounded-memory conversion:

1. It parses the graph-provided node and edge Avro schemas once.
2. It streams nodes into a temporary SQLite table, encoding each metadata record immediately. The table's byte-sorted primary key detects duplicate IDs and provides deterministic node order.
3. It emits node-ID, node-metadata, and offset sections to separate temporary files while assigning dense ordinals in bounded update batches.
4. It streams edges into the SQLite spool after resolving endpoint IDs to ordinals. Edge metadata is encoded immediately rather than retained as Python objects.
5. SQLite performs a disk-backed order by `(source, target, input order)`. The writer consumes that cursor once to emit outgoing CSR, targets, metadata, and metadata offsets. It records the resulting canonical edge indices in bounded batches.
6. If requested, a second disk-backed order by `(target, source, edge index)` emits the incoming CSR streams without duplicating edge metadata.
7. Once every section length is known, the writer creates one temporary destination, copies each section in fixed-size chunks, flushes it, and atomically replaces the requested path.

The implementation sets SQLite temporary storage to file and caps its page cache, so working memory does not grow with the number or total metadata size of edges. It still requires temporary disk capacity for the SQLite spool, its external-sort runs, the staged sections, and the final temporary CGFâ€”potentially several times the output size at peak. The supplied graph may itself be eager, but the writer does not add another full in-memory graph representation.

Node lookup during conversion is disk-backed as well; no unbounded Python ID-to-ordinal dictionary is required. A bounded least-recently-used cache retains up to 65,536 endpoint ordinals, accelerating graphs whose nodes recur across nearby edges without making memory proportional to graph size. This favors predictable memory use over an unrestricted in-memory lookup table.

## Validation and safety

Full validation checks structural bounds and counts, monotonic offsets, sorted unique UTF-8 IDs, endpoint ordinals, incoming references, valid schema JSON, and successful Avro decoding of every record. Readers should bound allocations and recursion performed by their Avro implementation and should treat metadata as untrusted data rather than executable content.

## Space and performance considerations

The primary CSR arrays provide sequential scans and direct outgoing-edge access with excellent locality. Fixed-width `u64` costs more than VarInts for small values but enables zero-copy views and constant-time element lookup. Metadata offsets isolate a single Avro record, so retrieving one record touches only its offset entries and byte range. The optional incoming index costs approximately 24 bytes per edge plus `(N + 1) * 8` bytes but avoids an `O(M)` scan.

If analytical column scans later become important, Arrow or Parquet should be a separate sidecar keyed by node ordinal or edge index. Embedding a second metadata model in CGF would add implementation and compatibility burden without improving the core graph traversal path.