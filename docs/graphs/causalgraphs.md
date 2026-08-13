# Causal Graphs

Causality extraction (see [Tasks](../tasks/causality_detection.md)) identifies causal relationships claimed in natural
language text. **Causal graphs** aggregate such relations (one edge per
`cause → effect` pair) into a knowledge graph that can be queried, traversed, and used as
background knowledge (for example to look up whether a candidate cause/effect pair is already
attested, or to expand a seed concept along its causal neighborhood).

`causalatee.graph` provides:

- A small, typed [`Graph`][causalatee.graph.Graph] / [`Node`][causalatee.graph.Node] / [`Edge`][causalatee.graph.Edge] interface that every backend below implements, so code
  written against it works regardless of how a graph is stored.
- [`load_causenet`][causalatee.graph.load_causenet], an eager loader for the
  [CauseNet](https://causenet.org) causality graph and other
  compatible JSONL resources.
- [`load_cause_effect_graph`][causalatee.graph.load_cause_effect_graph], a loader for the Lexical
  Cause-Effect Graph (CEG) format, mined at CausalBank scale.
- **CGF** (Causal Graph Format), a compact, memory-mappable binary format for storing a `Graph` on
  disk and traversing it without materializing it in Python objects — see the
  [CGF 1.0 specification](cgf_spec.md) for the full format.
- [`SQLGraph`][causalatee.graph.SQLGraph], a generic, mutable, SQLite-backed `Graph` you build
  yourself via `add_node`/`add_edge` — unlike the loaders above, it has no external source to load
  from (`causalatee.mining.GraphSink` uses one as its storage engine).

## Quick start

```python
from causalatee.graph import load_causenet, save_cgf, load_cgf

# Eagerly load a (bounded) slice of CauseNet into memory.
graph = load_causenet("causenet-precision.jsonl.bz2", limit=10_000)

# Convert it to CGF. The writer streams through a disk-backed sort, so memory
# use does not grow with the number of edges.
save_cgf(graph, "causenet.cgf")

# Memory-map the CGF file: nodes and edges are resolved lazily from disk.
with load_cgf("causenet.cgf", validate=True) as mapped:
    rain = mapped.get_node("rain")
    for edge in rain.outgoing_edges():
        print(edge.target.id, edge.metadata["support"])
```

`Graph` implementations are interchangeable: `save_cgf` accepts *any* `Graph`, not just a
`CauseNet` instance, and a `CGFGraph` loaded back with `load_cgf` is itself a `Graph` that can be
passed to `save_cgf` again (e.g. to add or drop the incoming-edge index).

## Choosing a backend

| | `CauseNet` | `CGFGraph` |
|---|---|---|
| Loaded from | Local or remote JSONL (optionally gzip/bzip2) | A local `.cgf` file |
| Memory use | Whole selection materialized as Python objects | Memory-mapped; independent of graph size |
| Node/edge lookup | Python `dict` | Binary search / CSR offsets |
| Best for | Building a graph, small samples, one-off exploration (`limit=...`) | Repeatedly querying or traversing a large graph |

In short: use `load_causenet` to *build* a graph from the source data, then `save_cgf` to persist
it once for fast, low-memory access afterwards via `load_cgf`.

## Example

See the [CauseNet and CGF](../examples/causenet_cgf.ipynb) notebook for a runnable walkthrough of
loading a CauseNet sample, saving it as CGF, and loading it back.

## Supported Causal Graphs

{{ graph_cards() }}

## API Reference

See [`causalatee.graph`](../reference/graph.md) for the full API.
