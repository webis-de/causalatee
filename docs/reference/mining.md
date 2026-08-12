# causalatee.mining

A streaming, concurrency-aware pipeline for turning a corpus of documents into an
aggregated causal graph: `source -> flat_map -> filter -> map -> map -> reduce`.

Nothing is materialized between stages — an entire corpus, however large, streams
through a bounded number of items in flight at once. Sync and async stages compose
uniformly; a stage backed by a single GPU model (any
[`causalatee.models`](models.md) protocol instance) should run with `concurrency=1`
(serialized calls, batched internally via `batch_size`), while I/O-bound stages
(fetching documents) can run with much higher `concurrency`.

## Quick start

```python
from causalatee.mining import Document, Pipeline, causal_predicate, graph_sink
from causalatee.graph import save_cgf

async def source():
    async for post in fetch_posts():                       # user-supplied, I/O-bound
        yield Document(id=post.id, text=post.title)

graph = await (
    Pipeline(source())
    .flat_map(split_into_sentences, concurrency=8)          # CPU-bound; fan out
    .filter(detection_model, batch_size=64, concurrency=1,   # 1 GPU: internal batching
            predicate=causal_predicate)
    .map(candidate_extraction_model, batch_size=64, concurrency=1)  # same GPU
    .map(identification_model, batch_size=64, concurrency=1)        # same GPU
    .reduce(graph_sink())
)
save_cgf(graph, "mined.cgf")
```

`identification_model` here is a [`PairwiseIdentification`][causalatee.models.PairwiseIdentification];
each mapped text must already carry candidate-span markers (`<e0>`, `<e1>`, ...) for
this to work directly — see
[`causalatee.models.identify_candidates`][causalatee.models.identify_candidates] for
the marking/enumeration/remapping glue, shared with
[`compose_extraction`][causalatee.models.compose_extraction], that a real
identification stage function should wrap around the raw model.

## Why not just materialize the corpus and loop?

Because a real corpus doesn't fit in memory, and a naive `for document in corpus:
run_everything(document)` loop runs every stage sequentially per item — the GPU sits
idle while the next document is fetched, and the next document isn't fetched until
the current one finishes identification. `Pipeline` overlaps stages (document N+1 is
being fetched while document N is mid-identification) and lets each stage set its
own concurrency independently, while guaranteeing a `concurrency=1` GPU stage never
receives more than one call at a time regardless of how concurrent its neighboring
stages are.

## Reducing into a graph

[`graph_sink`][causalatee.mining.graph_sink] is both the `Pipeline.reduce` sink and
(after `reduce` completes) a [`causalatee.graph.Graph`][causalatee.graph.Graph] —
`await pipeline.reduce(graph_sink())` returns something directly usable with
[`save_cgf`][causalatee.graph.save_cgf]. Every relation the pipeline emits is spooled
to a temp SQLite database as it arrives (bounded memory regardless of corpus size);
aggregation by `(cause, effect)` pair happens once, lazily, the first time `.nodes`/
`.edges` is accessed, and populates a
[`SQLGraph`][causalatee.graph.SQLGraph] with one `add_edge` call per aggregated pair
— `GraphSink` only owns the spooling/aggregation, `SQLGraph` is the generic,
reusable storage engine. Close it (or use it as a context manager) once done with the
resulting graph, the same way [`load_cgf`][causalatee.graph.load_cgf] is used.

## Example

See the [Mining a Causal Graph](../examples/mining.ipynb) notebook for a runnable
walkthrough of the full pipeline -> `graph_sink` -> `save_cgf` flow, using a
deterministic stand-in extraction function so it runs without a real model.

## API Reference

::: causalatee.mining
    options:
      members_order: source
      show_if_no_docstring: false
      filters: ["!^_"]
