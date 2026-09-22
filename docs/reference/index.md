# API Reference

Auto-generated from docstrings in the `causalatee` package.

| Module | Description |
|--------|-------------|
| [`causalatee.data.constants`](constants.md) | Shared enums every task/dataset/model uses: `Task`, `ClassLabel`, `Relation` (a relation *type* — see the [glossary](../glossary.md)). |
| [`causalatee.data.utils`](data_utils.md) | Convert between inter-sentence (whole-document/section) and intra-sentence (per-sentence) causality data — marker parsing, sentence splitting, `Dataset.map()`-ready batch adapters. |
| [`causalatee.graph`](graph.md) | Typed `Graph`/`Node`/`Edge` interface, an eager CauseNet loader, and CGF — a compact, memory-mappable binary format for causal graphs. |
| [`causalatee.models`](models.md) | Structural `Protocol` interfaces for causality models — one per task, plus pairwise and end-to-end variants. |
| [`causalatee.mining`](mining.md) | A streaming, concurrency-aware pipeline for turning a document corpus into an aggregated causal graph. |
