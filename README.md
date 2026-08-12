<p align="center">
   <img width=200px src="docs/assets/icon.png"/>
</p>

<p align="center">
  A Python library to simplify handling causality in natural language: extraction, training
  extraction models, unified datasets, and storing/querying results as causal graphs.
</p>

<p align="center">
  <a href="https://pypi.org/project/causalatee/"><img alt="PyPI" src="https://img.shields.io/pypi/v/causalatee"></a>
  <a href="https://causalatee.webis.de/"><img alt="Docs" src="https://img.shields.io/badge/docs-causalatee.webis.de-blue"></a>
  <img alt="Python" src="https://img.shields.io/pypi/pyversions/causalatee">
  <a href="https://github.com/webis-de/causalatee/actions/workflows/tests.yml"><img alt="Tests" src="https://github.com/webis-de/causalatee/actions/workflows/tests.yml/badge.svg"/></a>
  <a href="https://github.com/webis-de/causalatee/actions/workflows/linter.yml"><img alt="Linter" src="https://github.com/webis-de/causalatee/actions/workflows/linter.yml/badge.svg"/></a>
</p>

## What is this?

`causalatee` covers the full lifecycle of working with causality in text:

- **Datasets** — causality corpora, converted into one consistent HuggingFace-compatible schema
  across three standardized tasks:
  - **Causality Detection** — does a sentence express a causal relation at all?
  - **Causal Candidate Extraction** — which spans in a sentence are causes/effects?
  - **Causality Identification** — given two marked spans, does a causal relation hold between them?
- **`causalatee.models`** — batch-callable [`typing.Protocol`][protocol] interfaces
  (`Detection`, `CandidateExtraction`, `PairwiseIdentification`, `Identification`, `Extraction`)
  that any conforming model (rule-based, HuggingFace, or otherwise) satisfies with zero
  inheritance, plus `compose_extraction` to build an end-to-end extractor from the three
  sub-tasks.
- **`causalatee.graph`** — a typed `Graph`/`Node`/`Edge` interface with several interchangeable
  backends: an eager [CauseNet](https://causenet.org) loader, **CGF** (a compact,
  memory-mappable on-disk format), a [CausalBank](https://github.com/eecrazy/CausalBank)
  Cause-Effect Graph loader, and `SQLGraph`, a generic mutable graph you build yourself via
  `add_node`/`add_edge`.
- **`causalatee.mining`** — a streaming, concurrency-aware pipeline
  (`source -> flat_map -> filter -> map -> map -> reduce`) that turns a corpus of raw documents
  into an aggregated causal graph, without materializing the corpus in memory.
- **`causalatee.nn`** / **`causalatee.integrations`** — a biaffine span-grid extraction head, and
  ready-made HuggingFace `Pipeline` / PyTorch Lightning integrations for the three tasks above.

[protocol]: https://docs.python.org/3/library/typing.html#typing.Protocol

See **[causalatee.webis.de](https://causalatee.webis.de/)** for the full documentation,
including the dataset inventory, task/model reference, and runnable example notebooks.

## Installation

```bash
pip install causalatee
```

Optional extras pull in dependencies for specific pieces: `huggingface` (fine-tuning/inference
pipelines), `baselines` (dependency-parse baselines), `mining` (the corpus-mining pipeline), and
`docs` (building this documentation locally).

## Quick start

Every converted dataset is a standard HuggingFace dataset, indexed by task:

```python
from datasets import load_dataset

dataset = load_dataset("thagen/CausalNewsCorpus", "causality identification")
```

See the [Datasets](https://causalatee.webis.de/datasets/) page for the full list, and the
[Examples](https://causalatee.webis.de/examples/) notebooks for fine-tuning a model, building a
causal graph, and mining one from a raw corpus.

## Development

```bash
pip install -e ".[tests]"
ruff check .
mypy -p causalatee
pytest
```

## Building the Documentation

```bash
pip install -e ".[docs]"
mkdocs serve       # live-reloading local server
mkdocs build       # static site written to site/
```
