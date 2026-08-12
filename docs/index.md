<p align="center">
   <img width=200px src="assets/icon.png"/>
</p>

`causalatee` is a Python library to simplify handling causality in natural language: extraction,
training extraction models, unified datasets, and storing/querying results as causal graphs.

## Installation

```bash
pip install causalatee
```

## Where to go from here

- **[Tasks](tasks/causality_detection.md)** — the three standardized tasks (Detection, Candidate
  Extraction, Identification) that every dataset and model interface below shares.
- **[Datasets](datasets/index.md)** — causality corpora, converted into one consistent
  HuggingFace-compatible schema.
- **[Models](models/index.md)** — model approaches for the tasks above, from rule-based
  baselines to fine-tuned neural classifiers, all satisfying the same batch-callable
  [`causalatee.models`](reference/models.md) `Protocol` interfaces.
- **[Causal Graphs](graphs/causalgraphs.md)** — storing and querying causal relations as a graph,
  with several interchangeable backends (`CauseNet`, `CGF`, `SQLGraph`, ...).
- **[Examples](examples/index.md)** — runnable notebooks covering fine-tuning, baselines, and
  building/mining a causal graph.
- **[API Reference](reference/index.md)** — the full `causalatee` API.
