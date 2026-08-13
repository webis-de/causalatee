---
title: Countercausal News Corpus (CCNC)
repo: https://github.com/webis-de/arxiv-countercausality
domain: News
year: 2025
sentences: "3,415"
bib_key: hagen:2025a
note: The first dataset to explicitly distinguish causal, countercausal, and uncausal statements.
polarity: captured
strength: none
supported_tasks:
  causality-detection:
    tiraurl: https://www.tira.io/api/evaluations/causality-extraction-toucheclef26/task1-ccnc-test-dataset-20260605-test
  causal-candidate-extraction:
  causality-identification:
  causality-extraction:
---

# {{ page.meta.title }}

{{ dataset_badges() }}
{{ dataset_pills() }}

CCNC is the first corpus to explicitly distinguish *causal*, *countercausal*, and *uncausal* sentences, revealing that countercausal patterns are a major source of error in causality detection models (Cohen's κ = 0.74).

Converted directly from the paper's own repository ([github.com/webis-de/arxiv-countercausality](https://github.com/webis-de/arxiv-countercausality)), not the [Touché@CLEF26](https://touche.webis.de/clef26/touche26-web/causality-extraction.html) shared task's TIRA-hosted variant of the same corpus.

{{ dataset_overview() }}
{{ dataset_viewer() }}
{{ tira_leaderboard() }}

## Citation

{{ bibtex_entry(page.meta.bib_key) }}
