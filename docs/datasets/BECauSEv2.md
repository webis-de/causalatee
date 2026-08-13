---
title: BECauSE 2.0
repo: https://github.com/tanfiona/UniCausal
domain: News
year: 2017
sentences: "1,803"
bib_key: dunietz:2017
polarity: none
strength: none
hf_repo: thagen/BECauSEv2
hf_page: https://huggingface.co/datasets/thagen/BECauSEv2
supported_tasks:
  causality-detection:
    splits:
      train: 453
      test: 17
  causal-candidate-extraction:
    splits:
      train: 453
      test: 17
  causality-identification:
    splits:
      train: 453
      test: 17
  causality-extraction:
---

# {{ page.meta.title }}

{{ dataset_badges() }}
{{ dataset_pills() }}

BECauSE 2.0 annotates causal relations (three types: *Consequence*, *Motivation*, *Purpose*) alongside seven non-causal "overlapping" relations that share causal-like constructions (e.g. *Correlation*, *Hypothetical*) in English news text, making it one of the most relation-dense corpora for causal NLP.

The original annotation also tags each causal instance's `Degree` as `Facilitate` or `Inhibit`, but `Inhibit` ("the cause suppresses the effect") is negative-valence causation, not a denial that a causal relationship holds at all — so it isn't represented as `Relation.Countercausal` here (see the dataset's README for the full reasoning).

{{ dataset_overview() }}
{{ dataset_viewer() }}
{{ tira_leaderboard() }}

## Citation

{{ bibtex_entry(page.meta.bib_key) }}
