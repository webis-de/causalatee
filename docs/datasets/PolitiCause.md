---
title: PolitiCause
domain: Politics
year: 2024
sentences: "17,780"
polarity: none
strength: none
supported_tasks:
  causality-detection:
  causal-candidate-extraction:
  causality-identification:
---

<!-- TODO(user): add the PolitiCause bibtex entry to docs/references/datasets.bib
     (Garcia Corral et al., LREC-COLING 2024, "PolitiCause: An Annotation Scheme
     and Corpus for Causality in Political Texts",
     https://aclanthology.org/2024.lrec-main.1124/), then set bib_key: in the
     frontmatter above and add {{ dataset_citation() }} to the bottom of this page. -->

# {{ page.meta.title }}

{{ dataset_badges() }}
{{ dataset_pills() }}

PolitiCause annotates causal relations in political speech transcripts — UN General Debate statements and UK government press conferences — focusing on the domain-specific challenges of political discourse such as attribution, hedging, and complex multi-clause sentences.

Unlike its own paper (which only reports sentence-level classification results), causal-candidate-extraction and causality-identification here are derived from the dataset's underlying multi-annotator span file, reconciled into a single canonical cause/effect pair per sentence — see the conversion script's module docstring for the exact reconciliation policy and its ~9% coverage loss from genuinely incomplete source annotations.

{{ dataset_overview() }}
{{ dataset_viewer() }}
{{ tira_leaderboard() }}
