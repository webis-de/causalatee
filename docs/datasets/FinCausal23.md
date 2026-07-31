---
title: FinCausal 2023 (FinCausal-23)
domain: Financial news
year: 2023
sentences: "2,630"
granularity: inter-sentence
supported_tasks:
  causal-candidate-extraction:
  causality-identification:
---

<!-- TODO(user): add the FinCausal 2023 bibtex entry to docs/references/datasets.bib
     (Moreno-Sandoval et al., IEEE BigData 2023, "The Financial Document Causality
     Detection Shared Task", also arXiv:2401.13545), then set bib_key: in the
     frontmatter above and add {{ dataset_citation() }} to the bottom of this page. -->

# {{ page.meta.title }}

{{ dataset_badges() }}
{{ dataset_pills() }}

FinCausal 2023 annotates cause/effect spans in excerpts from financial disclosures and news, continuing [FinCausal 2020](FinCausal20.md)'s shared task with an expanded, re-annotated dataset.

Unlike FinCausal 2020, this release's public data is entirely pre-filtered causal segments — there are no non-causal examples at all, so causality-detection is **not** listed as a supported task above: a detection table built from this dataset alone would be single-class (every row Causal) and meaningless to train or evaluate on in isolation. The conversion script still writes `causality-detection/{train,test}.parquet` anyway (derived from the identification table via `causalatee.data.utils.identification_batch_to_detection`) so it's there for anyone who wants to pool it with another dataset's negatives for a combined detection table — it's just not advertised as a first-class supported task here, and this project's own evaluation sweep excludes it. The official host is a CodaLab competition gated behind shared-task registration; this project instead converts from a participant's public mirror of the labeled training data, since the organizers never released the equivalent of FinCausal 2020's plain download — see the conversion script for the exact provenance and the train/test split this project had to invent (no split exists in the source beyond an unlabeled blind test).

{{ dataset_overview() }}
{{ dataset_viewer() }}
{{ tira_leaderboard() }}
