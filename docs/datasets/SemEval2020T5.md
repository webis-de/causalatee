---
title: SemEval-2020 Task 5
repo: https://github.com/arielsho/SemEval-2020-Task-5
domain: News
year: 2020
sentences: "20,000"
bib_key: yang:2020
polarity: none
strength: none
supported_tasks:
  causality-detection:
  causal-candidate-extraction:
  causality-identification:
  causality-extraction:
---

# {{ page.meta.title }}

{{ dataset_badges() }}
{{ dataset_pills() }}

SemEval-2020 Task 5 asks systems to recognize counterfactual statements in news text and, for causal ones, extract the antecedent (hypothetical condition) and consequent (hypothetical result) spans — mapped here onto this project's cause/effect roles.

Detection draws from all 20,000 sentences (2,192 causal); extraction and identification draw from a separate, smaller set of 5,501 sentences re-collected specifically for span annotation — about 14% of those have only an antecedent span with no separate consequent, so they contribute a marked entity but no relation.

Every sentence here is counterfactual/irrealis by construction — a distinct modality axis from both polarity and strength that this project's schema doesn't yet have a category for, currently flattened to a plain `Relation.Causal` with no flag at all.

{{ dataset_overview() }}
{{ dataset_viewer() }}
{{ tira_leaderboard() }}

## Citation

{{ bibtex_entry(page.meta.bib_key) }}
