---
title: SemEval-2020 Task 5
domain: News
year: 2020
sentences: "20,000"
supported_tasks:
  causality-detection:
  causal-candidate-extraction:
  causality-identification:
---

<!-- TODO(user): add the SemEval-2020 Task 5 bibtex entry to docs/references/datasets.bib
     (Yang et al., SemEval-2020, "SemEval-2020 Task 5: Counterfactual Recognition",
     https://aclanthology.org/2020.semeval-1.40/), then set bib_key: in the
     frontmatter above and add {{ dataset_citation() }} to the bottom of this page. -->

# {{ page.meta.title }}

{{ dataset_badges() }}
{{ dataset_pills() }}

SemEval-2020 Task 5 asks systems to recognize counterfactual statements in news text and, for causal ones, extract the antecedent (hypothetical condition) and consequent (hypothetical result) spans — mapped here onto this project's cause/effect roles.

Detection draws from all 20,000 sentences (2,192 causal); extraction and identification draw from a separate, smaller set of 5,501 sentences re-collected specifically for span annotation — about 14% of those have only an antecedent span with no separate consequent, so they contribute a marked entity but no relation.

{{ dataset_overview() }}
{{ dataset_viewer() }}
{{ tira_leaderboard() }}
