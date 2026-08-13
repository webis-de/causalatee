---
title: SemEval-2007 Task 4
repo: https://github.com/davidsbatista/Annotated-Semantic-Relationships-Datasets
domain: Web
year: 2007
sentences: "220"
bib_key: girju:2007
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

SemEval-2007 Task 4 defines seven semantic relations between nominal pairs (including *Cause-Effect*) mined from web search results and asks systems to classify each pair. Only the Cause-Effect relation subset is used here.

Sentences already carry `<e1>`/`<e2>` markers in this project's own marker format, so no re-tagging was needed. The relation's argument order (`Cause-Effect(eX,eY)`) gives the cause/effect role assignment per sentence — it is not fixed to always be `(e2,e1)`, verified directly (10/140 training sentences use `(e1,e2)` instead).

{{ dataset_overview() }}
{{ dataset_viewer() }}
{{ tira_leaderboard() }}

## Citation

{{ bibtex_entry(page.meta.bib_key) }}
