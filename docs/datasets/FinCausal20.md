---
title: FinCausal 2020 (FinCausal-20)
repo: https://github.com/yseop/YseopLab
domain: Finance
year: 2020
sentences: "22,058"
bib_key: mariko:2020
granularity: inter-sentence
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

FinCausal annotates causal relations in financial disclosures and news, targeting quantified facts where a cause leads to a measurable financial effect. Converted directly from the FNP 2020 shared task organizers' own repository, using only the "practice" (train) and "trial" (test) splits — the "evaluation" split is an unlabeled blind test set that was never republished with gold labels.

Each row is a whole text section (see the granularity pill above), not a single sentence: causal relations can genuinely span multiple sentences within one section, so a per-sentence schema would risk dropping or corrupting them — the same reasoning already applied to [BioCause](BioCause.md) and [TCR](TCR.md).

{{ dataset_overview() }}
{{ dataset_viewer() }}
{{ tira_leaderboard() }}

## Citation

{{ bibtex_entry(page.meta.bib_key) }}
