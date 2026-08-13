---
title: Causal News Corpus V2 / RECESS (CNCv2)
repo: https://github.com/tanfiona/CausalNewsCorpus
domain: News
year: 2023
sentences: "3,415"
bib_key: tan:2023
hf_repo: thagen/CausalNewsCorpusV2
polarity: none
strength: none
hf_page: https://huggingface.co/datasets/thagen/CausalNewsCorpusV2
supported_tasks:
  causality-detection:
  causal-candidate-extraction:
  causality-identification:
  causality-extraction:
---

# {{ page.meta.title }}

{{ dataset_badges() }}
{{ dataset_pills() }}

The "V2" release of the Causal News Corpus — published as RECESS — annotates causal event sentences drawn from news, with span-level cause/effect/signal annotations for a much larger share of causal sentences than the original 2022 release.

This is the actively-maintained release the maintainers themselves recommend using ("For 2023 Shared Task, please use V2"), with far richer span annotations (2257 causal relations) than [CNC](CNC.md), the original 2022 release (183 causal relations) — kept as its own separate dataset for comparison rather than silently overwritten.

{{ dataset_overview() }}
{{ dataset_viewer() }}
{{ tira_leaderboard() }}

## Citation

{{ bibtex_entry(page.meta.bib_key) }}
