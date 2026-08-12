---
title: Causal News Corpus (CNC)
domain: News
year: 2022
sentences: "3,248"
bib_key: tan:2022a
polarity: none
strength: none
hf_repo: thagen/CausalNewsCorpus
hf_page: https://huggingface.co/datasets/thagen/CausalNewsCorpus
supported_tasks:
  causality-detection:
  causal-candidate-extraction:
  causality-identification:
---

# {{ page.meta.title }}

{{ dataset_badges() }}
{{ dataset_pills() }}

The Causal News Corpus annotates causal event sentences drawn from protest-event news and provides span-level cause/effect annotations for a subset of causal sentences.

This is the original 2022 ("V1") release, converted directly from the source repository. Its span (Cause/Effect) annotations are sparse — only 183 of its causal relations have spans at all. See [CNCv2](CNCv2.md) for "V2" / RECESS, the actively-maintained 2023 release with far richer span coverage (2257 relations), which the maintainers now recommend using.

{{ dataset_overview() }}
{{ dataset_viewer() }}
{{ tira_leaderboard() }}
{{ dataset_citation() }}
