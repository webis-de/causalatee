---
title: EventStoryLine (ESL)
domain: News
year: 2017
sentences: "2,247"
doi: 10.18653/V1/W17-2711
bib_key: caselli:2017
hf_repo: thagen/EventStoryLine
hf_page: https://huggingface.co/datasets/thagen/EventStoryLine
supported_tasks:
  causality-detection:
    splits:
      train: 2014
      test: 233
  causality-identification:
    splits:
      train: 2014
      test: 233
---

# {{ page.meta.title }}

{{ dataset_badges() }}

EventStoryLine annotates causal and temporal relations between events in 58 topics of ECB+ news clusters, targeting cross-document event coreference and storyline extraction.

{{ dataset_overview() }}
{{ dataset_viewer() }}
{{ tira_leaderboard() }}
{{ dataset_citation() }}
