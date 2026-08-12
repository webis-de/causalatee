---
title: EventStoryLine (ESL)
domain: News
year: 2017
sentences: "2,247"
doi: 10.18653/V1/W17-2711
bib_key: caselli:2017
polarity: documented
strength: none
hf_repo: thagen/EventStoryLine
hf_page: https://huggingface.co/datasets/thagen/EventStoryLine
supported_tasks:
  causality-detection:
    splits:
      train: 2014
      test: 233
  causal-candidate-extraction:
    splits:
      train: 780
      test: 84
  causality-identification:
    splits:
      train: 2014
      test: 233
---

# {{ page.meta.title }}

{{ dataset_badges() }}
{{ dataset_pills() }}

EventStoryLine annotates causal and temporal relations between events in 58 topics of ECB+ news clusters, targeting cross-document event coreference and storyline extraction.

The underlying ECB+ event markables distinguish negated event mentions (e.g. `NEG_ACTION_OCCURRENCE`) from their affirmative counterparts, but this converter currently ingests both undifferentiated, so that polarity signal is documented here without yet being surfaced in the converted relations.

causal-candidate-extraction is derived from the identification table above (`causalatee.data.utils.identification_batch_to_extraction`), restricted to the entities backing an actual causal relation — ESL2HF's direct ECB+ XML parser only implements detection and identification.

{{ dataset_overview() }}
{{ dataset_viewer() }}
{{ tira_leaderboard() }}
{{ dataset_citation() }}
