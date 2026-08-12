---
title: TCR
domain: News
year: 2018
sentences: "172"
granularity: inter-sentence
doi: 10.18653/V1/P18-1212
bib_key: ning:2018
hf_repo: thagen/TCR
hf_page: https://huggingface.co/datasets/thagen/TCR
polarity: none
strength: none
supported_tasks:
  causality-detection:
  causal-candidate-extraction:
  causality-identification:
---

# {{ page.meta.title }}

{{ dataset_badges() }}
{{ dataset_pills() }}

TCR (Temporal and Causal Reasoning) provides joint temporal and causal relation annotations over a small but densely annotated set of news articles, enabling reasoning that combines both relation types.

Extraction and identification rows are whole documents (see the granularity pill above): these 25 news articles were annotated specifically for long-range, cross-sentence causal reasoning, so a per-sentence schema would drop or corrupt a large share of the real links. Detection is derived at sentence granularity instead (via `causalatee.data.utils.identification_batch_to_detection_sentences`), since every document here was selected *for* containing causal chains and a whole-document label would otherwise never be negative.

TCR shares its 25 source documents with [EventCausality](EventCausality.md), but the two are **not** the same annotations — TCR re-extracted events (ClearTK) and independently re-annotated/pruned causal links (580 → 172) for a different paper. Don't conflate results across the two.

{{ dataset_overview() }}
{{ dataset_viewer() }}
{{ tira_leaderboard() }}
{{ dataset_citation() }}
