---
title: EventCausality
domain: Web
year: 2011
sentences: "583"
bib_key: do:2011
hf_repo: thagen/EventCausality
hf_page: https://huggingface.co/datasets/thagen/EventCausality
supported_tasks:
  causality-detection:
  causal-candidate-extraction:
  causality-identification:
---

# {{ page.meta.title }}

{{ dataset_badges() }}
{{ dataset_pills() }}

EventCausality provides minimally-supervised annotations of causal relations between events collected from web text, using a small set of seed patterns to bootstrap annotation.

!!! warning "No direct source available — data quality unresolved"
    Unlike most datasets in this toolkit, EventCausality could **not** be moved off the CREST
    aggregation to a direct source. The original release (Do, Chan & Roth, EMNLP 2011 — 580
    manually-annotated causal links on 25 CNN articles) was never publicly archived anywhere
    reachable: the paper only promises a future release via a CogComp resource page that is now
    dead. [TCR](TCR.md) covers the same 25 source documents but is a **different, later
    re-annotation** (different event set, different label scheme, 172 vs. 580 links) — using it
    here would silently conflate two distinct datasets under one citation, so this repo does not
    do that. The data below still comes from CREST's own aggregation, which has a confirmed
    corruption/completeness bug (0 train rows) — treat results on this dataset with that caveat
    in mind.

{{ dataset_overview() }}
{{ dataset_viewer() }}
{{ tira_leaderboard() }}
{{ dataset_citation() }}
