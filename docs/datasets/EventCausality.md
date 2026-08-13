---
title: EventCausality
repo: https://github.com/phosseini/CREST
domain: Web
year: 2011
sentences: "583"
bib_key: do:2011
hf_repo: thagen/EventCausality
hf_page: https://huggingface.co/datasets/thagen/EventCausality
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
    corruption/completeness bug (0 train rows) and a confirmed character-offset misalignment
    affecting causal-candidate-extraction and causality-identification alike (~21% of spans are
    fully out-of-bounds, now dropped rather than silently shipped) — treat results on this
    dataset with those caveats in mind.

{{ dataset_overview() }}
{{ dataset_viewer() }}
{{ tira_leaderboard() }}

## Citation

{{ bibtex_entry(page.meta.bib_key) }}
