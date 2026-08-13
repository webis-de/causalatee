---
title: BECauSE 1.0
repo: https://github.com/duncanka/BECauSE
domain: News / Government
year: 2015
sentences: "—"
bib_key: dunietz:2015
polarity: documented
strength: none
supported_tasks: {}
---

# {{ page.meta.title }}

{{ dataset_badges() }}
{{ dataset_pills() }}

BECauSE 1.0 is the original release of the BECauSE corpus, annotating causal constructions across three source corpora. It predates [BECauSE 2.0](BECauSEv2.md), the actively-maintained ~20%-larger release this project already has.

Each causal instance carries a `Degree = FACILITATE | INHIBIT` attribute — `INHIBIT` marking causation that prevents or hinders its effect, functionally a countercausal signal — a scheme that originates in this 1.0 release itself and is carried over unchanged into BECauSE 2.0; it is not yet converted at all, since v1.0's conversion is fully LDC-blocked (see below).

!!! warning "Blocked — mostly LDC-gated, remaining portion too small"
    Of v1.0's three source corpora, only CongressionalHearings (3 documents) ships its raw text directly in the
    release. The other two — NYT (59 documents, New York Times Annotated Corpus) and PTB (47 documents, Penn
    Treebank sections 2–23) — require separate LDC subscriptions this project does not have; only their
    annotations, not the underlying text, are in the repo. A conversion limited to the 3 free documents would be
    far too small to be a meaningful dataset and would not be representative of this corpus's cited size. Not
    attempted — see `resources/datasets/BECauSEv1/conversion_script.py` for the full trail.

{{ dataset_overview() }}

## Citation

{{ bibtex_entry(page.meta.bib_key) }}
