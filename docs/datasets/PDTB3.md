---
title: Penn Discourse Treebank 3.0 (PDTB 3.0)
domain: News / WSJ
year: 2019
sentences: "—"
bib_key: webber:2019
hf_repo: thagen/PennDiscourseTreebankv3
hf_page: https://huggingface.co/datasets/thagen/PennDiscourseTreebankv3
supported_tasks:
  causality-detection:
  causal-candidate-extraction:
  causality-identification:
---

# {{ page.meta.title }}

{{ dataset_badges() }}
{{ dataset_pills() }}

PDTB 3.0 annotates discourse relations — including causal, temporal, and contingency senses — across the full Penn Treebank WSJ corpus using both explicit connectives and implicit relation spans.

!!! note "Requires your own paid LDC license — not redistributed by causalatee"
    PDTB 3.0 is a paid Linguistic Data Consortium resource
    ([LDC2019T05](https://catalog.ldc.upenn.edu/LDC2019T05)) — distinct from PDTB 2.0's
    ([LDC2008T05](https://catalog.ldc.upenn.edu/LDC2008T05), see [PDTB-2](PDTBv2.md)); owning one
    does not grant access to the other. causalatee does not fetch or host the underlying text —
    obtain your own license, follow [UniCausal](UniCausal.md)'s preprocessing recipe to produce
    `pdtb_train.csv`/`pdtb_test.csv`, then run `resources/datasets/PDTB3/conversion_script.py`
    yourself.

{{ dataset_overview() }}
{{ dataset_viewer() }}
{{ tira_leaderboard() }}
{{ dataset_citation() }}
