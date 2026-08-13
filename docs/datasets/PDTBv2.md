---
title: Penn Discourse Treebank 2.0 (PDTB-2)
domain: News / WSJ
year: 2008
sentences: "—"
bib_key: prasad:2008
granularity: intra-sentence
polarity: none
strength: none
supported_tasks: {}
---

# {{ page.meta.title }}

{{ dataset_badges() }}
{{ dataset_pills() }}

PDTB-2.0 annotates discourse relations — including causal, temporal, and contingency senses — across the Penn Treebank WSJ corpus, using explicit connectives and implicit relation spans. It is the predecessor to [PDTB 3.0](PDTB3.md).

!!! warning "Blocked — separate paid LDC license, no existing preprocessing recipe"
    PDTB-2.0 is gated behind its own paid Linguistic Data Consortium license
    ([LDC2008T05](https://catalog.ldc.upenn.edu/LDC2008T05)) — distinct from PDTB 3.0's
    ([LDC2019T05](https://catalog.ldc.upenn.edu/LDC2019T05)); owning one does not grant access to
    the other. Unlike PDTB 3.0, [UniCausal](UniCausal.md) has no existing preprocessing recipe for
    PDTB-2.0's (different, earlier) annotation format, so even with a license, someone would need to
    write a PDTB-2-specific converter from scratch. Not yet attempted — see
    `resources/datasets/PDTBv2/conversion_script.py`.

{{ dataset_overview() }}

## Citation

{{ bibtex_entry(page.meta.bib_key) }}
