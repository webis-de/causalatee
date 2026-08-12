---
title: AltLex
domain: Wikipedia
year: 2016
sentences: "1,000"
doi: 10.18653/V1/P16-1135
bib_key: hidey:2016
polarity: none
strength: none
hf_repo: thagen/AltLex
hf_page: https://huggingface.co/datasets/thagen/AltLex
supported_tasks:
  causality-detection:
    splits:
      train: 596
      test: 404
  causal-candidate-extraction:
    splits:
      train: 596
      test: 404
  causality-identification:
    splits:
      train: 596
      test: 404
---

# {{ page.meta.title }}

{{ dataset_badges() }}
{{ dataset_pills() }}

AltLex is a Wikipedia-based corpus of 1,000 sentences annotated for causal relations identified through *alternative lexicalizations* — causal markers beyond the canonical connectives *because*, *so*, and *since*.

{{ dataset_overview() }}
{{ dataset_viewer() }}
{{ tira_leaderboard() }}
{{ dataset_citation() }}
