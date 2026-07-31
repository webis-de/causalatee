---
title: BECauSE 2.0
domain: News
year: 2017
sentences: "1,803"
doi: 10.18653/V1/W17-0812
bib_key: dunietz:2017
hf_repo: thagen/BECauSEv2
hf_page: https://huggingface.co/datasets/thagen/BECauSEv2
supported_tasks:
  causality-detection:
    splits:
      train: 453
      test: 17
  causal-candidate-extraction:
    splits:
      train: 453
      test: 17
  causality-identification:
    splits:
      train: 453
      test: 17
---

# {{ page.meta.title }}

{{ dataset_badges() }}
{{ dataset_pills() }}

BECauSE 2.0 annotates causal and five overlapping semantic relations (e.g. *motivation*, *purpose*) in English news text, making it one of the most relation-dense corpora for causal NLP.

{{ dataset_overview() }}
{{ dataset_viewer() }}
{{ tira_leaderboard() }}
{{ dataset_citation() }}
