---
title: SCITE
domain: Science
year: 2021
sentences: "5,236"
doi: 10.1016/J.NEUCOM.2020.08.078
bib_key: li:2021
hf_repo: thagen/SCITE
hf_page: https://huggingface.co/datasets/thagen/SCITE
supported_tasks:
  causality-detection:
    splits:
      train: 4450
      test: 786
  causal-candidate-extraction:
    splits:
      train: 4450
      test: 786
  causality-identification:
    splits:
      train: 4450
      test: 786
---

# {{ page.meta.title }}

{{ dataset_badges() }}
{{ dataset_pills() }}

SCITE annotates causal relations in scientific texts at the token level, providing BIO-tagged cause and effect spans for training sequence labelling models in the scientific domain.

{{ dataset_overview() }}
{{ dataset_viewer() }}
{{ tira_leaderboard() }}
{{ dataset_citation() }}
