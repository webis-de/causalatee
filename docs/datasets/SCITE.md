---
title: SCITE
repo: https://github.com/Das-Boot/scite
domain: Science
year: 2021
sentences: "5,236"
bib_key: li:2021
hf_repo: thagen/SCITE
hf_page: https://huggingface.co/datasets/thagen/SCITE
polarity: none
strength: none
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
  causality-extraction:
---

# {{ page.meta.title }}

{{ dataset_badges() }}
{{ dataset_pills() }}

SCITE annotates causal relations in scientific texts at the token level, providing BIO-tagged cause and effect spans for training sequence labeling models in the scientific domain.

{{ dataset_overview() }}
{{ dataset_viewer() }}
{{ tira_leaderboard() }}

## Citation

{{ bibtex_entry(page.meta.bib_key) }}
