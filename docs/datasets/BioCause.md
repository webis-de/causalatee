---
title: BioCause
domain: Medical
year: 2013
sentences: "851"
granularity: inter-sentence
bib_key: mihaila:2013
hf_repo: thagen/BioCause
hf_page: https://huggingface.co/datasets/thagen/BioCause
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

BioCause annotates causal relations in 19 open-access biomedical research articles, providing fine-grained span-level annotations for cause and effect event mentions.

Unlike most datasets in this toolkit, each row is a whole article **section**, not a single sentence: some cause/effect spans reference an adjacent sentence to their trigger, which a per-sentence schema cannot represent without either dropping or corrupting them. A per-sentence view can still be derived on demand via `causalatee.data.utils`'s sentence-splitting utilities.

{{ dataset_overview() }}
{{ dataset_viewer() }}
{{ tira_leaderboard() }}

## Citation

{{ bibtex_entry(page.meta.bib_key) }}
