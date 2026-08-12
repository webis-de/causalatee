---
title: SemEval-2010 Task 8
domain: General
year: 2010
sentences: "10,690"
bib_key: hendrickx:2010
hf_repo: thagen/SemEval2010T8
hf_page: https://huggingface.co/datasets/thagen/SemEval2010T8
polarity: none
strength: none
supported_tasks:
  causality-detection:
    splits:
      train: 7975
      test: 2715
  causal-candidate-extraction:
    splits:
      train: 999
      test: 328
  causality-identification:
    splits:
      train: 7975
      test: 2715
---

# {{ page.meta.title }}

{{ dataset_badges() }}
{{ dataset_pills() }}

SemEval-2010 Task 8 defines nine semantic relations between nominal pairs (including *Cause-Effect*) and asks systems to classify the relation and its directionality from a short context sentence.

causal-candidate-extraction is derived from the identification table above (`causalatee.data.utils.identification_batch_to_extraction`), restricted to the entities backing an actual causal relation — UniCausal's SemEval-2010 Task 8 CSVs don't carry span data for a standalone extraction table.

{{ dataset_overview() }}
{{ dataset_viewer() }}
{{ tira_leaderboard() }}
{{ dataset_citation() }}
