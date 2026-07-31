---
title: CausalTimeBank (CTB)
domain: News
year: 2014
sentences: "2,201"
doi: 10.3115/v1/W14-0702
bib_key: mirza:2014a
hf_repo: thagen/CausalTimeBank
hf_page: https://huggingface.co/datasets/thagen/CausalTimeBank
supported_tasks:
  causality-detection:
    splits:
      train: 1885
      test: 316
  causal-candidate-extraction:
    splits:
      train: 234
      test: 42
  causality-identification:
    splits:
      train: 1885
      test: 316
---

# {{ page.meta.title }}

{{ dataset_badges() }}
{{ dataset_pills() }}

CausalTimeBank extends the TempEval-3 TimeBank corpus with explicit causal relation annotations between event mentions, linking causality to temporal ordering.

causal-candidate-extraction is derived from the identification table above (`causalatee.data.utils.identification_batch_to_extraction`), restricted to the entities backing an actual causal relation — UniCausal's CTB CSVs don't carry span data for a standalone extraction table.

{{ dataset_overview() }}
{{ dataset_viewer() }}
{{ tira_leaderboard() }}
{{ dataset_citation() }}
