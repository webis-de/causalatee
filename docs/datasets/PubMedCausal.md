---
title: PubMedCausal
domain: Biomedical
year: 2026
sentences: "30,000"
doi: 10.48550/ARXIV.2605.28363
bib_key: kunle-john:2026
hf_repo: thagen/PubMedCausal
hf_page: https://huggingface.co/datasets/thagen/PubMedCausal
polarity: none
strength: none
supported_tasks:
  causality-detection:
    splits:
      train: 15000
      test: 15000
  causal-candidate-extraction:
    splits:
      train: 1972
      test: 1973
  causality-identification:
    splits:
      train: 1529
      test: 1528
---

# {{ page.meta.title }}

{{ dataset_badges() }}
{{ dataset_pills() }}

PubMedCausal is a biomedical causal relation extraction corpus built from PubMed abstracts, typing each cause-effect pair along two dimensions: Explicit/Implicit causality and Intra-/Inter-sentential sententiality. There are no countercausal annotations — every relation converts to `Relation.Causal`.

Each pair's `causality` field (`Explicit`/`Implicit`) is already present in the raw JSON causalatee fetches but is not yet read by the converter. This is not a strength/certainty signal, though — it records whether an explicit causal connective is present in the text, not whether the causal claim itself is hedged (e.g. "may cause") or asserted with confidence markers (e.g. "I think"). It's a distinct, currently unmodeled property (connective explicitness), left undocumented as a pill on purpose to avoid conflating the two.

{{ dataset_overview() }}
{{ dataset_viewer() }}
{{ tira_leaderboard() }}
{{ dataset_citation() }}
