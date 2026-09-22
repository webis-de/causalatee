---
title: Sentence Embeddings
type: Fine-tuned pretrained transformer encoder (dense sentence/token embeddings)
# TODO: no docs/references/*.bib entry exists yet for RoBERTa (Liu et al. 2019, "RoBERTa: A
# Robustly Optimized BERT Pretraining Approach", arXiv:1907.11692) -- add one to models.bib
# (bib_key liu:2019, matching this project's `author:year` key convention) and set bib_key
# below once it exists, so model_badges()/the Citation section below can render it.
supported_tasks:
  causality-detection:
  causal-candidate-extraction:
  causality-identification:
---

# {{ page.meta.title }}

{{ model_badges() }}

Unlike the other model pages in this section, this approach has no bespoke architecture of its
own to describe — it relies entirely on the dense contextual embeddings a pretrained transformer
encoder already produces for a sentence, with a lightweight, standard HuggingFace head fine-tuned
on top: `AutoModelForSequenceClassification` for [Causality Detection](../tasks/causality_detection.md)
and [Causality Identification](../tasks/causality_identification.md), `AutoModelForTokenClassification`
for [Causal Event Candidate Extraction](../tasks/causal_event_candidate_detection.md). It is the
approach `causalatee.integrations.huggingface` implements and the one this project's own example
notebooks demonstrate — this page is therefore mostly a set of links to those notebooks, plus the
one detail that differs from a plain off-the-shelf `Trainer` recipe: entity markers.

## How it works

The "embedding" here is just the encoder's own hidden states — no separate embedding model is
trained or kept frozen, and the whole encoder is fine-tuned end to end (gradients flow back into
every transformer layer, so the embedding itself adapts to the task rather than staying fixed):

- **Detection** — the sequence classification head projects the pooled sentence representation
  to `Causal`/`Uncausal`.
- **Candidate Extraction** — every subword token's own hidden state is classified individually
  into a BIO tag (`O` outside a span, `B-`/`I-` — undifferentiated, as in the example notebook
  below, or role-specific `B-CAUSE`/`B-EFFECT`, as illustrated in
  `causalatee.integrations.huggingface.CausalCandidateExtractionPipeline`'s docstring), decoded
  back into character-level spans.
- **Identification** — entity markers (`<e1>`, `</e1>`, `<e2>`, `</e2>`) are inserted into the
  input text ([`causalatee.data.utils.insert_entity_markers`][causalatee.data.utils.insert_entity_markers])
  and registered as tokenizer `additional_special_tokens` (so each marker is one token, never
  split into subwords) before the same sequence classification head is fine-tuned over
  `causalatee.data.constants.Relation`.

Any checkpoint compatible with the corresponding `AutoModelFor*` class works — nothing in
`causalatee.integrations.huggingface` is specific to one encoder. The example notebooks below
fine-tune `roberta-base` (Liu et al., 2019 — "RoBERTa: A Robustly Optimized BERT Pretraining
Approach") as a concrete choice, not a requirement. The resulting
pipelines (`CausalityDetectionPipeline`, `CausalCandidateExtractionPipeline`,
`CausalityIdentificationPipeline`, all in `causalatee.integrations.huggingface`)
already satisfy their respective `causalatee.models` protocols with no extra code — see the
[API reference](../reference/models.md).

## Examples

| Task | Notebook |
|------|----------|
| [Causality Detection](../tasks/causality_detection.md) | [Fine-tuning for Causality Detection](../examples/detection.ipynb) |
| [Causal Event Candidate Extraction](../tasks/causal_event_candidate_detection.md) | [Fine-tuning for Causal Candidate Extraction](../examples/candidate_extraction.ipynb) |
| [Causality Identification](../tasks/causality_identification.md) | [Fine-tuning for Causality Identification](../examples/identification.ipynb) |

Each notebook walks through loading a `causalatee` dataset from the HuggingFace Hub, tokenizing it
for the corresponding `AutoModelFor*` class, and fine-tuning with the HuggingFace `Trainer` —
including, for candidate extraction, `causalatee.integrations.huggingface.span_compute_metrics`
as the `Trainer`'s `compute_metrics`, which decodes BIO predictions back to character spans and
reports the same precision/recall/F1/granularity-penalized-F1 metrics as the Touché shared-task
evaluator. See each notebook for its own evaluation metric and expected numbers — they aren't
identical across tasks (e.g. detection's demo reports binary F1 on the positive class specifically,
since the example dataset is small and skewed).

## Strengths and Limitations

| | |
|---|---|
| **One recipe, three tasks** | The same fine-tune-a-pretrained-encoder recipe covers detection, extraction, and identification — no task-specific feature engineering |
| **Strong baseline for modest effort** | Contextual embeddings from a pretrained encoder already capture much of the signal that classical feature-based baselines (e.g. [SDP](sdp_causality_extraction.md)) rely on hand-crafting |
| **Backbone-agnostic** | Any `AutoModelForSequenceClassification`/`AutoModelForTokenClassification`-compatible checkpoint works, not just RoBERTa |
| **Opaque compared to SDP** | No explicit linguistic structure (dependency path, connective feature) to point to — a misclassification is harder to trace to a specific cause than with the [SDP baseline](sdp_causality_extraction.md) |
| **Plain BIO can't represent overlapping spans** | The candidate-extraction notebook decodes one tag per token, same limitation as any BIO scheme — see [Biaffine Span-Grid Extraction](biaffine_span_extraction.md) for a drop-in, backbone-agnostic read-out layer that removes it *without* giving up the fine-tuned encoder this page describes |
| **Needs a GPU and labeled data** | Fine-tuning the whole encoder is far more compute- and data-hungry than SDP's feature-based variant |

## Citation

The example notebooks fine-tune RoBERTa (Liu et al., 2019 — "RoBERTa: A Robustly Optimized BERT
Pretraining Approach", arXiv:1907.11692). No `docs/references/*.bib` entry exists for it yet —
see the TODO in this page's own frontmatter.
