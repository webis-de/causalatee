---
title: SCITE
repo: https://github.com/Das-Boot/scite
type: BiLSTM + char-CNN + Flair/BERT + self-attention + CRF
bib_key: li:2021
supported_tasks:
  causal-candidate-extraction:
---

# {{ page.meta.title }}

{{ model_badges() }}

**SCITE** (Self-attentive BiLSTM-CRF with Transferred Embeddings) frames causal event span
extraction as sequence tagging: every token gets a BIO tag over three span types — Cause (`C`),
Effect (`E`), and an ambiguous "Embedded" (`Emb`) category for spans that could plausibly act as
either — decoded by a linear-chain CRF.

Introduced by [@li:2021], SCITE combines transferred contextual embeddings (Flair or BERT) with a
character-level CNN and multi-head self-attention (MHSA) on top of a BiLSTM encoder. It's also the
origin of the [SCITE dataset](../datasets/SCITE.md) this project converts.

## Architecture

Input per word: a frozen 300-d word embedding (Komninos & Manandhar) concatenated with a 30-d
character-CNN embedding and a contextual embedding — either Flair (news-forward + news-backward,
2048-d stacked) or BERT.

1. **BiLSTM** — 1 layer, hidden size 256 per direction (512-d output).
2. **Multi-head self-attention (MHSA)** — 3 heads, 8-d per head (24-d attention output),
   *concatenated* onto the BiLSTM output (not added as a residual) before the final projection.
3. **CRF decoder** — linear-chain CRF over 7 tags (`O`, `B-C`, `I-C`, `B-E`, `I-E`, `B-Emb`,
   `I-Emb`).

Trained with Nadam (lr = 0.001), variational dropout (0.5) on the CNN and LSTM outputs, gradient
clipping at 5.0, up to 200 epochs with the checkpoint selected by 10-fold cross-validation F1.

## Evaluation

A predicted triplet (cause span, effect span, causal link) is scored correct only if **both**
spans exactly match a gold triplet — a strict criterion that compounds span-level errors into
larger triplet-level ones. [@li:2021] reports 0.8455 triplet F1 (5-run mean) for the full model on
the [SCITE dataset](../datasets/SCITE.md), roughly 6–7 points above a plain BiLSTM-CRF without the
transferred embeddings or attention. Most of that gap traces to the contextual embeddings, not the
attention mechanism itself: removing MHSA alone costs under 1 point, while removing Flair/BERT
costs several.

## Strengths and Limitations

| | |
|---|---|
| **Joint span+relation modeling** | One tagging pass produces both spans and, implicitly, their pairing |
| **Strong with contextual embeddings** | Flair/BERT features account for most of the gain over a plain BiLSTM-CRF |
| **Strict triplet scoring is punishing** | A single off-by-one span boundary error zeroes out an otherwise-correct triplet |
| **Ambiguous "Embedded" category** | The lowest-scoring tag class in the original paper (~0.29 F1) — spans that could plausibly be cause or effect depending on framing are genuinely hard to tag consistently |

## Citation

{{ bibtex_entry("li:2021") }}
