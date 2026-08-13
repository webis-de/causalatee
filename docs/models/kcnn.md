---
title: k-CNN
type: Knowledge- + data-channel CNN for relation classification
bib_key: li:2019
supported_tasks:
  causality-identification:
---

# {{ page.meta.title }}

{{ model_badges() }}

**k-CNN** classifies whether a causal relation holds between two already-marked entity spans by
combining two parallel convolutional channels: a *knowledge-oriented* (K) channel that encodes
linguistic cues from the text strictly between the two entities, and a *data-oriented* (D)
channel that encodes the whole sentence with position-relative embeddings.

Introduced by [@li:2019] for binary causal-vs-non-causal relation classification over an
already-identified entity pair, k-CNN directly targets
[Causality Identification](../tasks/causality_identification.md) — it needs both spans marked
ahead of time and does not itself detect or extract them.

## Architecture

**K-channel** (knowledge-oriented): only the lemmatized words strictly between the two entity
spans. Convolves with cosine similarity (unit-vector dot product) over 1/2/3-word windows, then
max-pools. Filters are pruned by an ANOVA F-test (keeping only filters whose class separation is
statistically significant) and the survivors are clustered (k-means, separately for
unigrams/bigrams) to control dimensionality.

**D-channel** (data-oriented): the full sentence, each word embedded together with its relative
position to both entity heads (20-d trainable position embeddings). Convolves with a standard
dot product + tanh over 3/4-word windows, then max-pools.

**Semantic features** (optional): WordNet top-level category one-hots for both entities (82-d)
plus four FrameNet causal-lexicon scores, summed separately over the before/between/after regions
of the sentence relative to the entity pair.

The K-channel, D-channel, and semantic features are concatenated and fed to a small fully
connected softmax classifier (dropout 0.4). Word embeddings are the frozen, unit-normalized
Levy & Goldberg dependency-based embeddings — not the more commonly substituted word2vec or
GloVe embeddings.

## Evaluation

[@li:2019] reports macro-averaged F1 (10-fold cross-validation, averaged over 10 seeds) of 92.64
on SemEval-2010 Task 8, 77.86 on CausalTimeBank, and 83.31 on Event StoryLine for the best
configuration (both channels plus semantic features). The knowledge-oriented channel —
restricting convolution to only the words between the two entities, rather than the whole
sentence — is the paper's central contribution over a plain sentence-level CNN baseline.

## Strengths and Limitations

| | |
|---|---|
| **Strong on SemEval-2010 Task 8** | Competitive F1 without any contextual (transformer) embeddings |
| **Identification-only** | Needs both entity spans already marked; cannot detect or extract spans itself |
| **Needs markers on negatives too** | Both channels require two marked entity spans on *every* example, including non-causal ones — datasets whose `causality-identification` conversion only marks entities on causal-relation rows (rather than on every candidate pair) can't be fed to k-CNN as-is; see `resources/models/kCNN/notes.txt` for the concrete blocker hit reproducing it against causalatee's own converted tables |
| **Sensitive to embedding choice** | Built specifically around Levy-Goldberg dependency-based embeddings, whose original distribution site is no longer reliably available |

## Citation

{{ bibtex_entry("li:2019") }}
