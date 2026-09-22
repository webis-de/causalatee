---
title: Causal Connective Lexicon
type: Rule-based lexicon lookup (`causalatee.nlp.LexiconDetector`)
bib_key: girju:2003
supported_tasks:
  causality-detection:
---

# {{ page.meta.title }}

{{ model_badges() }}

The simplest possible causality-detection baseline: a sentence is predicted **causal** iff it
contains any causal discourse connective — a word or phrase that lexically signals a causal
relation (*cause*, *lead to*, *because*, *therefore*, ...). No parsing, no training, no learned
parameters.

This technique never looks at syntactic structure at all — deliberately kept separate from the
[SDP baseline](sdp_causality_extraction.md), which uses the same connective lexicon but as one
feature among several *on the dependency path between two already-marked spans*, for
[Causality Identification](../tasks/causality_identification.md). Bundling the two under one name
is a real, previously-made mistake worth avoiding: they answer different questions (*"does this
sentence mention causality at all?"* vs. *"do these two specific spans stand in a causal
relation?"*) and only one of them touches a parse tree.

## How it works

`causalatee.nlp.LexiconDetector` wraps `find_causal_connectives` into a
[`causalatee.models.Detection`][causalatee.models.Detection]-conforming callable:

```python
from causalatee.nlp import LexiconDetector

detector = LexiconDetector()  # loads en_core_web_sm lazily; pass nlp=... to reuse one
detector("The storm caused significant flooding.")
# {'label': 'causal', 'score': 1.0}
detector("The cat sat on the mat.")
# {'label': 'uncausal', 'score': 1.0}
```

`score` is always `1.0` — this is a deterministic rule, not a calibrated probability. A **negated**
connective ("The vaccine did **not** cause autism") still predicts `causal`: `ClassLabel.Causal`
covers both causal and countercausal sentences alike (see
[Causality Detection](../tasks/causality_detection.md)) — only identification distinguishes the
two, via `Relation`.

**Deliberately used as a standalone decision rule, against the SDP page's own advice.** The SDP
baseline's documentation explicitly warns that a connective match is "a candidate signal, not
proof of a causal relation" and should be one feature among several, never used alone. This class
does exactly that anyway, as the cheapest possible floor baseline every learned model should beat.

## Strengths and Limitations

| | |
|---|---|
| **Zero cost** | No parsing, no training, no GPU — a single lemma/surface-form lookup per sentence |
| **High precision on canonical connectives** | If "caused"/"triggered"/"because" appears, the sentence usually is causal |
| **Capped recall** | Misses implicit causality and "alternative lexicalizations" that use no fixed connective at all — see the [AltLex dataset](../datasets/AltLex.md), whose entire premise is causal relations expressed without one |
| **No confidence estimate** | `score` is always `1.0`; not comparable to a calibrated classifier's probability |

## Citation

{{ bibtex_entry("girju:2003") }}
