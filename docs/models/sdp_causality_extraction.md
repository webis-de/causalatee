---
title: SDP Causality Extraction
type: Dependency-based baseline
bib_key: girju:2003
supported_tasks:
  causal-candidate-extraction:
  causality-identification:
---

# {{ page.meta.title }}

{{ model_badges() }}

The **Shortest Dependency Path (SDP)** approach is a classical, linguistically motivated baseline
for causal relation extraction. Given a sentence and two candidate entity spans, it finds the
shortest path connecting the two entities in the syntactic dependency parse tree and uses that
path as the feature representation for a downstream classifier.

The approach was introduced for general relation extraction by [@bunescu:2005] and later applied
specifically to causal relations by [@girju:2003] and others. It remains a strong, interpretable
baseline for the [Causal Event Candidate Detection](../tasks/causal_event_candidate_detection.md)
and [Causality Identification](../tasks/causality_identification.md) steps of the extraction
pipeline.

## Motivation

A dependency parse tree encodes the grammatical structure of a sentence. For a sentence such as

> *The storm caused significant flooding.*

the parse places *caused* as the root with *storm* and *flooding* as its subject and object
dependents. The shortest path between *storm* and *flooding* passes through *caused* — the very
word that expresses the causal relation.

More formally, the SDP between two node sets $A$ and $B$ in the dependency graph $G$ is

$$
\text{SDP}(A, B) = \arg\min_{a \in A,\, b \in B} d_G(a, b)
$$

where $d_G$ is the graph distance in the undirected version of $G$ and node sets $A$, $B$ are the
head tokens of each entity span [@bunescu:2005].

This path tends to strip away syntactic material irrelevant to the relation between the two
entities, yielding a compact, informative representation.

## Algorithm

```
Input:  sentence s, entity spans e₁ and e₂
Output: relation label (causal / no-relation)

1. Parse s with a dependency parser (spaCy, Stanza) to obtain tree T.
2. Find the head token h₁ of e₁ and h₂ of e₂ using the span's root.
3. Find the shortest path P = (h₁, …, h₂) in the undirected T.
4. Encode P as a sequence of (token, dep_label, direction) triples,
   where direction ∈ {↑, ↓} indicates whether each arc points toward
   or away from the root.
5. Extract features from P (see below).
6. Apply a trained classifier to the feature vector.
```

## Features

### Classical (feature-based) encoding

| Feature group | Description |
|---------------|-------------|
| **Path tokens** | Lemmas of tokens on the SDP (bag-of-words or positional) |
| **Dependency labels** | Sequence of arc labels on the path (e.g. `nsubj`, `obj`, `prep`) |
| **Arc directions** | Direction of each arc relative to the path traversal (↑ toward root, ↓ away) |
| **Path length** | Number of hops — short paths are more reliable [@bunescu:2005] |
| **POS tags** | Part-of-speech of each node on the path |
| **Causal lexicon** | Whether any known causal connective (*cause*, *lead to*, *result in*, …) appears on the path [@girju:2003] — see [Causal Lexicon](#causal-lexicon) below |
| **Entity types** | Named-entity labels of $e_1$ and $e_2$ (useful when entities are noun phrases) |

A linear SVM trained on these features achieves competitive performance on SemEval-2010 Task 8
[@rink:2010].

### Neural encoding

Rather than hand-crafting features, the SDP can be fed directly into a neural sequence model:

- **SDP-CNN** [@xu:2015a] — convolves over word and position embeddings of the SDP tokens.
- **SDP-LSTM** [@xu:2015b] — runs a bidirectional LSTM along the path; entity markers at
  the endpoints help the model locate the relation head. This variant achieves state-of-the-art
  results on SemEval-2010 Task 8 without additional linguistic features.

## Causal Lexicon

Whether a known causal connective appears on (or near) the path is one of the oldest and
cheapest features for causal relation classification [@girju:2003]. `causalatee.nlp.find_causal_connectives`
implements this over a small, hand-curated lexicon organized into the categories commonly
distinguished in the discourse-connective literature — causal verbs and nouns (*cause*, *trigger*,
*result in*, *factor*, *consequence*), subordinating cue phrases (*because*, *due to*, *since*),
and sentence-linking adverbials (*therefore*, *consequently*) — broadly following the connective
categories used by the Penn Discourse TreeBank's `Contingency.Cause` relation class [@pdtb:2008]
and the causal verb patterns cataloged by [@girju:2003]. Verbs and nouns are matched by lemma
(one entry, e.g. "lead to", automatically covers every inflection: "leads to"/"led to"/"leading
to"); cue phrases and adverbials are matched as literal surface forms, since they don't
meaningfully inflect.

```python
import spacy
from causalatee.nlp import find_causal_connectives

nlp = spacy.load("en_core_web_sm")
doc = nlp("The storm caused significant flooding.")

for match in find_causal_connectives(doc):
    print(match.text, match.category)
# caused verb
```

**A connective match is a candidate signal, not proof of a causal relation.** Several entries in
the lexicon are highly ambiguous outside of context — *since* and *as* are far more often temporal
or comparative than causal in general text. This is exactly why the classical feature-based
encoding above uses the causal lexicon as one feature among several (path length, POS tags,
dependency labels), never as a standalone decision rule. Absence of a match is equally
uninformative: [@hidey:2016] show that many real causal relations use no fixed connective at all
("alternative lexicalizations" — see the [AltLex dataset](../datasets/AltLex.md)), motivating the
neural encodings described above as a way to capture causality that lexical matching misses
entirely.

## Strengths and Limitations

| | |
|---|---|
| **Interpretable** | The extracted path is human-readable; misclassifications are easy to trace |
| **Low data requirement** | Feature-based variants work with a few hundred labeled examples |
| **Parse-dependent** | Errors in the dependency tree propagate directly to the relation representation |
| **Explicit markers only** | Fails on implicit causality where no lexical connective appears on the path |
| **Span detection required** | Assumes entity spans are already identified; not a stand-alone pipeline |

## Practical Tools

| Tool | Language | Notes |
|------|----------|-------|
| [spaCy](https://spacy.io) | Python | `en_core_web_trf` gives transformer-backed parses; `.root` attribute gives the span head |
| [Stanza](https://stanfordnlp.github.io/stanza/) | Python | Stanford NLP; supports 70+ languages; returns `DependencyRelation` objects |
| [NetworkX](https://networkx.org) | Python | `nx.shortest_path` on the token graph; convenient for SDP traversal |

`causalatee.nlp.shortest_dependency_path` implements the spaCy + NetworkX combination above directly,
taking character-offset spans rather than requiring the caller to locate head tokens manually:

```python
import spacy
from causalatee.nlp import format_sdp, shortest_dependency_path

nlp = spacy.load("en_core_web_sm")
doc = nlp("The storm caused significant flooding.")

path = shortest_dependency_path(doc, span_a=(0, 9), span_b=(29, 37))  # "The storm", "flooding"
print(format_sdp(path))
# storm --nsubj↑--> caused --dobj↓--> flooding
```

Requires the `baselines` extra (`pip install 'causalatee[baselines]'`) for `spacy` and `networkx`.

## References

<!-- CITE bunescu:2005  — Bunescu & Mooney (2005), "A Shortest Path Dependency Kernel for
     Relation Extraction", HLT/EMNLP 2005. The paper that introduced the SDP kernel for
     relation extraction and established why the SDP captures relation evidence. -->

<!-- CITE girju:2003  — Girju, Badulescu & Moldovan (2003), "Automatic Detection of Causal
     Relations for Question Answering", ACL Workshop on Multilingual Summarization and Question
     Answering 2003. First application of syntactic patterns (including dependency paths) to
     causal relation detection. -->

<!-- CITE pdtb:2008  — Prasad, Dinesh, Lee, Miltsakaki, Robaldo, Joshi & Webber (2008), "The Penn
     Discourse TreeBank 2.0", LREC 2008. Defines the Contingency.Cause discourse relation class
     and its connective inventory, the basis for causalatee.nlp's causal cue-phrase/adverbial categories. -->

<!-- CITE hidey:2016  — Hidey & McKeown (2016), "Identifying Causal Relations Using Parallel
     Wikipedia Corpora", ACL 2016. Introduces "alternative lexicalizations" (AltLex) — causal
     relations expressed without a fixed connective — motivating why connective matching alone
     has limited recall. -->

<!-- CITE rink:2010  — Rink & Harabagiu (2010), "UTD: Classifying Semantic Relations by
     Combining Lexical and Semantic Resources", SemEval 2010 Task 8. Winning system that used
     SDP features; good baseline reference for classification performance. -->

<!-- CITE xu:2015a  — Xu et al. (2015), "Classifying Relations by Ranking with Convolutional
     Neural Networks", ACL-IJCNLP 2015. SDP-CNN: convolutional model over the shortest
     dependency path. -->

<!-- CITE xu:2015b  — Xu et al. (2015), "Classifying Relations via Long Short Term Memory
     Networks along Shortest Dependency Path", EMNLP 2015. SDP-LSTM: strongest neural SDP
     baseline for relation classification. -->
