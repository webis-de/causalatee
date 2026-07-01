# SDP Causality Extraction

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
| **Causal lexicon** | Whether any known causal connective (*cause*, *lead to*, *result in*, …) appears on the path [@girju:2003] |
| **Entity types** | Named-entity labels of $e_1$ and $e_2$ (useful when entities are noun phrases) |

A linear SVM trained on these features achieves competitive performance on SemEval-2010 Task 8
[@rink:2010].

### Neural encoding

Rather than hand-crafting features, the SDP can be fed directly into a neural sequence model:

- **SDP-CNN** [@xu:2015a] — convolves over word and position embeddings of the SDP tokens.
- **SDP-LSTM** [@xu:2015b] — runs a bidirectional LSTM along the path; entity markers at
  the endpoints help the model locate the relation head. This variant achieves state-of-the-art
  results on SemEval-2010 Task 8 without additional linguistic features.

## Strengths and Limitations

| | |
|---|---|
| **Interpretable** | The extracted path is human-readable; misclassifications are easy to trace |
| **Low data requirement** | Feature-based variants work with a few hundred labelled examples |
| **Parse-dependent** | Errors in the dependency tree propagate directly to the relation representation |
| **Explicit markers only** | Fails on implicit causality where no lexical connective appears on the path |
| **Span detection required** | Assumes entity spans are already identified; not a stand-alone pipeline |

## Practical Tools

| Tool | Language | Notes |
|------|----------|-------|
| [spaCy](https://spacy.io) | Python | `en_core_web_trf` gives transformer-backed parses; `.root` attribute gives the span head |
| [Stanza](https://stanfordnlp.github.io/stanza/) | Python | Stanford NLP; supports 70+ languages; returns `DependencyRelation` objects |
| [NetworkX](https://networkx.org) | Python | `nx.shortest_path` on the token graph; convenient for SDP traversal |

A minimal spaCy snippet:

```python
import spacy, networkx as nx

nlp = spacy.load("en_core_web_sm")
doc = nlp("The storm caused significant flooding.")

# Build undirected token graph
G = nx.Graph()
for token in doc:
    if token.head != token:
        G.add_edge(token.i, token.head.i, dep=token.dep_)

# Entity head tokens (replace with actual span roots)
h1, h2 = doc[1].i, doc[4].i          # "storm", "flooding"
path_indices = nx.shortest_path(G, h1, h2)
path_tokens  = [doc[i] for i in path_indices]
print([t.text for t in path_tokens])  # ['storm', 'caused', 'flooding']
```

## References

<!-- CITE bunescu:2005  — Bunescu & Mooney (2005), "A Shortest Path Dependency Kernel for
     Relation Extraction", HLT/EMNLP 2005. The paper that introduced the SDP kernel for
     relation extraction and established why the SDP captures relation evidence. -->

<!-- CITE girju:2003  — Girju, Badulescu & Moldovan (2003), "Automatic Detection of Causal
     Relations for Question Answering", ACL Workshop on Multilingual Summarization and Question
     Answering 2003. First application of syntactic patterns (including dependency paths) to
     causal relation detection. -->

<!-- CITE rink:2010  — Rink & Harabagiu (2010), "UTD: Classifying Semantic Relations by
     Combining Lexical and Semantic Resources", SemEval 2010 Task 8. Winning system that used
     SDP features; good baseline reference for classification performance. -->

<!-- CITE xu:2015a  — Xu et al. (2015), "Classifying Relations by Ranking with Convolutional
     Neural Networks", ACL-IJCNLP 2015. SDP-CNN: convolutional model over the shortest
     dependency path. -->

<!-- CITE xu:2015b  — Xu et al. (2015), "Classifying Relations via Long Short Term Memory
     Networks along Shortest Dependency Path", EMNLP 2015. SDP-LSTM: strongest neural SDP
     baseline for relation classification. -->
