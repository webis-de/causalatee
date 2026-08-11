---
title: CauseEffectGraph
description: >-
  The Lexical Cause-Effect Graph (CEG) is a word-level cause/effect graph mined at CausalBank
  scale, with an occurrence count and two causal-strength scores ("necessity" and
  "sufficiency") per pair.
snippet: |
  from causalatee.graph import load_cause_effect_graph

  # Download the CEG file from the CausalBank project (see the link above). Omitting `limit`
  # is safe even for the full release: node vocabulary loads eagerly (cheap), but edges always
  # stream from disk rather than being materialized as Python objects.
  graph = load_cause_effect_graph("Lexical_Cause_Effect_Graph.txt")

  storm = graph.get_node("storm")
  for edge in storm.outgoing_edges():
      print(edge.target.id, edge.metadata["support"])
bib_key: li:2020
website: https://github.com/eecrazy/CausalBank
---

# {{ page.meta.title }}

{{ graph_page_icons() }}

{{ page.meta.description }}

```python
{{ page.meta.snippet }}
```

# Citation

{{ bibtex_entry(page.meta.bib_key) }}
