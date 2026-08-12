---
title: SQLGraph
description: >-
  SQLGraph is causalatee's generic, mutable, SQLite-backed causal graph, for building a graph
  incrementally rather than loading one from an existing external resource.
snippet: |
  import sqlite3

  from causalatee.graph import SQLGraph

  # Unlike causalatee's other graph implementations (CauseNet, CGF, CauseEffectGraph), SQLGraph has
  # no external source to load from -- build one yourself, one add_node()/add_edge() call at a
  # time. Use sqlite3.connect(":memory:") for a scratch graph, or a file path to persist it.
  graph = SQLGraph(sqlite3.connect("my_graph.sqlite3"))

  # add_edge auto-creates missing endpoints. Metadata is stored generically as JSON, so any
  # JSON-serializable fields work -- there's no fixed schema like CauseNet's/CGF's typed edges.
  graph.add_edge("storm", "flooding", metadata={"support": 12})

  storm = graph.get_node("storm")
  for edge in storm.outgoing_edges():
      print(edge.target.id, edge.metadata["support"])
---

# {{ page.meta.title }}

{{ graph_page_icons() }}

{{ page.meta.description }}

```python
{{ page.meta.snippet }}
```
