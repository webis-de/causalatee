---
title: CGF
icon: material-lightning-bolt
description: >-
  CGF (Causal Graph Format) is causalatee's own compact, memory-mappable binary format for
  storing a causal graph on disk and traversing it without materializing it in Python objects.
snippet: |
  from causalatee.graph import load_causenet, save_cgf, load_cgf

  # Build a Graph from any source -- here, a small sample of CauseNet (see the CauseNet card
  # above for the full download URL).
  graph = load_causenet(
      "https://groups.uni-paderborn.de/wdqa/causenet/causality-graphs/causenet-precision.jsonl.bz2",
      limit=10_000,
  )

  # The writer streams through a disk-backed sort, so memory use does not grow with edge count.
  save_cgf(graph, "causenet.cgf")

  # Nodes and edges are then resolved lazily from disk, memory-mapped.
  with load_cgf("causenet.cgf", validate=True) as mapped:
      rain = mapped.get_node("rain")
      for edge in rain.outgoing_edges():
          print(edge.target.id, edge.metadata["support"])
website: graphs/cgf_spec.md
---

# {{ page.meta.title }}

{{ graph_page_icons() }}

{{ page.meta.description }} See the [CGF 1.0 specification](../cgf_spec.md) for the full format.

```python
{{ page.meta.snippet }}
```
