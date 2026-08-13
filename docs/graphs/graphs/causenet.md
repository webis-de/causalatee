---
title: CauseNet
icon: material-web
description: >-
  CauseNet is a large-scale causality graph extracted from web and Wikipedia text, mapping cause
  concepts to effect concepts with supporting evidence for each relation.
snippet: |
  from causalatee.graph import load_causenet

  # The official CauseNet-precision release; `limit` bounds how many edges load eagerly.
  graph = load_causenet(
      "https://groups.uni-paderborn.de/wdqa/causenet/causality-graphs/causenet-precision.jsonl.bz2",
      limit=10_000,
  )

  rain = graph.get_node("rain")
  for edge in rain.outgoing_edges():
      print(edge.target.id, edge.metadata["support"])
bib_key: heindorf:2020
website: https://causenet.org
---

# {{ page.meta.title }}

{{ graph_page_icons() }}

{{ page.meta.description }}

```python
{{ page.meta.snippet }}
```

# Citation

{{ bibtex_entry(page.meta.bib_key) }}
