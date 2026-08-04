# Causality Extraction

Causality extraction is the **end-to-end** task of extracting structured causal relations from
natural language text. It is typically decomposed into three sequential subtasks
[@hagen:2025a]:

```mermaid
%%{init: {"flowchart": {"htmlLabels": true}, "themeVariables": {"fontSize": "18px", "fontFamily": "sans-serif"}}}%%
flowchart LR
    A(["`*'The storm caused
    significant flooding.'*`"])
    --> B["1. Causality\nDetection"]
    -- causal -->
    C["2. Causal Event\nCandidate Detection"]
    -- "`cause: *'The storm'*
    effect: *'significant flooding'*`" -->
    D["3. Causality\nIdentification"]
    --> E(["`*(The storm,
    significant flooding,
    Causal)*`"])

    B -- uncausal --> Z([Discard])

    click B href "../causality_detection/" "Causality Detection"
    click C href "../causal_event_candidate_detection/" "Causal Event Candidate Detection"
    click D href "../causality_identification/" "Causality Identification"

    style Z fill:#fdd,stroke:#c00,color:#600
    style A fill:#e8f4fd,stroke:#4a90d9
    style E fill:#e8fde8,stroke:#4a9d4a
```

Sentences classified as `Uncausal` at step 1 are discarded.

### Output

A fully extracted relation is a triple:

| Field | Type | Description |
|-------|------|-------------|
| `cause` | `str` | The causing event span |
| `effect` | `str` | The caused event span |
| `relation` | `Relation` | `Causal` or `Countercausal` |

### Example

```
Input:  "The storm caused significant flooding."

Output: {
    cause:    "The storm",
    effect:   "significant flooding",
    relation: Relation.Causal
}

Input:  "Sugar does not cause hyperactivity."

Output: {
    cause:    "Sugar",
    effect:   "hyperactivity",
    relation: Relation.Countercausal
}
```

A single sentence may yield multiple relation triples when several cause–effect pairs are present.

## Subtasks

| Step | Task | Page |
|------|------|------|
| 1 | Sentence-level classification | [Causality Detection](causality_detection.md) |
| 2 | Span extraction | [Causal Event Candidate Detection](causal_event_candidate_detection.md) |
| 3 | Relation classification | [Causality Identification](causality_identification.md) |

## Datasets

{{ task_datasets("causality-detection") }}

## Models

End-to-end causality extraction results are typically reported per subtask. See the individual task
pages for model benchmarks.
