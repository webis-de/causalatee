# Causal Event Candidate Extraction

Given a sentence that has been classified as causal (or countercausal), causal event candidate
extraction identifies the text spans that are plausible **cause** and **effect** candidates. The task
is typically modelled as sequence labelling (BIO tagging) or as direct span prediction.

The output spans are *candidates* — their actual causal relationship is determined in the subsequent
[Causality Identification](causality_identification.md) step.

### Data Schema

Each split is stored as a Parquet file with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `index` | `str` | Unique sentence identifier |
| `text` | `str` | The input sentence |
| `entity` | `list[list[int, int]]` | Character-level `[start, end]` spans of candidate events |

### Example

```
Input:  "The storm caused significant flooding."

Output: entity = [[0, 9], [17, 36]]
        # "The storm" → [0, 9]
        # "significant flooding" → [17, 36]
```

Spans can overlap when a phrase participates in multiple causal pairs within the same sentence.

## Datasets

{{ task_datasets("causal-candidate-extraction") }}

## Models

Models are evaluated using **macro-averaged F₁** over extracted spans.

| Model | F₁ | Reference |
|-------|-----|-----------|
| DistilBERT | 35.8% | [@sanh2019distilbert] |
| RoBERTa | 44.0% | [@liu2019roberta] |

!!! note
    Span extraction is substantially harder than sentence classification. The relatively low F₁
    scores reflect the difficulty of localising exact event boundaries in free text.
