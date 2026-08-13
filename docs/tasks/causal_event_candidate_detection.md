# Causal Event Candidate Extraction

Given a sentence that has been classified as causal (or countercausal), causal event candidate
extraction identifies the text spans that are plausible **cause** and **effect** candidates. The task
is typically modeled as sequence labeling (BIO tagging) or as direct span prediction.

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

{{ all_datasets(filter_task="causal-candidate-extraction") }}

## Models

A model for this task implements the
[`causalatee.models.CandidateExtraction`][causalatee.models.CandidateExtraction] protocol —
see the [API reference](../reference/models.md) for the full interface.

Models are evaluated using **macro-averaged F₁** over extracted spans.

## Example

See the [Fine-tuning for Causal Candidate Extraction](../examples/candidate_extraction.ipynb) notebook
for a step-by-step walkthrough of fine-tuning RoBERTa as a token classifier with BIO labels and
span-overlap evaluation metrics.
