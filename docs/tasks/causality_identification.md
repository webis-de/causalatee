# Causality Identification

Given a causal sentence and a pair of candidate event spans, causality identification determines
whether the two events stand in a causal relation — and, if so, the *direction* and *polarity* of
that relation.

The task is typically modelled as classification over marked-up text: entity markers are inserted
into the sentence to delimit the two candidate spans, and a classifier is applied to the resulting
string.

### Data Schema

Each split is stored as a Parquet file with the following columns:

| Column      | Type         | Description                                                                                  |
|-------------|--------------|----------------------------------------------------------------------------------------------|
| `index`     | `str`        | Unique sentence identifier                                                                   |
| `text`      | `str`        | Sentence with entity markers `<e1>...</e1>`, `<e2>...</e2>`, ...                             |
| `relations` | `list[dict]` | Each dict has `relationship` (`Relation` int), `first` (e.g. `"e1"`), `second` (e.g. `"e2"`) |

Relation values come from `causalatee.data.constants.Relation`:

| Value | Constant | Meaning |
|-------|----------|---------|
| 0 | `Relation.NoRelation` | The spans are not causally related |
| 1 | `Relation.Causal` | `first` causes `second` |
| 2 | `Relation.Countercausal` | `first` countercausally relates to `second` |

### Example

```
Input text:  "<e1>The storm</e1> caused <e2>significant flooding</e2>."

Output relations: [
    {"relationship": 1, "first": "e1", "second": "e2"}   # Causal
]

Input text:  "<e1>Sugar</e1> does not cause <e2>hyperactivity</e2>."

Output relations: [
    {"relationship": 2, "first": "e1", "second": "e2"}   # Countercausal
]
```

A sentence may contain multiple entity pairs and therefore multiple relation entries.

## Datasets

{{ task_datasets("causality-identification") }}

## Models

Models are evaluated using **macro-averaged F₁**.

| Model | F₁ | Reference |
|-------|-----|-----------|
| DistilBERT | 90.1% | [@sanh2019distilbert] |
| RoBERTa | 92.1% | [@liu2019roberta] |
| Mistral-7B-Instruct | 56.0% | [@jiang2023mistral] |

## Example

See the [Fine-tuning for Causality Identification](../examples/identification.ipynb) notebook for a
step-by-step walkthrough of fine-tuning RoBERTa with entity-marker special tokens using the
HuggingFace `Trainer`.
