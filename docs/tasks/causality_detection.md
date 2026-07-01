# Causality Detection

Causality detection is a **sentence classification** task: given a natural language sentence, does it
express causal information?

The task can be framed as:

- **Binary classification** — `Causal` (procausal *or* countercausal) vs. `Uncausal`
- **Ternary classification** — `Procausal` / `Countercausal` / `Uncausal`

The library currently exposes the binary formulation via `ClassLabel`.

### Data Schema

Each split is stored as a Parquet file with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `index` | `str` | Unique sentence identifier |
| `text` | `str` | The input sentence |
| `label` | `int` | `ClassLabel.Causal` (1) or `ClassLabel.Uncausal` (0) |

### What counts as causal?

A sentence is considered causal when it expresses a relation between two events A and B satisfying
three conditions (following Grivaz):

1. **Temporal order** — A precedes B (the effect cannot occur before the cause)
2. **Counterfactuality** — B is less likely without A
3. **Ontological asymmetry** — A causing B does not imply B causes A

Sentences that *negate* such a relation are **countercausal** and are still labelled `Causal` in the
binary scheme. Common countercausal patterns include [@hagen:2025a]:

| Pattern | Example |
|---------|---------|
| Direct negation | "A does not cause B" |
| Lack of effect | "A happened and B did not happen" |
| Inverse expected cause | "B happened though A did not happen" |
| Usual inverse effect | "B happened despite A" |
| Negated context | "It is falsely believed that A causes B" |
| Violation of counterfactuality | "A and B happened coincidentally" |

### Example

```
Input:  "The storm caused significant flooding."
Output: ClassLabel.Causal (1)

Input:  "She went to the store and bought milk."
Output: ClassLabel.Uncausal (0)

Input:  "Sugar does not cause hyperactivity."
Output: ClassLabel.Causal (1)   # countercausal
```

## Datasets

{{ task_datasets("causality-detection") }}

**CCNC** [@hagen:2025a] is the first dataset to explicitly distinguish procausal,
countercausal, and uncausal sentences (inter-annotator agreement: Cohen's κ = 0.74).

## Models

Models are evaluated using **macro-averaged F₁**.

| Model | F₁ | Reference |
|-------|-----|-----------|
| DistilBERT | 80.0% | [@sanh2019distilbert] |
| RoBERTa | 87.4% | [@liu2019roberta] |
| Mistral-7B-Instruct | 66.2% | [@jiang2023mistral] |

!!! note
    Models trained without countercausal examples misclassify countercausal sentences as causal
    more than **10× as often** as models trained on CCNC [@hagen:2025a].

## Example

See the [Fine-tuning for Causality Detection](../examples/detection.ipynb) notebook for a
step-by-step walkthrough of fine-tuning RoBERTa on this task using the HuggingFace `Trainer`.
