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

| Corpus | Sentences | Domain | Year | Reference | Links |
|--------|-----------|--------|------|-----------|-------|
| AltLex | 1,000 | Wikipedia | 2016 | [@hidey:2016] | [:hugging:](https://huggingface.co/datasets/thagen/AltLex) |
| BECauSE 2.0 | 1,803 | News | 2017 | [@dunietz:2017] | [:hugging:](https://huggingface.co/datasets/thagen/BECauSEv2) |
| BioCause | 851 | Medical | 2013 | [@mihaila:2013] | |
| CaTeRs | 488 | Fiction | 2016 | [@mostafazadeh:2016] | |
| Causal News Corpus (CNC) | 1,957 | News | 2022 | [@tan:2022a] | [:hugging:](https://huggingface.co/datasets/thagen/CausalNewsCorpus) |
| CausalTimeBank (CTB) | 2,201 | News | 2014 | [@mirza:2014a] | [:hugging:](https://huggingface.co/datasets/thagen/CausalTimeBank) |
| Countercausal News Corpus (CCNC) | 3,415 | News | 2025 | [@hagen:2025counterclaims] | |
| EventStoryLine (ESL) | 2,247 | News | 2017 | [@caselli:2017] | [:hugging:](https://huggingface.co/datasets/thagen/EventStoryLine) |
| FinCausal | 2,136 | Finance | 2020 | [@mariko:2020] | |
| PDTB 3.0 | — | News/WSJ | 2019 | [@webber:2019] | [:hugging:](https://huggingface.co/datasets/thagen/PennDiscourseTreebankv3) |
| PolitiCause | 5,070 | Politics | 2024 | — | |
| SCITE | 5,236 | Science | 2021 | [@li:2021] | [:hugging:](https://huggingface.co/datasets/thagen/SCITE) |
| SemEval 2010 Task 8 | 10,690 | General | 2010 | [@hendrickx:2010] | [:hugging:](https://huggingface.co/datasets/thagen/SemEval2010T8) |
| UniCausal | 14,903 | Multiple | 2023 | [@tan:2023] | |

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
