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

Relation values come from `ctk.data.constants.Relation`:

| Value | Constant | Meaning |
|-------|----------|---------|
| 0 | `Relation.NoRelation` | The spans are not causally related |
| 1 | `Relation.Procausal` | `first` causes `second` |
| 2 | `Relation.Concausal` | `first` countercausally relates to `second` |

### Example

```
Input text:  "<e1>The storm</e1> caused <e2>significant flooding</e2>."

Output relations: [
    {"relationship": 1, "first": "e1", "second": "e2"}   # Procausal
]

Input text:  "<e1>Sugar</e1> does not cause <e2>hyperactivity</e2>."

Output relations: [
    {"relationship": 2, "first": "e1", "second": "e2"}   # Concausal
]
```

A sentence may contain multiple entity pairs and therefore multiple relation entries.

## Datasets

| Corpus                           | Sentences | Domain    | Citation  | Links |
|:---------------------------------|----------:|:----------|-----------|-------|
| AltLex                           |     1,000 | Wikipedia | [@hidey:2016] | [:hugging:](https://huggingface.co/datasets/thagen/AltLex) |
| BECauSE 2.0                      |     1,803 | News      | [@dunietz:2017] | [:hugging:](https://huggingface.co/datasets/thagen/BECauSEv2) |
| BioCause                         |       851 | Medical   | [@mihaila:2013] | [:hugging:](https://huggingface.co/datasets/thagen/BioCause) |
| CaTeRs                           |       488 | Fiction   | [@mostafazadeh:2016] | [:hugging:](https://huggingface.co/datasets/thagen/CaTeRS) |
| Causal News Corpus (CNC)         |     1,957 | News      | [@tan:2022a] | [:hugging:](https://huggingface.co/datasets/thagen/CausalNewsCorpus) |
| CausalTimeBank (CTB)             |     2,201 | News      | [@mirza:2014a] | [:hugging:](https://huggingface.co/datasets/thagen/CausalTimeBank) |
| COPA                             |     2,000 | General   | [@roemmele:2011a] | [:hugging:](https://huggingface.co/datasets/thagen/COPA) |
| Countercausal News Corpus (CCNC) |     3,415 | News      | [@hagen:2025a] | |
| EventCausality                   |       583 | Web       | [@do:2011] | [:hugging:](https://huggingface.co/datasets/thagen/EventCausality) |
| EventStoryLine (ESL)             |     2,247 | News      | [@caselli:2017] | [:hugging:](https://huggingface.co/datasets/thagen/EventStoryLine) |
| FinCausal                        |     2,136 | Finance   | [@mariko:2020] | |
| PDTB 3.0                         |        —  | News/WSJ  | [@webber:2019] | [:hugging:](https://huggingface.co/datasets/thagen/PennDiscourseTreebankv3) |
| SCITE                            |     5,236 | Science   | [@li:2021] | [:hugging:](https://huggingface.co/datasets/thagen/SCITE) |
| SemEval 2010 Task 8              |    10,690 | General   | [@hendrickx:2010] | [:hugging:](https://huggingface.co/datasets/thagen/SemEval2010T8) |
| TCR                              |       172 | News      | [@ning:2018] | [:hugging:](https://huggingface.co/datasets/thagen/TCR) |
| UniCausal                        |    14,903 | Multiple  | [@tan:2023] | |

## Models

Models are evaluated using **macro-averaged F₁**.

| Model | F₁ | Reference |
|-------|-----|-----------|
| DistilBERT | 90.1% | [@sanh2019distilbert] |
| RoBERTa | 92.1% | [@liu2019roberta] |
| Mistral-7B-Instruct | 56.0% | [@jiang2023mistral] |
