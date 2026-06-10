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

| Corpus | Sentences | Domain | Year | Reference | Links |
|--------|-----------|--------|------|-----------|-------|
| AltLex | 1,000 | Wikipedia | 2016 | [@hidey:2016] | [:hugging:](https://huggingface.co/datasets/thagen/AltLex) |
| BECauSE 2.0 | 1,803 | News | 2017 | [@dunietz:2017] | [:hugging:](https://huggingface.co/datasets/thagen/BECauSEv2) |
| BioCause | 851 | Medical | 2013 | [@mihaila:2013] | |
| CaTeRs | 488 | Fiction | 2016 | [@mostafazadeh:2016] | |
| Causal News Corpus (CNC) | 1,957 | News | 2022 | [@tan:2022a] | [:hugging:](https://huggingface.co/datasets/thagen/CausalNewsCorpus) |
| Countercausal News Corpus (CCNC) | 3,415 | News | 2025 | [@hagen:2025counterclaims] | |
| FinCausal | 2,136 | Finance | 2020 | [@mariko:2020] | |
| PDTB 3.0 | — | News/WSJ | 2019 | [@webber:2019] | [:hugging:](https://huggingface.co/datasets/thagen/PennDiscourseTreebankv3) |
| SCITE | 5,236 | Science | 2021 | [@li:2021] | [:hugging:](https://huggingface.co/datasets/thagen/SCITE) |
| UniCausal | 14,903 | Multiple | 2023 | [@tan:2023] | |

## Models

Models are evaluated using **macro-averaged F₁** over extracted spans.

| Model | F₁ | Reference |
|-------|-----|-----------|
| DistilBERT | 35.8% | [@sanh2019distilbert] |
| RoBERTa | 44.0% | [@liu2019roberta] |

!!! note
    Span extraction is substantially harder than sentence classification. The relatively low F₁
    scores reflect the difficulty of localising exact event boundaries in free text.
