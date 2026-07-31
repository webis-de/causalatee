# Models

Model approaches for causality extraction, ranging from interpretable rule-based baselines to
fine-tuned neural classifiers.

| Model | Type | Tasks | Page |
|-------|------|-------|------|
| [SDP Causality Extraction](sdp_causality_extraction.md) | Dependency-based baseline | E · I | [→](sdp_causality_extraction.md) |
| [Biaffine Span-Grid Extraction](biaffine_span_extraction.md) | Neural read-out layer (`causalatee.nn.BiaffineSpanHead`) | E | [→](biaffine_span_extraction.md) |

**Task codes**: E = Causal Event Candidate Extraction · I = Causality Identification

Neural fine-tuning approaches (RoBERTa etc.) are covered in the [Examples](../examples/detection.ipynb)
section.
