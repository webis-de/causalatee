# Biaffine Span-Grid Extraction

The **biaffine span-grid** approach reformulates causal event span extraction as a
dependency-parsing-style problem: instead of tagging each token with a BIO label, every
candidate ``(start, end)`` token pair is scored independently by a biaffine classifier over an
``L x L`` grid.

The formulation is due to [@yu:2020], introduced for flat and nested named entity recognition
("NER as dependency parsing"). It is implemented in `causalatee.nn.BiaffineSpanHead` as a small,
backbone-agnostic read-out layer — attach it to the final hidden states of any pretrained
transformer encoder and fine-tune end to end.

## Motivation

BIO tagging decodes a span as a chain of local transitions (`B-Entity`, `I-Entity`, `O`), which
has two structural limitations for causal event spans:

- **Cannot represent overlapping or nested spans** — a token can carry only one label, so two
  candidate spans sharing a token are mutually exclusive by construction.
- **Decoding needs heuristics** — adjacent spans separated only by punctuation or a short gap
  require ad-hoc gap-tolerance rules to avoid being merged or split incorrectly.

Scoring every ``(start, end)`` pair independently removes both limitations: a cell is a
self-contained prediction, so overlapping spans are just two "on" cells sharing a token, and
decoding is a threshold (optionally followed by greedy non-max suppression) rather than a
transition-based decoder.

## Architecture

$$
\text{grid}[i, j] = \sigma\big(\hat{x}_i^\top U \hat{y}_j\big), \qquad
x_i = \text{MLP}_{\text{start}}(h_i), \quad y_j = \text{MLP}_{\text{end}}(h_j)
$$

where $h_i$ is the backbone's hidden state at token $i$, $\text{MLP}_{\text{start}}$ and
$\text{MLP}_{\text{end}}$ are independent projections down to a smaller working dimension, $U$ is
a learned bilinear form, and $\hat{x}, \hat{y}$ denote the projections with a constant ``1``
appended (giving the biaffine map its bias terms). A cell above threshold on the upper triangle
(``i <= j``) is decoded as a predicted span from token $i$ to token $j$.

```
Input:  hidden states h_1, ..., h_L from any transformer encoder
Output: predicted spans as (start, end) token index pairs

1. x_i = MLP_start(h_i),  y_j = MLP_end(h_j)         for all i, j
2. grid[i, j] = biaffine(x_i, y_j)                    raw score, all i, j
3. probs = sigmoid(grid)
4. candidates = { (i, j) : probs[i, j] >= threshold, i <= j }
5. (optional) greedy NMS: sort candidates by score descending;
   keep a candidate only if it does not character-overlap
   an already-accepted span
6. Map surviving (i, j) token indices to character offsets
```

Training minimizes binary cross-entropy over the upper-triangle cells against a sparse gold grid
(one ``1`` per gold span, at its ``(start, end)`` cell), with a dynamic positive-class weight to
counteract the extreme sparsity of the target ($O(L^2)$ cells, $O(1)$ positives per sentence).

## Backbone independence

`BiaffineSpanHead` takes only `hidden_size` at construction time and consumes generic
`(batch, seq_len, hidden_size)` tensors — it has no dependency on any specific encoder
architecture:

```python
from transformers import AutoModel
from causalatee.nn import BiaffineSpanHead

backbone = AutoModel.from_pretrained("bert-base-uncased")
head = BiaffineSpanHead(backbone.config.hidden_size)

hidden_states = backbone(input_ids, attention_mask=attention_mask).last_hidden_state
grid = head(hidden_states)  # (batch, seq_len, seq_len) raw logits
```

!!! note "The layer is portable; trained weights are not"
    `BiaffineSpanHead`'s *architecture* attaches to any backbone, but a *trained* head's
    projections are learned jointly with the specific hidden space they were fine-tuned on.
    Swapping to a different backbone (or even a differently-initialized run of the same one)
    requires fine-tuning the head again — the pretrained weights themselves do not transfer.

## Strengths and Limitations

| | |
|---|---|
| **Overlapping/nested spans** | Native — each cell is an independent prediction |
| **No decoding heuristics** | Threshold + optional NMS, no BIO transition rules or gap tolerance |
| **Backbone-agnostic** | Read-out layer only needs `hidden_size`; works with any encoder |
| **Quadratic cost** | $O(L^2)$ scored cells per sentence; costly for very long sequences |
| **Sparse supervision** | Requires care (dynamic `pos_weight`, or focal loss) to avoid the head collapsing to all-negative |
| **Needs training** | Unlike a rule-based baseline, the head must be fit (or fine-tuned) on labeled span data |

## Citation

{{ bibtex_entry("yu:2020") }}