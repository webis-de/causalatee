"""SCITE model implementations.

Three public classes
---------------------
:class:`SCITEForTokenClassification`
    Token-level sequence labelling with a CRF output layer.  Used for causal span
    extraction (7-class BIO scheme: O, B-C, I-C, B-E, I-E, B-Emb, I-Emb).  This is
    the faithful Li et al. (2021) reproduction -- not touched by the ablation below.

:class:`SCITEForSequenceClassification`
    Sentence-level binary classification.  The first and last valid LSTM hidden
    states are concatenated and fed to a linear head.

:class:`SCITEForBiaffineSpanClassification`
    Ablation (added 2026-07-21, see conf-causality-repro/notes.txt): replaces
    ``SCITEForTokenClassification``'s CRF-based sequential BIO decoding with
    ``causalatee.nn.BiaffineSpanHead``-based parallel span-grid decoding, one
    grid per span type (Cause/Effect/Emb), while keeping the SAME embedding +
    BiLSTM + attention trunk and the SAME downstream triplet-pairing heuristic
    (``tags_to_triplets`` in repro.py) -- isolates whether CRF sequential
    decoding specifically drives SCITE's reported numbers, since decoded spans
    are converted back into the identical BIO tag convention
    ``SCITEForTokenClassification`` produces, so every consumer downstream of
    ``logits`` (triplet pairing, span metrics) needs zero changes.

Architecture (shared)
---------------------
1. **Word embeddings** — learnable lookup table initialised from Wiki-extvec vectors
   (loaded once from a ``numpy`` file at construction time).
2. **Character CNN** — randomly initialised character embeddings → 1-D convolution →
   ReLU → max-pooling → one vector per word.
3. **Contextual embeddings** — one of three sources:

   * ``embedding_source="flair"`` with ``precompute_contextual_embeddings=False``:
     Flair news-forward + news-backward embeddings are generated inside ``forward()``
     from the already-tokenised ``tokens`` argument.  No text tokenisation occurs in
     the model.
   * ``embedding_source="flair"`` with ``precompute_contextual_embeddings=True``:
     a pre-computed tensor ``precomputed_contextual_embeddings`` is passed directly.
   * ``embedding_source="bert"``:
     a frozen BERT encoder is an internal sub-module.  ``forward()`` receives
     ``bert_input_ids``, ``bert_attention_mask``, and ``bert_token_to_word`` (produced
     by :class:`SciteTokenizer`) and averages sub-word hidden states to word level.
     **No tokeniser is called inside the model.**

4. **Bidirectional LSTM** — over the concatenated embeddings.
5. **Multi-head attention** — two variants (see :class:`SCITEConfig`).
6. **CRF** (:class:`SCITEForTokenClassification`) or linear head
   (:class:`SCITEForSequenceClassification`).
"""

import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from transformers import AutoModel, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput

try:
    from .configuration_scite import SCITEConfig
except ImportError:
    from configuration_scite import SCITEConfig

try:
    import flair
    from flair.data import Sentence
    from flair.embeddings import FlairEmbeddings, StackedEmbeddings

    _FLAIR_AVAILABLE = True
except ImportError:
    _FLAIR_AVAILABLE = False


# ---------------------------------------------------------------------------
# Shared backbone
# ---------------------------------------------------------------------------


class _SCITEBackbone(PreTrainedModel):
    """Shared BiLSTM + attention trunk used by both output heads."""

    config_class = SCITEConfig

    def __init__(self, config: SCITEConfig, word_embeddings_path: Optional[str] = None):
        super().__init__(config)

        # --- Word embeddings ---
        if config.use_word_embeddings and word_embeddings_path is not None:
            pretrained = torch.tensor(
                np.load(word_embeddings_path, allow_pickle=True), dtype=torch.float32
            )
            self.word_embedding = nn.Embedding.from_pretrained(pretrained, padding_idx=0)
        else:
            self.word_embedding = nn.Embedding(config.word_vocab_size, config.word_embedding_dim, padding_idx=0)

        # --- Character embeddings + CNN ---
        char_init = np.random.uniform(
            low=-math.sqrt(3 / config.char_embedding_dim),
            high=math.sqrt(3 / config.char_embedding_dim),
            size=(config.char_vocab_size, config.char_embedding_dim),
        ).astype(np.float32)
        char_init[0] = 0.0  # PAD → zero vector
        self.char_embedding = nn.Embedding.from_pretrained(
            torch.tensor(char_init), freeze=False, padding_idx=0
        )
        self.char_cnn = nn.Conv1d(
            config.char_embedding_dim,
            config.char_cnn_out_channels,
            config.char_cnn_kernel_size,
            padding=config.char_cnn_padding,
        )
        self.char_relu = nn.ReLU()
        self.dropout_cnn = nn.Dropout(config.dropout_cnn)

        # --- Contextual embeddings (Flair or BERT) ---
        self.stacked_flair = None
        self.bert_model = None

        if config.embedding_source == "flair" and not config.precompute_contextual_embeddings:
            if not _FLAIR_AVAILABLE:
                raise ImportError("Install 'flair' to use Flair contextual embeddings.")
            flair.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.flair_forward = FlairEmbeddings("news-forward")
            self.flair_backward = FlairEmbeddings("news-backward")
            self.stacked_flair = StackedEmbeddings([self.flair_forward, self.flair_backward])
            # Override config so downstream dim calculations use the actual model size.
            config.flair_embedding_dim = self.stacked_flair.embedding_length

        elif config.embedding_source == "bert":
            self.bert_model = AutoModel.from_pretrained(config.bert_model_name)
            self.bert_model.eval()
            for p in self.bert_model.parameters():
                p.requires_grad = False

        # --- Determine combined embedding dimension for BiLSTM input ---
        self._context_dim = (
            config.flair_embedding_dim
            if config.embedding_source == "flair"
            else config.bert_embedding_dim
        )
        combined_dim = self._context_dim
        if config.use_word_embeddings:
            combined_dim += config.word_embedding_dim
        combined_dim += config.char_cnn_out_channels  # char CNN always active

        # --- BiLSTM ---
        self.bilstm = nn.LSTM(
            input_size=combined_dim,
            hidden_size=config.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout_lstm,
        )
        self._lstm_out_dim = config.hidden_size * 2

        # --- Multi-head attention ---
        if config.smha_concat:
            # Official implementation (Das-Boot/scite, models/MHSA.py): the
            # Q/K/V projections and the output projection are bias-free and
            # ReLU-activated; an extra W_o (feature_dim × feature_dim) follows
            # the attention read-out before concatenation with the LSTM output.
            feature_dim = config.concat_num_heads * config.concat_head_dim
            self.q_proj = nn.Linear(self._lstm_out_dim, feature_dim, bias=False)
            self.k_proj = nn.Linear(self._lstm_out_dim, feature_dim, bias=False)
            self.v_proj = nn.Linear(self._lstm_out_dim, feature_dim, bias=False)
            self.o_proj = nn.Linear(feature_dim, feature_dim, bias=False)
            self._attn_out_dim = self._lstm_out_dim + feature_dim
        else:
            self.attention = nn.MultiheadAttention(
                embed_dim=self._lstm_out_dim,
                num_heads=config.num_attention_heads,
                batch_first=True,
            )
            self._attn_out_dim = self._lstm_out_dim

    # ------------------------------------------------------------------
    # Forward: shared trunk → returns attended LSTM output
    # ------------------------------------------------------------------

    def _forward_trunk(
        self,
        word_ids: torch.Tensor,
        char_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        tokens: Optional[List[List[str]]],
        precomputed_contextual_embeddings: Optional[torch.Tensor],
        bert_input_ids: Optional[torch.Tensor],
        bert_attention_mask: Optional[torch.Tensor],
        bert_token_to_word: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run the shared trunk and return ``[B, seq_len, attn_out_dim]``."""
        config = self.config
        batch_size, seq_len = char_ids.shape[:2]
        device = char_ids.device

        # 1. Word embeddings
        word_emb = self.word_embedding(word_ids)

        # 2. Character CNN — official code uses a linear (no activation)
        #    convolution with max-over-time pooling and no dropout on this
        #    path; ReLU/dropout are kept behind config flags for ablations.
        flat_chars = char_ids.view(batch_size * seq_len, config.max_clen)
        char_emb = self.char_embedding(flat_chars).permute(0, 2, 1)
        char_conv = self.char_cnn(char_emb)
        if config.char_cnn_relu:
            char_conv = self.char_relu(char_conv)
        if config.dropout_cnn > 0:
            char_conv = self.dropout_cnn(char_conv)
        char_feat = F.max_pool1d(char_conv, kernel_size=char_emb.size(2))
        char_feat = char_feat.squeeze(-1).view(batch_size, seq_len, -1)

        # 3. Contextual embeddings
        ctx = self._contextual_embeddings(
            batch_size, seq_len, device, tokens,
            precomputed_contextual_embeddings,
            bert_input_ids, bert_attention_mask, bert_token_to_word,
        )

        # 4. Concatenate all features
        parts = [word_emb, char_feat, ctx] if config.use_word_embeddings else [char_feat, ctx]
        full_emb = torch.cat(parts, dim=-1)

        # 5. BiLSTM. The official Keras LSTM uses dropout=0.5 on the input
        #    transformations with a mask shared across timesteps (variational);
        #    replicated here as timestep-constant dropout on the LSTM input.
        #    (Keras' recurrent_dropout=0.5 has no efficient cuDNN equivalent
        #    and is NOT replicated — a documented deviation.)
        if self.training and config.dropout_lstm_input > 0:
            keep = 1.0 - config.dropout_lstm_input
            mask = torch.bernoulli(
                torch.full((batch_size, 1, full_emb.size(-1)), keep, device=device)
            ) / keep
            full_emb = full_emb * mask
        lstm_out, _ = self.bilstm(full_emb)

        # 6. Multi-head attention
        if config.smha_concat:
            lstm_out = self._smha_concat(lstm_out, attention_mask)
        else:
            lstm_out = self._smha_residual(lstm_out, attention_mask)

        return lstm_out

    # ------------------------------------------------------------------
    # Contextual embedding generation (no tokenisation)
    # ------------------------------------------------------------------

    def _contextual_embeddings(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        tokens: Optional[List[List[str]]],
        precomputed: Optional[torch.Tensor],
        bert_input_ids: Optional[torch.Tensor],
        bert_attention_mask: Optional[torch.Tensor],
        bert_token_to_word: Optional[torch.Tensor],
    ) -> torch.Tensor:
        config = self.config

        if precomputed is not None:
            return precomputed.to(device)

        if config.embedding_source == "flair":
            return self._flair_embeddings(tokens, batch_size, seq_len, device)
        elif config.embedding_source == "bert":
            return self._bert_embeddings(batch_size, seq_len, device, bert_input_ids, bert_attention_mask, bert_token_to_word)

        raise ValueError(f"Unknown embedding_source: {config.embedding_source!r}")

    def _flair_embeddings(
        self,
        tokens: List[List[str]],
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        if tokens is None:
            raise ValueError("`tokens` must be provided for Flair embedding generation.")
        sentences = [Sentence(" ".join(toks), use_tokenizer=False) for toks in tokens]
        with torch.no_grad():
            self.stacked_flair.embed(sentences)
        emb_dim = self.config.flair_embedding_dim
        result = torch.zeros(batch_size, seq_len, emb_dim, device=device)
        for i, sent in enumerate(sentences):
            for j, tok in enumerate(sent):
                if j >= seq_len:
                    break
                result[i, j] = tok.embedding.to(device)
        return result

    def _bert_embeddings(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        bert_input_ids: Optional[torch.Tensor],
        bert_attention_mask: Optional[torch.Tensor],
        bert_token_to_word: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Average frozen BERT sub-word hidden states to word level.

        No tokeniser is called here.  ``bert_input_ids``, ``bert_attention_mask``,
        and ``bert_token_to_word`` must be supplied as tensors (produced by
        :class:`SciteTokenizer`).
        """
        if bert_input_ids is None:
            raise ValueError(
                "bert_input_ids, bert_attention_mask, and bert_token_to_word must be "
                "provided when embedding_source='bert'."
            )
        with torch.no_grad():
            hidden = self.bert_model(
                input_ids=bert_input_ids.to(device),
                attention_mask=bert_attention_mask.to(device),
            ).last_hidden_state  # [B, bert_len, bert_dim]

        emb_dim = self.config.bert_embedding_dim
        result = torch.zeros(batch_size, seq_len, emb_dim, device=device)
        counts = torch.zeros(batch_size, seq_len, device=device)

        # bert_token_to_word: [B, bert_len], values are word indices or -1
        w2w = bert_token_to_word.to(device)
        for b in range(batch_size):
            for j in range(w2w.shape[1]):
                wi = w2w[b, j].item()
                if wi < 0 or wi >= seq_len:
                    continue
                result[b, wi] += hidden[b, j]
                counts[b, wi] += 1
        # Avoid divide-by-zero for positions with no BERT tokens
        denom = counts.clamp(min=1).unsqueeze(-1)
        return result / denom

    # ------------------------------------------------------------------
    # Attention variants
    # ------------------------------------------------------------------

    def _smha_concat(self, lstm_out: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Concatenated SMHA (SCITE paper variant, official MHSA.py semantics).

        The official layer applies ReLU to the (bias-free) Q/K/V projections,
        runs scaled dot-product attention per head, then applies a bias-free
        ReLU-activated output projection before concatenating with the LSTM
        output.
        """
        config = self.config
        B, T, _ = lstm_out.shape
        Q = F.relu(self.q_proj(lstm_out)).view(B, T, config.concat_num_heads, config.concat_head_dim).transpose(1, 2)
        K = F.relu(self.k_proj(lstm_out)).view(B, T, config.concat_num_heads, config.concat_head_dim).transpose(1, 2)
        V = F.relu(self.v_proj(lstm_out)).view(B, T, config.concat_num_heads, config.concat_head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (config.concat_head_dim ** 0.5)
        if attention_mask is not None:
            mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, -1e9)
        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, V).transpose(1, 2).contiguous().view(B, T, -1)
        out = F.relu(self.o_proj(out))
        return torch.cat([lstm_out, out], dim=-1)

    def _smha_residual(self, lstm_out: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Vanilla MHA with residual connection."""
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out, key_padding_mask=key_padding_mask)
        else:
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        return lstm_out + attn_out


# ---------------------------------------------------------------------------
# Token classification (CRF)
# ---------------------------------------------------------------------------


class SCITEForTokenClassification(_SCITEBackbone):
    """SCITE for token-level causal span extraction with a CRF output layer.

    Parameters
    ----------
    config:
        Model configuration.
    word_embeddings_path:
        Path to a ``.pkl`` / ``.npy`` file containing a pre-trained word embedding
        matrix (shape ``[vocab_size, word_embedding_dim]``).  Required when
        ``config.use_word_embeddings=True``.

    Inputs to ``forward()``
    -----------------------
    word_ids : Tensor ``[B, seq_len]``
        Word vocabulary indices (from :class:`SciteTokenizer`).
    char_ids : Tensor ``[B, seq_len, max_clen]``
        Character vocabulary indices.
    attention_mask : Tensor ``[B, seq_len]``, optional
        1 for real tokens, 0 for padding.
    labels : Tensor ``[B, seq_len]``, optional
        BIO label integers (−100 for positions to ignore / padding).
    tokens : ``List[List[str]]``, optional
        Pre-tokenised word lists — required for live Flair embedding generation.
    precomputed_contextual_embeddings : Tensor ``[B, seq_len, ctx_dim]``, optional
        Pre-computed Flair or BERT contextual embeddings.
    bert_input_ids : Tensor ``[B, bert_len]``, optional
    bert_attention_mask : Tensor ``[B, bert_len]``, optional
    bert_token_to_word : Tensor ``[B, bert_len]``, optional
        Sub-word to word alignment produced by :class:`SciteTokenizer` (−1 for
        special tokens).  Passed together with ``bert_input_ids`` in BERT mode.
    """

    def __init__(self, config: SCITEConfig, word_embeddings_path: Optional[str] = None):
        super().__init__(config, word_embeddings_path)
        self.classifier = nn.Linear(self._attn_out_dim, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

    def forward(
        self,
        word_ids: torch.Tensor,
        char_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        tokens: Optional[List[List[str]]] = None,
        precomputed_contextual_embeddings: Optional[torch.Tensor] = None,
        bert_input_ids: Optional[torch.Tensor] = None,
        bert_attention_mask: Optional[torch.Tensor] = None,
        bert_token_to_word: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> TokenClassifierOutput:
        seq_len = char_ids.shape[1]
        trunk = self._forward_trunk(
            word_ids, char_ids, attention_mask, tokens,
            precomputed_contextual_embeddings,
            bert_input_ids, bert_attention_mask, bert_token_to_word,
        )
        emissions = self.classifier(trunk)  # [B, seq_len, num_labels]

        crf_mask = attention_mask.bool() if attention_mask is not None else None

        loss = None
        if labels is not None:
            # CRF does not accept −100 ignore indices; replace with 0 for the CRF pass
            crf_labels = labels.clone()
            crf_labels[crf_labels == -100] = 0
            loss = -self.crf(emissions, crf_labels, mask=crf_mask, reduction="mean")

        decoded = self.crf.decode(emissions, mask=crf_mask)
        # Pad decoded sequences back to seq_len (CRF returns variable-length lists)
        padded = [seq + [-100] * (seq_len - len(seq)) for seq in decoded]
        logits = torch.tensor(padded, dtype=torch.long, device=char_ids.device)

        return TokenClassifierOutput(loss=loss, logits=logits)


# ---------------------------------------------------------------------------
# Token classification ablation: per-type biaffine span grids instead of CRF
# ---------------------------------------------------------------------------

# BIO tag ids, matching SCITEForTokenClassification / configuration_scite.py's
# 7-class scheme exactly, so decoded output is a drop-in replacement for its
# `logits`: every downstream consumer (tags_to_triplets, compute_triplet_f1,
# the shared span metrics) needs zero changes.
_TAG_O = 0
_TAG_TYPES = {"C": (1, 2), "E": (3, 4), "Emb": (5, 6)}  # name -> (B_tag, I_tag)


def _bio_span_ids_for_type(labels: torch.Tensor, b_tag: int, i_tag: int) -> torch.Tensor:
    """Per-token span-id tensor for ONE BIO type, for :meth:`BiaffineSpanHead.grid_targets`.

    ``0`` = background (including ``-100`` ignore/padding positions -- safe,
    since ``valid_cell_mask`` already excludes those via the attention mask
    regardless of what span id they're assigned here). A positive integer is
    the 1-based index of the (contiguous B, then I*) run it belongs to,
    matching the module's own BIO parsing convention (see ``tags_to_triplets``
    in repro.py, which every consumer of this class's output still uses).
    """
    span_ids = torch.zeros_like(labels)
    for b in range(labels.shape[0]):
        current_id = 0
        in_span = False
        for t in range(labels.shape[1]):
            tag = labels[b, t].item()
            if tag == b_tag:
                current_id += 1
                span_ids[b, t] = current_id
                in_span = True
            elif tag == i_tag and in_span:
                span_ids[b, t] = current_id
            else:
                in_span = False
    return span_ids


def _sample_loss_cells(
    target: torch.Tensor, valid: torch.Tensor, neg_per_pos: int = 20, min_negatives: int = 50
) -> torch.Tensor:
    """Subsample negative cells for the LOSS ONLY (never for decoding),
    keeping every positive cell.

    SCITE's own training data is mostly all-O sentences by design (see
    module docstring), which the biaffine grid formulation turns into an
    extreme per-cell imbalance -- confirmed directly on BioCause: up to
    ~75000 negative cells per positive one. A plain BCE loss over every
    valid cell collapses to predicting background everywhere (verified:
    it did, 2026-07-21). ``pos_weight`` at that scale causes its own
    numerical instability, so this subsamples negatives instead (the
    standard remedy in the NER-as-parsing / span-grid literature for
    this exact problem) -- per example, keep ALL positive cells plus at
    most ``max(min_negatives, neg_per_pos * n_positives)`` randomly
    chosen negative cells. An all-negative example (no gold span of this
    type at all) still contributes ``min_negatives`` random cells, a
    light reinforcement signal rather than either the full imbalanced
    set or none at all.
    """
    loss_mask = torch.zeros_like(valid)
    for b in range(target.shape[0]):
        pos_idx = (target[b] == 1) & valid[b]
        neg_idx = (target[b] == 0) & valid[b]
        n_pos = int(pos_idx.sum().item())
        loss_mask[b] |= pos_idx
        neg_positions = neg_idx.nonzero(as_tuple=False)
        n_neg_keep = min(neg_positions.shape[0], max(min_negatives, neg_per_pos * n_pos))
        if n_neg_keep > 0:
            chosen = neg_positions[torch.randperm(neg_positions.shape[0], device=target.device)[:n_neg_keep]]
            loss_mask[b, chosen[:, 0], chosen[:, 1]] = True
    return loss_mask


def _decode_token_spans(scores: torch.Tensor, real_token_mask: torch.Tensor, threshold: float = 0.5) -> list:
    """Greedy non-overlap decode of (start, end) TOKEN-index spans for ONE
    example, from ONE type's grid -- same ranking/suppression logic as
    :meth:`BiaffineSpanHead.decode_spans_from_grid`, but staying in token
    index space (no ``offset_mapping``/character conversion) since the
    caller converts straight back to a per-token BIO array.
    """
    valid = (real_token_mask.unsqueeze(1) & real_token_mask.unsqueeze(0)).triu()
    probs = torch.sigmoid(scores)
    hits = ((probs >= threshold) & valid).nonzero(as_tuple=False)
    candidates = sorted(
        ((probs[i, j].item(), int(i), int(j)) for i, j in hits.tolist()),
        key=lambda c: c[0], reverse=True,
    )
    accepted: list = []
    for _, s, e in candidates:
        if not any(s < ae and e > as_ for as_, ae in accepted):
            accepted.append((s, e))
    return sorted(accepted)


class SCITEForBiaffineSpanClassification(_SCITEBackbone):
    """SCITE ablation: per-type ``BiaffineSpanHead`` grids instead of a CRF.

    See the module docstring for the rationale. Inputs to ``forward()`` are
    IDENTICAL to :class:`SCITEForTokenClassification`'s (same trunk, same
    ``labels`` BIO convention) -- only the read-out after the shared trunk
    differs.

    One independent ``BiaffineSpanHead`` per span type (Cause/Effect/Emb):
    ``BiaffineSpanHead`` itself has no label dimension (one grid, one binary
    boundary score per cell -- see causalatee.nn's own docstring), unlike the
    CRF's single 7-way-per-token classification, so span TYPE needs one grid
    per type rather than one shared grid.

    Decoded per-type spans are merged back into a single ``[B, seq_len]``
    BIO tag array (fixed priority Emb > C > E when two types' decoded spans
    disagree on the same token -- Emb represents "serves as both", so it's
    the least arbitrary tie-break available; a token already claimed by a
    higher-priority type is never overwritten). This keeps ``logits`` in
    exactly the same shape/convention ``SCITEForTokenClassification``
    returns, so ``tags_to_triplets``'s pairing heuristic and the shared span
    metrics need zero changes -- the ablation isolates ONLY the span
    detection readout, not the (already deterministic, non-learned) pairing
    step.
    """

    def __init__(self, config: SCITEConfig, word_embeddings_path: Optional[str] = None):
        super().__init__(config, word_embeddings_path)
        from causalatee.nn import BiaffineSpanHead

        self.cause_head = BiaffineSpanHead(self._attn_out_dim)
        self.effect_head = BiaffineSpanHead(self._attn_out_dim)
        self.emb_head = BiaffineSpanHead(self._attn_out_dim)
        self._heads = {"C": self.cause_head, "E": self.effect_head, "Emb": self.emb_head}
        self._loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        word_ids: torch.Tensor,
        char_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        tokens: Optional[List[List[str]]] = None,
        precomputed_contextual_embeddings: Optional[torch.Tensor] = None,
        bert_input_ids: Optional[torch.Tensor] = None,
        bert_attention_mask: Optional[torch.Tensor] = None,
        bert_token_to_word: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> TokenClassifierOutput:
        from causalatee.nn import BiaffineSpanHead

        seq_len = char_ids.shape[1]
        trunk = self._forward_trunk(
            word_ids, char_ids, attention_mask, tokens,
            precomputed_contextual_embeddings,
            bert_input_ids, bert_attention_mask, bert_token_to_word,
        )
        real_token_mask = attention_mask.bool() if attention_mask is not None else torch.ones(
            trunk.shape[:2], dtype=torch.bool, device=trunk.device
        )
        valid_cells = BiaffineSpanHead.valid_cell_mask(real_token_mask)

        grids = {name: head(trunk) for name, head in self._heads.items()}

        loss = None
        if labels is not None:
            losses = []
            for name, (b_tag, i_tag) in _TAG_TYPES.items():
                span_ids = _bio_span_ids_for_type(labels, b_tag, i_tag)
                target = BiaffineSpanHead.grid_targets(span_ids)
                loss_cells = _sample_loss_cells(target, valid_cells)
                losses.append(self._loss_fn(grids[name][loss_cells], target[loss_cells]))
            loss = torch.stack(losses).mean()

        # Decode each example independently (variable per-example real-token
        # count), merge per-type spans into one BIO tag row per SCITE's own
        # convention, priority Emb > C > E on conflicting tokens.
        batch_size = trunk.shape[0]
        tag_rows = []
        for b in range(batch_size):
            tags = [_TAG_O] * seq_len
            mask_b = real_token_mask[b]
            for name in ("Emb", "C", "E"):
                b_tag, i_tag = _TAG_TYPES[name]
                for s, e in _decode_token_spans(grids[name][b], mask_b):
                    if tags[s] != _TAG_O:
                        continue
                    tags[s] = b_tag
                    for t in range(s + 1, e + 1):
                        if tags[t] == _TAG_O:
                            tags[t] = i_tag
            tag_rows.append(tags)
        logits = torch.tensor(tag_rows, dtype=torch.long, device=char_ids.device)

        return TokenClassifierOutput(loss=loss, logits=logits)


# ---------------------------------------------------------------------------
# Sequence classification (binary)
# ---------------------------------------------------------------------------


class SCITEForSequenceClassification(_SCITEBackbone):
    """SCITE for binary sentence-level causality classification.

    The first and last valid LSTM hidden states (after attention) are concatenated
    and fed to a linear classifier.  This mirrors the ``binary_classification=True``
    branch of the original ``SCITEModel``.

    Inputs and output follow the same conventions as
    :class:`SCITEForTokenClassification`, except:

    * ``labels`` is a 1-D tensor of shape ``[B]`` (sentence-level class ids).
    * The ``logits`` in the output are of shape ``[B, num_labels]``.
    """

    def __init__(self, config: SCITEConfig, word_embeddings_path: Optional[str] = None):
        super().__init__(config, word_embeddings_path)
        self.classifier = nn.Linear(self._attn_out_dim * 2, config.num_labels)

    def forward(
        self,
        word_ids: torch.Tensor,
        char_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        tokens: Optional[List[List[str]]] = None,
        precomputed_contextual_embeddings: Optional[torch.Tensor] = None,
        bert_input_ids: Optional[torch.Tensor] = None,
        bert_attention_mask: Optional[torch.Tensor] = None,
        bert_token_to_word: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        trunk = self._forward_trunk(
            word_ids, char_ids, attention_mask, tokens,
            precomputed_contextual_embeddings,
            bert_input_ids, bert_attention_mask, bert_token_to_word,
        )

        # Extract first token and last valid token hidden states
        first = trunk[:, 0, :]
        if attention_mask is not None:
            last_pos = attention_mask.sum(dim=1) - 1
            last = trunk[torch.arange(trunk.size(0), device=trunk.device), last_pos]
        else:
            last = trunk[:, -1, :]

        pooled = torch.cat([first, last], dim=-1)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)
