"""SCITE model implementations.

Two public classes
------------------
:class:`SCITEForTokenClassification`
    Token-level sequence labelling with a CRF output layer.  Used for causal span
    extraction (7-class BIO scheme: O, B-C, I-C, B-E, I-E, B-Emb, I-Emb).

:class:`SCITEForSequenceClassification`
    Sentence-level binary classification.  The first and last valid LSTM hidden
    states are concatenated and fed to a linear head.

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
            self.q_proj = nn.Linear(self._lstm_out_dim, config.concat_num_heads * config.concat_head_dim)
            self.k_proj = nn.Linear(self._lstm_out_dim, config.concat_num_heads * config.concat_head_dim)
            self.v_proj = nn.Linear(self._lstm_out_dim, config.concat_num_heads * config.concat_head_dim)
            self._attn_out_dim = self._lstm_out_dim + config.concat_num_heads * config.concat_head_dim
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

        # 2. Character CNN
        flat_chars = char_ids.view(batch_size * seq_len, config.max_clen)
        char_emb = self.char_embedding(flat_chars).permute(0, 2, 1)
        char_feat = F.max_pool1d(self.dropout_cnn(self.char_relu(self.char_cnn(char_emb))), kernel_size=char_emb.size(2))
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

        # 5. BiLSTM
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
        """Concatenated SMHA (SCITE paper variant)."""
        config = self.config
        B, T, _ = lstm_out.shape
        Q = self.q_proj(lstm_out).view(B, T, config.concat_num_heads, config.concat_head_dim).transpose(1, 2)
        K = self.k_proj(lstm_out).view(B, T, config.concat_num_heads, config.concat_head_dim).transpose(1, 2)
        V = self.v_proj(lstm_out).view(B, T, config.concat_num_heads, config.concat_head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (config.concat_head_dim ** 0.5)
        if attention_mask is not None:
            mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, -1e9)
        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, V).transpose(1, 2).contiguous().view(B, T, -1)
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
