from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

try:
    from .configuration_kcnn import KCNNConfig
except ImportError:
    from configuration_kcnn import KCNNConfig


class KCNNForSequenceClassification(PreTrainedModel):
    config_class = KCNNConfig

    def __init__(
        self,
        config: KCNNConfig,
        knowledge_filters: dict = None,
        word_embeddings: torch.Tensor = None,
    ):
        super().__init__(config)

        if word_embeddings is not None:
            self.word_embedding = nn.Embedding.from_pretrained(
                word_embeddings.float(), padding_idx=0, freeze=True
            )
        else:
            self.word_embedding = nn.Embedding(
                config.vocab_size, config.embedding_dim, padding_idx=0
            )
            self.word_embedding.weight.requires_grad_(False)

        # Trainable position embeddings.
        # Distances range from -(max_seq_length-1) to +(max_seq_length-1);
        # we shift by (max_seq_length-1) so the minimum maps to index 0.
        self.position_embedding = nn.Embedding(
            2 * config.max_seq_length - 1, config.pos_embedding_dim
        )

        # K-channel: frozen knowledge-oriented filters.
        # Weights are L2-normalised once at init so that dot-product = cosine similarity.
        self.k_filters = nn.ModuleDict()
        if config.k_channel_output_dim > 0:
            for ws, n_filters in zip(config.k_filter_sizes, config.k_filters_per_size):
                if n_filters == 0:
                    continue
                conv = nn.Conv1d(
                    in_channels=config.embedding_dim,
                    out_channels=n_filters,
                    kernel_size=ws,
                    bias=False,
                )
                if knowledge_filters is not None and str(ws) in knowledge_filters:
                    w = knowledge_filters[str(ws)].float()
                    # Normalise each filter (shape [n_filters, emb_dim, ws]) to unit vectors
                    # over the (emb_dim × ws) dimensions so dot-product = cosine similarity.
                    w = F.normalize(w.reshape(n_filters, -1), dim=1).reshape_as(w)
                    with torch.no_grad():
                        conv.weight.copy_(w)
                conv.weight.requires_grad_(False)
                self.k_filters[f"ws_{ws}"] = conv

        # D-channel: trainable data-oriented filters.
        d_in_channels = config.embedding_dim + 2 * config.pos_embedding_dim
        self.d_convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=d_in_channels,
                out_channels=config.num_d_filters,
                kernel_size=k,
            )
            for k in config.d_filter_sizes
        ])

        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(config.classifier_input_dim, config.num_labels)

        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,           # [B, d_seq_len]  full sentence word ids
        k_channel_ids: torch.Tensor,       # [B, k_seq_len]  between-entity word ids
        d_channel_position_ids: torch.Tensor,  # [B, d_seq_len, 2]  relative positions
        wordnet_features: Optional[torch.Tensor] = None,
        framenet_scores: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        B = input_ids.size(0)
        device = input_ids.device
        parts = []

        # -------------------------------------------------------------------
        # K-channel: words between the two entities
        # Cosine similarity convolution: L2-normalise word embeddings so that
        # the standard dot-product inside Conv1d equals cosine similarity / k.
        # The /k division is equivalent to sharing the scale uniformly across
        # the window; since filters are also unit-normalised, the output is
        # the average cosine similarity over the window (Eq. 1 of paper).
        # -------------------------------------------------------------------
        if self.config.k_channel_output_dim > 0 and self.k_filters:
            k_emb = self.word_embedding(k_channel_ids)          # [B, k_seq, emb]
            k_emb_norm = F.normalize(k_emb, p=2, dim=-1)        # unit vectors
            x_k = k_emb_norm.permute(0, 2, 1)                   # [B, emb, k_seq]
            for ws_str, conv in self.k_filters.items():
                ws = int(ws_str.split("_")[1])
                if x_k.size(-1) < ws:
                    # Between-entity sequence shorter than filter window
                    parts.append(torch.zeros(B, conv.out_channels, device=device))
                else:
                    out = conv(x_k) / ws                         # [B, n_filt, k_seq - ws + 1]
                    parts.append(out.max(dim=2).values)          # [B, n_filt]

        # -------------------------------------------------------------------
        # D-channel: full sentence with position embeddings
        # Standard dot-product convolution + tanh activation.
        # -------------------------------------------------------------------
        d_word_emb = self.word_embedding(input_ids)              # [B, d_seq, emb]
        pos_offset = self.config.max_seq_length - 1
        pos_e1 = self.position_embedding(d_channel_position_ids[:, :, 0] + pos_offset)
        pos_e2 = self.position_embedding(d_channel_position_ids[:, :, 1] + pos_offset)
        d_input = torch.cat([d_word_emb, pos_e1, pos_e2], dim=-1)  # [B, d_seq, d_in]
        x_d = d_input.permute(0, 2, 1)                             # [B, d_in, d_seq]
        for conv in self.d_convs:
            out = torch.tanh(conv(x_d))                            # [B, r, d_seq-k+1]
            parts.append(out.max(dim=2).values)                    # [B, r]

        # -------------------------------------------------------------------
        # Optional semantic features
        # -------------------------------------------------------------------
        if self.config.use_wordnet_features:
            if wordnet_features is not None and wordnet_features.shape == (B, self.config.wordnet_dim):
                parts.append(wordnet_features.float())
            else:
                parts.append(torch.zeros(B, self.config.wordnet_dim, device=device))

        if self.config.use_framenet_scores:
            if framenet_scores is not None and framenet_scores.shape == (B, self.config.framenet_dim):
                parts.append(framenet_scores.float())
            else:
                parts.append(torch.zeros(B, self.config.framenet_dim, device=device))

        combined = torch.cat(parts, dim=-1)
        logits = self.classifier(self.dropout(combined))

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)
