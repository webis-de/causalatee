from transformers import PretrainedConfig

LABEL_NAMES = ["O", "B-C", "I-C", "B-E", "I-E", "B-Emb", "I-Emb"]


class SCITEConfig(PretrainedConfig):
    """Configuration for SCITE (Semantic Causality Information Token Extraction).

    SCITE combines word embeddings, character-level CNN features, contextual embeddings
    (Flair or frozen BERT), a bidirectional LSTM, and multi-head attention.  For token
    classification the output is decoded through a CRF layer; for sequence classification
    the first and last LSTM hidden states are concatenated and fed to a linear head.

    Embedding sources
    -----------------
    ``embedding_source="flair"``
        Flair news-forward + news-backward embeddings are generated on the fly inside the
        model from the already-tokenised ``tokens`` argument.  Set
        ``precompute_contextual_embeddings=True`` to cache them once before training and
        pass a ``precomputed_contextual_embeddings`` tensor directly to ``forward()``.

    ``embedding_source="bert"``
        A frozen BERT encoder is used.  The model expects ``bert_input_ids``,
        ``bert_attention_mask``, and ``bert_token_to_word`` (produced by
        :class:`SciteTokenizer`) and averages sub-word hidden states to word level.
        No tokeniser is called inside the model.

    Attention variants
    ------------------
    ``smha_concat=False`` (default)
        Vanilla ``nn.MultiheadAttention`` with a residual connection over the BiLSTM
        output.  ``num_attention_heads`` heads.

    ``smha_concat=True``
        Scaled multi-head attention whose output is *concatenated* to the BiLSTM output
        instead of added.  Uses ``concat_num_heads`` heads of dimension ``concat_head_dim``.
    """

    model_type = "scite"

    def __init__(
        self,
        # Vocabulary sizes — set after calling SciteTokenizer.build_vocab()
        word_vocab_size: int = 30522,
        char_vocab_size: int = 128,
        # Word / character embedding dimensions
        word_embedding_dim: int = 300,
        char_embedding_dim: int = 30,
        # Character CNN
        char_cnn_out_channels: int = 30,
        char_cnn_kernel_size: int = 3,
        char_cnn_padding: int = 1,
        # BiLSTM
        hidden_size: int = 256,
        # Multi-head attention
        smha_concat: bool = True,
        num_attention_heads: int = 4,
        concat_num_heads: int = 3,
        concat_head_dim: int = 8,
        # Output
        num_labels: int = 7,
        # Sequence lengths (used for padding in the data collator)
        max_wlen: int = 58,
        max_clen: int = 23,
        # Contextual embeddings
        embedding_source: str = "flair",
        flair_embedding_dim: int = 4096,
        bert_embedding_dim: int = 768,
        bert_model_name: str = "google-bert/bert-base-uncased",
        # Regularisation.  Official-code defaults: the char CNN is a linear
        # convolution with no dropout (char_cnn_relu=False, dropout_cnn=0.0);
        # the LSTM input gets variational dropout 0.5 (Keras LSTM dropout=0.5).
        dropout_cnn: float = 0.0,
        char_cnn_relu: bool = False,
        dropout_lstm: float = 0.5,
        dropout_lstm_input: float = 0.5,
        # Flags
        use_word_embeddings: bool = True,
        precompute_contextual_embeddings: bool = False,
        **kwargs,
    ):
        _id2label = kwargs.pop("id2label", {i: l for i, l in enumerate(LABEL_NAMES[:num_labels])})
        _label2id = kwargs.pop("label2id", {l: i for i, l in _id2label.items()})
        super().__init__(id2label=_id2label, label2id=_label2id, **kwargs)

        self.word_vocab_size = word_vocab_size
        self.char_vocab_size = char_vocab_size
        self.word_embedding_dim = word_embedding_dim
        self.char_embedding_dim = char_embedding_dim
        self.char_cnn_out_channels = char_cnn_out_channels
        self.char_cnn_kernel_size = char_cnn_kernel_size
        self.char_cnn_padding = char_cnn_padding
        self.hidden_size = hidden_size
        self.smha_concat = smha_concat
        self.num_attention_heads = num_attention_heads
        self.concat_num_heads = concat_num_heads
        self.concat_head_dim = concat_head_dim
        self.num_labels = num_labels
        self.max_wlen = max_wlen
        self.max_clen = max_clen
        self.embedding_source = embedding_source
        self.flair_embedding_dim = flair_embedding_dim
        self.bert_embedding_dim = bert_embedding_dim
        self.bert_model_name = bert_model_name
        self.dropout_cnn = dropout_cnn
        self.char_cnn_relu = char_cnn_relu
        self.dropout_lstm = dropout_lstm
        self.dropout_lstm_input = dropout_lstm_input
        self.use_word_embeddings = use_word_embeddings
        self.precompute_contextual_embeddings = precompute_contextual_embeddings
