from typing import List

from transformers import PretrainedConfig


class KCNNConfig(PretrainedConfig):
    model_type = "kcnn"

    def __init__(
        self,
        vocab_size: int = 30000,
        embedding_dim: int = 300,
        pos_embedding_dim: int = 20,
        max_seq_length: int = 200,
        num_labels: int = 2,
        dropout_rate: float = 0.4,
        k_filter_sizes: List[int] = None,
        k_filters_per_size: List[int] = None,
        d_filter_sizes: List[int] = None,
        num_d_filters: int = 50,
        use_wordnet_features: bool = True,
        use_framenet_scores: bool = True,
        num_wordnet_noun_categories: int = 26,
        num_wordnet_verb_categories: int = 15,
        num_framenet_scores: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.pos_embedding_dim = pos_embedding_dim
        self.max_seq_length = max_seq_length
        self.num_labels = num_labels
        self.dropout_rate = dropout_rate
        self.k_filter_sizes = k_filter_sizes if k_filter_sizes is not None else [1, 2, 3]
        self.k_filters_per_size = k_filters_per_size if k_filters_per_size is not None else [0, 0, 0]
        self.d_filter_sizes = d_filter_sizes if d_filter_sizes is not None else [3, 4]
        self.num_d_filters = num_d_filters
        self.use_wordnet_features = use_wordnet_features
        self.use_framenet_scores = use_framenet_scores
        self.num_wordnet_noun_categories = num_wordnet_noun_categories
        self.num_wordnet_verb_categories = num_wordnet_verb_categories
        self.num_framenet_scores = num_framenet_scores

        self.k_channel_output_dim = sum(self.k_filters_per_size)
        self.d_channel_output_dim = num_d_filters * len(self.d_filter_sizes)
        self.wordnet_dim = (
            2 * (num_wordnet_noun_categories + num_wordnet_verb_categories)
            if use_wordnet_features
            else 0
        )
        self.framenet_dim = num_framenet_scores if use_framenet_scores else 0
        self.classifier_input_dim = (
            self.k_channel_output_dim
            + self.d_channel_output_dim
            + self.wordnet_dim
            + self.framenet_dim
        )
