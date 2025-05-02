from transformers import PretrainedConfig


class CehrBertConfig(PretrainedConfig):
    model_type = "cehrbert"

    def __init__(
        self,
        vocab_size=20000,
        n_visit_segments=3,
        hidden_size=128,
        n_time_embd=16,
        num_hidden_layers=12,
        num_attention_heads=8,
        intermediate_size=2048,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        cls_token_id=None,
        lab_token_ids=None,
        tie_word_embeddings=True,
        num_labels=2,
        classifier_dropout=0.1,
        bidirectional=True,
        include_value_prediction=False,
        mlm_probability=0.15,
        time_embedding_scaling_factor: float = 1000,
        age_embedding_scaling_factor: float = 100,
        sample_packing_max_positions=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_time_embd = n_time_embd
        self.n_visit_segments = n_visit_segments
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.sample_packing_max_positions = (
            sample_packing_max_positions if sample_packing_max_positions else max_position_embeddings
        )
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.tie_word_embeddings = tie_word_embeddings
        self.num_labels = num_labels
        self.classifier_dropout = classifier_dropout
        self.bidirectional = bidirectional
        self.include_value_prediction = include_value_prediction
        self.mlm_probability = mlm_probability
        self.time_embedding_scaling_factor = time_embedding_scaling_factor
        self.age_embedding_scaling_factor = age_embedding_scaling_factor

        self.cls_token_id = cls_token_id
        self.lab_token_ids = lab_token_ids

        super().__init__(pad_token_id=pad_token_id, **kwargs)
