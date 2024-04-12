from transformers import PretrainedConfig


class CehrBertConfig(PretrainedConfig):
    model_type = "cehr_bert"

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
            tie_word_embeddings=True,
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
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.tie_word_embeddings = tie_word_embeddings

        super().__init__(pad_token_id=pad_token_id, **kwargs)
