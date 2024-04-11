from transformers import PretrainedConfig


class CehrBertConfig(PretrainedConfig):
    model_type = "cehr_bert"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
            self,
            vocab_size=20000,
            n_positions=512,
            n_visit_segments=3,
            n_embd=128,
            n_time_embd=16,
            n_layer=12,
            n_head=8,
            n_inner=None,
            activation_function="gelu_new",
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            scale_attn_weights=True,
            use_cache=True,
            bos_token_id=50256,
            eos_token_id=50256,
            scale_attn_by_inverse_layer_idx=False,
            reorder_and_upcast_attn=False,
            **kwargs,
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_visit_segments = n_visit_segments
        self.n_embd = n_embd
        self.n_time_embd = n_time_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.reorder_and_upcast_attn = reorder_and_upcast_attn

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
