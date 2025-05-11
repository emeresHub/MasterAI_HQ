from transformers import PretrainedConfig

class MyGPTConfig(PretrainedConfig):
    model_type = "my_gpt"

    def __init__(
        self,
        vocab_size: int = 50257,
        block_size: int = 1024,
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 768,
        dropout: float = 0.0,
        bias: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size  = vocab_size
        self.block_size  = block_size
        self.n_layer     = n_layer
        self.n_head      = n_head
        self.n_embd      = n_embd
        self.dropout     = dropout
        self.bias        = bias
