# modeling_my_gpt.py

import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoTokenizer, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from config import MyGPTConfig
from model import GPT  # your original GPT & GPTConfig dataclass

class MyGPTForCausalLM(PreTrainedModel):
    config_class = MyGPTConfig
    base_model_prefix = "gpt"

    def __init__(self, config: MyGPTConfig):
        super().__init__(config)
        # instantiate your original GPT
        self.gpt = GPT(config)

        # tie token embeddings → LM head
        # (your GPT already did this, but HF expects resize methods)
        self.tie_weights()

        # run HF weight-init checks (no-op if loading weights next)
        self.post_init()

    def tie_weights(self):
        # ensure HF knows about tied embeddings
        self.gpt.lm_head.weight = self.gpt.transformer.wte.weight

    def forward(
        self,
        input_ids=None,
        attention_mask=None,   # unused by your GPT, but HF Trainer may pass it
        labels=None,
        **kwargs
    ):
        # your GPT forward returns (logits, loss)
        logits, loss = self.gpt(input_ids, targets=labels)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    @classmethod
    def from_pretrained_checkpoint(cls, checkpoint_path: str, config: MyGPTConfig):
        # 1) load the raw pytorch ckpt
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        raw_state = ckpt["model"]

        # 2) strip any DDP prefixes
        unwanted = "_orig_mod."
        new_state = {}
        for k, v in raw_state.items():
            if k.startswith(unwanted):
                new_state[k[len(unwanted):]] = v
            else:
                new_state[k] = v

        # 3) instantiate model & load weights
        model = cls(config)
        model.gpt.load_state_dict(new_state, strict=True)
        return model

    def resize_token_embeddings(self, new_num_tokens: int):
        """
        If you add special tokens to your tokenizer, call this to resize
        both wte and lm_head to new_num_tokens, preserving existing weights.
        """
        old_embed = self.gpt.transformer.wte
        old_num, dim = old_embed.weight.shape

        if new_num_tokens == old_num:
            return self

        # create new embedding
        new_embed = nn.Embedding(new_num_tokens, dim)
        new_embed.weight.data[:old_num] = old_embed.weight.data
        new_embed.weight.data[old_num:] = old_embed.weight.data[-1:].expand(new_num_tokens - old_num, -1)

        self.gpt.transformer.wte = new_embed
        self.gpt.lm_head = nn.Linear(dim, new_num_tokens, bias=False)
        self.tie_weights()
        return self


# ------------------------------
# Example usage script: load, save, then fine-tune with Trainer
# ------------------------------
if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # 1) load checkpoint hyperparams
    ckpt = torch.load("out/ckpt.pt", map_location="cpu")
    hf_config = MyGPTConfig(**ckpt["model_args"])

    # 2) load model from raw checkpoint
    model = MyGPTForCausalLM.from_pretrained_checkpoint("out/ckpt.pt", hf_config)

    # 3) load tokenizer (from your `training-gpt-2/out` directory)
    tokenizer = AutoTokenizer.from_pretrained("training-gpt-2/out")
    # make sure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))

    # 4) save in HF format
    save_dir = "hf_my_gpt/"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print(f"✅ Model and tokenizer saved to {save_dir}")
