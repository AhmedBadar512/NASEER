import sys, os
# ensure repo root on path for direct script execution
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from transformers import GPTNeoForCausalLM, GPTNeoConfig
from models.layers.naseer_modules import NASEERLayer
import torch.nn as nn

class NASEERAttentionAdapter(nn.Module):
    def __init__(self, naseer_layer):
        super().__init__()
        self.naseer = naseer_layer

    def forward(self, hidden_states, *args, **kwargs):
        # ignore HF-specific extra args, apply NASEER layer
        output = self.naseer(hidden_states)
        # mimic Attention return: (attn_output, present)
        return output, None

def load_naseer_gpt_neo(pretrained_model="EleutherAI/gpt-neo-125M",
                        entangle_method='gated',
                        top_k=None,
                        rank=8,
                        hidden_size=None,
                        num_heads=None):
    """
    Load GPT‑Neo‑125M and replace each block.attn with a NASEERLayer of matching size.
    """
    # Load config & base model
    config = GPTNeoConfig.from_pretrained(pretrained_model)
    model = GPTNeoForCausalLM.from_pretrained(pretrained_model, config=config)

    eff_hidden_size = hidden_size or config.hidden_size
    eff_num_heads = num_heads or config.num_attention_heads

    # Swap out attention modules
    for block in model.transformer.h:
        block.attn = NASEERAttentionAdapter(
            NASEERLayer(
                hidden_size=eff_hidden_size,
                num_heads=eff_num_heads,
                top_k=top_k,
                entangle_method=entangle_method,
                rank=rank
            )
        )

    return model

if __name__ == "__main__":
    # quick check of NASEER‑GPT‑Neo structure
    model = load_naseer_gpt_neo()
    print("NASEER‑GPT‑Neo‑125M block attention modules:")
    print("Block\tAttentionLayer")
    for idx, block in enumerate(model.transformer.h):
        print(f"{idx}\t{block.attn.__class__.__name__}")
