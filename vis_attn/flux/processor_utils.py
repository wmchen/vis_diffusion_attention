import torch
from diffusers.models.attention_processor import Attention


class AttnProcessorWithHook:
    def __init__(self, hook):
        self.hook = hook

    @staticmethod
    def reshape_seq_to_heads(x: torch.Tensor, attn: Attention):
        assert len(x.shape) == 3
        batch_size, _, inner_dim = x.shape
        head_dim = inner_dim // attn.heads
        x = x.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        return x
    
    @staticmethod
    def reshape_heads_to_seq(x: torch.Tensor):
        assert len(x.shape) == 4
        batch_size, heads, _, head_dim = x.shape
        x = x.transpose(1, 2).reshape(batch_size, -1, heads * head_dim)
        return x
