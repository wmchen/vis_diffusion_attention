from typing import Optional, Union, List

import numpy as np
from diffusers import FluxPipeline
from diffusers.models.attention_processor import FluxAttnProcessor2_0

from .processor_utils import AttnProcessorWithHook


def renorm_attention(attn: np.ndarray, renorm: bool):
    if renorm:
        attn *= 255.
        attn = attn.astype(np.uint8)
        if (np.max(attn) - np.min(attn)) > 0:
            attn = attn.astype(np.float32)
            attn = (attn - np.min(attn)) / (np.max(attn) - np.min(attn))
            attn *= 255.
            attn = attn.astype(np.uint8)
    else:
        attn = (attn - np.min(attn)) / (np.max(attn) - np.min(attn))
        attn *= 255.
        attn = attn.astype(np.uint8)
    return attn


class AttentionHook:
    def __init__(
        self, 
        pipe: FluxPipeline, 
        hook_steps: Optional[Union[int, List[int]]] = None,
        hook_trans_block: bool = True,
        hook_single_trans_block: bool = False,
        num_inference_steps: int = 50,
        max_sequence_length: int = 512,
        height: int = 1024,
        width: int = 1024,
    ):
        assert hook_steps is None or isinstance(hook_steps, (int, list)), "hook_steps must be int or list of int if specified."

        self.pipe = pipe
        self.hook_trans_block = hook_trans_block
        self.hook_single_trans_block = hook_single_trans_block
        self.num_inference_steps = num_inference_steps
        self.max_sequence_length = max_sequence_length
        self.height = height
        self.width = width

        num_transformer_blocks = self.pipe.transformer.config.num_layers
        num_single_transformer_blocks = self.pipe.transformer.config.num_single_layers
        if hook_steps is None:
            hook_steps = [i for i in range(num_inference_steps)]
        else:
            if isinstance(hook_steps, int):
                hook_steps = [hook_steps]
            for step in hook_steps:
                assert step < num_inference_steps, f"step {step} is out of range."
        self.hook_steps = hook_steps
        self.trans_selected_ids = []
        self.single_trans_selected_ids = []
        for step in hook_steps:
            if hook_trans_block:
                self.trans_selected_ids.extend([i for i in range(step*num_transformer_blocks, (step+1)*num_transformer_blocks)])
            if hook_single_trans_block:
                self.single_trans_selected_ids.extend([i for i in range(step*num_single_transformer_blocks, (step+1)*num_single_transformer_blocks)])

        self.transformer = pipe.transformer
        self.trans_block_counter = 0
        self.single_trans_block_counter = 0

    def apply(
        self, 
        trans_attn_processor: AttnProcessorWithHook, 
        single_trans_attn_processor: AttnProcessorWithHook
    ):
        def parse(net, block_type: str):
            if net.__class__.__name__ == "Attention":
                if block_type == "transformer_block":
                    if self.hook_trans_block:
                        net.set_processor(trans_attn_processor(self))
                elif block_type == "single_transformer_block":
                    if self.hook_single_trans_block:
                        net.set_processor(single_trans_attn_processor(self))
            elif hasattr(net, "children"):
                for child in net.children():
                    parse(child, block_type)
        
        for name, module in self.transformer.named_children():
            if name == "transformer_blocks":
                parse(module, "transformer_block")
            elif name == "single_transformer_blocks":
                parse(module, "single_transformer_block")

    def remove(self):
        def parse(net):
            if net.__class__.__name__ == "Attention":
                net.set_processor(FluxAttnProcessor2_0())
            elif hasattr(net, "children"):
                for child in net.children():
                    parse(child)
        
        for name, module in self.transformer.named_children():
            if name == "transformer_blocks" or name == "single_transformer_blocks":
                parse(module)

    def reset(self):
        self.trans_block_counter = 0
        self.single_trans_block_counter = 0
