from typing import Optional, Union, List

import numpy as np
from diffusers import DiffusionPipeline
from diffusers.models.attention_processor import AttnProcessor2_0

from .attn_processor import AttnProcessorWithHook


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
        pipe: DiffusionPipeline, 
        hook_steps: Optional[Union[int, List[int]]] = None,
        hook_attn_scale: int = 16,
        hook_down_blocks: bool = True,
        hook_mid_block: bool = True,
        hook_up_blocks: bool = True,
        num_inference_steps: int = 50,
        height: int = 512,
        width: int = 512,
    ):
        assert hook_steps is None or isinstance(hook_steps, (int, list)), "hook_steps must be int or list of int if specified."

        self.pipe = pipe
        self.num_inference_steps = num_inference_steps
        self.height = height
        self.width = width
        self.hook_attn_scale = hook_attn_scale

        # each CrossAttnDownBlock2D has 2 Transformer2DModel
        num_down_blocks = len(list(filter(lambda x: x == "CrossAttnDownBlock2D", self.pipe.unet.config.down_block_types))) * 2
        # each UNetMidBlock2DCrossAttn has 1 Transformer2DModel
        num_mid_block = 1
        # each CrossAttnUpBlock2D has 3 Transformer2DModel
        num_up_blocks = len(list(filter(lambda x: x == "CrossAttnUpBlock2D", self.pipe.unet.config.up_block_types))) * 3
        if hook_steps is None:
            hook_steps = [i for i in range(num_inference_steps)]
        else:
            if isinstance(hook_steps, int):
                hook_steps = [hook_steps]
            for step in hook_steps:
                assert step < num_inference_steps, f"step {step} is out of range."
        self.hook_steps = hook_steps
        self.hook_down_blocks = hook_down_blocks
        self.hook_mid_block = hook_mid_block
        self.hook_up_blocks = hook_up_blocks
        self.down_selected_ids = dict(self_attn=[], cross_attn=[])
        self.mid_selected_ids = dict(self_attn=[], cross_attn=[])
        self.up_selected_ids = dict(self_attn=[], cross_attn=[])
        for step in hook_steps:
            if hook_down_blocks:
                self.down_selected_ids["self_attn"].extend([i for i in range(step*num_down_blocks, (step+1)*num_down_blocks)])
                self.down_selected_ids["cross_attn"].extend([i for i in range(step*num_down_blocks, (step+1)*num_down_blocks)])
            if hook_mid_block:
                self.mid_selected_ids["self_attn"].extend([i for i in range(step*num_mid_block, (step+1)*num_mid_block)])
                self.mid_selected_ids["cross_attn"].extend([i for i in range(step*num_mid_block, (step+1)*num_mid_block)])
            if hook_up_blocks:
                self.up_selected_ids["self_attn"].extend([i for i in range(step*num_up_blocks, (step+1)*num_up_blocks)])
                self.up_selected_ids["cross_attn"].extend([i for i in range(step*num_up_blocks, (step+1)*num_up_blocks)])

        self.down_block_counter = dict(self_attn=0, cross_attn=0)
        self.mid_block_counter = dict(self_attn=0, cross_attn=0)
        self.up_block_counter = dict(self_attn=0, cross_attn=0)

    def apply(
        self, 
        down_block_attn_processor: AttnProcessorWithHook,
        mid_block_attn_processor: AttnProcessorWithHook,
        up_block_attn_processor: AttnProcessorWithHook,
    ):
        def parse(net, block_type: str):
            if net.__class__.__name__ == "Attention":
                if block_type == "down_blocks":
                    if self.hook_down_blocks:
                        net.set_processor(down_block_attn_processor(self))
                elif block_type == "mid_block":
                    if self.hook_mid_block:
                        net.set_processor(mid_block_attn_processor(self))
                elif block_type == "up_blocks":
                    if self.hook_up_blocks:
                        net.set_processor(up_block_attn_processor(self))
            elif hasattr(net, "children"):
                for child in net.children():
                    parse(child, block_type)
        
        for name, module in self.pipe.unet.named_children():
            if name == "down_blocks":
                parse(module, "down_blocks")
            elif name == "mid_block":
                parse(module, "mid_block")
            elif name == "up_blocks":
                parse(module, "up_blocks")

    def remove(self):
        def parse(net):
            if net.__class__.__name__ == "Attention":
                net.set_processor(AttnProcessor2_0())
            elif hasattr(net, "children"):
                for child in net.children():
                    parse(child)
        
        for name, module in self.pipe.unet.named_children():
            if name in ["down_blocks", "mid_block", "up_blocks"]:
                parse(module)

    def reset(self):
        self.down_block_counter = dict(self_attn=0, cross_attn=0)
        self.mid_block_counter = dict(self_attn=0, cross_attn=0)
        self.up_block_counter = dict(self_attn=0, cross_attn=0)
