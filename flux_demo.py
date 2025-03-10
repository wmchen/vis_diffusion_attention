import torch
from diffusers import FluxPipeline

from vis_attn import VisAttentionHook


pipe = FluxPipeline.from_pretrained("/home/ailab/ailab_weights/flux/FLUX.1-dev/", torch_dtype=torch.bfloat16)
pipe.to("cuda")

hook = VisAttentionHook(pipe, vis_cross_attn=True, vis_self_attn=True)
prompt = "Two tigers sitting on the ground, the left tiger is wearing a blue hat, the right tiger is wearing a red hat."
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
).images[0]
hook.vis_attention_maps(image, prompt=prompt, max_columns=16)
