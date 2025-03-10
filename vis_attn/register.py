from mlcbase import Registry
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, FluxPipeline

from .stable_diffusion import StableDiffusionVisAttentionHook
from .flux import FluxVisAttentionHook


PIPE = Registry('Pipeline')
PIPE.register_module("StableDiffusionPipeline", StableDiffusionPipeline.from_pretrained)
PIPE.register_module("StableDiffusionXLPipeline", StableDiffusionXLPipeline.from_pretrained)
PIPE.register_module("FluxPipeline", FluxPipeline.from_pretrained)

HOOK = Registry('Hook')
HOOK.register_module("StableDiffusionVisAttentionHook", StableDiffusionVisAttentionHook)
HOOK.register_module("FluxVisAttentionHook", FluxVisAttentionHook)
