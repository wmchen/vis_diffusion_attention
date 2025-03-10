import os
import argparse
import random

import torch
from mlcbase import load_toml, create
from accelerate.utils import set_seed

from vis_attn import PIPE, HOOK


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/vis_sdxl_cross_attention.toml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--offload", action="store_true")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    assert os.path.exists(args.config), f"Config file {args.config} does not exist."

    if args.seed is not None:
        seed = args.seed
    else:
        seed = random.randint(0, 2**32 - 1)
    set_seed(seed)
    print(f"[VISUALIZER] seed: {seed}")

    cfg = load_toml(args.config)

    # init model
    model_cfg = cfg.model
    if model_cfg.torch_dtype == "fp32":
        model_cfg.torch_dtype = torch.float32
    elif model_cfg.torch_dtype == "fp16":
        model_cfg.torch_dtype = torch.float16
    elif model_cfg.torch_dtype == "bf16":
        model_cfg.torch_dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported torch_dtype: {model_cfg.torch_dtype}")
    print(f"[VISUALIZER] torch_dtype: {model_cfg.torch_dtype}")
    
    pipe = PIPE.build(model_cfg)
    if args.offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(args.device)
    print(f"[VISUALIZER] model loaded")

    # init hook
    hook_cfg = cfg.hook
    hook_cfg.pipe = pipe
    hook_cfg.save_dir = "attention_outputs/" if hook_cfg.save_dir is None else hook_cfg.save_dir
    save_dir = hook_cfg.save_dir
    hook = HOOK.build(hook_cfg)
    print(f"[VISUALIZER] hook loaded")

    # run
    counter = 0
    for prompt, target_token in zip(cfg.execute.prompts, cfg.execute.target_token):
        print(f"[VISUALIZER] prompt: {prompt}")
        print(f"[VISUALIZER] target token: {target_token}")
        hook.save_dir = os.path.join(save_dir, f"{counter}")
        create(hook.save_dir, "dir")
        print(f"[VISUALIZER] save directory: {hook.save_dir}")

        image = pipe(prompt, **cfg.execute.model).images[0]
        image.save(os.path.join(hook.save_dir, "image.png"))
        hook.vis_attention_maps(image, prompt, target_token, **cfg.execute.hook)
        hook.reset()

        counter += 1


if __name__ == "__main__":
    main()
