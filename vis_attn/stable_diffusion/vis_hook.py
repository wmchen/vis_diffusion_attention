import os.path as osp
import math
from typing import Optional, Union, List

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import DiffusionPipeline
from mlcbase import EmojiProgressBar, create

from .hook_utils import AttentionHook, renorm_attention
from .attn_processor import VisDownBlockAttnProcessor, VisMidBlockAttnProcessor, VisUpBlockAttnProcessor


class StableDiffusionVisAttentionHook(AttentionHook):
    def __init__(
        self, 
        pipe: DiffusionPipeline, 
        vis_cross_attn: bool = False, 
        vis_self_attn: bool = False, 
        save_dir: str = "vis_sd_attn_outputs/",
        **kwargs
    ):
        assert vis_cross_attn or vis_self_attn, "Either vis_cross_attn or vis_self_attn must be True."
        super().__init__(pipe, **kwargs)
        self.apply(VisDownBlockAttnProcessor, VisMidBlockAttnProcessor, VisUpBlockAttnProcessor)
        self.vis_cross_attn = vis_cross_attn
        self.vis_self_attn = vis_self_attn
        self.cross_attn = dict(down=[], mid=[], up=[])
        self.self_attn = dict(down=[], mid=[], up=[])
        self.save_dir = save_dir
        create(save_dir, "dir")
    
    def set_prompt(self, prompt: str, target_token: Optional[Union[str, List[str]]] = None):
        assert target_token is None or isinstance(target_token, (str, list)), "target_token must be a string or a list of strings if provided"
        if isinstance(target_token, str):
            target_token = [target_token]
        if isinstance(target_token, list):
            target_text_list = []
            for text in target_token:
                assert isinstance(text, str), "target_token must be a list of strings"
                assert text in prompt, f"target_token '{text}' not found in prompt {prompt}"
                assert text not in target_text_list, f"duplicate target_token '{text}' found"
                target_text_list.append(text)
        
        tokens = self.pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids

        selected_ids = []
        if target_token is not None:
            cache_pool = {}
            for target_text in target_text_list:
                cache_pool[target_text] = {"text_cache": None, "id_cache": []}
        for i in range(self.pipe.tokenizer.model_max_length):
            token_id = tokens[0, i].item()
            if token_id == 0:
                break

            if target_token is None:
                selected_ids.append(i)
            else:
                text = self.pipe.tokenizer.decode(token_id)
                if text == "":
                    continue
                for target_text in target_text_list:
                    if text == target_text:
                        selected_ids.append(i)
                        for k in cache_pool.keys():
                            cache_pool[k] = {"text_cache": None, "id_cache": []}
                    else:
                        if cache_pool[target_text]["text_cache"] is None:
                            if target_text.startswith(text):
                                cache_pool[target_text]["text_cache"] = text
                                cache_pool[target_text]["id_cache"].append(i)
                        else:
                            cache_pool[target_text]["text_cache"] += text
                            if cache_pool[target_text]["text_cache"] == target_text:
                                cache_pool[target_text]["id_cache"].append(i)
                                selected_ids.append(cache_pool[target_text]["id_cache"])
                                cache_pool[target_text]["text_cache"] = None
                                cache_pool[target_text]["id_cache"] = []
                            elif target_text.startswith(cache_pool[target_text]["text_cache"]):
                                cache_pool[target_text]["id_cache"].append(i)
                            else:
                                cache_pool[target_text]["text_cache"] = None
                                cache_pool[target_text]["id_cache"] = []

        return tokens, selected_ids

    def vis_attention_maps(
        self,
        image: Optional[Union[str, Image.Image]] = None,
        prompt: Optional[str] = None,
        target_token: Optional[Union[str, List[str]]] = None,
        stride: int = 16,
        is_horizontal: bool = True,
        renorm: bool = True,
        base_figsize: int = 3,
        max_columns: int = 5,
        apply_heat_map: bool = True,
        alpha: float = 0.5,
        beta: float = 0.5,
        out_suffix: str = ".png"
    ):
        assert self.cross_attn is not None or self.self_attn is not None, "inference the model before visualizing attention maps"
        if apply_heat_map:
            assert image is not None, "image must be provided if apply_heat_map is True"
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            h, w, _ = image.shape

        if self.vis_cross_attn:
            print("[HOOK] visualizing cross attention maps...")
            assert prompt is not None, "prompt must be provided if vis_cross_attn is True"

            save_root = osp.join(self.save_dir, "cross_attention")
            create(save_root, "dir")
            create(osp.join(save_root, "attention_map"), "dir")
            if apply_heat_map:
                create(osp.join(save_root, "heat_map"), "dir")

            print("[HOOK] setting prompt...")
            tokens, selected_ids = self.set_prompt(prompt, target_token)
            print("[HOOK] done")

            if len(selected_ids) <= max_columns:
                rows = 1
                columns = len(selected_ids)
            else:
                rows = math.ceil(len(selected_ids) / max_columns)
                columns = max_columns
            figsize = (base_figsize * columns, base_figsize * rows)
            fig = plt.figure(figsize=figsize)
            axs = fig.subplots(rows, columns)

            print("[HOOK] calculating attention maps...")
            counter = 0
            attns = torch.zeros((len(selected_ids), self.hook_attn_scale, self.hook_attn_scale))
            if self.hook_down_blocks:
                for attn_maps in self.cross_attn["down"]:
                    if attn_maps.shape[0] == self.hook_attn_scale ** 2:
                        for fig_id, index in enumerate(selected_ids):
                            if isinstance(index, int):
                                attn = attn_maps.view(self.hook_attn_scale, self.hook_attn_scale, self.pipe.tokenizer.model_max_length)
                                attn = attn[:, :, index]  # sqrt(S)*sqrt(S)
                                attns[fig_id, :, :] += attn
                            elif isinstance(index, list):
                                attn = torch.zeros(self.hook_attn_scale, self.hook_attn_scale)
                                for i in index:
                                    attn_ = attn_maps.view(self.hook_attn_scale, self.hook_attn_scale, self.pipe.tokenizer.model_max_length)
                                    attn_ = attn_[:, :, i]  # sqrt(S)*sqrt(S)
                                    attn = attn + attn_
                                attn /= len(index)
                                attns[fig_id, :, :] += attn
                        counter += 1
            if self.hook_mid_block:
                for attn_maps in self.cross_attn["mid"]:
                    if attn_maps.shape[0] == self.hook_attn_scale ** 2:
                        for fig_id, index in enumerate(selected_ids):
                            if isinstance(index, int):
                                attn = attn_maps.view(self.hook_attn_scale, self.hook_attn_scale, self.pipe.tokenizer.model_max_length)
                                attn = attn[:, :, index]  # sqrt(S)*sqrt(S)
                                attns[fig_id, :, :] += attn
                            elif isinstance(index, list):
                                attn = torch.zeros(self.hook_attn_scale, self.hook_attn_scale)
                                for i in index:
                                    attn_ = attn_maps.view(self.hook_attn_scale, self.hook_attn_scale, self.pipe.tokenizer.model_max_length)
                                    attn_ = attn_[:, :, i]  # sqrt(S)*sqrt(S)
                                    attn = attn + attn_
                                attn /= len(index)
                                attns[fig_id, :, :] += attn
                        counter += 1
            if self.hook_up_blocks:
                for attn_maps in self.cross_attn["up"]:
                    if attn_maps.shape[0] == self.hook_attn_scale ** 2:
                        for fig_id, index in enumerate(selected_ids):
                            if isinstance(index, int):
                                attn = attn_maps.view(self.hook_attn_scale, self.hook_attn_scale, self.pipe.tokenizer.model_max_length)
                                attn = attn[:, :, index]  # sqrt(S)*sqrt(S)
                                attns[fig_id, :, :] += attn
                            elif isinstance(index, list):
                                attn = torch.zeros(self.hook_attn_scale, self.hook_attn_scale)
                                for i in index:
                                    attn_ = attn_maps.view(self.hook_attn_scale, self.hook_attn_scale, self.pipe.tokenizer.model_max_length)
                                    attn_ = attn_[:, :, i]  # sqrt(S)*sqrt(S)
                                    attn = attn + attn_
                                attn /= len(index)
                                attns[fig_id, :, :] += attn
                        counter += 1
            attns /= counter
            print("[HOOK] done")

            with EmojiProgressBar(total=len(selected_ids)) as pbar:
                for fig_id, index in enumerate(selected_ids):
                    if isinstance(index, int):
                        text = self.pipe.tokenizer.decode(tokens[0, index].item())
                    elif isinstance(index, list):
                        text = "".join([self.pipe.tokenizer.decode(tokens[0, i].item()) for i in index])
                    attn = attns[fig_id, :, :].numpy()
                    attn = renorm_attention(attn, renorm)
                    text = "{END}" if text == "</s>" else text

                    if rows == 1:
                        if columns == 1:
                            ax = axs
                        else:
                            ax = axs[fig_id]
                    else:
                        ax = axs[fig_id // columns, fig_id % columns]
                    ax.axis("off")
                    ax.imshow(attn, cmap="gray")
                    ax.set_title(text)
                    cv2.imwrite(osp.join(save_root, "attention_map", f"{fig_id}_{text}{out_suffix}"), attn)
                    if apply_heat_map:
                        heat_image = cv2.applyColorMap(cv2.resize(attn, (w, h)), cv2.COLORMAP_JET)
                        added_image = cv2.addWeighted(image, alpha, heat_image, beta, 0)
                        cv2.imwrite(osp.join(save_root, "heat_map", f"{fig_id}_{text}{out_suffix}"), added_image)
                    pbar.update(1)

            print("[HOOK] saving cross attention maps...")
            plt.savefig(osp.join(save_root, f"attention_maps{out_suffix}"), bbox_inches="tight")
            print("[HOOK] done")

        if self.vis_self_attn:
            print("[HOOK] visualizing self attention maps...")
            assert stride > 0 and (stride & (stride - 1)) == 0, f"{stride} is not a power of 2"

            save_root = osp.join(self.save_dir, "self_attention")
            create(save_root, "dir")
            create(osp.join(save_root, "attention_map"), "dir")
            if apply_heat_map:
                create(osp.join(save_root, "heat_map"), "dir")

            
            selected_ids = list(range(0, self.hook_attn_scale ** 2, stride))
            if len(selected_ids) <= max_columns:
                rows = 1
                columns = len(selected_ids)
            else:
                rows = math.ceil(len(selected_ids) / max_columns)
                columns = max_columns
            figsize = (base_figsize * columns, base_figsize * rows)
            fig = plt.figure(figsize=figsize)
            axs = fig.subplots(rows, columns)
            
            print("[HOOK] calculating attention maps...")
            counter = 0
            attns = torch.zeros((len(selected_ids), self.hook_attn_scale, self.hook_attn_scale))
            if self.hook_down_blocks:
                for attn_maps in self.self_attn["down"]:
                    if attn_maps.shape[0] == self.hook_attn_scale ** 2:
                        for fig_id, index in enumerate(selected_ids):
                            if is_horizontal:
                                attn = attn_maps[index, :]  # S
                            else:
                                attn = attn_maps[:, index]  # S
                            attn = attn.view(self.hook_attn_scale, self.hook_attn_scale)
                            attns[fig_id, :, :] += attn
                        counter += 1
            if self.hook_mid_block:
                for attn_maps in self.self_attn["mid"]:
                    if attn_maps.shape[0] == self.hook_attn_scale ** 2:
                        for fig_id, index in enumerate(selected_ids):
                            if is_horizontal:
                                attn = attn_maps[index, :]  # S
                            else:
                                attn = attn_maps[:, index]  # S
                            attn = attn.view(self.hook_attn_scale, self.hook_attn_scale)
                            attns[fig_id, :, :] += attn
                        counter += 1
            if self.hook_up_blocks:
                for attn_maps in self.self_attn["up"]:
                    if attn_maps.shape[0] == self.hook_attn_scale ** 2:
                        for fig_id, index in enumerate(selected_ids):
                            if is_horizontal:
                                attn = attn_maps[index, :]  # S
                            else:
                                attn = attn_maps[:, index]  # S
                            attn = attn.view(self.hook_attn_scale, self.hook_attn_scale)
                            attns[fig_id, :, :] += attn
                        counter += 1
            attns /= counter
            print("[HOOK] done")

            with EmojiProgressBar(total=len(selected_ids)) as pbar:
                for fig_id, index in enumerate(selected_ids):
                    attn = attns[fig_id, :, :].numpy()
                    attn = renorm_attention(attn, renorm)

                    if rows == 1:
                        if columns == 1:
                            ax = axs
                        else:
                            ax = axs[fig_id]
                    else:
                        ax = axs[fig_id // columns, fig_id % columns]
                    ax.axis("off")
                    ax.imshow(attn, cmap="gray")
                    ax.set_title(str(fig_id))
                    cv2.imwrite(osp.join(save_root, "attention_map", f"{fig_id}{out_suffix}"), attn)
                    if apply_heat_map:
                        heat_image = cv2.applyColorMap(cv2.resize(attn, (w, h)), cv2.COLORMAP_JET)
                        added_image = cv2.addWeighted(image, alpha, heat_image, beta, 0)
                        cv2.imwrite(osp.join(save_root, "heat_map", f"{fig_id}{out_suffix}"), added_image)
                    pbar.update(1)
            print("[HOOK] saving self attention maps...")
            plt.savefig(osp.join(save_root, f"attention_maps{out_suffix}"), bbox_inches="tight")
            print("[HOOK] done")

    def reset(self):
        self.cross_attn = dict(down=[], mid=[], up=[])
        self.self_attn = dict(down=[], mid=[], up=[])
        super().reset()
