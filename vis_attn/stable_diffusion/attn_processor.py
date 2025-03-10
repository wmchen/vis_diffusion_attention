import math
from typing import Optional

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention


class AttnProcessorWithHook:
    def __init__(self, hook):
        self.hook = hook

    @staticmethod
    def reshape_seq_to_heads(x: torch.Tensor, attn: Attention, inner_dim: int):
        assert len(x.shape) == 3
        batch_size = x.shape[0]
        head_dim = inner_dim // attn.heads
        x = x.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        return x
    
    @staticmethod
    def reshape_heads_to_seq(x: torch.Tensor, attn: Attention, inner_dim: int):
        assert len(x.shape) == 4
        head_dim = inner_dim // attn.heads
        batch_size = x.shape[0]
        x = x.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        return x


class VisDownBlockAttnProcessor(AttnProcessorWithHook):
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]

        query = self.reshape_seq_to_heads(query, attn, inner_dim)
        key = self.reshape_seq_to_heads(key, attn, inner_dim)
        value = self.reshape_seq_to_heads(value, attn, inner_dim)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        dtype = query.dtype
        query = query.float()
        key = key.float()
        value = value.float()
        scale_factor = 1 / math.sqrt(query.shape[-1])
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight = torch.softmax(attn_weight, dim=-1)

        if self.hook.vis_cross_attn and attn.is_cross_attention:
            if self.hook.down_block_counter["cross_attn"] in self.hook.down_selected_ids["cross_attn"]:
                cross_attn = attn_weight.mean(dim=1).mean(dim=0)  # S*77
                self.hook.cross_attn["down"].append(cross_attn.detach().cpu())

        if self.hook.vis_self_attn and not attn.is_cross_attention:
            if self.hook.down_block_counter["self_attn"] in self.hook.down_selected_ids["self_attn"]:
                self_attn = attn_weight.mean(dim=1).mean(dim=0)  # S*S
                self.hook.self_attn["down"].append(self_attn.detach().cpu())

        if attn.is_cross_attention:
            self.hook.down_block_counter["cross_attn"] += 1
        else:
            self.hook.down_block_counter["self_attn"] += 1
        
        hidden_states = attn_weight @ value
        hidden_states = self.reshape_heads_to_seq(hidden_states, attn, inner_dim)
        hidden_states = hidden_states.to(dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class VisMidBlockAttnProcessor(AttnProcessorWithHook):
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]

        query = self.reshape_seq_to_heads(query, attn, inner_dim)
        key = self.reshape_seq_to_heads(key, attn, inner_dim)
        value = self.reshape_seq_to_heads(value, attn, inner_dim)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        dtype = query.dtype
        query = query.float()
        key = key.float()
        value = value.float()
        scale_factor = 1 / math.sqrt(query.shape[-1])
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight = torch.softmax(attn_weight, dim=-1)

        if self.hook.vis_cross_attn and attn.is_cross_attention:
            if self.hook.mid_block_counter["cross_attn"] in self.hook.mid_selected_ids["cross_attn"]:
                cross_attn = attn_weight.mean(dim=1).mean(0)  # S*77
                self.hook.cross_attn["mid"].append(cross_attn.detach().cpu())

        if self.hook.vis_self_attn and not attn.is_cross_attention:
            if self.hook.mid_block_counter["self_attn"] in self.hook.mid_selected_ids["self_attn"]:
                self_attn = attn_weight.mean(dim=1).mean(0)  # S*S
                self.hook.self_attn["mid"].append(self_attn.detach().cpu())

        if attn.is_cross_attention:
            self.hook.mid_block_counter["cross_attn"] += 1
        else:
            self.hook.mid_block_counter["self_attn"] += 1
        
        hidden_states = attn_weight @ value
        hidden_states = self.reshape_heads_to_seq(hidden_states, attn, inner_dim)
        hidden_states = hidden_states.to(dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class VisUpBlockAttnProcessor(AttnProcessorWithHook):
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]

        query = self.reshape_seq_to_heads(query, attn, inner_dim)
        key = self.reshape_seq_to_heads(key, attn, inner_dim)
        value = self.reshape_seq_to_heads(value, attn, inner_dim)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        dtype = query.dtype
        query = query.float()
        key = key.float()
        value = value.float()
        scale_factor = 1 / math.sqrt(query.shape[-1])
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight = torch.softmax(attn_weight, dim=-1)

        if self.hook.vis_cross_attn and attn.is_cross_attention:
            if self.hook.up_block_counter["cross_attn"] in self.hook.up_selected_ids["cross_attn"]:
                cross_attn = attn_weight.mean(dim=1).mean(dim=0)  # S*77
                self.hook.cross_attn["up"].append(cross_attn.detach().cpu())

        if self.hook.vis_self_attn and not attn.is_cross_attention:
            if self.hook.up_block_counter["self_attn"] in self.hook.up_selected_ids["self_attn"]:
                self_attn = attn_weight.mean(dim=1).mean(dim=0)  # S*S
                self.hook.self_attn["up"].append(self_attn.detach().cpu())

        if attn.is_cross_attention:
            self.hook.up_block_counter["cross_attn"] += 1
        else:
            self.hook.up_block_counter["self_attn"] += 1
        
        hidden_states = attn_weight @ value
        hidden_states = self.reshape_heads_to_seq(hidden_states, attn, inner_dim)
        hidden_states = hidden_states.to(dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
