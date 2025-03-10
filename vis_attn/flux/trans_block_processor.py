import math
from typing import Optional

import torch
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb

from .processor_utils import AttnProcessorWithHook


class VisTransBlockAttnProcessor(AttnProcessorWithHook):
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        query = self.reshape_seq_to_heads(query, attn)
        key = self.reshape_seq_to_heads(key, attn)
        value = self.reshape_seq_to_heads(value, attn)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # `context` projections.
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
        encoder_hidden_states_query_proj = self.reshape_seq_to_heads(encoder_hidden_states_query_proj, attn)
        encoder_hidden_states_key_proj = self.reshape_seq_to_heads(encoder_hidden_states_key_proj, attn)
        encoder_hidden_states_value_proj = self.reshape_seq_to_heads(encoder_hidden_states_value_proj, attn)

        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

        # attention
        query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        scale_factor = 1 / math.sqrt(query.shape[-1])
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight = torch.softmax(attn_weight, dim=-1)

        if self.hook.vis_cross_attn:
            if self.hook.trans_block_counter in self.hook.trans_selected_ids:
                ci_region = attn_weight[:, :, :self.hook.max_sequence_length, self.hook.max_sequence_length:].mean(dim=1)  # N*512*4096
                ic_region = attn_weight[:, :, self.hook.max_sequence_length:, :self.hook.max_sequence_length].mean(dim=1)  # N*4096*512
                cross_attn = ic_region + ci_region.transpose(-2, -1)  # N*4096*512
                if self.hook.cross_attn is None:
                    self.hook.cross_attn = cross_attn.detach().cpu()
                else:
                    self.hook.cross_attn += cross_attn.detach().cpu()
        if self.hook.vis_self_attn:
            if self.hook.trans_block_counter in self.hook.trans_selected_ids:
                self_attn = attn_weight[:, :, self.hook.max_sequence_length:, self.hook.max_sequence_length:].mean(dim=1)  # N*4096*4096
                if self.hook.self_attn is None:
                    self.hook.self_attn = self_attn.detach().cpu()
                else:
                    self.hook.self_attn += self_attn.detach().cpu()
        self.hook.trans_block_counter += 1

        hidden_states = attn_weight @ value
        hidden_states = self.reshape_heads_to_seq(hidden_states)
        hidden_states = hidden_states.to(query.dtype)

        encoder_hidden_states, hidden_states = (
            hidden_states[:, : encoder_hidden_states.shape[1]],
            hidden_states[:, encoder_hidden_states.shape[1] :],
        )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states
