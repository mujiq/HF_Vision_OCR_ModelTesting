# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.checkpoint
from peft import LoraConfig, get_peft_model
from timm.models.layers import DropPath
from torch import nn
from transformers import GenerationConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging

from .configuration_internvl import InternVLConfig
from .modeling_intern_vit import (InternVisionEmbeddings, InternVisionEncoder,
                                  InternVisionModel)
from .modeling_qllama import LlamaForCausalLM, _expand_mask, _make_causal_mask

try:
    from .flash_attention import FlashAttention  # v1/v2
except:
    print('FlashAttention is not installed.')

logger = logging.get_logger(__name__)


class InternVLPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = InternVLConfig
    base_model_prefix = 'internvl'
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [
        r'position_ids',
    ]
    _no_split_modules = ['InternAttention', 'LlamaDecoderLayer', 'LlamaForCausalLM']
    _skip_keys_device_placement = 'past_key_values'
    _keep_in_fp32_modules = ['wo']

    # def _init_weights(self, module):
    #     """Initialize the weights"""
    #     factor = self.config.initializer_range
    #     if isinstance(module, nn.Conv2d) or isinstance(module, nn.Embedding) or isinstance(module, nn.Linear):
    #         module.weight.data.normal_(mean=0.0, std=factor)
    #         if hasattr(module, 'bias') and module.bias is not None:
    #             module.bias.data.zero_()
    #     if isinstance(module, InternVisionEmbeddings):
    #         if hasattr(self.config, 'vision_config'):
    #             factor = self.config.vision_config.initializer_range
    #         nn.init.trunc_normal_(module.position_embedding, mean=0.0, std=factor)
    #         nn.init.trunc_normal_(module.class_embedding, mean=0.0, std=factor)
    #     elif isinstance(module, nn.LayerNorm):
    #         module.bias.data.zero_()
    #         module.weight.data.fill_(1.0)
    #     elif isinstance(module, nn.Linear) and module.bias is not None:
    #         module.bias.data.zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, InternVisionModel):
            module.gradient_checkpointing = value
        if isinstance(module, InternVisionEncoder):
            module.gradient_checkpointing = value


class CrossAttention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None, out_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        assert all_head_dim == dim

        self.q = nn.Linear(dim, all_head_dim, bias=False)
        self.k = nn.Linear(dim, all_head_dim, bias=False)
        self.v = nn.Linear(dim, all_head_dim, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.k_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, k=None, v=None):
        B, N, C = x.shape
        N_k = k.shape[1]
        N_v = v.shape[1]

        q_bias, k_bias, v_bias = None, None, None
        if self.q_bias is not None:
            q_bias = self.q_bias
            k_bias = self.k_bias
            v_bias = self.v_bias

        q = F.linear(input=x, weight=self.q.weight, bias=q_bias)
        q = q.reshape(B, N, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)  # (B, N_head, N_q, dim)

        k = F.linear(input=k, weight=self.k.weight, bias=k_bias)
        k = k.reshape(B, N_k, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)

        v = F.linear(input=v, weight=self.v.weight, bias=v_bias)
        v = v.reshape(B, N_v, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B, N_head, N_q, N_k)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class AttentiveBlock(nn.Module):

    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, attn_head_dim=None, out_dim=None):
        super().__init__()

        self.norm1_q = norm_layer(dim)
        self.norm1_k = norm_layer(dim)
        self.norm1_v = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=drop, attn_head_dim=attn_head_dim, out_dim=out_dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x_q, x_kv, pos_q, pos_k, bool_masked_pos, rel_pos_bias=None):
        x_q = self.norm1_q(x_q + pos_q)
        x_k = self.norm1_k(x_kv + pos_k)
        x_v = self.norm1_v(x_kv)
        x = self.cross_attn(x_q, k=x_k, v=x_v)

        return x


class AttentionPoolingBlock(AttentiveBlock):

    def forward(self, x):
        x_q = x.mean(1, keepdim=True)
        x_kv, pos_q, pos_k = x, 0, 0
        x = super().forward(x_q, x_kv, pos_q, pos_k, bool_masked_pos=None, rel_pos_bias=None)
        x = x.squeeze(1)
        return x


@dataclass
class InternVLModelOutput(ModelOutput):
    """
    Class defining the outputs of [`InternVLModelOutput`].
    """

    loss: Optional[torch.FloatTensor] = None
    loss_itm: Optional[torch.FloatTensor] = None
    loss_itc: Optional[torch.FloatTensor] = None
    loss_itg: Optional[torch.FloatTensor] = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k]
            if k not in ['loss', 'loss_itm', 'loss_itc', 'loss_itg']
            else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation.
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return torch.stack(output, 0)

    @staticmethod
    def backward(ctx, grads):
        input, = ctx.saved_tensors
        dist.all_reduce(grads)
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class InternVLModel(InternVLPreTrainedModel):
    config_class = InternVLConfig
    main_input_name = 'pixel_values'

    def __init__(self, config: InternVLConfig):
        super().__init__(config)

        text_hidden_size = config.qllama_config.hidden_size
        vision_hidden_size = config.vision_config.hidden_size
        clip_embed_dim = config.clip_embed_dim
        attn_pool_num_heads = config.attn_pool_num_heads
        config.qllama_config.num_query_token = config.num_query_token
        self.num_query_token = config.num_query_token
        self.label_smoothing = config.label_smoothing

        self.vision_model = InternVisionModel(config.vision_config)  # frozen
        self.qllama = LlamaForCausalLM(config.qllama_config)  # frozen
        self.query_tokens = nn.Parameter(  # trainable
            torch.zeros(1, config.num_query_token, text_hidden_size)
        )

        self.text_projection = nn.Parameter(torch.empty(text_hidden_size, clip_embed_dim))  # frozen
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))  # trainable
        self.clip_projector = AttentionPoolingBlock(  # frozen
            dim=vision_hidden_size, num_heads=attn_pool_num_heads, qkv_bias=True, qk_scale=None,
            drop=0., attn_drop=0., norm_layer=partial(nn.LayerNorm, eps=1e-5), out_dim=clip_embed_dim)
        self.clip_projector2 = AttentionPoolingBlock(  # trainable
            dim=text_hidden_size, num_heads=attn_pool_num_heads, qkv_bias=True, qk_scale=None,
            drop=0., attn_drop=0., norm_layer=partial(nn.LayerNorm, eps=1e-5), out_dim=clip_embed_dim)
        self.itm_head = nn.Linear(text_hidden_size, 2)  # trainable
        self.gradient_checkpointing = True

        # Initialize weights and apply final processing
        # self.post_init()

        if config.use_backbone_lora:
            self.wrap_backbone_lora(r=config.use_backbone_lora, lora_alpha=config.use_backbone_lora * 2)
        if config.use_qllama_lora:
            self.wrap_qllama_lora(r=config.use_qllama_lora,  lora_alpha=config.use_qllama_lora * 2)
        if config.force_image_size:
            self.vision_model.resize_pos_embeddings(
                old_size=config.vision_config.image_size,
                new_size=config.force_image_size,
                patch_size=config.vision_config.patch_size
            )

    def wrap_backbone_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.vision_model = get_peft_model(self.vision_model, lora_config)
        self.vision_model.print_trainable_parameters()

    def wrap_qllama_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                            'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.qllama = get_peft_model(self.qllama, lora_config)
        self.qllama.print_trainable_parameters()

    def get_input_embeddings(self):
        return self.qllama.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.qllama.set_input_embeddings(value)

    def set_output_embeddings(self, new_embeddings):
        self.qllama.set_output_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.qllama.get_output_embeddings()

    @torch.no_grad()
    def _prepare_attention_mask(
            self,
            image_attention_mask: torch.LongTensor,
            attention_mask: torch.LongTensor,
            input_embeds: torch.FloatTensor,
            repeat_time: int,
    ):
        # itm, itc, itg
        attention_mask = torch.cat([image_attention_mask, attention_mask], dim=1)
        expand_mask = _expand_mask(attention_mask, input_embeds.dtype).to(
            input_embeds.device)  # [bsz, 1, tgt_seq_len, src_seq_len]
        itm_mask, itc_mask, itg_mask = torch.chunk(expand_mask, repeat_time, dim=0)

        itc_mask[:, :, :self.num_query_token, self.num_query_token:] = torch.finfo(input_embeds.dtype).min
        itc_mask[:, :, self.num_query_token:, :self.num_query_token] = torch.finfo(input_embeds.dtype).min
        itc_mask_causal = _make_causal_mask(
            (itc_mask.shape[0], itc_mask.shape[2] - self.num_query_token),
            input_embeds.dtype,
            device=input_embeds.device
        )
        # use causal mask for text in itc
        itc_mask[:, :, self.num_query_token:, self.num_query_token:] += itc_mask_causal

        itg_mask_causal = _make_causal_mask(
            (itg_mask.shape[0], itg_mask.shape[2]),
            input_embeds.dtype,
            device=input_embeds.device
        )
        itg_mask = itg_mask + itg_mask_causal
        itg_mask[:, :, :self.num_query_token, :self.num_query_token] = 0
        attention_mask = torch.cat([itm_mask, itc_mask, itg_mask], dim=0)

        return attention_mask

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            positive_input_ids: torch.FloatTensor,
            positive_attention_mask: torch.LongTensor,
            negative_input_ids: torch.FloatTensor,
            negative_attention_mask: torch.LongTensor,
            summarize_input_ids: torch.FloatTensor,
            summarize_attention_mask: torch.LongTensor,
            input_ids: torch.FloatTensor,
            attention_mask: torch.LongTensor,
            labels: torch.LongTensor,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, InternVLModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)
        image_embeds = vision_outputs[0]
        backbone_embeds = self.clip_projector(image_embeds)

        # step 2: prepare input_ids and attention_mask for three sub-tasks:
        # 1) image-text matching; 2) image-text contrastive learning; 3) image-grounded text generation.
        batch_size = input_ids.shape[0]
        self.positive_num = batch_size // 2
        input_ids = torch.cat([negative_input_ids[:-self.positive_num], positive_input_ids[-self.positive_num:],
                               summarize_input_ids, input_ids], dim=0)  # [3 * batch_size, seq_len]
        itm_attention_mask = torch.cat(
            [negative_attention_mask[:-self.positive_num], positive_attention_mask[-self.positive_num:]], dim=0)
        selected = itm_attention_mask.sum(1) - 1
        attention_mask = torch.cat(
            [itm_attention_mask, summarize_attention_mask, attention_mask], dim=0)  # [3 * batch_size, seq_len]

        repeat_time = input_ids.size(0) // batch_size
        # step 3: forward the input_ids and attention_mask through the text encoder.
        input_embeds = self.get_input_embeddings()(input_ids)
        query_tokens = self.query_tokens.repeat(repeat_time * batch_size, 1, 1)
        input_embeds = torch.cat([query_tokens, input_embeds], dim=1)
        image_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=image_embeds.device)
        attention_mask = self._prepare_attention_mask(
            image_attention_mask, attention_mask, input_embeds, repeat_time
        )
        if type(self.qllama.model) == LlamaForCausalLM:
            outputs = self.qllama.model.model.forward_train(
                inputs_embeds=input_embeds,
                vision_hidden_states=image_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                repeat_time=repeat_time,
            ).last_hidden_state
        else:
            outputs = self.qllama.model.forward_train(
                inputs_embeds=input_embeds,
                vision_hidden_states=image_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                repeat_time=repeat_time,
            ).last_hidden_state
        image_embeds = outputs[:, :self.num_query_token]
        text_embeds = outputs[:, self.num_query_token:]
        image_itm, image_itc, image_itg = image_embeds.chunk(repeat_time, dim=0)
        text_itm, text_itc, text_itg = text_embeds.chunk(repeat_time, dim=0)

        ###============== Image-Text Matching ===================###
        image_itm = self.itm_head(image_itm)
        logits = image_itm.mean(dim=1)
        itm_labels = torch.cat([
            torch.zeros(batch_size - self.positive_num, dtype=torch.long, device=logits.device),
            torch.ones(self.positive_num, dtype=torch.long, device=logits.device)
        ], dim=0)
        itm_labels[selected == 1] = -100  # ignore empty texts
        loss_itm = F.cross_entropy(logits, itm_labels)
        neg_match_acc = ((logits[:batch_size - self.positive_num].argmax(dim=-1) == 0) / (
                batch_size - self.positive_num)).sum()
        pos_match_acc = ((logits[-self.positive_num:].argmax(dim=-1) == 1) / self.positive_num).sum()

        ###============== Image-Text Contrastive ===================###
        image_itc = self.clip_projector2(image_itc)

        selected = summarize_attention_mask.sum(1) - 1
        text_itc = text_itc[torch.arange(text_itc.shape[0]), selected]
        text_itc = text_itc @ self.text_projection

        # normalized features
        image_itc = image_itc / image_itc.norm(dim=1, keepdim=True)
        text_itc = text_itc / text_itc.norm(dim=1, keepdim=True)
        image_itc_all = GatherLayer.apply(image_itc).flatten(0, 1)
        text_itc_all = GatherLayer.apply(text_itc).flatten(0, 1)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        sim_i2t = logit_scale * (image_itc @ text_itc_all.t())
        sim_t2i = logit_scale * (text_itc @ image_itc_all.t())
        bs = image_itc.size(0)
        rank = dist.get_rank() if dist.is_initialized() else 0
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=torch.long, device=sim_i2t.device)
        targets[selected == 4] = -100  # ignore empty texts
        loss_itc = (
                           F.cross_entropy(sim_i2t, targets, label_smoothing=self.label_smoothing)
                           + F.cross_entropy(sim_t2i, targets, label_smoothing=self.label_smoothing)
                   ) / 2

        ###============== Image-grounded Text Generation ===================###
        logits = self.qllama.lm_head(text_itg)
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        shift_logits = shift_logits.view(-1, self.qllama.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss_itg = F.cross_entropy(shift_logits, shift_labels)

        vision_sim = F.cosine_similarity(backbone_embeds.detach(), image_itc).mean()

        loss = loss_itm + loss_itc + loss_itg
        if dist.get_rank() == 0:
            print(f'loss: {loss.item()}, loss_itm: {loss_itm.item()}, loss_itc: {loss_itc.item()}, '
                  f'loss_itg: {loss_itg.item()}, vision_similarity: {round(vision_sim.item(), 5)}, '
                  f'logit scale: {round(1.0 / logit_scale.item(), 5)}, '
                  f'pos_match_acc: {round(pos_match_acc.item(), 4)}, '
                  f'neg_match_acc: {round(neg_match_acc.item(), 4)}')

        return InternVLModelOutput(
            loss=loss,
            loss_itc=loss_itc.detach(),
            loss_itm=loss_itm.detach(),
            loss_itg=loss_itg.detach(),
        )

    @torch.no_grad()
    def generate(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.FloatTensor,
            attention_mask: torch.LongTensor,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)
        image_embeds = vision_outputs[0]

        batch_size = image_embeds.shape[0]
        input_embeds = self.get_input_embeddings()(input_ids)
        query_tokens = self.query_tokens.repeat(batch_size, 1, 1)
        input_embeds = torch.cat([query_tokens, input_embeds], dim=1)
        image_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=image_embeds.device)
        attention_mask = torch.cat([image_attention_mask, attention_mask], dim=1)

        outputs = self.qllama.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            vision_hidden_states=image_embeds,
            generation_config=generation_config,
            use_zero_attention_mask=True,
            **generate_kwargs,
        )

        return outputs

    def get_text_features(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        r"""
        Returns:
            text_outputs (`CausalLMOutputWithPast`, or `tuple(torch.FloatTensor)` if `return_dict=False`):
                The language model outputs. If `return_dict=True`, the output is a [`CausalLMOutputWithPast`] that
                contains the language model logits, the past key values and the hidden states if
                `output_hidden_states=True`.
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_embeds = self.get_input_embeddings()(input_ids)
        attention_mask = _expand_mask(attention_mask, input_embeds.dtype).to(
            input_embeds.device)  # [bsz, 1, tgt_seq_len, src_seq_len]
        attention_mask += _make_causal_mask(
            (attention_mask.shape[0], attention_mask.shape[2]),
            input_embeds.dtype,
            device=input_embeds.device
        )
        if type(self.qllama.model) == LlamaForCausalLM:
            outputs = self.qllama.model.model.forward_train(
                inputs_embeds=input_embeds,
                vision_hidden_states=None,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            ).last_hidden_state
        else:
            outputs = self.qllama.model.forward_train(
                inputs_embeds=input_embeds,
                vision_hidden_states=None,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            ).last_hidden_state
        return outputs

    def get_image_features(
            self,
            pixel_values: torch.FloatTensor,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)
        image_embeds = vision_outputs[0]
        backbone_embeds = image_embeds

        batch_size = image_embeds.shape[0]
        input_embeds = self.query_tokens.repeat(batch_size, 1, 1)

        attention_mask = torch.ones(input_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        attention_mask = _expand_mask(attention_mask, input_embeds.dtype).to(
            input_embeds.device)  # [bsz, 1, tgt_seq_len, src_seq_len]
        if type(self.qllama.model) == LlamaForCausalLM:
            outputs = self.qllama.model.model.forward_train(
                inputs_embeds=input_embeds,
                vision_hidden_states=image_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            ).last_hidden_state
        else:
            outputs = self.qllama.model.forward_train(
                inputs_embeds=input_embeds,
                vision_hidden_states=image_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            ).last_hidden_state
        return backbone_embeds, outputs


class InternVL_C(InternVLModel):

    def encode_image(self, image):
        vision_outputs = self.vision_model(
            pixel_values=image,
            output_hidden_states=False,
            return_dict=True)
        image_embeds = vision_outputs[0]
        image_embeds = self.clip_projector(image_embeds)
        return image_embeds

    def encode_text(self, text):
        attention_mask = text > 0
        text_embeds = self.get_text_features(
            input_ids=text,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        text_embeds = text_embeds[torch.arange(text_embeds.shape[0]), attention_mask.sum(1) - 1]
        text_embeds = text_embeds @ self.text_projection
        return text_embeds

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text


class InternVL_G(InternVLModel):

    def encode_image(self, image):
        backbone_embeds, image_embeds = self.get_image_features(
            pixel_values=image,
            output_hidden_states=False,
            return_dict=True,
        )
        backbone_embeds = self.clip_projector(backbone_embeds)
        image_embeds = self.clip_projector2(image_embeds)
        # ensemble
        backbone_embeds = backbone_embeds / backbone_embeds.norm(dim=1, keepdim=True)
        image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True)
        image_embeds = image_embeds + backbone_embeds
        return image_embeds

    def encode_text(self, text):
        attention_mask = text > 0
        text_embeds = self.get_text_features(
            input_ids=text,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        text_embeds = text_embeds[torch.arange(text_embeds.shape[0]), attention_mask.sum(1) - 1]
        text_embeds = text_embeds @ self.text_projection
        return text_embeds

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text
