import logging
import fvcore.nn.weight_init as weight_init
import math
from typing import Optional, Tuple, Any
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import List, Tuple
import copy
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from detectron2.config import configurable
from detectron2.layers import Conv2d

from .position_encoding import PositionEmbeddingSine
from .SimpleMaskDecoder import SimpleMaskDecoder, CAResBlock, MultiScaleDownsamplingModule
from ..point_encoder.point_feature_map_encoder import (
    FeatureProjectionMLP, get_resize_feature_map, get_point_feature,
)
from ..point_encoder.point_encoder import PromptEncoder


def deepcopy_tensor_list(tensor_list):
    return [tensor.clone() for tensor in tensor_list]


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class MoE_CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False, num_experts=5):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dropout_val = dropout # Store dropout rate for experts
        self.normalize_before = normalize_before
        self.num_experts = num_experts

        # Each expert is a MultiheadAttention layer
        self.experts = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=dropout) 
            for _ in range(self.num_experts)
        ])

        # Gating network: takes `tgt` features to produce weights for each expert's output
        self.gating_network = nn.Linear(d_model, self.num_experts)

        self.norm_module = nn.LayerNorm(d_model)
        self.dropout_module = nn.Dropout(dropout) # Dropout for the combined expert output

        # activation is stored as in original CrossAttentionLayer but not directly used in attention logic
        self.activation = _get_activation_fn(activation) 
        self.register_buffer("gate_sum", torch.zeros(num_experts))
        self.register_buffer("gate_count", torch.tensor(0.))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Initialize experts' parameters (MultiheadAttention has its own init)
        for expert_module in self.experts:
            for p_expert in expert_module.parameters():
                 if p_expert.dim() > 1:
                    nn.init.xavier_uniform_(p_expert)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None, # Positional encoding for memory (key/value)
                     query_pos: Optional[Tensor] = None,
                     return_weight: Optional[bool]= False): # Positional encoding for tgt (query)

        seq_len_tgt, batch_size, _ = tgt.shape
        
        # Gating based on tgt features
        tgt_flat_for_gating = tgt.reshape(-1, self.d_model)
        gate_logits = self.gating_network(tgt_flat_for_gating) 
        gate_weights = F.softmax(gate_logits, dim=-1) # Shape: (seq_len_tgt * batch_size, num_experts)
        # if gate_weights.shape[0]==131072:
        ave_gate_weight = torch.mean(gate_weights,dim=0)
        self.gate_sum += ave_gate_weight
        self.gate_count += 1
        expert_outputs_collected = []
        q_embedded = self.with_pos_embed(tgt, query_pos)
        k_embedded = self.with_pos_embed(memory, pos)
        # Value is memory itself, without positional encoding typically for value in MHA

        for expert_idx in range(self.num_experts):
            expert_output_attn, _ = self.experts[expert_idx](
                query=q_embedded,
                key=k_embedded,
                value=memory, 
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask
            ) # output shape: (seq_len_tgt, batch_size, d_model)
            expert_outputs_collected.append(expert_output_attn)
        
        # Stack expert outputs: list of (S_tgt, B, D) -> (S_tgt, B, num_experts, D)
        expert_outputs_tensor = torch.stack(expert_outputs_collected, dim=2)

        # Reshape gate_weights for broadcasting: (S_tgt*B, num_experts) -> (S_tgt, B, num_experts)
        gate_weights_reshaped = gate_weights.view(seq_len_tgt, batch_size, self.num_experts)
        
        # Weighted sum of expert outputs
        # gate_weights_reshaped.unsqueeze(3) shape: (S_tgt, B, num_experts, 1)
        # expert_outputs_tensor shape: (S_tgt, B, num_experts, D)
        mo_output = torch.sum(gate_weights_reshaped.unsqueeze(3) * expert_outputs_tensor, dim=2)
        # mo_output shape: (S_tgt, B, D)
        
        tgt = tgt + self.dropout_module(mo_output) # Apply dropout to the attention output mixture
        tgt = self.norm_module(tgt)
        # if return_weight:
        #     return tgt, ave_gate_weight
        # else:
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt_norm = self.norm_module(tgt) # Normalize input tgt first
        
        seq_len_tgt, batch_size, _ = tgt_norm.shape
        
        tgt_norm_flat_for_gating = tgt_norm.reshape(-1, self.d_model)
        gate_logits = self.gating_network(tgt_norm_flat_for_gating)
        gate_weights = F.softmax(gate_logits, dim=-1)

        expert_outputs_collected = []
        # Use normalized tgt for query embedding
        q_embedded = self.with_pos_embed(tgt_norm, query_pos) 
        k_embedded = self.with_pos_embed(memory, pos)

        for expert_idx in range(self.num_experts):
            expert_output_attn, _ = self.experts[expert_idx](
                query=q_embedded,
                key=k_embedded,
                value=memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask
            )
            expert_outputs_collected.append(expert_output_attn)
        
        expert_outputs_tensor = torch.stack(expert_outputs_collected, dim=2)
        
        gate_weights_reshaped = gate_weights.view(seq_len_tgt, batch_size, self.num_experts)
        mo_output = torch.sum(gate_weights_reshaped.unsqueeze(3) * expert_outputs_tensor, dim=2)
        
        tgt = tgt + self.dropout_module(mo_output) # Apply dropout to the attention output mixture
        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


class MoE_FFNLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False, num_experts=4):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.normalize_before = normalize_before
        self.num_experts = num_experts

        # Each expert is a feed-forward network similar to the core of FFNLayer
        self.experts = nn.ModuleList()
        for _ in range(self.num_experts):
            activation_module: nn.Module
            if activation == "relu":
                activation_module = nn.ReLU()
            elif activation == "gelu":
                activation_module = nn.GELU()
            # elif activation == "glu":
                # nn.GLU is a bit different as it changes dimensions and is often used with specific network designs.
                # If GLU is truly needed here, the expert structure might need adjustment or dim_feedforward assumptions.
                # For now, raising an error if 'glu' is passed for MoE experts to prevent unexpected behavior.
            #    raise ValueError(f"Activation 'glu' is not directly supported in this MoE expert nn.Sequential structure. Please use 'relu' or 'gelu'.")
            else:
                # This case should ideally be caught by _get_activation_fn if an unknown string is passed,
                # but added here for robustness within MoE_FFNLayer if activation arg is somehow not validated prior.
                raise ValueError(f"Unsupported activation type '{activation}' for MoE expert. Supported: 'relu', 'gelu'.")
            
            self.experts.append(
                nn.Sequential(
                    nn.Linear(d_model, dim_feedforward),
                    activation_module, # Use the instantiated module
                    nn.Dropout(dropout),
                    nn.Linear(dim_feedforward, d_model)
                )
            )

        # Gating network to produce weights for each expert
        self.gating_network = nn.Linear(d_model, self.num_experts)

        self.dropout_module = nn.Dropout(dropout)
        self.norm_module = nn.LayerNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Initialize experts' parameters
        for expert_module in self.experts:
            for p in expert_module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward_post(self, tgt):
        seq_len, batch_size, d_model = tgt.shape
        tgt_flat = tgt.reshape(-1, d_model)

        gate_logits = self.gating_network(tgt_flat)
        gate_weights = F.softmax(gate_logits, dim=-1)

        expert_outputs_collected = []
        for expert_idx in range(self.num_experts):
            expert_outputs_collected.append(self.experts[expert_idx](tgt_flat))
        
        expert_outputs_tensor = torch.stack(expert_outputs_collected, dim=1)
        
        mo_output_flat = torch.sum(gate_weights.unsqueeze(2) * expert_outputs_tensor, dim=1)

        tgt2 = mo_output_flat.view(seq_len, batch_size, d_model)
        
        tgt = tgt + self.dropout_module(tgt2)
        tgt = self.norm_module(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt_norm = self.norm_module(tgt)
        
        seq_len, batch_size, d_model = tgt_norm.shape
        tgt_flat = tgt_norm.reshape(-1, d_model)

        gate_logits = self.gating_network(tgt_flat)
        gate_weights = F.softmax(gate_logits, dim=-1)

        expert_outputs_collected = []
        for expert_idx in range(self.num_experts):
            expert_outputs_collected.append(self.experts[expert_idx](tgt_flat))
        
        expert_outputs_tensor = torch.stack(expert_outputs_collected, dim=1)
        mo_output_flat = torch.sum(gate_weights.unsqueeze(2) * expert_outputs_tensor, dim=1)
        
        tgt2 = mo_output_flat.view(seq_len, batch_size, d_model)
        
        tgt = tgt + self.dropout_module(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


class PixelFFN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.conv = CAResBlock(dim, dim)

    def forward(self, pixel_flat: torch.Tensor, feature_size) -> torch.Tensor:
        h, w = feature_size
        hw, bs, embed_dim = pixel_flat.shape

        # make sure that the input has the proper size
        assert hw == h * w, f"Expected {h * w}, but got {hw}"
        # adjust the size map
        pixel_flat = pixel_flat.view(h, w, bs, self.dim)
        pixel_flat = pixel_flat.permute(2, 3, 0, 1).contiguous()  # [Batch_size, dim, H, W]

        x = self.conv(pixel_flat)

        # return to the original map
        x = x.permute(2, 3, 0, 1).contiguous()  # [H, W, Batch_size, dim]
        x = x.view(hw, bs, self.dim)  # [(H*W), Batch_size, dim]

        return x


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=scale_factor, stride=scale_factor)

    def forward(self, x):
        return self.upsample(x)


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super(DownsampleBlock, self).__init__()
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=scale_factor, stride=scale_factor)

    def forward(self, x):
        return self.downsample(x)


def create_block_list(in_channels, out_channels):
    block_list = nn.ModuleList()
    block_list.append(UpsampleBlock(in_channels, out_channels, scale_factor=2))
    block_list.append(UpsampleBlock(out_channels, out_channels, scale_factor=2))
    block_list.append(DownsampleBlock(out_channels, out_channels, scale_factor=4))
    block_list.append(UpsampleBlock(out_channels, out_channels, scale_factor=2))
    block_list.append(UpsampleBlock(out_channels, out_channels, scale_factor=2))
    return block_list


class MultiScaleMaskedTransformerDecoder(nn.Module):
    def __init__(
            self,
            model_cfg,
            *,
            pre_norm=False,
            enforce_input_project=False,
            training_mode=True
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        in_channels = model_cfg.model.transformer_decoder.in_channels
        num_classes = model_cfg.model.task_encoder.num_classes
        hidden_dim = model_cfg.model.transformer_decoder.hidden_dim
        num_queries = model_cfg.model.task_encoder.num_queries
        nheads = model_cfg.model.transformer_decoder.nheads
        dim_feedforward = model_cfg.model.transformer_decoder.dim_feedforward
        num_experts = model_cfg.model.transformer_decoder.get("num_experts", 3)

        assert model_cfg.model.transformer_decoder.dec_layers >= 1
        dec_layers = model_cfg.model.transformer_decoder.dec_layers
        mask_dim = model_cfg.model.transformer_decoder.mask_dim

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # define Transformer decoder here
        self.num_summarizer = model_cfg.model.object_summarizer.num_summaries
        self.num_heads = nheads
        self.point_embedding_dim = hidden_dim
        self.num_layers = dec_layers
        #######
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        self.transformer_feature_query_cross_attention_layers = nn.ModuleList()
        self.transformer_pixel_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                MoE_CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                    activation="relu",
                    num_experts=num_experts
                )
            )
            self.transformer_ffn_layers.append(
                MoE_FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                    activation="relu",
                    num_experts=num_experts
                )
            )
            self.transformer_feature_query_cross_attention_layers.append(
                MoE_CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                    activation="relu",
                    num_experts=num_experts
                )
            )

            self.transformer_pixel_ffn_layers.append(
                PixelFFN(dim=hidden_dim)
            )
        #########################
        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.single_object_query = num_queries
        self.num_queries = num_classes * num_queries
        # learnable query features
        self.icl_query_feat = nn.Embedding(self.num_summarizer, hidden_dim)
        self.icl_query_embed = nn.Embedding(self.num_summarizer, hidden_dim)
        self.summary_to_query_init = nn.Linear(hidden_dim, hidden_dim)
        self.summary_to_query_emb = nn.Linear(hidden_dim, hidden_dim)
        self.query_feat = nn.Embedding(num_classes * num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_classes * num_queries, hidden_dim)
        self.num_ref = model_cfg.model.object_summarizer.num_ref
        # self.task_query_embed = nn.Embedding(num_queries + 1, hidden_dim)
        self.num_classes = num_classes
        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())
        # if self.mask_classification:
        #     self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        # self.aggregate_class_layer = MLP(num_queries, num_queries, 1, num_layers=2)

        self.simple_decoders = nn.ModuleList()
        for i in range(self.num_feature_levels):
            self.simple_decoders.append(SimpleMaskDecoder(in_channels=hidden_dim, num_res_blocks=2, out_channels=32))

        self.sample_layer_list = create_block_list(hidden_dim, hidden_dim)
        self.res_query_layer_list = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(dec_layers - 1)])

        self.rescale_list = model_cfg.model.transformer_decoder.rescale
        # click mode
        self.point_encoder = MultiScaleDownsamplingModule(in_channels=3, out_channels=hidden_dim)
        self.training_mode = training_mode
        self.image_size = model_cfg.setting.image_size
        self.mask_classification = True
        # Old checkpoints did not store these modules. Give their fallback
        # initialization a stable seed without changing any global RNG state
        # or the initialization order of the existing model parameters.
        self.click_prompt_seed = int(
            model_cfg.model.transformer_decoder.get("click_prompt_seed", 0)
        )
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.click_prompt_seed)
            self.point_embed_encoder = PromptEncoder(
                embed_dim=hidden_dim, input_image_size=self.image_size,
            )
            # The original forward created separate projections in this order:
            # refresh round, feature level, then positive/negative polarity.
            # Keep those independent weights, including the final refresh,
            # while registering and reusing them for deterministic inference.
            self.click_feature_projection = nn.ModuleList([
                nn.ModuleList([
                    nn.ModuleList([
                        FeatureProjectionMLP(
                            input_dim=hidden_dim,
                            hidden_dim=hidden_dim // 2,
                            output_dim=hidden_dim,
                        )
                        for _ in range(2)
                    ])
                    for _ in range(self.num_feature_levels)
                ])
                for _ in range(1 + self.num_layers // self.num_feature_levels)
            ])

    def forward(self,
                x,
                point_feature_mask,
                point_tuple_list=None,
                pos_point_tuple_list=None,
                neg_point_tuple_list=None,
                click_mode='0',
                query_index=None,
                object_summarizer=None):

        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []
        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            # remain the H*W
            pos.append(self.pe_layer(x[i], None).flatten(2))
            # flatten NxCxHxW to NxCxHW
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])
            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        if click_mode == '0':
            # if click_sub_mode == "semantic":
            object_query_output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
            object_query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            # task_query_embed = self.task_query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            # Q*C --> QxBxC
            pixel_feature_list = deepcopy_tensor_list(src)
            query_output = self.extract_query_vector(object_query_output, label_index_list=query_index)
            query_embed = self.extract_query_vector(object_query_embed, label_index_list=query_index)

            # now pixel_feature_list = [tensor[B,C,H,W], ...., ....]
            # query_output, query_embed = task_query, task_query_embed
            single_object_query_num = query_output.shape[0]
            output_masks_all_layer = self.process_object_query_layer(
                query_output,
                query_embed,
                pixel_feature_list=pixel_feature_list,
                size_list=size_list,
                pos_list=pos,
                attn_mask=None,
                query_num=single_object_query_num)

            predictions_mask = []
            for layer_index in range(self.num_layers):
                predictions_mask.append(output_masks_all_layer[layer_index])  # concat to [16, num_objects, H, W]
            assert len(predictions_mask) == self.num_layers
            out = {
                'pred_masks': predictions_mask[-1],
                'aux_outputs': self._set_aux_loss(predictions_mask)
            }

        elif click_mode == '1':
            # task_query_embed = self.task_query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            #
            # object_query_output, object_query_embed = task_query, task_query_embed
            all_object_query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            all_object_query_output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
            object_query_output = self.extract_query_vector(all_object_query_output, label_index_list=query_index)
            object_query_embed = self.extract_query_vector(all_object_query_embed, label_index_list=query_index)
            single_object_query_num = object_query_output.shape[0]

            pos_click_query_embed, neg_click_query_embed = self.get_point_queries(
                pos_point_tuple_list,
                neg_point_tuple_list,
                input_image_size=self.image_size
            )
            click_query_embed = (pos_click_query_embed, neg_click_query_embed)

            pixel_feature_list = deepcopy_tensor_list(src)
            pos_click_query_output_list, neg_click_query_output_list = self.dense_click_query_embed(
                pixel_feature_list,
                pos_point_tuple_list,
                neg_point_tuple_list,
                rescale_list=self.rescale_list)

            update_point_query_tuple = (pos_point_tuple_list, neg_point_tuple_list)
            click_query_output = (pos_click_query_output_list, neg_click_query_output_list)

            output_masks_all_layer = self.process_click_and_object_layer(
                object_query_output,
                object_query_embed,
                click_query_output,
                click_query_embed,
                pixel_feature_list,
                size_list,
                pos,
                point_feature_mask=point_feature_mask,
                query_num=single_object_query_num,
                update_point_query_tuple=update_point_query_tuple)

            predictions_mask = []

            for layer_index in range(self.num_layers):
                predictions_mask.append(output_masks_all_layer[layer_index])

            assert len(predictions_mask) == self.num_layers

            out = {
                'pred_masks': predictions_mask[-1],
                'aux_outputs': self._set_aux_loss(predictions_mask)
            }

        elif click_mode == '2':
            pos_click_query_embed, neg_click_query_embed = self.get_point_queries(
                pos_point_tuple_list, neg_point_tuple_list, input_image_size=self.image_size
            )
            pixel_feature_list = deepcopy_tensor_list(src)
            # now pixel_feature_list = [tensor[HW,B,C], ...., ....]
            # click_query_num = pos_click_query_embed.shape[0]
            pos_click_query_output_list, neg_click_query_output_list = self.dense_click_query_embed(
                pixel_feature_list,
                pos_point_tuple_list,
                neg_point_tuple_list,
                rescale_list=self.rescale_list)

            update_point_query_tuple = (pos_point_tuple_list, neg_point_tuple_list)

            click_query_embed = (pos_click_query_embed, neg_click_query_embed)
            click_query_output = (pos_click_query_output_list, neg_click_query_output_list)

            output_masks_all_layer = self.process_click_layer(
                click_query_output,
                click_query_embed,
                pixel_feature_list,
                size_list,
                pos,
                attn_mask=None,
                # point_feature_mask=point_feature_mask,
                point_tuple_list=point_tuple_list,
                update_point_query_tuple=update_point_query_tuple)

            predictions_mask = []

            for layer_index in range(self.num_layers):
                predictions_mask.append(output_masks_all_layer[layer_index])

            assert len(predictions_mask) == self.num_layers
            out = {
                'pred_masks': predictions_mask[-1],
                'aux_outputs': self._set_aux_loss(predictions_mask,
                                                  )
            }
        elif click_mode == '3':
            # if click_sub_mode == "semantic":
            object_query_output = self.icl_query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
            object_query_embed = self.icl_query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            # task_query_embed = self.task_query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            # Q*C --> QxBxC
            pixel_feature_list = deepcopy_tensor_list(src)
            # now pixel_feature_list = [tensor[B,C,H,W], ...., ....]
            # query_output, query_embed = task_query, task_query_embed
            single_object_query_num = object_query_output.shape[0]
            output_masks_all_layer = self.process_icl_query_layer(
                object_query_output,
                object_query_embed,
                object_summarizer,
                pixel_feature_list=pixel_feature_list,
                size_list=size_list,
                pos_list=pos,
                attn_mask=None,
                query_num=single_object_query_num)

            predictions_mask = []
            for layer_index in range(self.num_layers):
                predictions_mask.append(output_masks_all_layer[layer_index])  # concat to [16, num_objects, H, W]
            assert len(predictions_mask) == self.num_layers
            out = {
                'pred_masks': predictions_mask[-1],
                'aux_outputs': self._set_aux_loss(predictions_mask)
            }

        elif click_mode == '4':
            # task_query_embed = self.task_query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            #
            # object_query_output, object_query_embed = task_query, task_query_embed
            object_query_output = self.icl_query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
            object_query_embed = self.icl_query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            single_object_query_num = object_query_output.shape[0]

            pos_click_query_embed, neg_click_query_embed = self.get_point_queries(
                pos_point_tuple_list,
                neg_point_tuple_list,
                input_image_size=self.image_size
            )
            click_query_embed = (pos_click_query_embed, neg_click_query_embed)

            pixel_feature_list = deepcopy_tensor_list(src)
            pos_click_query_output_list, neg_click_query_output_list = self.dense_click_query_embed(
                pixel_feature_list,
                pos_point_tuple_list,
                neg_point_tuple_list,
                rescale_list=self.rescale_list)

            update_point_query_tuple = (pos_point_tuple_list, neg_point_tuple_list)
            click_query_output = (pos_click_query_output_list, neg_click_query_output_list)

            output_masks_all_layer = self.process_click_and_icl_layer(
                object_query_output,
                object_query_embed,
                object_summarizer,
                click_query_output,
                click_query_embed,
                pixel_feature_list,
                size_list,
                pos,
                point_feature_mask=point_feature_mask,
                query_num=single_object_query_num,
                update_point_query_tuple=update_point_query_tuple)

            predictions_mask = []

            for layer_index in range(self.num_layers):
                predictions_mask.append(output_masks_all_layer[layer_index])

            assert len(predictions_mask) == self.num_layers

            out = {
                'pred_masks': predictions_mask[-1],
                'aux_outputs': self._set_aux_loss(predictions_mask)
            }

        return out

    def extract_query_vector(self, query_vector, label_index_list):
        object_num = self.num_classes
        batch_size = query_vector.shape[1]
        Q = query_vector.shape[0] // object_num
        query_list = []
        for i in range(batch_size):
            label_index = label_index_list[i]
            index = label_index - 1
            start = index * Q
            end = (index + 1) * Q
            extracted_vector = query_vector[start:end, i, :].unsqueeze(1)
            query_list.append(extracted_vector)
        extracted_query_vector = torch.cat(query_list, dim=1)
        return extracted_query_vector

    def process_object_query_layer(self,
                                   object_query_output,
                                   object_query_embed,
                                   pixel_feature_list,
                                   size_list,
                                   pos_list,
                                   attn_mask=None,
                                   query_num=4):
        output_masks_all_layer = []
        num_of_level = len(size_list)
        attn_mask = attn_mask
        res_pixel_feature = None  # Will be initialized in the first iteration

        for i in range(self.num_layers):
            level_index = i % num_of_level
            attn_masks_level_index = (level_index + 1) % 3
            if isinstance(attn_mask, torch.Tensor):
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            original_pixel_feature = pixel_feature_list[level_index]
            pixel_feature = original_pixel_feature
            # Now the pixel_feature is the size (HW, B, C)
            pixel_feature = self.add_resized_pixel_feature(pixel_feature, res_pixel_feature, iteration=i)
            # Now the pixel_feature is the size (HW, B, C)
            pos = pos_list[level_index]
            object_query_output = self.transformer_cross_attention_layers[i](
                object_query_output, pixel_feature,
                memory_mask=attn_mask,
                memory_key_padding_mask=None,
                pos=pos, query_pos=object_query_embed
            )

            object_query_output = self.transformer_self_attention_layers[i](
                object_query_output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=object_query_embed
            )

            object_query_output = self.transformer_ffn_layers[i](object_query_output)

            pixel_feature = self.transformer_feature_query_cross_attention_layers[i](
                pixel_feature, object_query_output,
                pos=object_query_embed, query_pos=pos
            )

            pixel_feature = self.transformer_pixel_ffn_layers[i](pixel_feature, size_list[level_index])
            # (HW,B,C)
            pixel_feature_list[level_index] = pixel_feature
            pixel_feature_size = size_list[level_index]
            res_pixel_feature = pixel_feature.clone()
            single_outputs_mask = self.single_forward_prediction_heads(pixel_feature, pixel_feature_size, level_index)
            attn_mask = F.interpolate(single_outputs_mask, size=size_list[attn_masks_level_index], mode="bilinear",
                                      align_corners=False)
            attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, query_num,
                                                                            1).flatten(0, 1) < 0.5).bool()
            attn_mask = attn_mask.detach()
            output_masks_all_layer.append(single_outputs_mask)

        return output_masks_all_layer

    def process_icl_query_layer(self,
                                object_query_output,
                                object_query_embed,
                                object_summarizer_list,
                                pixel_feature_list,
                                size_list,
                                pos_list,
                                attn_mask=None,
                                query_num=4):
        output_masks_all_layer = []
        num_of_level = len(size_list)
        attn_mask = attn_mask
        res_pixel_feature = None  # Will be initialized in the first iteration
        # (b, num_ref, num_summaries, v_dim)
        b, num_ref, num_summaries, v_dim = object_summarizer_list[0].shape
        object_query_output = object_query_output.repeat(num_ref, 1, 1)
        object_query_embed = object_query_embed.repeat(num_ref, 1, 1)
        # (R*Q,B,C)
        # obj_value = obj_value.view(b, num_ref * num_summaries, v_dim)
        for i in range(self.num_layers):
            level_index = i % num_of_level
            attn_masks_level_index = (level_index + 1) % 3
            if isinstance(attn_mask, torch.Tensor):
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            original_pixel_feature = pixel_feature_list[level_index]
            obj_values = object_summarizer_list[level_index]
            # (b, num_ref, num_summaries, v_dim)
            obj_values = obj_values.view(b, num_ref * num_summaries, v_dim)
            obj_values = obj_values.transpose(0, 1)
            # (R*Q,B,C)
            obj_init = self.summary_to_query_init(obj_values)
            obj_emb = self.summary_to_query_emb(obj_values)
            query_out = object_query_output + obj_init
            query_embed = object_query_embed + obj_emb
            pixel_feature = original_pixel_feature
            # Now the pixel_feature is the size (HW, B, C)
            pixel_feature = self.add_resized_pixel_feature(pixel_feature, res_pixel_feature, iteration=i)
            # Now the pixel_feature is the size (HW, B, C)
            pos = pos_list[level_index]
            query_out = self.transformer_cross_attention_layers[i](
                query_out, pixel_feature,
                memory_mask=attn_mask,
                memory_key_padding_mask=None,
                pos=pos, query_pos=query_embed
            )

            query_out = self.transformer_self_attention_layers[i](
                query_out, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            query_out = self.transformer_ffn_layers[i](query_out)

            pixel_feature = self.transformer_feature_query_cross_attention_layers[i](
                pixel_feature, query_out,
                pos=query_embed, query_pos=pos
            )

            pixel_feature = self.transformer_pixel_ffn_layers[i](pixel_feature, size_list[level_index])
            # (HW,B,C)
            pixel_feature_list[level_index] = pixel_feature
            pixel_feature_size = size_list[level_index]
            res_pixel_feature = pixel_feature.clone()
            single_outputs_mask = self.single_forward_prediction_heads(pixel_feature, pixel_feature_size, level_index)
            attn_mask = F.interpolate(single_outputs_mask, size=size_list[attn_masks_level_index], mode="bilinear",
                                      align_corners=False)
            # b,1,h,w
            fore_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, num_summaries//2,
                                                                            1).flatten(0, 1) < 0.5).bool()
            back_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, num_summaries//2,
                                                                            1).flatten(0, 1) > 0.5).bool()
            attn_mask = torch.cat([fore_mask, back_mask], dim=1)
            attn_mask = attn_mask.repeat(1, num_ref, 1)
            attn_mask = attn_mask.detach()
            output_masks_all_layer.append(single_outputs_mask)

        return output_masks_all_layer

    def process_click_and_object_layer(self,
                                       object_query_output,
                                       object_query_embed,
                                       click_query_output,
                                       click_query_embed,
                                       pixel_feature_list,
                                       size_list,
                                       pos_list,
                                       point_feature_mask=None,
                                       query_num=4,
                                       update_point_query_tuple=None):
        output_masks_all_layer = []
        num_of_level = len(size_list)

        pos_click_query_output_list, neg_click_query_output_list = click_query_output[0], click_query_output[1]
        pos_query_embed, neg_query_embed = click_query_embed[0], click_query_embed[1]
        point_feature_mask = point_feature_mask.squeeze(1)
        prev_mask = point_feature_mask[:, 2, :, :]  # [B, 256, 256]
        # attn_mask = F.interpolate(prev_mask.unsqueeze(1), size=size_list[0], mode="bilinear",
        #                           align_corners=False)
        # attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, query_num,
        #                                                                 1).flatten(0, 1) < 0.5).bool()
        # # (B*h, query_num, size)
        attn_mask = None
        res_pixel_feature = None
        # Will be initialized in the first iteration
        for i in range(self.num_layers):
            level_index = i % num_of_level
            attn_masks_level_index = (level_index + 1) % 3

            pos_query_output = pos_click_query_output_list[level_index]
            # (Q,B,C)
            neg_query_output = neg_click_query_output_list[level_index]
            # original_click_query_output = torch.cat((original_pos_query_output, original_neg_query_output), dim=0)
            pos_query_num, neg_query_num = pos_query_output.shape[0], neg_query_output.shape[0]
            pos_click_attn_mask = None
            # (B*H,Q,HW)
            neg_click_attn_mask = None

            combine_attn_mask = None
            # if isinstance(combine_attn_mask, torch.Tensor):
            #     combine_attn_mask[torch.where(combine_attn_mask.sum(-1) == combine_attn_mask.shape[-1])] = False

            original_pixel_feature = pixel_feature_list[level_index]
            pixel_feature = original_pixel_feature
            # Now the pixel_feature is the size (HW, B, C)
            pixel_feature = self.add_resized_pixel_feature(pixel_feature, res_pixel_feature, iteration=i)
            # Now the pixel_feature is the size (HW, B, C)

            pos = pos_list[level_index]
            combine_query_output = torch.concat(
                (object_query_output, pos_query_output, neg_query_output),
                dim=0)
            combine_query_embed = torch.concat(
                (object_query_embed, pos_query_embed, neg_query_embed),
                dim=0)

            combine_query_output = self.transformer_cross_attention_layers[i](
                combine_query_output, pixel_feature,
                memory_mask=combine_attn_mask,
                memory_key_padding_mask=None,
                pos=pos, query_pos=combine_query_embed
            )
            combine_query_output = self.transformer_self_attention_layers[i](
                combine_query_output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=combine_query_embed
            )
            combine_query_output = self.transformer_ffn_layers[i](combine_query_output)

            pixel_feature = self.transformer_feature_query_cross_attention_layers[i](
                pixel_feature, combine_query_output,
                pos=combine_query_embed, query_pos=pos
            )

            pixel_feature = self.transformer_pixel_ffn_layers[i](pixel_feature, size_list[level_index])

            pixel_feature_list[level_index] = pixel_feature
            pixel_feature_size = size_list[level_index]
            res_pixel_feature = pixel_feature.clone()

            single_outputs_mask = self.single_forward_prediction_heads(pixel_feature, pixel_feature_size, level_index)

            attn_mask = F.interpolate(single_outputs_mask, size=size_list[attn_masks_level_index], mode="bilinear",
                                      align_corners=False)
            attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, query_num,
                                                                            1).flatten(0, 1) < 0.5).bool()
            attn_mask = attn_mask.detach()

            if level_index == 2:
                (pos_point_tuple_list, neg_point_tuple_list) = update_point_query_tuple
                pos_click_query_output_list, neg_click_query_output_list = self.dense_click_query_embed(
                    pixel_feature_list, pos_point_tuple_list,
                    neg_point_tuple_list, self.rescale_list,
                    projection_round=(i + 1) // self.num_feature_levels)
            output_masks_all_layer.append(single_outputs_mask)

        return output_masks_all_layer

    def process_click_and_icl_layer(self,
                                    object_query_output,
                                    object_query_embed,
                                    object_summarizer_list,
                                    click_query_output,
                                    click_query_embed,
                                    pixel_feature_list,
                                    size_list,
                                    pos_list,
                                    point_feature_mask=None,
                                    query_num=4,
                                    update_point_query_tuple=None):
        output_masks_all_layer = []
        num_of_level = len(size_list)

        pos_click_query_output_list, neg_click_query_output_list = click_query_output[0], click_query_output[1]
        pos_query_embed, neg_query_embed = click_query_embed[0], click_query_embed[1]

        point_feature_mask = point_feature_mask.squeeze(1)
        prev_mask = point_feature_mask[:, 2, :, :]  # [B, 256, 256]

        b, num_ref, num_summaries, v_dim = object_summarizer_list[0].shape
        object_query_output = object_query_output.repeat(num_ref, 1, 1)
        object_query_embed = object_query_embed.repeat(num_ref, 1, 1)

        attn_mask = F.interpolate(prev_mask.unsqueeze(1), size=size_list[0], mode="bilinear",
                                  align_corners=False)
        # (B*h, query_num, size)
        fore_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, num_summaries // 2,
                                                                        1).flatten(0, 1) < 0.5).bool()
        back_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, num_summaries // 2,
                                                                        1).flatten(0, 1) > 0.5).bool()
        attn_mask = torch.cat([fore_mask, back_mask], dim=1)
        attn_mask = attn_mask.repeat(1, num_ref, 1)
        attn_mask = attn_mask.detach()

        res_pixel_feature = None
        # Will be initialized in the first iteration
        for i in range(self.num_layers):
            level_index = i % num_of_level
            attn_masks_level_index = (level_index + 1) % 3

            pos_query_output = pos_click_query_output_list[level_index]
            # (Q,B,C)
            neg_query_output = neg_click_query_output_list[level_index]
            # original_click_query_output = torch.cat((original_pos_query_output, original_neg_query_output), dim=0)

            pos_query_num, neg_query_num = pos_query_output.shape[0], neg_query_output.shape[0]
            pos_click_attn_mask = attn_mask[:, 0, :].unsqueeze(1).repeat(1, pos_query_num, 1)
            # (B*H,Q,HW)
            neg_click_attn_mask = ~(attn_mask[:, 0, :].unsqueeze(1).repeat(1, neg_query_num, 1))

            combine_attn_mask = torch.concat(
                (attn_mask, pos_click_attn_mask, neg_click_attn_mask),
                dim=1)
            if isinstance(combine_attn_mask, torch.Tensor):
                combine_attn_mask[torch.where(combine_attn_mask.sum(-1) == combine_attn_mask.shape[-1])] = False

            original_pixel_feature = pixel_feature_list[level_index]
            pixel_feature = original_pixel_feature
            # Now the pixel_feature is the size (HW, B, C)
            pixel_feature = self.add_resized_pixel_feature(pixel_feature, res_pixel_feature, iteration=i)
            # Now the pixel_feature is the size (HW, B, C)

            pos = pos_list[level_index]
            obj_values = object_summarizer_list[level_index]
            # (b, num_ref, num_summaries, v_dim)
            obj_values = obj_values.view(b, num_ref * num_summaries, v_dim)
            obj_values = obj_values.transpose(0, 1)
            # (R*Q,B,C)
            obj_init = self.summary_to_query_init(obj_values)
            obj_emb = self.summary_to_query_emb(obj_values)
            query_out = object_query_output + obj_init
            query_embed = object_query_embed + obj_emb
            combine_query_output = torch.concat(
                (query_out, pos_query_output, neg_query_output),
                dim=0)
            combine_query_embed = torch.concat(
                (query_embed, pos_query_embed, neg_query_embed),
                dim=0)

            combine_query_output = self.transformer_cross_attention_layers[i](
                combine_query_output, pixel_feature,
                memory_mask=combine_attn_mask,
                memory_key_padding_mask=None,
                pos=pos, query_pos=combine_query_embed
            )
            combine_query_output = self.transformer_self_attention_layers[i](
                combine_query_output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=combine_query_embed
            )
            combine_query_output = self.transformer_ffn_layers[i](combine_query_output)

            pixel_feature = self.transformer_feature_query_cross_attention_layers[i](
                pixel_feature, combine_query_output,
                pos=combine_query_embed, query_pos=pos
            )

            pixel_feature = self.transformer_pixel_ffn_layers[i](pixel_feature, size_list[level_index])

            pixel_feature_list[level_index] = pixel_feature
            pixel_feature_size = size_list[level_index]
            res_pixel_feature = pixel_feature.clone()

            single_outputs_mask = self.single_forward_prediction_heads(pixel_feature, pixel_feature_size, level_index)

            attn_mask = F.interpolate(single_outputs_mask, size=size_list[attn_masks_level_index], mode="bilinear",
                                      align_corners=False)
            fore_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, num_summaries//2,
                                                                            1).flatten(0, 1) < 0.5).bool()
            back_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, num_summaries//2,
                                                                            1).flatten(0, 1) > 0.5).bool()
            attn_mask = torch.cat([fore_mask, back_mask], dim=1)
            attn_mask = attn_mask.repeat(1, num_ref, 1)
            attn_mask = attn_mask.detach()

            if level_index == 2:
                (pos_point_tuple_list, neg_point_tuple_list) = update_point_query_tuple
                pos_click_query_output_list, neg_click_query_output_list = self.dense_click_query_embed(
                    pixel_feature_list, pos_point_tuple_list,
                    neg_point_tuple_list, self.rescale_list,
                    projection_round=(i + 1) // self.num_feature_levels)
            output_masks_all_layer.append(single_outputs_mask)

        return output_masks_all_layer

    def process_click_layer(self,
                            click_query_output,
                            click_query_embed,
                            pixel_feature_list,
                            size_list,
                            pos_list,
                            attn_mask=None,
                            # point_feature_mask=None,
                            point_tuple_list=None,
                            update_point_query_tuple=None):

        output_masks_all_layer = []
        num_of_level = len(size_list)
        pos_click_query_output_list, neg_click_query_output_list = click_query_output[0], click_query_output[1]
        pos_query_embed, neg_query_embed = click_query_embed[0], click_query_embed[1]

        attn_mask = attn_mask
        res_pixel_feature = None  # Will be initialized in the first iteration

        for i in range(self.num_layers):
            level_index = i % num_of_level
            attn_masks_level_index = (level_index + 1) % 3

            original_pixel_feature = pixel_feature_list[level_index]
            pos_query_output = pos_click_query_output_list[level_index]
            neg_query_output = neg_click_query_output_list[level_index]
            pos_query_num, neg_query_num = pos_query_output.shape[0], neg_query_output.shape[0]

            if isinstance(attn_mask, torch.Tensor):
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            combine_query_output = torch.concat(
                (pos_query_output, neg_query_output),
                dim=0)
            combine_query_embed = torch.concat(
                (pos_query_embed, neg_query_embed),
                dim=0)

            pixel_feature = original_pixel_feature
            # Now the pixel_feature is the size (HW, B, C)
            pos = pos_list[level_index]
            pixel_feature = self.add_resized_pixel_feature(pixel_feature, res_pixel_feature, iteration=i)

            combine_query_output = self.transformer_cross_attention_layers[i](
                combine_query_output, pixel_feature,
                memory_mask=attn_mask,
                memory_key_padding_mask=None,
                pos=pos, query_pos=combine_query_embed
            )

            # Self Attention
            combine_query_output = self.transformer_self_attention_layers[i](
                combine_query_output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=combine_query_embed
            )

            combine_query_output = self.transformer_ffn_layers[i](combine_query_output)

            pixel_feature = self.transformer_feature_query_cross_attention_layers[i](
                pixel_feature, combine_query_output,
                pos=combine_query_embed, query_pos=pos
            )
            pixel_feature = self.transformer_pixel_ffn_layers[i](pixel_feature, size_list[level_index])

            pixel_feature_list[level_index] = pixel_feature
            pixel_feature_size = size_list[level_index]
            res_pixel_feature = pixel_feature.clone()

            single_outputs_mask = self.single_forward_prediction_heads(pixel_feature, pixel_feature_size, level_index)

            binary_attn_mask = (single_outputs_mask.sigmoid() > 0.5).float()
            resize_scale = self.image_size[0] / size_list[attn_masks_level_index][0]

            point_feature_mask = get_resize_feature_map(point_tuple_list,
                                                        resize_scale=resize_scale,
                                                        target_size=size_list[attn_masks_level_index],
                                                        mask=binary_attn_mask.squeeze(1))
            point_feature_mask = point_feature_mask.squeeze(1)

            attn_mask = self.generate_new_attn_mask(point_feature_mask,
                                                    pos_query_num=pos_query_num,
                                                    neg_query_num=neg_query_num)
            del point_feature_mask
            if level_index == 2:
                (pos_point_tuple_list, neg_point_tuple_list) = update_point_query_tuple
                pos_click_query_output_list, neg_click_query_output_list = self.dense_click_query_embed(
                    pixel_feature_list, pos_point_tuple_list,
                    neg_point_tuple_list, self.rescale_list,
                    projection_round=(i + 1) // self.num_feature_levels)

            output_masks_all_layer.append(single_outputs_mask)

        return output_masks_all_layer

    def single_forward_prediction_heads(self,
                                        pixel_features,
                                        pixel_features_size,
                                        level_index):
        pixel_features = self.process_feature_dim(pixel_features, pixel_features_size)
        simple_decoder = self.simple_decoders[level_index]
        outputs_mask = simple_decoder(pixel_features)
        return outputs_mask

    def generate_new_attn_mask(self, point_feature, pos_query_num=4, neg_query_num=4):
        existing_mask = point_feature[:, 2, :, :]  # shape: (B, H, W)
        new_attn_mask = existing_mask.clamp(min=0, max=1)
        attn_mask = new_attn_mask.unsqueeze(1)
        #
        pos_attn_mask = (attn_mask.flatten(2).unsqueeze(1).repeat(1, self.num_heads, pos_query_num,
                                                                  1).flatten(0, 1) < 0.5).bool()
        neg_attn_mask = (attn_mask.flatten(2).unsqueeze(1).repeat(1, self.num_heads, neg_query_num,
                                                                  1).flatten(0, 1) > 0.5).bool()
        attn_mask = torch.concat((pos_attn_mask, neg_attn_mask), dim=1)
        attn_mask = attn_mask.detach()

        return attn_mask

    def add_resized_pixel_feature(self,
                                  pixel_feature: torch.Tensor,
                                  res_pixel_feature: Optional[torch.Tensor],
                                  iteration: int,
                                  ) -> torch.Tensor:
        """
        Adds a resized residual pixel feature from the previous iteration to the current pixel feature.

        Arguments:
            pixel_feature (torch.Tensor): Current pixel feature with shape (H*W, B, C).
            res_pixel_feature (Optional[torch.Tensor]): Residual pixel feature from the previous iteration.
            iteration (int): Current iteration index.

        Returns:
            torch.Tensor: Updated pixel feature with the resized residual added.
        """
        # Initialize `res_pixel_feature` if it is None
        if res_pixel_feature is None:
            res_pixel_feature = torch.zeros_like(pixel_feature)

        # Only resize and add residual feature after the first iteration
        if iteration > 0:
            # Determine the dimensions for reshaping
            feature_height = feature_width = math.isqrt(res_pixel_feature.shape[0])
            bs, channel = pixel_feature.shape[1], pixel_feature.shape[2]

            # Select the appropriate sampling layer
            sample_layer = self.sample_layer_list[iteration - 1]
            # Reshape and permute residual for resizing
            res_pixel_feature_reshaped = res_pixel_feature.view(feature_height, feature_width, bs, channel).permute(2,
                                                                                                                    3,
                                                                                                                    0,
                                                                                                                    1).contiguous()
            # Resize using the sampling layer
            resized_res_pixel_feature = sample_layer(res_pixel_feature_reshaped)
            # Flatten and permute back to match the shape of `pixel_feature`
            resized_res_pixel_feature = resized_res_pixel_feature.flatten(2).permute(2, 0, 1).contiguous()
            # Add resized residual feature to the current pixel feature
            pixel_feature = pixel_feature + resized_res_pixel_feature

        return pixel_feature

    def get_point_queries(self,
                          pos_point_tuple_list: List[List[Tuple[torch.Tensor, torch.Tensor]]],
                          neg_point_tuple_list: List[List[Tuple[torch.Tensor, torch.Tensor]]],
                          input_image_size: Tuple[int, int],
                          ) -> tuple[Any, Any]:
        """
        Encodes positive and negative points into query embeddings.

        Arguments:
            pos_point_tuple_list (List[Tuple[torch.Tensor, torch.Tensor]]): List of positive point tuples.
            neg_point_tuple_list (List[Tuple[torch.Tensor, torch.Tensor]]): List of negative point tuples.
            input_image_size (Tuple[int, int]): The input image size for normalization.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            click_query_embed, click_query_output, neg_click_query_embed, neg_click_query_output
            Each of shape (Q, B, C), where Q is the number of points, B is the batch size, and C is the embedding dimension.
        """
        # Stack positive and negative points and labels into batches
        batch_pos_points = torch.stack([item[0][0] for item in pos_point_tuple_list], dim=0)  # [B, N, 2]
        batch_pos_labels = torch.stack([item[0][1] for item in pos_point_tuple_list], dim=0)  # [B, N, 1]
        batch_neg_points = torch.stack([item[0][0] for item in neg_point_tuple_list], dim=0)  # [B, N, 2]
        batch_neg_labels = torch.stack([item[0][1] for item in neg_point_tuple_list], dim=0)  # [B, N, 1]

        # Embed positive and negative points
        pos_point_query = self.point_embed_encoder(
            batch_pos_points, batch_pos_labels, pad=False, input_image_size=input_image_size,
        )
        neg_point_query = self.point_embed_encoder(
            batch_neg_points, batch_neg_labels, pad=False, input_image_size=input_image_size,
        )
        # Permute to match desired shape (Q, B, C) for both positive and negative embeddings
        click_query_embed = pos_point_query.permute(1, 0, 2).contiguous()
        # click_query_output = click_query_embed.clone()  # Optionally separate if needed
        neg_click_query_embed = neg_point_query.permute(1, 0, 2).contiguous()
        # neg_click_query_output = neg_click_query_embed.clone()  # Optionally separate if needed
        return click_query_embed, neg_click_query_embed

    @torch.jit.unused
    def _set_aux_loss(self, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_masks": a}
                for a in zip(outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]

    def dense_click_query_embed(
            self,
            pixel_feature_list,
            pos_point_tuple_list: List[List[Tuple[torch.Tensor, torch.Tensor]]],
            neg_point_tuple_list: List[List[Tuple[torch.Tensor, torch.Tensor]]],
            rescale_list,
            *,
            projection_round=0,
    ):
        pos_click_query_list = []
        neg_click_query_list = []
        for level_index in range(len(pixel_feature_list)):
            original_pixel_feature = pixel_feature_list[level_index]

            dense_pos_point_query_output = get_point_feature(original_pixel_feature, pos_point_tuple_list,
                                                             rescale_list[level_index],
                                                             projection=self.click_feature_projection[projection_round][level_index][0])
            dense_neg_point_query_output = get_point_feature(original_pixel_feature, neg_point_tuple_list,
                                                             rescale_list[level_index],
                                                             projection=self.click_feature_projection[projection_round][level_index][1])
            pos_point_query_output = dense_pos_point_query_output
            neg_point_query_output = dense_neg_point_query_output
            pos_click_query_list.append(pos_point_query_output)
            neg_click_query_list.append(neg_point_query_output)
            del dense_pos_point_query_output, dense_neg_point_query_output, pos_point_query_output, neg_point_query_output

        return pos_click_query_list, neg_click_query_list

    @staticmethod
    def process_feature_dim(pixel_feature, pixel_features_size):
        max_hw, batch_size, embed_dim = pixel_feature.shape
        max_height, max_width = pixel_features_size
        assert max_hw == max_height * max_width, "The hw dimension must be equal to height * width"
        # Reshape the tensor to (batch_size, embed_dim, h, w)
        max_pixel_features = pixel_feature.view(max_height, max_width, batch_size, embed_dim)
        processed_features = max_pixel_features.permute(2, 3, 0, 1).contiguous()
        return processed_features
