import logging
import fvcore.nn.weight_init as weight_init
import math
from typing import Optional
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
from .Attn_Block import *
from ..transformer_decoder.position_encoding import PositionEmbeddingSine


class TaskEncoder(nn.Module):
    def __init__(
            self,
            model_cfg,
            *,
            pre_norm=True,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            num_classes: number of classes
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            channels and hidden dim is identical
        """
        super().__init__()
        in_channels = model_cfg.model.task_encoder.in_channels
        embed_dim = model_cfg.model.task_encoder.embed_dim
        num_queries = model_cfg.model.task_encoder.num_queries
        nheads = model_cfg.model.task_encoder.num_heads
        dim_feedforward = model_cfg.model.task_encoder.dim_feedforward
        dec_layers = model_cfg.model.task_encoder.dec_layers
        num_class = model_cfg.model.task_encoder.num_class

        # positional encoding
        N_steps = embed_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        #######
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=embed_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=embed_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=embed_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
        self.proj_conv = nn.Conv2d(96, in_channels, kernel_size=1)
        self.mask_conv = nn.Sequential(
            nn.Conv2d(1, in_channels, kernel_size=3, stride=2, padding=1),
            # (1, 1, 256, 256) -> (1, in_channels//2, 128, 128)
            nn.ReLU(),  # Activation function (optional)
            # nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, stride=2, padding=1),
            # # (1, in_channels//2, 128, 128) -> (1, in_channels, 64, 64)
            # nn.ReLU()  # Activation function (optional)
        )
        # Convolutional layer to fuse concatenated features
        self.fusion_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        # self.single_query_num  = num_queries
        self.num_queries = num_queries
        # learnable query features
        # self.query_feat = nn.Embedding(num_classes * num_queries, embed_dim)
        # # learnable query p.e.
        # self.query_embed = nn.Embedding(num_classes * num_queries, embed_dim)
        self.query_feat = nn.Embedding(num_queries, embed_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, embed_dim)
        self.memory_bank = nn.Parameter(torch.zeros(num_class, num_queries+1,  embed_dim), requires_grad=False)


    def forward(self, ref_feat, ref_mask, query_index):
        """
        NOTE: this interface is experimental.
        Args:
            ref_feat: shapes (channels and stride) of the input features (B,C_1,H/R,W/R)
            ref_mask: number of classes to predict, shape of (B,1,H,W)
        """

        # assert not torch.isnan(ref_feat).any(), "ref_feat contains NaN!"
        # assert not torch.isinf(ref_feat).any(), "ref_feat contains Inf!"
        # assert not torch.isnan(ref_mask).any(), "ref_mask contains NaN!"
        # assert not torch.isinf(ref_mask).any(), "ref_mask contains Inf!"

        ref_feat = self.proj_conv(ref_feat)
        # (B,C,H/R,W/R)
        bs, c, h, w = ref_feat.shape
        pos = self.pe_layer(ref_feat).flatten(2)
        pos = pos.permute(2, 0, 1).contiguous()
        # hw*bs*c
        # ref_mask = ref_mask.to(torch.float32)
        target_h, target_w = ref_mask.shape[-2:]
        original_feat, original_mask = ref_feat, ref_mask
        # ref_feat = self.proj_conv(original_feat)

        icl_query_output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        icl_query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        # (cls*Q)*C --> (cls*Q)xBxC
        # icl_query_output = self.extract_query_vector(icl_query_output, label_index_list=query_index)
        # icl_query_embed = self.extract_query_vector(icl_query_embed, label_index_list=query_index)
        # QxBxC
        ref_feat = F.interpolate(ref_feat, size=(target_h, target_w), mode='bilinear', align_corners=False)
        # (B, C, H, W)
        ref_mask = ref_mask.expand(-1, c, -1, -1)  # Expand mask to match feature channels (B, C, H, W)
        # Compute dot product element-wise
        ref_feat = ref_feat * ref_mask  # Shape: (B, C, H, W)
        ref_query = ref_feat.mean(dim=[2, 3]).unsqueeze(0)  # Shape: (B, C)
        # (B, C, H, W)
        # Make the queries
        attn_mask = self.generate_attn_mask(original_mask, target=(h, w))
        mask_feat = self.mask_conv(original_mask)
        # Concatenate ref_feat and ref_mask along channel dimension
        # merged_feat = torch.cat([ref_feat, ref_mask], dim=1)  # Shape: (B, 2C, H, W)
        # proj_feat = self.proj_conv(original_feat)
        merged_feat = original_feat + mask_feat  # Shape: (B, C, H, W)
        # Fuse the features with another convolution
        merged_feat = self.fusion_conv(merged_feat)
        merged_feat = merged_feat.flatten(2).permute(2, 0, 1).contiguous()
        # (hw, B, C)
        # hw, B, C = merged_feat.shape
        #
        # merged_feat = torch.randn(hw, B, C, device=merged_feat.device, dtype=merged_feat.dtype)
        if isinstance(attn_mask, torch.Tensor):
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

        for i in range(self.num_layers):
            icl_query_output = self.transformer_cross_attention_layers[i](
                icl_query_output, merged_feat,
                memory_mask=attn_mask,
                memory_key_padding_mask=None,
                pos=pos, query_pos=icl_query_embed
            )

            icl_query_output = self.transformer_self_attention_layers[i](
                icl_query_output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=icl_query_embed
            )

            icl_query_output = self.transformer_ffn_layers[i](icl_query_output)
            # the click part
        icl_query = torch.cat([ref_query, icl_query_output], dim=0)
        self.update_memory(query_index, icl_query)
        return icl_query

    def generate_attn_mask(self, ref_mask, target):
        # (B, H, W)
        new_attn_mask = ref_mask
        new_attn_mask = F.interpolate(new_attn_mask, size=target, mode='bilinear', align_corners=True)
        new_attn_mask = new_attn_mask.clamp(min=0, max=1)
        # attn_mask = new_attn_mask.unsqueeze(1)
        attn_mask = (new_attn_mask.flatten(2).unsqueeze(1).repeat(1, self.num_heads, self.num_queries,
                                                                  1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return attn_mask

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

    def update_memory(self, query_index, task_embeddings, alpha=0.999):
        """
        query_indices:  query class [batch_size]
        task_embeddings: task embedding, [query_num, batch_size, embedding_dim]
        """
        for i in range(len(query_index)):
            class_index = query_index[i]-1
            embedding = task_embeddings[:, i, :]
            if torch.all(self.memory_bank[class_index] == 0):
                self.memory_bank[class_index] = embedding
            else:
                self.memory_bank[class_index] = alpha * self.memory_bank[class_index] + (1 - alpha) * embedding

            self.memory_bank.detach()

    def get_memory_bank(self, query_index):
        """
        query_indices:  query class [batch_size]
        task_embeddings: task embedding, [query_num, batch_size, embedding_dim]
        """
        query_out_list = []
        for i in range(len(query_index)):
            class_index = query_index[i]-1
            embedding = self.memory_bank[class_index, :, :]
            # [5, 256]
            query_out_list.append(embedding)

        query_out_vector = torch.stack(query_out_list, dim=0)  # Stack along batch dim
        query_out_vector = query_out_vector.permute(1, 0, 2).contiguous()
        return query_out_vector  # Shape: [B, 5, 256]
