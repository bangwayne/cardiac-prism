from typing import List, Dict, Optional
from omegaconf import DictConfig
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_emb(sin_inp: torch.Tensor) -> torch.Tensor:
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding(nn.Module):
    def __init__(self,
                 dim: int,
                 scale: float = math.pi * 2,
                 temperature: float = 10000,
                 normalize: bool = True,
                 channel_last: bool = True,
                 transpose_output: bool = False):
        super().__init__()
        dim = int(np.ceil(dim / 4) * 2)
        self.dim = dim
        inv_freq = 1.0 / (temperature ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.normalize = normalize
        self.scale = scale
        self.eps = 1e-6
        self.channel_last = channel_last
        self.transpose_output = transpose_output

        self.cached_penc = None  # the cache is irrespective of the number of objects

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        :param tensor: A 4/5d tensor of size
            channel_last=True: (batch_size, h, w, c) or (batch_size, k, h, w, c)
            channel_last=False: (batch_size, c, h, w) or (batch_size, k, c, h, w)
        :return: positional encoding tensor that has the same shape as the input if the input is 4d
                 if the input is 5d, the output is broadcastable along the k-dimension
        """
        if len(tensor.shape) != 4 and len(tensor.shape) != 5:
            raise RuntimeError(f'The input tensor has to be 4/5d, got {tensor.shape}!')

        if len(tensor.shape) == 5:
            # take a sample from the k dimension
            num_objects = tensor.shape[1]
            tensor = tensor[:, 0]
        else:
            num_objects = None

        if self.channel_last:
            batch_size, h, w, c = tensor.shape
        else:
            batch_size, c, h, w = tensor.shape

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            if num_objects is None:
                return self.cached_penc
            else:
                return self.cached_penc.unsqueeze(1)

        self.cached_penc = None

        pos_y = torch.arange(h, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_x = torch.arange(w, device=tensor.device, dtype=self.inv_freq.dtype)
        if self.normalize:
            pos_y = pos_y / (pos_y[-1] + self.eps) * self.scale
            pos_x = pos_x / (pos_x[-1] + self.eps) * self.scale

        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_y = get_emb(sin_inp_y).unsqueeze(1)
        emb_x = get_emb(sin_inp_x)

        emb = torch.zeros((h, w, self.dim * 2), device=tensor.device, dtype=tensor.dtype)
        emb[:, :, :self.dim] = emb_x
        emb[:, :, self.dim:] = emb_y

        if not self.channel_last and self.transpose_output:
            # cancelled out
            pass
        elif (not self.channel_last) or (self.transpose_output):
            emb = emb.permute(2, 0, 1).contiguous()

        self.cached_penc = emb.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        if num_objects is None:
            return self.cached_penc
        else:
            return self.cached_penc.unsqueeze(1)


def _weighted_pooling(masks: torch.Tensor, value: torch.Tensor,
                      logits: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
    Perform weighted masked pooling over feature maps to summarize object regions.

    Args:
        masks (torch.Tensor):
            Binary or soft masks indicating allowed regions for pooling.
            Shape: (B * num_ref, 1, H, W, num_summaries)
            - B: batch size
            - H, W: height and width of feature maps
            - num_summaries: number of regions (e.g., foreground/background splits)

        value (torch.Tensor):
            Feature map values to be pooled.
            Shape: (B * num_ref, H, W, value_dim)
            - value_dim: feature dimension after projection

        logits (torch.Tensor):
            Raw attention logits per pixel and summary.
            Shape: (B * num_ref, H, W, num_summaries)
            - Used to generate soft attention weights via sigmoid activation.

    Returns:
        obj_values (torch.Tensor):
            Object-level summarized features after weighted pooling.
            Shape: (B * num_ref, num_summaries, value_dim)
            - Each summary corresponds to one foreground/background region.
    """
    masks = masks.squeeze(1)
    # (B, H, W, num_summaries)
    weights = logits.sigmoid() * masks
    # (B, H, W, num_summaries)
    sums = torch.einsum('bhwq,bhwc->bqc', weights, value)
    # B*H*W*num_summaries -> (B, num_summaries, value_dim)
    area = weights.flatten(start_dim=1, end_dim=2).sum(1).unsqueeze(-1)
    # (B, num_summaries, 1)
    obj_values = sums / (area + 1e-4)
    # (B, num_summaries, value_dim)
    return obj_values


class ObjectSummarizer(nn.Module):
    def __init__(self, model_cfg: DictConfig):
        super().__init__()

        this_cfg = model_cfg.model.object_summarizer
        # self.value_dim = model_cfg.value_dim
        self.embed_dim = this_cfg.embed_dim
        self.num_summaries = this_cfg.num_summaries
        self.add_pe = this_cfg.add_pe
        self.pixel_pe_scale = this_cfg.pixel_pe_scale
        self.pixel_pe_temperature = this_cfg.pixel_pe_temperature
        self.multi_in_channels = model_cfg.model.backbone.out_feature_channels

        if self.add_pe:
            self.pos_enc = PositionalEncoding(self.embed_dim,
                                              scale=self.pixel_pe_scale,
                                              temperature=self.pixel_pe_temperature)

        input_proj_list = []
        feature_pred_list = []
        weights_pred_list = []
        # from low resolution to high resolution (res5 -> res2)
        for name in list(self.multi_in_channels.keys())[::-1]:
            in_channels = self.multi_in_channels[name]
            input_proj_list.append(nn.Sequential(
                nn.Linear(in_channels, self.embed_dim)
            ))
            feature_pred_list.append(nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dim, self.embed_dim),
            ))
            weights_pred_list.append(nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dim, self.num_summaries),
            ))
        self.input_proj = nn.ModuleList(input_proj_list)
        self.feature_pred = nn.ModuleList(feature_pred_list)
        self.weights_pred = nn.ModuleList(weights_pred_list)

    def forward(self,
                masks: torch.Tensor,
                value: torch.Tensor,
                scale: int,
                num_ref: int) -> (torch.Tensor, Optional[torch.Tensor]):
        """
        Forward pass for ObjectSummarizer.

        Args:
            masks (torch.Tensor):
                Binary or soft masks for support images.
                Shape: (B * num_ref, 1, H, W)
                - B: batch size
                - num_ref: number of reference images per batch
                - H, W: height and width of the feature maps

            value (torch.Tensor):
                Feature maps extracted from the reference images.
                Shape: (B * num_ref, value_dim, H, W)
                - value_dim: feature embedding dimension (e.g., from backbone)

            scale (int):
                Feature scale index (e.g., corresponding to res3, res4, res5).

            num_ref (int):
                Number of reference images per batch (N).

        Returns:
            obj_value (torch.Tensor):
                Summarized object-level embeddings for foreground/background regions.
                Shape: (B, num_ref, num_summaries, embed_dim)
                - num_summaries: number of summaries per ref (typically foreground/background or more)
                - embed_dim: projected embedding dimension

            Optional[torch.Tensor]:
                Placeholder for any additional outputs (currently None).
        """
        h, w = value.shape[-2:]
        masks = F.interpolate(masks, size=(h, w), mode='area')
        masks = masks.unsqueeze(-1)
        # b1hw1
        inv_masks = 1 - masks  # inverse mask with shape b1hw1
        repeated_masks = torch.cat([
            masks.expand(-1, -1, -1, -1, self.num_summaries // 2),
            inv_masks.expand(-1, -1, -1, -1, self.num_summaries // 2),
        ], dim=-1)
        # b1hw*num_summaries

        # value = value.permute(0, 2, 3, 1).contiguous()
        value = value.permute(0, 2, 3, 1)
        # (B * num_ref, H, W, initial_dim)
        value = self.input_proj[scale](value)
        # (B * num_ref, H, W, value_dim = 256)
        if self.add_pe:
            pe = self.pos_enc(value)
            value = value + pe
        # with torch.cuda.amp.autocast(enabled=False):
        #     value = value.float()
        feature = self.feature_pred[scale](value)
        # (B * num_ref, H, W, value_dim = 256)
        logits = self.weights_pred[scale](value)
        # (B * num_ref, H, W, num_summaries)
        obj_value = _weighted_pooling(repeated_masks, feature, logits)
        # (B * num_ref, num_summaries, value_dim)
        bn, num_summaries, v_dim = obj_value.shape  # or get it from context
        b = int(bn // num_ref)
        obj_value = obj_value.view(b, num_ref, num_summaries, v_dim)
        # obj_value = obj_value.transpose(0, 1)
        return obj_value
