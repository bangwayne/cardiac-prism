# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from ..transformer_decoder.position_encoding import PositionEmbeddingSine


class MultiScaleDownsamplingModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleDownsamplingModule, self).__init__()
        # Reduce spatial dimensions by half and double the channels
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.GroupNorm(32, out_channels)
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.GroupNorm(32, out_channels)
        )
        # Maintain current size, apply same convolutional settings
        self.down3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.GroupNorm(32, out_channels)
        )
        # Reduce to 1/8th the original dimensions
        self.down4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.GroupNorm(32, out_channels)
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        outputs = {}
        if h == 256:
            x = self.down1(x)
            # outputs['res2'] = x
            outputs['res3'] = x  # [B, C, H/2, W/2]
            x = self.down2(x)
            outputs['res4'] = x  # [B, C, H/4, W/4]
            x = self.down3(x)
            outputs['res5'] = x  # [B, C, H/8, W/8] maintained
        elif h == 512:
        # x = self.down4(x)
            x = self.down1(x)
            # output_dict['res2'] = x1
            x = self.down2(x)
            outputs['res3'] = x  # [B, C, H/2, W/2]
            x = self.down3(x)
            outputs['res4'] = x  # [B, C, H/4, W/4]
            x = self.down4(x)
            outputs['res5'] = x  # [B, C, H/8, W/8] maintained
        return outputs


class PixelFuser(nn.Module):
    def __init__(
            self,
            model_cfg,
            input_shape: Dict[str, ShapeSpec],
    ):
        super().__init__()
        input_shape = {
            k: v for k, v in input_shape.items() if k in model_cfg.model.sem_seg_head.in_features
        }
        transformer_in_features = model_cfg.model.sem_seg_head.in_features
        transformer_input_shape = {
            k: v for k, v in input_shape.items() if k in transformer_in_features
        }
        conv_dim = model_cfg.model.sem_seg_head.conv_dim
        transformer_input_shape = sorted(transformer_input_shape.items(), key=lambda x: x[1].stride)
        self.transformer_in_features = [k for k, v in transformer_input_shape]  # starting from "res2" to "res5"
        transformer_in_channels = [v.channels for k, v in transformer_input_shape]

        self.transformer_num_feature_levels = len(self.transformer_in_features)
        if self.transformer_num_feature_levels > 1:
            input_proj_list = []
            # from low resolution to high resolution (res5 -> res2)
            for in_channels in transformer_in_channels[::-1]:
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                ))
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(transformer_in_channels[-1], conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                )])
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        self.mask_downsampler = MultiScaleDownsamplingModule(in_channels=3, out_channels=conv_dim)
        N_steps = conv_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

    def forward_features(self, features, point_feature=None):
        point_feature = point_feature.squeeze(1)
        multi_scale_features = []

        for idx, f in enumerate(self.transformer_in_features[::-1]):
            x = features[f].float()
            if point_feature is not None:
                point_feature_dict = self.mask_downsampler(point_feature)
                muti_level_point_feature = point_feature_dict[f].float()
                multi_scale_features.append(self.input_proj[idx](x) + self.pe_layer(x) + muti_level_point_feature)

        return multi_scale_features[-1], multi_scale_features[0], multi_scale_features
