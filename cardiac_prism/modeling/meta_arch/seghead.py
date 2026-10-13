# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY
from ..pixel_fuser.pixelfuser import *
from ..transformer_decoder.transformer_decoder import MultiScaleMaskedTransformerDecoder


# from ..transformer_decoder.maskformer_transformer_decoder import build_transformer_decoder


# from ..pixel_decoder.fpn import build_pixel_decoder

class SegHead(nn.Module):
    def __init__(
            self,
            model_cfg,
            input_shape: Dict[str, ShapeSpec],
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()
        input_shape = {k: v for k, v in input_shape.items() if k in model_cfg.model.sem_seg_head.in_features}
        num_classes = model_cfg.model.sem_seg_head.num_classes
        # loss_weight = model_cfg.model.sem_seg_head.loss_weight
        transformer_in_feature = model_cfg.model.sem_seg_head.transformer_in_feature
        # self.transformer_predictor = MultiScaleMaskedTransformerDecoder(model_cfg)
        self.pixel_decoder = PixelFuser(model_cfg, input_shape)
        # input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in sorted(input_shape.items(), key=lambda x: x[1].stride)]
        # self.loss_weight = loss_weight

        self.predictor = MultiScaleMaskedTransformerDecoder(model_cfg)
        self.transformer_in_feature = transformer_in_feature
        self.num_classes = num_classes

    def forward(self,
                features,
                point_feature=None,
                point_tuple_list=None,
                pos_point_tuple_list=None,
                neg_point_tuple_list=None,
                click_mode='0',
                query_index=[],
                # mask=None,
                task_query=None):

        return self.layers(features,
                           point_feature,
                           point_tuple_list,
                           pos_point_tuple_list,
                           neg_point_tuple_list,
                           click_mode=click_mode,
                           query_index=query_index,
                           # mask=mask,
                           task_query=task_query)

    def layers(self,
               features,
               point_feature=None,
               point_tuple_list=None,
               pos_point_tuple_list=None,
               neg_point_tuple_list=None,
               click_mode='0',
               query_index=[],
               # mask=None,
               task_query=None):

        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(
            features, point_feature)
        predictions = self.predictor(multi_scale_features,
                                     point_feature,
                                     point_tuple_list,
                                     pos_point_tuple_list,
                                     neg_point_tuple_list,
                                     click_mode=click_mode,
                                     query_index=query_index,
                                     object_summarizer=task_query)
        return predictions
