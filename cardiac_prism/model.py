# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple
import torch
import random
from torch import nn
from torch.nn import functional as F
from .modeling.loss import PointSampleLoss
from .utils.point_sampler import PointSampler
from .modeling.point_encoder.point_feature_map_encoder import get_batch_point_feature_map
from .modeling.pixel_fuser.pixelfuser import PixelFuser
from torch.cuda.amp import GradScaler, autocast
from fvcore.nn import FlopCountAnalysis

from .utils.memory_utils import *
from .modeling.backbone.utnet import UTNet
# from .modeling.task_encoder.task_encoder import TaskEncoder
from .modeling.meta_arch.seghead import SegHead
from .modeling.task_encoder.mask_encoder import MaskEncoder
from .modeling.task_encoder.object_summarier import ObjectSummarizer
# from .utils.memory_utils import UpsampleSupport


class CardiacPrism(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    def __init__(
            self,
            cfg,
            *,
            deep_supervision=True,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
        """
        super().__init__()
        model_cfg = cfg.model
        self.backbone = UTNet(model_cfg)
        # self.task_encoding = TaskEncoder(model_cfg)
        self.sem_seg_head = SegHead(model_cfg, self.backbone.output_shape())
        self.criterion = PointSampleLoss()
        self.object_summarizer = ObjectSummarizer(model_cfg)
        self.mask_encoder = MaskEncoder(model_cfg)
        self.num_queries = model_cfg.model.task_encoder.num_queries
        self.size_divisibility = model_cfg.setting.size_divisibility
        self.feature_idx = model_cfg.model.object_summarizer.feature_idx
        # self.upsample_support = UpsampleSupport(feat_channel=384)

        pixel_mean = model_cfg.setting.pixel_mean
        pixel_std = model_cfg.setting.pixel_std
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.point_sample_method = model_cfg.setting.point_sample_method

        dice_weight = model_cfg.setting.dice_weight
        mask_weight = model_cfg.setting.mask_weight
        weight_dict = {"loss_mask": mask_weight, "loss_dice": dice_weight}
        if deep_supervision:
            dec_layers = model_cfg.model.transformer_decoder.dec_layers
            aux_weight_dict = {}
            for i in range(dec_layers):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

    @property
    def device(self):
        return self.pixel_mean.device

    def single_inference(self, batched_inputs, outputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
            outputs
        """
        with torch.no_grad():
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = torch.stack(images, dim=0)

            mask_pred_results = outputs["pred_masks"]
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.shape[-2], images.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
        return mask_pred_results

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image['masks'].to(self.device)
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image['labels'].to(self.device),
                    # "unique_labels": targets_per_image['unique_labels'].to(self.device),
                    "masks": padded_masks,
                }
            )
        return new_targets

    @staticmethod
    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        sem_seg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return sem_seg

    @staticmethod
    def read_support(query_key: torch.Tensor,
                     memory_key: torch.Tensor,
                     msk_value: torch.Tensor) -> torch.Tensor:
        """
        query_key       : B * CK * H * W
        memory_key      : B * CK * T * H * W
        msk_value       : B * CV * H * W
        pixel_feature   : B * C * H * W
        """
        # batch_size, num_objects = msk_value.shape[:2]
        # diff = (memory_key[:, 0] - query_key).abs().mean()
        # read using visual attention
        # with torch.cuda.amp.autocast(enabled=False):
        affinity = get_affinity(memory_key.float(), query_key.float())
        # avg_affinity = affinity.mean(dim=(-2, -3))  # B x T
        msk_value = msk_value.float()
        # B * (num_objects*CV) * H * W
        pixel_readout = readout(affinity, msk_value)
        return pixel_readout

    @staticmethod
    def read_sim_rank(query_key: torch.Tensor,
                     memory_key: torch.Tensor) -> torch.Tensor:
        """
        query_key       : B * CK * H * W
        memory_key      : B * CK * T * H * W
        msk_value       : B * CV * H * W
        pixel_feature   : B * C * H * W
        """
        memory_key = memory_key.transpose(1, 2)
        _, best_match_idx = get_similarity(memory_key.float(), query_key.float(), rank=True)

        return best_match_idx

    def forward(self, *args, **kwargs):
        raise NotImplementedError
