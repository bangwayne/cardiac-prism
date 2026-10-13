import logging
import os
import sys
from omegaconf import DictConfig
import numpy as np
import torch
from typing import Dict
from torch.nn import functional as F
from detectron2.structures import ImageList
import random
from .resize_transform import SegmentationPreprocessor
# Ensure local package imports work when launched via streamlit script
_HERE = os.path.abspath(os.path.dirname(__file__))
_PKG_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_PARENT_ROOT = os.path.abspath(os.path.join(_PKG_ROOT, ".."))
for p in (_PKG_ROOT, _PARENT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    # package import
    from Verse_2_MOE_ui.Verse_model import Verse
    from Verse_2_MOE_ui.utils.point_sampler import PointSampler
    from Verse_2_MOE_ui.modeling.point_encoder.point_feature_map_encoder import get_batch_point_feature_map
except Exception:
    try:
        from ..Verse_model import Verse
        from ..utils.point_sampler import PointSampler
        from ..modeling.point_encoder.point_feature_map_encoder import get_batch_point_feature_map
    except Exception:
        # script fallback
        from Verse_model import Verse
        from utils.point_sampler import PointSampler
        from modeling.point_encoder.point_feature_map_encoder import get_batch_point_feature_map

log = logging.getLogger()


def _pack_user_point_tokens(pos_pts, neg_pts, device):
    """Match the sampler's token count for a complete click history.

    After K clicks, training supplies K positive and K negative tokens. Each
    real click has an ignore token in the opposite polarity. Ignore tokens
    participate in attention, so dropping them changes inference.
    """
    slots = max(1, len(pos_pts) + len(neg_pts))

    def encode(points, label):
        coordinates = torch.full((slots, 2), -10, device=device, dtype=torch.long)
        labels = torch.full((slots,), -1, device=device, dtype=torch.long)
        if points:
            coordinates[:len(points)] = torch.as_tensor(points, device=device, dtype=torch.long)
            labels[:len(points)] = label
        return coordinates, labels

    return encode(pos_pts, 1), encode(neg_pts, 0)


class InferenceCore(Verse):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.iter_num = cfg.testing.iter_num
        self.testing_click_mode = cfg.testing.testing_click_mode
        self.num_ref = cfg.testing.num_ref
        self.long_size = cfg.model.setting.long_side_size
        self.size_preprocessor = SegmentationPreprocessor(long_size=self.long_size)

    def forward(self, batched_inputs):
        result_dict, point_dict, extra_dict = self.iter_inference(batched_inputs)
        return result_dict, point_dict, extra_dict

    def iter_inference(self, batched_inputs, unseen_class=False):
        with torch.no_grad():
            images = [x["image"].to(self.device) for x in batched_inputs]
            # images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = torch.stack(images, dim=0)
            height, width = images[0].shape[-2], images[0].shape[-1]
            # -------------------------
            # Branch 0: user-provided clicks (skip PointSampler)
            # -------------------------
            if any("user_points" in x for x in batched_inputs):
                query_index_raw = [x.get("q_index", None) for x in batched_inputs]
                query_index = [int(q) for q in query_index_raw if q is not None]
                has_qindex = any(q is not None for q in query_index_raw)
                has_clicks = False
                batch_data_with_point = []
                for idx, item in enumerate(batched_inputs):
                    pos_pts = item.get("user_points", {}).get("pos", [])
                    neg_pts = item.get("user_points", {}).get("neg", [])
                    if len(pos_pts) + len(neg_pts) > 0:
                        has_clicks = True

                    # Label convention in get_coord_features: 0 -> negative, 1 -> positive
                    (pos_points, pos_labels), (neg_points, neg_labels) = _pack_user_point_tokens(
                        pos_pts, neg_pts, self.device,
                    )
                    points = torch.cat([pos_points, neg_points], dim=0)
                    point_labels = torch.cat([pos_labels, neg_labels], dim=0)

                    # Accept prev_mask and click_index from caller for stateful iteration
                    prev_mask = item.get("prev_mask", None)
                    click_index = item.get("click_index", 0)
                    batch_data_with_point.append({
                        "image": item["image"].to(self.device),
                        "points_list": [(points, point_labels)],
                        "pos_point_list": [(pos_points, pos_labels)],
                        "neg_point_list": [(neg_points, neg_labels)],
                        "size_info": item["size_info"],
                        "pad_info": item["pad_info"],
                        "scale_factor": item["scale_factor"],
                        "seg_result": prev_mask.to(self.device) if prev_mask is not None else None,
                        "click_index": click_index,
                    })

                # Decide click_mode based on presence of q-index and clicks
                # User rule: q-index only -> mode 1; clicks only -> mode 3; both -> mode 2
                if has_qindex and not has_clicks:
                    click_mode = "0"
                elif has_qindex and has_clicks:
                    click_mode = "1"
                elif has_clicks and not has_qindex:
                    click_mode = "2"
                else:
                    click_mode = "2"

                # Ensure query_index list is non-empty when needed; fallback to [1]
                if len(query_index) == 0:
                    query_index = [1]

                # User flow: single-pass using provided clicks (batch size 1 assumed)
                result_dict, point_dict = {}, {}
                features = self.backbone(images)
                # Use original encoder to build point feature maps
                point_feature_batch_data = get_batch_point_feature_map(batch_data=batch_data_with_point).to(self.device)
                point_tuple_list = [x["points_list"] for x in batch_data_with_point]
                outputs = self.sem_seg_head(
                    features,
                    point_feature_batch_data,
                    point_tuple_list,
                    click_mode=click_mode,
                    query_index=query_index,
                    pos_point_tuple_list=[x["pos_point_list"] for x in batch_data_with_point],
                    neg_point_tuple_list=[x["neg_point_list"] for x in batch_data_with_point],
                )

                processed_mask_results = self.single_inference(batched_inputs, outputs).sigmoid()
                seg_result_list, point_list, padded_result_list = [], [], []
                for bs_index, item in enumerate(batch_data_with_point):
                    seg_result_padded = processed_mask_results[bs_index].detach().cpu()  # (1, long_H, long_W)
                    size_info = item["size_info"]
                    pad_info = item["pad_info"]
                    scale_factor = item["scale_factor"]
                    restored_seg = self.size_preprocessor.unpad_and_resize(
                        seg_result_padded,
                        original_size=size_info,
                        pad=pad_info,
                    )
                    restore_point = self.size_preprocessor.map_valid_points_back(
                        item["points_list"][0][0],
                        item["points_list"][0][1],
                        original_size=size_info,
                        pad=pad_info,
                        scale_factor=scale_factor,
                    )
                    seg_result_list.append(restored_seg)
                    point_list.append(restore_point)
                    padded_result_list.append(seg_result_padded)

                result_dict[0] = seg_result_list
                point_dict[0] = point_list
                extra_dict = {"padded_results": {0: padded_result_list}}
                return result_dict, point_dict, extra_dict

            return result_dict, point_dict, {}

    def process_ref_inputs(self, batched_inputs, images, mode="train"):
        """
        Process reference images and masks, extract support features.

        Args:
            batched_inputs (list[dict]): each dict should contain keys 'ref_img' and 'ref_mask',
                                         where 'ref_img' is shape [num_ref, 3, H, W] and
                                         'ref_mask' is [num_ref, 1, H, W].
            images (Tensor): query image tensor of shape [B, 3, H, W]

        Returns:
            object_summaries (list[Tensor]): list of object-level features extracted from support
            pixel_readout (list[Tensor]): list of pixel-level support feature readouts
        """

        ref_img = [x['ref_img'].to(self.device) for x in batched_inputs]
        ref_mask = [x['ref_mask'].to(self.device) for x in batched_inputs]

        # Stack into shape [B, R, 3, H, W] and [B, R, 1, H, W]
        ref_img = torch.stack(ref_img, dim=0)
        ref_mask = torch.stack(ref_mask, dim=0)
        b, all_num_ref, c, h, w = ref_img.shape
        # === Randomly sample how many references to use ===
            # Randomly decide how many refs to pick: 1, 2, or all 3
        if mode == "train":
            k = random.choice([1, 2, 3])
            indices = random.sample(range(all_num_ref), k)
            # === Select the sampled refs ===
            ref_img = ref_img[:, indices]  # Select same indices for all batch
            ref_mask = ref_mask[:, indices]
        else:
            ref_img = ref_img
            ref_mask = ref_mask
        # Update num_ref for later usage
        # === Flatten B × R dimension ===
        b, num_ref, _, h, w = ref_img.shape
        ref_img = ref_img.reshape(-1, 3, h, w)  # [B*R, 3, H, W]
        ref_mask = ref_mask.reshape(-1, 1, h, w)  # [B*R, 1, H, W]

        # === Extract features from query and reference images ===
        query_features = self.backbone(images)  # dict of multi-scale features
        key_features = self.backbone(ref_img)  # features for reference images
        # === Concatenate ref image and mask, pass through mask encoder ===
        ref_combine = torch.cat((ref_img, ref_mask), dim=1)  # [B*R, 4, H, W]
        value_features = self.mask_encoder(ref_combine)  # dict of multi-scale values
        object_summaries = []
        pixel_readout = []
        ordered_keys = ["res5", "res4", "res3"]
        # === For each feature level (e.g., res3, res4, res5) ===
        for key in ordered_keys:
            scale = self.feature_idx[key]
            query_feat = query_features[key]  # [B, C, H, W]
            key_feat = key_features[key]  # [B*R, C, H, W]
            value_feat = value_features[key]  # [B*R, C, H, W]

            # === Get object-level summary features from support ===
            object_summary = self.object_summarizer(ref_mask, value_feat, scale, num_ref)
            # # [B*R, 1, H, W] [B*R, C, H, W] -> [B,R*Q,C]
            object_summaries.append(object_summary)
            if key == "res5":
                # query_feat = query_features[key]  # [B, C, H, W]
                # key_feat = key_features[key]  # [B*R, C, H, W]
            # === Reshape key/value features to [B, R, C, H, W] ===
                key_feat = key_feat.reshape(b, num_ref, *key_feat.shape[1:]).contiguous()
                value_feat = value_feat.reshape(b, num_ref, *value_feat.shape[1:]).contiguous()
            #
                align_data = self.read_support(query_feat, key_feat, value_feat)
                # align_data_list = self.upsample_support(align_data)
                # # === Read pixel-level features from support ===
                # 加上 pixel readout
                pixel_readout.append(align_data)
            else:
                pixel_readout.append(query_feat)

        pixel_readout = dict(zip(ordered_keys, pixel_readout))
        return object_summaries, pixel_readout


    @staticmethod
    def extract_best(ref_img, best_match_idx):
        bs = ref_img.shape[0]
        ref_img_list = []
        for i in range(bs):
            best_idx = best_match_idx[i]
            best_ref_img = ref_img[i, best_idx].unsqueeze(0)
            ref_img_list.append(best_ref_img)
        ref_img_stack = torch.stack(ref_img_list, dim=0)
        return ref_img_stack
