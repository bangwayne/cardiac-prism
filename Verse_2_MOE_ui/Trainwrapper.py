import logging
from omegaconf import DictConfig
import numpy as np
import torch
from typing import Dict
from detectron2.structures import ImageList
import random
from .Verse_model import Verse
from .modeling.loss import PointSampleLoss
from .utils.point_sampler import PointSampler
from .modeling.point_encoder.point_feature_map_encoder import get_batch_point_feature_map
# from .utils.memory_utils import UpsampleSupport

log = logging.getLogger()


class VerseTrainWrapper(Verse):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.iter_num = cfg.training.iter_num
        self.training_click_mode = cfg.training.training_click_mode
        self.sampling_probs = cfg.training.sampling_probs
        self.click_loss_weight = cfg.training.click_loss_weight
        self.num_ref = cfg.data.num_ref
        # self.upsample_support = UpsampleSupport(feat_channel=384)

    def forward(self, batched_inputs):
        losses = self.iter_training(batched_inputs)
        return losses

    def iter_training(self, batched_inputs):
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
            click_mode: indicate that if training or not
            epoch: indicate that epoch
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        # images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = torch.stack(images, dim=0)
        height, width = images[0].shape[-2], images[0].shape[-1]
        # 50% chance to sample and add an element
        # Sample an element (this is just an example, modify the sampling logic as needed)
        mode = []
        # sampled_element = random.choice(self.training_click_mode)  # Replace with actual logic to sample the element
        # mode.append(sampled_element)
        sampled_element = random.choices(self.training_click_mode, weights=self.sampling_probs, k=1)[0]
        mode.append(sampled_element)
        query_index = [x["q_index"] for x in batched_inputs]
        Sampler = PointSampler()
        batch_data_with_point = Sampler.initial_test_points(batch_data=batched_inputs, device=self.device)

        epoch_num = batch_data_with_point[0]['epoch']
        total_iter_num = self.iter_num

        # del batch_data_with_mask
        # stage 1: query_base seg and inter click
        if '1' in mode:
            print("mode 1 is on")
            features = self.backbone(images)
            for iter_num in range(total_iter_num):
                for data_dict in batch_data_with_point:
                    data_dict['click_index'] = iter_num
                # batch_data_with_point = [{**data, 'click_index': iter_num} for data in batch_data_with_point]
                if iter_num == 0:
                    point_feature_batch_data = get_batch_point_feature_map(batch_data_with_point).to(
                        self.device)
                    point_tuple_list = [x['points_list'] for x in batch_data_with_point]
                    pos_point_tuple_list, neg_point_tuple_list = None, None
                else:
                    point_feature_batch_data = get_batch_point_feature_map(batch_data_with_next_point).to(
                        self.device)
                    point_tuple_list = [x['points_list'] for x in batch_data_with_next_point]
                    pos_point_tuple_list = [x['pos_point_list'] for x in batch_data_with_next_point]
                    neg_point_tuple_list = [x['neg_point_list'] for x in batch_data_with_next_point]

                outputs = self.sem_seg_head(
                    features,
                    point_feature_batch_data,
                    point_tuple_list,
                    pos_point_tuple_list,
                    neg_point_tuple_list,
                    click_mode='1' if iter_num != 0 else '0',
                    query_index=query_index,
                    # task_query=object_summaries
                )

                if "target" in batched_inputs[0]:
                    gt_instances = [x["target"] for x in batched_inputs]
                    targets = self.prepare_targets(gt_instances, images)
                else:
                    targets = None

                if iter_num == 0:
                    losses = self.criterion(outputs, targets)
                elif iter_num > 0:
                    iter_losses = self.criterion(outputs, targets)
                    for key in iter_losses:
                        losses[key] += iter_losses[key] * self.click_loss_weight

                processed_mask_results = self.single_inference(batched_inputs, outputs).sigmoid()

                for bs_index in range(
                        len(batch_data_with_point if iter_num == 0 else batch_data_with_next_point)):
                    (batch_data_with_point if iter_num == 0 else batch_data_with_next_point)[bs_index][
                        'seg_result'] = \
                        processed_mask_results[bs_index]

                if iter_num < total_iter_num - 1:
                    if self.point_sample_method == "min_dis":
                        batch_data_with_next_point = Sampler.get_next_points(
                            batch_data_with_point if iter_num == 0 else batch_data_with_next_point,
                            device=self.device,
                            click_index=(iter_num + 1)
                        )
                    elif self.point_sample_method == "largest_component":
                        batch_data_with_next_point = Sampler.get_next_points_component(
                            batch_data_with_point if iter_num == 0 else batch_data_with_next_point,
                            device=self.device,
                            click_index=(iter_num + 1)
                        )
        if '2' in mode:
            print("mode 2 is on")
            # features = self.backbone(images)
            object_summaries, pixel_readout = self.process_ref_inputs(batched_inputs, images)
            # icl_query = self.task_encoding(ref_features, ref_mask, query_index)
            for iter_num in range(total_iter_num):
                for data_dict in batch_data_with_point:
                    data_dict['click_index'] = iter_num
                # batch_data_with_point = [{**data, 'click_index': iter_num} for data in batch_data_with_point]
                if iter_num == 0:
                    point_feature_batch_data = get_batch_point_feature_map(batch_data_with_point).to(
                        self.device)
                    point_tuple_list = [x['points_list'] for x in batch_data_with_point]
                    pos_point_tuple_list, neg_point_tuple_list = None, None
                else:
                    point_feature_batch_data = get_batch_point_feature_map(batch_data_with_next_point).to(
                        self.device)
                    point_tuple_list = [x['points_list'] for x in batch_data_with_next_point]
                    pos_point_tuple_list = [x['pos_point_list'] for x in batch_data_with_next_point]
                    neg_point_tuple_list = [x['neg_point_list'] for x in batch_data_with_next_point]

                outputs = self.sem_seg_head(
                    pixel_readout,
                    point_feature_batch_data,
                    point_tuple_list,
                    pos_point_tuple_list,
                    neg_point_tuple_list,
                    click_mode='4' if iter_num != 0 else '3',
                    query_index=query_index,
                    task_query=object_summaries
                )

                if "target" in batched_inputs[0]:
                    gt_instances = [x["target"] for x in batched_inputs]
                    targets = self.prepare_targets(gt_instances, images)
                else:
                    targets = None

                if iter_num == 0:
                    losses = self.criterion(outputs, targets)
                elif iter_num > 0:
                    iter_losses = self.criterion(outputs, targets)
                    for key in iter_losses:
                        losses[key] += iter_losses[key] * self.click_loss_weight

                processed_mask_results = self.single_inference(batched_inputs, outputs).sigmoid()

                for bs_index in range(
                        len(batch_data_with_point if iter_num == 0 else batch_data_with_next_point)):
                    (batch_data_with_point if iter_num == 0 else batch_data_with_next_point)[bs_index][
                        'seg_result'] = \
                        processed_mask_results[bs_index]

                if iter_num < total_iter_num - 1:
                    if self.point_sample_method == "min_dis":
                        batch_data_with_next_point = Sampler.get_next_points(
                            batch_data_with_point if iter_num == 0 else batch_data_with_next_point,
                            device=self.device,
                            click_index=(iter_num + 1)
                        )
                    elif self.point_sample_method == "largest_component":
                        batch_data_with_next_point = Sampler.get_next_points_component(
                            batch_data_with_point if iter_num == 0 else batch_data_with_next_point,
                            device=self.device,
                            click_index=(iter_num + 1)
                        )
                # in the last training, we make it wrong to batch_data_with_point, should be with_nbe
        # stage 2: no query_base inter click:
        if '3' in mode:
            print("mode 3 is on")
            features = self.backbone(images)
            query_index = [x["q_index"] for x in batched_inputs]
            for iter_num in range(total_iter_num):
                for data_dict in batch_data_with_point:
                    data_dict['click_index'] = iter_num
                if iter_num == 0:
                    for bs_index in range(len(batch_data_with_point)):
                        batch_data_with_point[bs_index]['seg_result'] = torch.zeros(1, height, width)

                    batch_data_with_next_point_mode2 = Sampler.get_next_points_component(batch_data_with_point,
                                                                                         device=self.device,
                                                                                         click_index=(iter_num + 1))

                    point_feature_batch_data = get_batch_point_feature_map(batch_data_with_next_point_mode2).to(
                        self.device)
                    point_tuple_list = [x['points_list'] for x in batch_data_with_next_point_mode2]
                    pos_point_tuple_list = [x['pos_point_list'] for x in batch_data_with_next_point_mode2]
                    neg_point_tuple_list = [x['neg_point_list'] for x in batch_data_with_next_point_mode2]

                    # with autocast():
                    # features = self.backbone(images)
                    outputs = self.sem_seg_head(features,
                                                point_feature_batch_data,
                                                point_tuple_list,
                                                pos_point_tuple_list,
                                                neg_point_tuple_list,
                                                click_mode='2',
                                                query_index=query_index)

                    if "target" in batched_inputs[0]:
                        gt_instances = [x["target"] for x in batched_inputs]
                        targets = self.prepare_targets(gt_instances, images)
                    # change the key here, and move the to device into the prepare_targets function
                    else:
                        targets = None

                    losses2 = self.criterion(outputs, targets)
                    if 'losses' not in locals():
                        losses = {key: torch.zeros_like(losses2[key]) for key in losses2}
                    for key in losses2:
                        losses[key] += losses2[key]
                    processed_mask_results = self.single_inference(batched_inputs, outputs)
                    processed_mask_results = processed_mask_results.sigmoid()
                    #
                    for bs_index in range(len(batch_data_with_next_point_mode2)):
                        batch_data_with_next_point_mode2[bs_index]['seg_result'] = processed_mask_results[bs_index]

                    if self.point_sample_method == "min_dis":
                        batch_data_with_next_point_mode2 = Sampler.get_next_points(batch_data_with_next_point_mode2,
                                                                                   device=self.device,
                                                                                   click_index=(iter_num + 2))
                    elif self.point_sample_method == "largest_component":
                        batch_data_with_next_point_mode2 = Sampler.get_next_points_component(
                            batch_data_with_next_point_mode2,
                            device=self.device,
                            click_index=(iter_num + 2))
                #
                else:
                    # weight = 0.8
                    point_feature_batch_data = get_batch_point_feature_map(batch_data_with_next_point_mode2).to(
                        self.device)
                    point_tuple_list = [x['points_list'] for x in batch_data_with_next_point_mode2]
                    pos_point_tuple_list = [x['pos_point_list'] for x in batch_data_with_next_point_mode2]
                    neg_point_tuple_list = [x['neg_point_list'] for x in batch_data_with_next_point_mode2]
                    # features = self.backbone(images)
                    outputs = self.sem_seg_head(features,
                                                point_feature_batch_data,
                                                point_tuple_list,
                                                pos_point_tuple_list,
                                                neg_point_tuple_list,
                                                click_mode='2',
                                                query_index=query_index)

                    if "target" in batched_inputs[0]:
                        gt_instances = [x["target"] for x in batched_inputs]
                        targets = self.prepare_targets(gt_instances, images)
                    # change the key here, and move the to device into the prepare_targets function
                    else:
                        targets = None

                    iter_losses = self.criterion(outputs, targets, cal_class_loss=False)
                    for key in iter_losses:
                        losses[key] += iter_losses[key] * self.click_loss_weight

                    processed_mask_results = self.single_inference(batched_inputs, outputs)
                    processed_mask_results = processed_mask_results.sigmoid()
                    for bs_index in range(len(batch_data_with_next_point_mode2)):
                        batch_data_with_next_point_mode2[bs_index]['seg_result'] = processed_mask_results[bs_index]

                    if iter_num < total_iter_num - 1:
                        if self.point_sample_method == "min_dis":
                            batch_data_with_next_point_mode2 = Sampler.get_next_points(batch_data_with_next_point_mode2,
                                                                                       device=self.device,
                                                                                       click_index=(iter_num + 2))
                        elif self.point_sample_method == "largest_component":
                            batch_data_with_next_point_mode2 = Sampler.get_next_points_component(
                                batch_data_with_next_point_mode2,
                                device=self.device,
                                click_index=(iter_num + 2))

        return losses

    def process_ref_inputs(self, batched_inputs, images):
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
        # k = random.choice([1, 2, 3])
        # indices = random.sample(range(all_num_ref), k)
        k = 1
        indices = random.sample(range(all_num_ref), k)
        # # === Select the sampled refs ===
        ref_img = ref_img[:, indices]  # Select same indices for all batch
        ref_mask = ref_mask[:, indices]
        # Update num_ref for later usage
        # === Flatten B × R dimension ===
        b, num_ref, _, h, w = ref_img.shape
        ref_img = ref_img.reshape(-1, 3, h, w) # [B*R, 3, H, W]
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
                pixel_readout.append(align_data)
            else:
                pixel_readout.append(query_feat)
            #     pixel_readout.append(self.read_support(query_feat, key_feat, value_feat) + query_feat)
            #     pixel_readout.append(self.read_support(query_feat, key_feat, value_feat))
            # else:
            #     pixel_readout.append(query_feat)

        pixel_readout = dict(zip(ordered_keys, pixel_readout))
        return object_summaries, pixel_readout
