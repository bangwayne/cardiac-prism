import warnings
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple, Union
import torch
import torch.nn.functional as F
import random
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from scipy.ndimage import label, center_of_mass


class PointSampler:

    def __init__(self,
                 # batch_data,
                 # num_iter,
                 num_positive_extra=2,
                 num_negative_extra=3,
                 negative_bg_prob=1,
                 negative_other_prob=0.2,
                 negative_border_prob=0.4,
                 spacial_dim=2,
                 use_border_masks=True,
                 fix_extra_point_num=None):

        # self.batch_data = batch_data
        # self.num_iter = num_iter
        self.batch_size = 8
        self.num_positive_extra = num_positive_extra
        self.num_negative_extra = num_negative_extra
        self.fix_extra_point_num = fix_extra_point_num
        # self.batch_size = len(batch_data)
        self.use_border_masks = use_border_masks
        self.negative_bg_prob = negative_bg_prob
        self.negative_other_prob = negative_other_prob
        self.negative_border_prob = negative_border_prob
        self.spacial_dim = spacial_dim

    def get_border_masks(self, batch_data, device, kernel_size=3, ero_rate=0.12):
        """
        The function to first expand the mask, then reduce the original ones
        to get the border mask.

        Arguments:
          batch_data (List of Dict): the data dict, batch_data[i] is a Dict, and have keys
          {'image': the input image, with the size (C,H,W);
           'sem_seg': the ground-truth label, with the size [H,W];
           'image_transforms': the record of the transform to this image;
           'file_name': the path of the image,
           'width': 384,
           'height': 384,
           'target': is a dict having key: 'labels' and 'masks'
            {'labels': tensor([1, 2], device='cuda:0'),
             'masks': tensor of True and False, with the size [num_labels, H, W]
           }
          kernel_size: the parameters of erosion
          ero_rate: the parameters of erosion
        """
        self.batch_size = len(batch_data)
        kernel = torch.ones((1, 1, kernel_size, kernel_size), device=device, dtype=torch.float32)

        for bs in range(self.batch_size):
            masks = batch_data[bs]['target']['masks'].int().to(device)
            if not isinstance(masks, torch.Tensor):
                raise TypeError("Input must be a PyTorch tensor")

            num_label, height, width = masks.shape
            borders = []

            for i in range(num_label):
                single_mask = masks[i].float().unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]
                expand_r = int(torch.ceil(ero_rate * torch.sqrt(single_mask.sum())).item())

                # Dilate the mask using convolution
                expanded_mask = single_mask
                for _ in range(expand_r):
                    expanded_mask = F.conv2d(expanded_mask, kernel, padding=kernel_size // 2)
                    expanded_mask = (expanded_mask > 0).float()

                # Subtract the original mask to get the border
                border_mask = expanded_mask.squeeze(0).squeeze(0) - single_mask.squeeze(0).squeeze(0)
                borders.append(border_mask)

            border_masks = torch.stack(borders, dim=0).to(device)
            batch_data[bs]['target']['border_mask'] = border_masks

        return batch_data

    def get_mask_dict(self, batch_data, device):
        """
        The function to get the mask of 3 types: border_mask, other_mask, and background_mask.

        Arguments:
          batch_data (List of Dict): the data dict, batch_data[i] is a Dict, and have keys
          {'image': the input image, with the size (C,H,W);
           'sem_seg': the ground-truth label, with the size [H,W];
           'image_transforms': the record of the transform to this image;
           'file_name': the path of the image,
           'width': 384,
           'height': 384,
           'target': is a dict having key: 'labels' and 'masks'
            {'labels': tensor([1, 2], device='cuda:0'),
             'masks': tensor of True and False, with the size [num_labels, H, W]
           }
        """
        self.batch_size = len(batch_data)
        for bs in range(self.batch_size):
            masks = batch_data[bs]['target']['masks'].to(device).int()
            if not isinstance(masks, torch.Tensor):
                raise TypeError("Input must be a PyTorch tensor")

            if self.use_border_masks:
                batch_data = self.get_border_masks(batch_data, device)
                border_masks = batch_data[bs]['target']['border_mask'].to(device)
            else:
                border_masks = None
            # when choose don;t use border mask, these seems to be a problem, need to fix. April 22
            sem_seg = batch_data[bs]['sem_seg'].to(device)
            num_label, height, width = masks.shape
            labels = batch_data[bs]['target']['labels'].to(device)
            neg_mask_list = []

            for i in range(num_label):
                if i > 1:
                    neg_masks = ((sem_seg != 0) & (sem_seg != labels[i]))
                else:
                    neg_masks = ((sem_seg != 0) & (sem_seg != labels.item()))

                merged_neg_mask = neg_masks.to(torch.int)
                labels_cls = masks[i]

                if torch.all(labels_cls == 0):
                    all_bg = torch.logical_not(labels_cls).to(torch.int)
                    neg_masks = {
                        'bg': all_bg,
                        'other': all_bg,
                        'border': all_bg,
                    }
                else:
                    merged_all_mask = torch.any(masks, dim=0).to(torch.int)
                    if self.use_border_masks:
                        combined_mask = torch.logical_or(merged_all_mask, border_masks[i])
                    else:
                        combined_mask = merged_all_mask

                    neg_mask_bg = torch.logical_not(combined_mask).to(torch.int)
                    neg_masks = {
                        'bg': neg_mask_bg,
                        'other': merged_neg_mask,
                        'border': border_masks[i] if self.use_border_masks else torch.zeros_like(merged_neg_mask),
                    }

                neg_mask_list.append(neg_masks)

            batch_data[bs]['neg_masks_list'] = neg_mask_list

        return batch_data

    def initial_select_points(self, batch_data, device, fix_extra_point_num=None):
        """
        The function to get the mask of 3 types: border_mask, other_mask, and background_mask.

        Arguments:
          batch_data (List of Dict): the data dict, batch_data[i] is a Dict, and have keys
          {'image': the input image, with the size (C,H,W);
           'sem_seg': the ground-truth label, with the size [H,W];
           'image_transforms': the record of the transform to this image;
           'file_name': the path of the image,
           'width': 384,
           'height': 384,
           'target': is a dict having key: 'labels' and 'masks'
            {'labels': tensor([1, 2], device='cuda:0'),
             'masks': tensor of True and False, with the size [num_labels, H, W]
           }
           'neg_mask_list': is a list of dict, the number of dict is the same as the numbers of labels, each dict includes:
            neg_masks = {
                'bg': neg_mask_bg,
                'other': merged_neg_mask,
                'border': border_masks[i]}
            after this function, the batch_data dict will add a key ['point_list']:
            which is a list of tuple (points: (N,2), labels: (N,1))
            label: 0 represent the positive point, label 1 represent the negtive point

        """
        self.batch_size = len(batch_data)
        pos_thred = 0.9
        neg_thred = 0.1
        probabilities = [self.negative_bg_prob, self.negative_other_prob, self.negative_border_prob]

        for bs in range(self.batch_size):
            masks = batch_data[bs]['target']['masks'].to(device).int()
            neg_masks_list = batch_data[bs]['neg_masks_list']
            if not isinstance(masks, torch.Tensor):
                raise TypeError("Input must be a PyTorch tensor")

            num_label, height, width = masks.shape
            labels = batch_data[bs]['target']['labels'].to(device)
            points_list = []

            for i in range(num_label):
                points = torch.zeros((0, 2), device=device, dtype=torch.long)
                point_labels = torch.zeros((0), device=device, dtype=torch.long)
                single_masks = masks[i]
                neg_masks = neg_masks_list[i]

                # Get positive indices
                positive_indices = torch.nonzero(single_masks > pos_thred, as_tuple=True)

                if positive_indices[0].numel() == 0:
                    selected_positive_point = torch.tensor([-1, -1], device=device).unsqueeze(0)
                    points = torch.cat((points, selected_positive_point), dim=0)
                    point_labels = torch.cat((point_labels, torch.tensor([-1], device=device)))
                else:
                    random_idx = torch.randint(len(positive_indices[0]), (1,))
                    selected_positive_point = torch.stack([positive_indices[dim][random_idx] for dim in range(2)],
                                                          dim=1).squeeze(0)
                    points = torch.cat((points, selected_positive_point.unsqueeze(0)), dim=0)
                    point_labels = torch.cat((point_labels, torch.tensor([1], device=device)))

                # Add extra positive points
                if self.num_positive_extra > 0:
                    pos_idx_list = torch.randperm(len(positive_indices[0]))[:self.num_positive_extra]
                    extra_positive_points = torch.stack([positive_indices[dim][pos_idx_list] for dim in range(2)],
                                                        dim=1)
                    points = torch.cat((points, extra_positive_points), dim=0)
                    point_labels = torch.cat(
                        (point_labels, torch.ones((extra_positive_points.shape[0]), device=device)))

                # Add extra negative points
                if self.num_negative_extra > 0:
                    extra_negative_points = []
                    for _ in range(self.num_negative_extra):
                        selected_key = random.choices(list(neg_masks.keys()), weights=probabilities, k=1)[0]
                        selected_neg_mask = neg_masks[selected_key]

                        if torch.all(selected_neg_mask == 0):
                            selected_neg_mask = neg_masks['bg']

                        negative_indices = torch.nonzero(selected_neg_mask > neg_thred, as_tuple=True)
                        neg_idx = torch.randint(len(negative_indices[0]), (1,))
                        extra_negative_points.append(
                            torch.stack([negative_indices[dim][neg_idx] for dim in range(2)], dim=1).squeeze(0))

                    extra_negative_points = torch.stack(extra_negative_points, dim=0)
                    points = torch.cat((points, extra_negative_points), dim=0)
                    point_labels = torch.cat(
                        (point_labels, torch.zeros((extra_negative_points.shape[0]), device=device)))

                if fix_extra_point_num is None:
                    left_point_num = self.num_positive_extra + self.num_negative_extra + 1 - point_labels.shape[0]
                else:
                    left_point_num = fix_extra_point_num + 1 - point_labels.shape[0]

                ignore_points = torch.full((left_point_num, 2), -1, device=device, dtype=torch.long)
                points = torch.cat((points, ignore_points), dim=0)
                ignore_labels = torch.full((left_point_num,), -1, device=device, dtype=torch.long)
                point_labels = torch.cat((point_labels, ignore_labels))

                points_list.append((points, point_labels))

            batch_data[bs]['points_list'] = points_list

        return batch_data

    def get_next_points(self, batch_data, device, click_index, pred_thresh=0.499):
        """
        The function to get next point in the iteration.

        Arguments:
          batch_data (List of Dict): the data dict, batch_data[i] is a Dict, and have keys
          {'image': the input image, with the size (C,H,W);
           'sem_seg': the ground-truth label, with the size [H,W];
           'image_transforms': the record of the transform to this image;
           'file_name': the path of the image,
           'width': 384,
           'height': 384,
           'target': is a dict having key: 'labels' and 'masks'
            {'labels': tensor([1, 2], device='cuda:0'),
             'masks': tensor of True and False, with the size [num_labels, H, W]
           }
           'neg_mask_list': is a list of dict, the number of dict is the same as the numbers of labels, each dict includes:
            neg_masks = {
                'bg': neg_mask_bg,
                'other': merged_neg_mask,
                'border': border_masks[i]}
            after this function, the batch_data dict will add a key ['point_list']:
            which is a list of tuple (points: (N,2), labels: (N,1))
            label: 0 represent the positive point, label 1 represent the negative point.
            'seg_result': the seg result in last iteration, should be (num_class+1, H, W), the first class is empty.
        """
        self.batch_size = len(batch_data)
        assert click_index > 0
        for bs in range(self.batch_size):
            file_name = batch_data[bs]['file_name']
            masks = batch_data[bs]['target']['masks'].int().to(device)
            seg_result = batch_data[bs]['seg_result']
            seg_result = (seg_result > 0.5).int().to(device)

            if not isinstance(masks, torch.Tensor):
                raise TypeError("Input must be a PyTorch tensor")

            num_label, height, width = masks.shape
            labels = batch_data[bs]['target']['labels'].to(device)
            points_list = batch_data[bs]['points_list']
            new_point_list = []

            for i in range(num_label):
                points, point_labels = points_list[i]
                gt_array = masks[i].cpu().numpy()
                pred_array = seg_result.cpu().numpy()

                fn_mask = np.logical_and(gt_array, pred_array < pred_thresh)
                fp_mask = np.logical_and(np.logical_not(gt_array), pred_array > pred_thresh)

                fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), 'constant').astype(np.uint8)
                fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), 'constant').astype(np.uint8)

                fn_mask_dt = cv2.distanceTransform(fn_mask, cv2.DIST_L2, 5)[1:-1, 1:-1]
                fp_mask_dt = cv2.distanceTransform(fp_mask, cv2.DIST_L2, 5)[1:-1, 1:-1]

                fn_max_dist = np.max(fn_mask_dt)
                fp_max_dist = np.max(fp_mask_dt)

                is_positive = fn_max_dist > fp_max_dist
                dt = fn_mask_dt if is_positive else fp_mask_dt
                inner_mask = dt > max(fn_max_dist, fp_max_dist) / 3.0
                indices = np.argwhere(inner_mask)

                if len(indices) > 0:
                    coords = indices[np.random.randint(0, len(indices))]
                    new_point = torch.tensor([coords[0], coords[1]], device=device).unsqueeze(0)
                    if is_positive:
                        new_points = torch.cat((points, new_point), dim=0)
                        new_labels = torch.cat((point_labels, torch.tensor([1], device=device)), dim=0)
                    else:
                        new_points = torch.cat((points, new_point), dim=0)
                        new_labels = torch.cat((point_labels, torch.tensor([0], device=device)), dim=0)
                else:
                    new_point = torch.tensor([-1, -1], device=device).unsqueeze(0)
                    new_points = torch.cat((points, new_point), dim=0)
                    new_labels = torch.cat((point_labels, torch.tensor([-1], device=device)), dim=0)

                new_points_tuple = (new_points, new_labels)
                new_point_list.append(new_points_tuple)

            batch_data[bs]['points_list'] = new_point_list

            batch_data[bs]['click_index'] = click_index

        return batch_data

    def get_next_points_component(self, batch_data, device, click_index, pred_thresh=0.499):
        """
        The function to get next point in the iteration.

        Arguments:
          batch_data (List of Dict): the data dict, batch_data[i] is a Dict, and have keys
          {'image': the input image, with the size (C,H,W);
           'sem_seg': the ground-truth label, with the size [num_class, H,W];
           'image_transforms': the record of the transform to this image;
           'file_name': the path of the image,
           'width': 384,
           'height': 384,
           'target': is a dict having key: 'labels' and 'masks'
            {'labels': tensor([1, 2], device='cuda:0'),
             'masks': tensor of True and False, with the size [num_labels, H, W]
           }
           device: device;
           click_index: click_index:
           pred_thresh:pred_thresh
        after this function, the batch_data dict will add a key ['point_list']:
        which is a list of tuple (points: (N,2), labels: (N,1))
        label: 0 represent the positive point, label 1 represent the negative point.
        'seg_result': the seg result in last iteration, should be (num_class+1, H, W), the first class is empty.
        """
        self.batch_size = len(batch_data)
        assert click_index > 0
        for bs in range(self.batch_size):
            file_name = batch_data[bs]['file_name']
            masks = batch_data[bs]['target']['masks'].int().to(device)
            ## (num_label, H, W)
            seg_result = batch_data[bs]['seg_result']
            seg_result = (seg_result > 0.5).int().to(device)
            ## should has shape (num_class, H, W)
            assert seg_result.shape == masks.shape, (f"Shape mismatch: Expected masks of shape {masks.shape}, "
                                                     f"but got segmentation results of shape {seg_result.shape}")
            if not isinstance(masks, torch.Tensor):
                raise TypeError("Input must be a PyTorch tensor")

            num_label, height, width = masks.shape
            points_list = batch_data[bs]['points_list']
            pos_point_list = batch_data[bs]['pos_point_list']
            neg_point_list = batch_data[bs]['neg_point_list']
            new_point_list = []
            new_pos_point_list = []
            new_neg_point_list = []

            for i in range(num_label):
                points, point_labels = points_list[i]
                pos_points, pos_labels = pos_point_list[i]
                neg_points, neg_labels = neg_point_list[i]
                gt_array = masks[i]
                pred_array = seg_result[i]
                gt_array = gt_array.to(torch.bool)
                pred_array = pred_array.to(torch.bool)

                fn_mask = torch.logical_and(gt_array, torch.logical_not(pred_array))
                fp_mask = torch.logical_and(torch.logical_not(gt_array), pred_array)
                del gt_array, pred_array
                # fn_mask = torch.logical_and(gt_array, pred_array < pred_thresh)
                # fp_mask = torch.logical_and(~gt_array, pred_array > pred_thresh)

                fn_mask = fn_mask.cpu().numpy()
                fp_mask = fp_mask.cpu().numpy()

                largest_fn_component = self.get_largest_connected_component(fn_mask)
                largest_fp_component = self.get_largest_connected_component(fp_mask)

                fn_size = largest_fn_component.sum()
                fp_size = largest_fp_component.sum()

                is_positive = fn_size > fp_size
                largest_component = largest_fn_component if is_positive else largest_fp_component

                largest_component = largest_component.astype(np.uint8)

                padded_component = np.pad(largest_component, ((1, 1), (1, 1)), 'constant')
                dt = cv2.distanceTransform(padded_component, cv2.DIST_L2, 5)[1:-1, 1:-1]

                inner_mask = dt > dt.max() / 3.0
                indices = np.argwhere(inner_mask)

                if inner_mask.sum() > 0:
                    centroid = center_of_mass(inner_mask)
                    centroid = np.round(centroid).astype(int)
                    if largest_component[centroid[0], centroid[1]]:  # Check if the centroid is in the component
                        new_point = torch.tensor([centroid[0], centroid[1]], device=device).unsqueeze(0)
                    else:
                        coords = indices[np.random.randint(0, len(indices))]
                        new_point = torch.tensor([coords[0], coords[1]], device=device).unsqueeze(0)

                    if is_positive:
                        new_points = torch.cat((points, new_point), dim=0)
                        new_labels = torch.cat((point_labels, torch.tensor([1], device=device)), dim=0)
                        pos_points = torch.cat((new_point, pos_points), dim=0)  # 先进先出，新的点替换第一个
                        pos_labels = torch.cat((torch.tensor([1], device=device), pos_labels), dim=0)  # label 是 1

                        neg_points = torch.cat((torch.tensor([-10, -10], device=device).unsqueeze(0), neg_points), dim=0)  # 先进先出，新的点替换第一个
                        neg_labels = torch.cat((torch.tensor([-1], device=device), neg_labels), dim=0)
                        ## so, the positive point, which label is 1
                    else:
                        neg_points = torch.cat((new_point, neg_points), dim=0)  # 先进先出，新的点替换第一个
                        neg_labels = torch.cat((torch.tensor([0], device=device), neg_labels), dim=0)  # label 是 0
                        new_points = torch.cat((points, new_point), dim=0)
                        new_labels = torch.cat((point_labels, torch.tensor([0], device=device)), dim=0)

                        pos_points = torch.cat((torch.tensor([-10, -10], device=device).unsqueeze(0), pos_points), dim=0)  # 先进先出，新的点替换第一个
                        pos_labels = torch.cat((torch.tensor([-1], device=device), pos_labels), dim=0)
                        ## so, the negative point, which label is 0

                else:
                    new_point = torch.tensor([-10, -10], device=device).unsqueeze(0)
                    new_points = torch.cat((points, new_point), dim=0)
                    new_labels = torch.cat((point_labels, torch.tensor([-1], device=device)), dim=0)
                    pos_points = torch.cat((torch.tensor([-10, -10], device=device).unsqueeze(0), pos_points),
                                           dim=0)  # 先进先出，新的点替换第一个
                    pos_labels = torch.cat((torch.tensor([-1], device=device), pos_labels), dim=0)
                    neg_points = torch.cat((torch.tensor([-10, -10], device=device).unsqueeze(0), neg_points),
                                           dim=0)  # 先进先出，新的点替换第一个
                    neg_labels = torch.cat((torch.tensor([-1], device=device), neg_labels), dim=0)

                new_point_list.append((new_points, new_labels))
                new_pos_point_list.append((pos_points, pos_labels))
                new_neg_point_list.append((neg_points, neg_labels))

            batch_data[bs]['points_list'] = new_point_list
            batch_data[bs]['pos_point_list'] = new_pos_point_list
            batch_data[bs]['neg_point_list'] = new_neg_point_list
            batch_data[bs]['click_index'] = click_index

        return batch_data


    def initial_test_points(self, batch_data, device, fix_extra_point_num=20):
        """
        The function to get the mask of 3 types: border_mask, other_mask, and background_mask.
        Arguments:
        batch_data (List of Dict): the data dict, batch_data[i] is a Dict, and have keys
        {'image': the input image, with the size (C,H,W);
        'sem_seg': the ground-truth label, with the size [H,W];
        'image_transforms': the record of the transform to this image;
        'file_name': the path of the image,
        'width': the width of the image,
        'height': the height of the image,
        'target': is a dict having key: 'labels' and 'masks'
            {'labels': tensor([1, 2], device='cuda:0'),
            'masks': tensor of True and False, with the size [num_labels, H, W]
        }
        'neg_mask_list': is a list of dict, the number of dict is the same as the numbers of labels, each dict includes:
            neg_masks = {
                'bg': neg_mask_bg,
                'other': merged_neg_mask,
                'border': border_masks[i]}
            after this function, the batch_data dict will add a key ['point_list']:
            which is a list of tuple (points: (N,2), labels: (N,1))
            label: 0 represent the positive point, label 1 represent the negative point
        """
        self.batch_size = len(batch_data)

        for bs in range(self.batch_size):
            masks = batch_data[bs]['target']['masks'].to(device).int()
            if not isinstance(masks, torch.Tensor):
                raise TypeError("Input must be a PyTorch tensor")

            num_class, height, width = masks.shape
            labels = batch_data[bs]['target']['labels'].to(device)
            points_list = []
            pos_point_list = []
            neg_point_list = []

            # 将fix_extra_point_num除以2
            half_fix_extra_point_num = 1

            for i in range(num_class):
                points = torch.zeros((0, 2), device=device, dtype=torch.long)
                point_labels = torch.zeros((0), device=device, dtype=torch.long)

                if fix_extra_point_num is None:
                    left_point_num = self.num_positive_extra + self.num_negative_extra + 1 - labels.shape[0]
                else:
                    left_point_num = fix_extra_point_num + 1 - labels.shape[0]

                ignore_points = torch.full((left_point_num, 2), -10, device=device, dtype=torch.long)
                points = torch.cat((points, ignore_points), dim=0)
                ignore_labels = torch.full((left_point_num,), -1, device=device, dtype=torch.long)
                point_labels = torch.cat((point_labels, ignore_labels))

                points_list.append((points, point_labels))

                # 创建 pos_point_list 和 neg_point_list，忽略点数量为 fix_extra_point_num / 2
                pos_ignore_points = torch.empty((0, 2), device=device, dtype=torch.long)
                pos_ignore_labels = torch.empty((0,), device=device, dtype=torch.long)
                pos_point_list.append((pos_ignore_points, pos_ignore_labels))

                neg_ignore_points = torch.empty((0, 2), device=device, dtype=torch.long)
                neg_ignore_labels = torch.empty((0,), device=device, dtype=torch.long)
                neg_point_list.append((neg_ignore_points, neg_ignore_labels))

            batch_data[bs]['points_list'] = points_list
            batch_data[bs]['pos_point_list'] = pos_point_list
            batch_data[bs]['neg_point_list'] = neg_point_list

        return batch_data

    def initial_select_test_points(self, batch_data, device, fix_extra_point_num=20):
        """
        The function to get the mask of 3 types: border_mask, other_mask, and background_mask.
        Arguments:
        batch_data (List of Dict): the data dict, batch_data[i] is a Dict, and have keys
        {'image': the input image, with the size (C,H,W);
        'sem_seg': the ground-truth label, with the size [H,W];
        'image_transforms': the record of the transform to this image;
        'file_name': the path of the image,
        'width': the width of the image,
        'height': the height of the image,
        'target': is a dict having key: 'labels' and 'masks'
            {'labels': tensor([1, 2], device='cuda:0'),
            'masks': tensor of True and False, with the size [num_labels, H, W]
        }
        'neg_mask_list': is a list of dict, the number of dict is the same as the numbers of labels, each dict includes:
            neg_masks = {
                'bg': neg_mask_bg,
                'other': merged_neg_mask,
                'border': border_masks[i]}
            after this function, the batch_data dict will add a key ['point_list']:
            which is a list of tuple (points: (N,2), labels: (N,1))
            label: 0 represent the positive point, label 1 represent the negative point
        """
        self.batch_size = len(batch_data)

        for bs in range(self.batch_size):
            masks = batch_data[bs]['target']['masks'].to(device).int()
            if not isinstance(masks, torch.Tensor):
                raise TypeError("Input must be a PyTorch tensor")

            num_class, height, width = masks.shape
            labels = batch_data[bs]['target']['labels'].to(device)
            points_list = []
            pos_point_list = []
            neg_point_list = []

            # 将fix_extra_point_num除以2
            half_fix_extra_point_num = 20

            for i in range(num_class):
                points = torch.zeros((0, 2), device=device, dtype=torch.long)
                point_labels = torch.zeros((0), device=device, dtype=torch.long)

                if fix_extra_point_num is None:
                    left_point_num = self.num_positive_extra + self.num_negative_extra + 1 - labels.shape[0]
                else:
                    left_point_num = fix_extra_point_num + 1 - labels.shape[0]

                ignore_points = torch.full((left_point_num, 2), -10, device=device, dtype=torch.long)
                points = torch.cat((points, ignore_points), dim=0)
                ignore_labels = torch.full((left_point_num,), -1, device=device, dtype=torch.long)
                point_labels = torch.cat((point_labels, ignore_labels))

                points_list.append((points, point_labels))

                # 创建 pos_point_list 和 neg_point_list，忽略点数量为 fix_extra_point_num / 2
                pos_ignore_points = torch.full((half_fix_extra_point_num, 2), -10, device=device, dtype=torch.long)
                pos_ignore_labels = torch.full((half_fix_extra_point_num,), -1, device=device, dtype=torch.long)
                pos_point_list.append((pos_ignore_points, pos_ignore_labels))

                neg_ignore_points = torch.full((half_fix_extra_point_num, 2), -10, device=device, dtype=torch.long)
                neg_ignore_labels = torch.full((half_fix_extra_point_num,), -1, device=device, dtype=torch.long)
                neg_point_list.append((neg_ignore_points, neg_ignore_labels))

            batch_data[bs]['points_list'] = points_list
            batch_data[bs]['pos_point_list'] = pos_point_list
            batch_data[bs]['neg_point_list'] = neg_point_list

        return batch_data

    @staticmethod
    def get_largest_connected_component(mask):
        """
        This function finds and returns the largest connected component in a binary mask.

        Parameters:
        mask (np.ndarray): Binary mask of shape (H, W)

        Returns:
        largest_component (np.ndarray): Binary mask of the largest connected component
        """
        labeled_array, num_features = label(mask)
        if num_features == 0:
            return mask  # No connected components found, return the original mask

        sizes = [np.sum(labeled_array == i + 1) for i in range(num_features)]
        largest_component_label = np.argmax(sizes) + 1

        largest_component = (labeled_array == largest_component_label).astype(np.uint8)
        return largest_component
