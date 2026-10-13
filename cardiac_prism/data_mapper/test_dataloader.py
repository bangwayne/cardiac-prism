import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import SimpleITK as sitk
from PIL import Image
import json
import yaml
import math
import random
import logging
import copy
from pathlib import Path
import cv2
from ..inference.resize_transform import SegmentationPreprocessor


class SemanticTestDataset(Dataset):
    def __init__(self,
                 cfg,
                 *,
                 images_dir_name="image",
                 masks_dir_name="annotations",
                 mode='test',
                 transform=None,
                 three_chanel=True):
        dataset_cfg = cfg.dataset
        self.mode = mode
        self.data_name = dataset_cfg.dataset_name
        self.data_dir = dataset_cfg.dataset_path
        self.data_config = dataset_cfg.data_config
        self.dataset_path = Path(self.data_dir + "/" + self.data_name + "/Data")
        self.images_path = self.dataset_path / mode / images_dir_name
        self.masks_path = self.dataset_path / mode / masks_dir_name
        self.three_channel = three_chanel
        self.transform = transform
        self.dataset_samples = []
        self.metadata = self.load_metadata()
        self.resize_processor = SegmentationPreprocessor(long_size=512)
        # for i in range(len(self.data_name_list)):
        #     images_path = self.images_path_list[i]
        #     for x in sorted(images_path.glob('*.nii.gz')):
        #         self.dataset_samples.append(x.name)

    def load_metadata(self):
        metadata = []
        weight_list = []
        dataset_slice_counts = {name: 0 for name in self.data_config}

        # if data_name in self.data_name_list:
        images_path = self.images_path
        masks_path = self.masks_path
        ndim = self.data_config[self.data_name]["ndim"]
        if ndim == 3:
            img_name_list = [x.name for x in sorted(images_path.glob('*.nii.gz'))]
            print(f"Start loading {self.data_name} {self.mode} metadata")

            for filename in img_name_list:
                img_path = os.path.join(images_path, filename)
                mask_name = filename.split('.')[0] + "_gt.nii.gz"
                mask_path = os.path.join(masks_path, mask_name)

                itk_mask = sitk.ReadImage(mask_path)
                array_mask = sitk.GetArrayFromImage(itk_mask)

                unique_labels = self.data_config[self.data_name]["unique_labels"]
                slices_num = array_mask.shape[0]

                for slice_index in range(slices_num):
                    slice_mask = array_mask[slice_index]
                    slice_unique_labels = np.unique(slice_mask[slice_mask != 0])
                    slice_unique_labels = [label for label in slice_unique_labels if label in unique_labels]
                    if len(slice_unique_labels) > 0:
                        for test_label in slice_unique_labels:
                            dataset_slice_counts[self.data_name] += 1
                            metadata.append(
                                (img_path,
                                 mask_path,
                                 slice_index,
                                 unique_labels,
                                 test_label,
                                 filename,
                                 self.data_name,
                                 ndim)
                            )
        elif ndim == 2:
            d2_img_name_list = [x.name for x in sorted(images_path.glob('*.png'))]
            print(f"Start loading {self.data_name} {self.mode} metadata")

            for filename in d2_img_name_list:
                img_path = os.path.join(images_path, filename)
                mask_name = filename.split('.')[0] + "_gt.png"
                mask_path = os.path.join(masks_path, mask_name)
                # --- Read image and convert to tensor (C, H, W) ---
                # --- Read mask (as grayscale image) ---
                # mask = Image.open(mask_path).convert("L")
                array_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                # array_mask = np.array(mask)  # shape (H, W)
                unique_labels = self.data_config[self.data_name]["unique_labels"]
                # Since it's 2D, treat whole image as one "slice"
                slice_index = 0
                slice_mask = array_mask
                slice_unique_labels = np.unique(slice_mask[slice_mask != 0])
                slice_unique_labels = [label for label in slice_unique_labels if label in unique_labels]
                if len(slice_unique_labels) > 0:
                    for test_label in slice_unique_labels:
                        dataset_slice_counts[self.data_name] += 1
                        metadata.append(
                            (img_path,
                             mask_path,
                             slice_index,
                             unique_labels,
                             test_label,
                             filename,
                             self.data_name,
                             ndim)
                        )

        print(f"Load done, length of dataset: {len(metadata)}")
        print("Slice counts per dataset:")
        for dataset, count in dataset_slice_counts.items():
            print(f"  {dataset}: {count} slices")
        return metadata

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_path, mask_path, slice_index, labels, test_label, filename, data_name, ndim = self.metadata[idx]
        query_mapping = self.data_config[data_name]["query_mapping"]
        # if self.mode == "train":
        unique_label = test_label
        unique_query = query_mapping[int(unique_label)][0]
        if ndim == 3:
            itk_img, itk_mask = sitk.ReadImage(img_path), sitk.ReadImage(mask_path)
            img, mask = sitk.GetArrayFromImage(itk_img), sitk.GetArrayFromImage(itk_mask)  #
            # num_layer, h, w
            height, width = img.shape[1], img.shape[2]
            gray_image = torch.from_numpy(img[slice_index]).float()
            # (h, w)
            if self.three_channel:
                gray_image = gray_image.unsqueeze(0)
                ori_image = gray_image.repeat(3, 1, 1)  # (3, h, w)
            else:
                ori_image = gray_image
            ori_sem_seg = torch.from_numpy(mask[slice_index]).long()
            # sem_seg: (h, w)
        else:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # shape: (H, W, 3)
            # Convert BGR -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # now (H, W, 3)
            # Transpose to (3, H, W) and convert to float32 Tensor
            ori_image = torch.from_numpy(img.transpose(2, 0, 1)).float()
            height, width = img.shape[1], img.shape[2]
            # Read mask in grayscale
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # shape: (H, W)
            # mask = np.array(mask)
            # (h, w)
            ori_sem_seg = torch.from_numpy(mask).long()
        image, sem_seg, original_size, pad_info, scale_factor = self.resize_processor.resize_and_pad(ori_image,
                                                                                                     ori_sem_seg)
        sample = {
            'image': image,
            'ori_image': ori_image,
            'ori_sem_seg': ori_sem_seg,
            'sem_seg': sem_seg,
            'file_name': filename,
            'slice_index': slice_index,
            'width': width,
            'height': height,
            'q_index': unique_query,
            'unique_label': unique_label,
            'size_info': original_size,
            'pad_info': pad_info,
            'scale_factor': scale_factor
            # 'ref_img': ref_images,
            # 'ref_mask': ref_masks
        }
        if self.transform:
            sample = self.transform(sample)

        single_mask = torch.stack([(sample['sem_seg'] == unique_label).bool()])

        target = {
            "labels": torch.tensor(labels).long(),
            # 'unique_labels': self.get_label([unique_label]).long(),
            # this place is very important to keep the label dtype long, else it will report some error
            "masks": single_mask
        }
        sample['target'] = target

        return sample

    @staticmethod
    def preprocess(itk_img, itk_mask):
        img = sitk.GetArrayFromImage(itk_img)
        mask = sitk.GetArrayFromImage(itk_mask)
        tensor_img = torch.from_numpy(img).float()
        tensor_mask = torch.from_numpy(mask).long()
        return tensor_img, tensor_mask


class FewShotTestDataset(Dataset):
    def __init__(self,
                 cfg,
                 *,
                 images_dir_name="image",
                 masks_dir_name="annotations",
                 mode='test',
                 transform=None,
                 ref_transform=None,
                 three_chanel=True):
        dataset_cfg = cfg.dataset
        self.mode = mode
        self.ref_mode = dataset_cfg.ref_mode
        self.data_name = dataset_cfg.dataset_name
        self.data_dir = dataset_cfg.dataset_path
        self.data_config = dataset_cfg.data_config
        self.dataset_path = Path(self.data_dir + "/" + self.data_name + "/Data")
        self.images_path = self.dataset_path / mode / images_dir_name
        self.masks_path = self.dataset_path / mode / masks_dir_name
        self.json_path = str(self.dataset_path / mode / self.ref_mode) + "_ref.json"
        self.three_channel = three_chanel
        self.transform = transform
        self.ref_transform = ref_transform
        self.dataset_samples = []
        self.metadata = self.load_metadata()
        self.resize_processor = SegmentationPreprocessor(long_size=512)


    def load_metadata(self):
        metadata = []
        dataset_slice_counts = {name: 0 for name in self.data_config}

        # if data_name in self.data_name_list:
        # images_path = self.images_path
        # masks_path = self.masks_path
        ndim = self.data_config[self.data_name]["ndim"]
        if ndim == 3:
            img_name_list = [x.name for x in sorted(self.images_path.glob('*.nii.gz'))]
            print(f"Start loading {self.data_name} {self.mode} metadata")

            for filename in img_name_list:
                img_path = os.path.join(self.images_path, filename)
                mask_name = filename.split('.')[0] + "_gt.nii.gz"
                mask_path = os.path.join(self.masks_path, mask_name)
                # import image path and mask path
                itk_mask = sitk.ReadImage(mask_path)
                array_mask = sitk.GetArrayFromImage(itk_mask)
                # load the mask
                unique_labels = self.data_config[self.data_name]["unique_labels"]
                slices_num = array_mask.shape[0]

                for slice_index in range(slices_num):
                    slice_ratio = slice_index / array_mask.shape[0]
                    slice_mask = array_mask[slice_index]
                    # slice_unique_labels = np.unique(slice_mask[slice_mask != 0])
                    slice_unique_labels = [label for label in np.unique(slice_mask[slice_mask != 0]) if label in unique_labels]
                    if len(slice_unique_labels) > 0:
                        for test_label in slice_unique_labels:
                            dataset_slice_counts[self.data_name] += 1
                            metadata.append(
                                (img_path,
                                 mask_path,
                                 slice_index,
                                 unique_labels,
                                 test_label,
                                 filename,
                                 self.data_name,
                                 ndim,
                                 slice_ratio)
                            )
        elif ndim == 2:
            d2_img_name_list = [x.name for x in sorted(self.images_path.glob('*.png'))]
            print(f"Start loading {self.data_name} {self.mode} metadata")

            for filename in d2_img_name_list:
                img_path = os.path.join(self.images_path, filename)
                mask_name = filename.split('.')[0] + "_gt.png"
                mask_path = os.path.join(self.masks_path, mask_name)
                # --- Read image and convert to tensor (C, H, W) ---
                # --- Read mask (as grayscale image) ---
                # mask = Image.open(mask_path).convert("L")
                array_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                # array_mask = np.array(mask)  # shape (H, W)
                unique_labels = self.data_config[self.data_name]["unique_labels"]
                # Since it's 2D, treat whole image as one "slice"
                slice_index = 0
                slice_mask = array_mask
                slice_unique_labels = np.unique(slice_mask[slice_mask != 0])
                slice_unique_labels = [label for label in slice_unique_labels if label in unique_labels]
                if len(slice_unique_labels) > 0:
                    for test_label in slice_unique_labels:
                        dataset_slice_counts[self.data_name] += 1
                        metadata.append(
                            (img_path,
                             mask_path,
                             slice_index,
                             unique_labels,
                             test_label,
                             filename,
                             self.data_name,
                             ndim,
                             1)
                        )

        print(f"Load done, length of dataset: {len(metadata)}")
        print("Slice counts per dataset:")
        for dataset, count in dataset_slice_counts.items():
            print(f"  {dataset}: {count} slices")
        return metadata

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_path, mask_path, slice_index, labels, test_label, filename, data_name, ndim, _ = self.metadata[idx]
        query_mapping = self.data_config[data_name]["query_mapping"]
        # if self.mode == "train":
        unique_label = test_label
        unique_query = query_mapping[int(unique_label)][0]
        ref_images, ref_masks = self.sample_test_reference(
            json_path=self.json_path,
            unique_label=unique_label,
            ref_transform=self.ref_transform,
            ndim=ndim,
            current_idx=idx,
        )

        if ndim == 3:
            itk_img, itk_mask = sitk.ReadImage(img_path), sitk.ReadImage(mask_path)
            img, mask = sitk.GetArrayFromImage(itk_img), sitk.GetArrayFromImage(itk_mask)  #
            # num_layer, h, w
            height, width = img.shape[1], img.shape[2]
            gray_image = torch.from_numpy(img[slice_index]).float()
            # (h, w)
            slices_num = img.shape[0]
            slice_ratio = slice_index / slices_num
            if self.three_channel:
                gray_image = gray_image.unsqueeze(0)
                ori_image = gray_image.repeat(3, 1, 1)  # (3, h, w)
            else:
                ori_image = gray_image
            ori_sem_seg = torch.from_numpy(mask[slice_index]).long()
            # sem_seg: (h, w)
        else:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # shape: (H, W, 3)
            # Convert BGR -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # now (H, W, 3)
            # Transpose to (3, H, W) and convert to float32 Tensor
            ori_image = torch.from_numpy(img.transpose(2, 0, 1)).float()
            height, width = img.shape[1], img.shape[2]
            # Read mask in grayscale
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # shape: (H, W)
            # mask = np.array(mask)
            # (h, w)
            ori_sem_seg = torch.from_numpy(mask).long()
            slice_ratio = 1
        image, sem_seg, original_size, pad_info, scale_factor = self.resize_processor.resize_and_pad(ori_image,
                                                                                                     ori_sem_seg)

        # (num_ref, 3, h, w)
        # (num_ref, 1, h, w)
        sample = {
            'image': image,
            'ori_image': ori_image,
            'ori_sem_seg': ori_sem_seg,
            'sem_seg': sem_seg,
            'file_name': filename,
            'slice_index': slice_index,
            'width': width,
            'height': height,
            'q_index': unique_query,
            'unique_label': unique_label,
            'size_info': original_size,
            'pad_info': pad_info,
            'scale_factor': scale_factor,
            'ref_img': ref_images,
            'ref_mask': ref_masks,
            'slice_ratio': slice_ratio
        }
        if self.transform:
            sample = self.transform(sample)

        single_mask = torch.stack([(sample['sem_seg'] == unique_label).bool()])

        target = {
            "labels": torch.tensor(labels).long(),
            # 'unique_labels': self.get_label([unique_label]).long(),
            # this place is very important to keep the label dtype long, else it will report some error
            "masks": single_mask
        }
        sample['target'] = target

        return sample

    def sample_test_reference(self,
                              json_path: str,
                              unique_label: int,
                              # ref_transform: bool,
                              current_idx: int,
                              ref_transform: bool=True,
                              ndim: int = 3
                              ) -> (torch.Tensor, torch.Tensor):
        """
        Sample multiple reference images and masks from the same dataset.

        Args:
            metadata (list): List of metadata tuples. Each item includes:
                             (img_path, mask_path, slice_index, ... , data_name)
            current_idx (int): Index of the current (query) image to exclude from sampling.
            data_name (str): Name of the current dataset (used to filter reference candidates).
            preprocess_fn (callable): Function to preprocess image and mask, returns tensors.
            unique_label (int): The label to extract from the mask (for binary masks).
            num_refs (int): Number of reference samples to draw.
            three_channel (bool): Whether to convert image to 3-channel.

        Returns:
            ref_images: (num_refs, 3, H, W)
            ref_masks:  (num_refs, 1, H, W)
        """
        with open(json_path, "r") as f:
            ref_dict = json.load(f)
        ref_images, ref_masks = [], []
        for pid, entry in ref_dict.items():
            img_path, mask_path, ndim = entry["image_path"], entry["mask_path"], entry["ndim"]
            if ndim == 3:
                # Load image and mask
                itk_ref_img = sitk.ReadImage(img_path)
                itk_ref_mask = sitk.ReadImage(mask_path)
                ref_img_array = sitk.GetArrayFromImage(itk_ref_img)  # [D, H, W]
                ref_mask_array = sitk.GetArrayFromImage(itk_ref_mask)  # [D, H, W]

                slice_ratio = self.metadata[current_idx][-1]
                num_slices = ref_mask_array.shape[0]
                center = int(slice_ratio * num_slices)
                # center = 3
                # ref_image, binary_mask = self._sample_slice_with_label(ref_img_array, ref_mask_array, unique_label, center)
                # #
                # ref_image, binary_mask = self._sample_optimize_slice(ref_img_array, ref_mask_array, unique_label, center)
                # ref_image, binary_mask = self._sample_slices_by_percentile(ref_img_array, ref_mask_array, unique_label, num_slices=1)
                ref_image, binary_mask = self._sample_center_slice(ref_img_array, ref_mask_array, unique_label)
                #
                ref_images.append(ref_image)
                ref_masks.append(binary_mask)
                # Check slices in the range [center_slice - 2, center_slice + 2]
            else:
                ref_img = Image.open(img_path).convert("RGB")  # ensure it's 3-channel RGB
                # --- Read mask (as grayscale image) ---
                ref_img = np.array(ref_img).transpose(2, 0, 1)
                ref_image = torch.from_numpy(ref_img).float()
                height, width = ref_image.shape[1], ref_image.shape[2]
                mask = Image.open(mask_path).convert("L")
                mask = np.array(mask)
                ref_mask = torch.from_numpy(mask).long()

                if ref_transform:
                    ref_sample = {
                        'ref_img': ref_image,  # (3, H, W)
                        'sem_seg': ref_mask  # (H, W)
                    }
                    ref_sample = self.ref_transform(ref_sample)
                    ref_image = ref_sample['image']
                    ref_mask = ref_sample['sem_seg']

                binary_mask = (ref_mask == unique_label).float().unsqueeze(0)  # (1, H, W)
                ref_images.append(ref_image)
                ref_masks.append(binary_mask)

            if ndim == 2:
                ref_images = torch.stack(ref_images, dim=0)  # (num_refs, 3, H, W)
                ref_masks = torch.stack(ref_masks, dim=0)  # (num_refs, 1, H, W)
            else:
                ref_images = torch.stack(ref_images, dim=0)  # (num_refs, 3, H, W)
                ref_masks = torch.stack(ref_masks, dim=0)  # (num_refs, 1, H, W)
                ref_images = ref_images.view(-1, *ref_images.shape[2:])
                ref_masks = ref_masks.view(-1, *ref_masks.shape[2:])
            return ref_images, ref_masks

    def _sample_slice_with_label(self, ref_img, ref_mask, unique_label, center, ref_transform=True):
        offsets = list(range(-2, 3))
        num_slices = ref_mask.shape[0]
        slice_indices = [np.clip(center + offset, 0, num_slices - 1) for offset in offsets]

        if np.any(np.isin(ref_mask[center], unique_label)):
            ref_image = torch.from_numpy(ref_img[center]).float()
            ref_mask = torch.from_numpy(ref_mask[center]).long()
            if self.three_channel:
                ref_image = ref_image.unsqueeze(0).repeat(3, 1, 1)
            else:
                ref_image = ref_image.unsqueeze(0)
            if ref_transform:
                ref_sample = {'ref_img': ref_image, 'sem_seg': ref_mask}
                ref_sample = self.ref_transform(ref_sample)
                ref_image = ref_sample['image']
                ref_mask = ref_sample['sem_seg']

            binary_mask = (ref_mask == unique_label).float().unsqueeze(0)
            return ref_image, binary_mask

        else:
            shuffled_indices = random.sample(slice_indices, len(slice_indices))
            for i in shuffled_indices:
                if np.any(np.isin(ref_mask[i], unique_label)):
                    ref_image = torch.from_numpy(ref_img[i]).float()
                    ref_mask = torch.from_numpy(ref_mask[i]).long()
                    if self.three_channel:
                        ref_image = ref_image.unsqueeze(0).repeat(3, 1, 1)
                    else:
                        ref_image = ref_image.unsqueeze(0)
                    if ref_transform:
                        ref_sample = {'ref_img': ref_image, 'sem_seg': ref_mask}
                        ref_sample = self.ref_transform(ref_sample)
                        ref_image = ref_sample['image']
                        ref_mask = ref_sample['sem_seg']

                    binary_mask = (ref_mask == unique_label).float().unsqueeze(0)
                    return ref_image, binary_mask

            # Randomly select slices until a matching label is found
            valid_slices = [i for i in range(num_slices) if np.any(np.isin(ref_mask[i], unique_label))]
            if valid_slices:
                sampled_slice = np.random.choice(valid_slices)
            else:
                sampled_slice = np.random.randint(0, num_slices)
            ref_image = torch.from_numpy(ref_img[sampled_slice]).float()
            ref_mask = torch.from_numpy(ref_mask[sampled_slice]).long()
            if self.three_channel:
                ref_image = ref_image.unsqueeze(0).repeat(3, 1, 1)
            else:
                ref_image = ref_image.unsqueeze(0)
            if ref_transform:
                ref_sample = {'ref_img': ref_image, 'sem_seg': ref_mask}
                ref_sample = self.ref_transform(ref_sample)
                ref_image = ref_sample['image']
                ref_mask = ref_sample['sem_seg']

            binary_mask = (ref_mask == unique_label).float().unsqueeze(0)
            return ref_image, binary_mask

    def _sample_optimize_slice(self, ref_img, ref_mask, unique_label, center, ref_transform=True):
        offsets = list(range(-5, 5))  # [-2, -1, 0, 1, 2]
        num_slices = ref_mask.shape[0]
        slice_indices = [np.clip(center + offset, 0, num_slices - 1) for offset in offsets]
        # Collect valid slices with the desired label
        selected_images = []
        selected_masks = []

        for i in slice_indices:
            if np.any(np.isin(ref_mask[i], unique_label)):
                img = torch.from_numpy(ref_img[i]).float()
                msk = torch.from_numpy(ref_mask[i]).long()

                if self.three_channel:
                    img = img.unsqueeze(0).repeat(3, 1, 1)
                else:
                    img = img.unsqueeze(0)

                if ref_transform:
                    ref_sample = {'ref_img': img, 'sem_seg': msk}
                    ref_sample = self.ref_transform(ref_sample)
                    img = ref_sample['image']
                    msk = ref_sample['sem_seg']

                binary_mask = (msk == unique_label).float().unsqueeze(0)
                selected_images.append(img)
                selected_masks.append(binary_mask)

        # If not enough valid slices found, try to search whole volume
        if len(selected_images) < 5:
            all_valid = [i for i in range(num_slices) if np.any(np.isin(ref_mask[i], unique_label))]
            for i in all_valid:
                if len(selected_images) >= 5:
                    break
                if i in slice_indices:
                    continue  # already processed
                img = torch.from_numpy(ref_img[i]).float()
                msk = torch.from_numpy(ref_mask[i]).long()

                if self.three_channel:
                    img = img.unsqueeze(0).repeat(3, 1, 1)
                else:
                    img = img.unsqueeze(0)

                if ref_transform:
                    ref_sample = {'ref_img': img, 'sem_seg': msk}
                    ref_sample = self.ref_transform(ref_sample)
                    img = ref_sample['image']
                    msk = ref_sample['sem_seg']

                binary_mask = (msk == unique_label).float().unsqueeze(0)
                selected_images.append(img)
                selected_masks.append(binary_mask)

        # If still not enough, repeat existing ones
        while len(selected_images) < 5:
            selected_images.append(selected_images[-1])
            selected_masks.append(selected_masks[-1])

        # If more than 5, randomly select 5
        if len(selected_images) > 5:
            selected = random.sample(list(zip(selected_images, selected_masks)), 5)
            selected_images, selected_masks = zip(*selected)

        # Stack into tensors: [5, C, H, W]
        ref_images = torch.stack(selected_images, dim=0)
        binary_masks = torch.stack(selected_masks, dim=0)

        return ref_images, binary_masks

    def _sample_center_slice(self, ref_img, ref_mask, unique_label, ref_transform=True):
        num_slices = ref_mask.shape[0]
        # 找到所有包含 unique_label 的切片索引
        valid_indices = [i for i in range(num_slices) if np.any(np.isin(ref_mask[i], unique_label))]
        if not valid_indices:
            raise ValueError(f"No slices with label {unique_label} found in the reference mask.")
        # 取中间的那个索引
        middle_idx = valid_indices[len(valid_indices) // 2]
        img = torch.from_numpy(ref_img[middle_idx]).float()
        msk = torch.from_numpy(ref_mask[middle_idx]).long()

        if self.three_channel:
            img = img.unsqueeze(0).repeat(3, 1, 1)
        else:
            img = img.unsqueeze(0)

        if ref_transform:
            ref_sample = {'ref_img': img, 'sem_seg': msk}
            ref_sample = self.ref_transform(ref_sample)
            img = ref_sample['image']
            msk = ref_sample['sem_seg']

        binary_mask = (msk == unique_label).float().unsqueeze(0)
        # 添加 batch 维度
        ref_image = img.unsqueeze(0)         # [1, C, H, W]
        binary_mask = binary_mask.unsqueeze(0)  # [1, 1, H, W]

        return ref_image, binary_mask
    

        def _sample_center_slice(self, ref_img, ref_mask, unique_label, ref_transform=True):
            num_slices = ref_mask.shape[0]
            # 找到所有包含 unique_label 的切片索引
            valid_indices = [i for i in range(num_slices) if np.any(np.isin(ref_mask[i], unique_label))]
            if not valid_indices:
                raise ValueError(f"No slices with label {unique_label} found in the reference mask.")
            # 取中间的那个索引
            middle_idx = valid_indices[len(valid_indices) // 2]
            img = torch.from_numpy(ref_img[middle_idx]).float()
            msk = torch.from_numpy(ref_mask[middle_idx]).long()

            if self.three_channel:
                img = img.unsqueeze(0).repeat(3, 1, 1)
            else:
                img = img.unsqueeze(0)

            if ref_transform:
                ref_sample = {'ref_img': img, 'sem_seg': msk}
                ref_sample = self.ref_transform(ref_sample)
                img = ref_sample['image']
                msk = ref_sample['sem_seg']

            binary_mask = (msk == unique_label).float().unsqueeze(0)
            # 添加 batch 维度
            ref_image = img.unsqueeze(0)         # [1, C, H, W]
            binary_mask = binary_mask.unsqueeze(0)  # [1, 1, H, W]

            return ref_image, binary_mask

    def _sample_center_opt_slice(self, ref_img, ref_mask, unique_label, ref_transform=True):
        num_slices = ref_mask.shape[0]
        # 找到所有包含 unique_label 的切片索引
        valid_indices = [i for i in range(num_slices) if np.any(np.isin(ref_mask[i], unique_label))]
        if not valid_indices:
            raise ValueError(f"No slices with label {unique_label} found in the reference mask.")

        # 计算中心索引（不是原始 volume 的 center，而是 valid_indices 中的）
        center = valid_indices[len(valid_indices) // 2]

        # 从 valid_indices 中选出离 center 最近的最多5个切片
        sorted_by_distance = sorted(valid_indices, key=lambda x: abs(x - center))
        selected_indices = sorted_by_distance[:5]

        selected_images = []
        selected_masks = []

        for idx in selected_indices:
            img = torch.from_numpy(ref_img[idx]).float()
            msk = torch.from_numpy(ref_mask[idx]).long()

            if self.three_channel:
                img = img.unsqueeze(0).repeat(3, 1, 1)
            else:
                img = img.unsqueeze(0)

            if ref_transform:
                ref_sample = {'ref_img': img, 'sem_seg': msk}
                ref_sample = self.ref_transform(ref_sample)
                img = ref_sample['image']
                msk = ref_sample['sem_seg']

            binary_mask = (msk == unique_label).float().unsqueeze(0)
            selected_images.append(img)
            selected_masks.append(binary_mask)

        # 如果不足5个，重复最后一张填满
        while len(selected_images) < 5:
            selected_images.append(selected_images[-1])
            selected_masks.append(selected_masks[-1])

        # 拼接为 tensor: [5, C, H, W]
        ref_images = torch.stack(selected_images, dim=0)
        binary_masks = torch.stack(selected_masks, dim=0)

        return ref_images, binary_masks
    
    def _sample_slices_by_percentile(self, ref_img, ref_mask, unique_label, num_slices, ref_transform=True):
        num_total_slices = ref_mask.shape[0]
        valid_indices = [i for i in range(num_total_slices) if np.any(np.isin(ref_mask[i], unique_label))]
        if not valid_indices:
            raise ValueError(f"No slices with label {unique_label} found in the reference mask.")

        index_min, index_max = valid_indices[0], valid_indices[-1]

        def choose_positions(count: int) -> list[float]:
            if count <= 1:
                return [0.5]
            if count == 2:
                return [0.5, 0.2]
            if count == 3:
                return [0.5, 0.2, 0.8]
            if count == 4:
                return [0.2, 0.4, 0.5, 0.6]
            if count == 5:
                return [0.2, 0.4, 0.5, 0.6, 0.8]
            if count == 6:
                return [0.2, 0.4, 0.5, 0.6, 0.8, 0.1]
            if count == 7:
                return [0.2, 0.4, 0.5, 0.6, 0.8, 0.1, 0.3]
            if count == 8:
                return [0.2, 0.4, 0.5, 0.6, 0.8, 0.1, 0.3, 0.7]
            if count == 9:
                return [0.2, 0.4, 0.5, 0.6, 0.8, 0.1, 0.3, 0.7, 0.9]
            if count == 10:
                return [0.2, 0.4, 0.5, 0.6, 0.8, 0.1, 0.3, 0.7, 0.9, 0.5]

        positions = choose_positions(num_slices)
        selected_indices = []
        for pct in positions[:num_slices]:
            raw_idx = index_min + pct * (index_max - index_min)
            clamped_idx = int(round(max(min(raw_idx, index_max), index_min)))
            selected_indices.append(clamped_idx)

        while len(selected_indices) < num_slices:
            selected_indices.append(selected_indices[-1])

        selected_indices = selected_indices[:num_slices]

        images = []
        masks = []
        for idx in selected_indices:
            img = torch.from_numpy(ref_img[idx]).float()
            msk = torch.from_numpy(ref_mask[idx]).long()

            if self.three_channel:
                img = img.unsqueeze(0).repeat(3, 1, 1)
            else:
                img = img.unsqueeze(0)

            if ref_transform:
                ref_sample = {'ref_img': img, 'sem_seg': msk}
                ref_sample = self.ref_transform(ref_sample)
                img = ref_sample['image']
                msk = ref_sample['sem_seg']

            binary_mask = (msk == unique_label).float().unsqueeze(0)
            images.append(img)
            masks.append(binary_mask)

        stacked_images = torch.stack(images, dim=0)
        stacked_masks = torch.stack(masks, dim=0)
        return stacked_images, stacked_masks

    @staticmethod
    def preprocess(itk_img, itk_mask):
        img = sitk.GetArrayFromImage(itk_img)
        mask = sitk.GetArrayFromImage(itk_mask)
        tensor_img = torch.from_numpy(img).float()
        tensor_mask = torch.from_numpy(mask).long()
        return tensor_img, tensor_mask

