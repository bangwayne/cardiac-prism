import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import SimpleITK as sitk
from PIL import Image
import yaml
import math
import random
import logging
import copy
from pathlib import Path
import cv2

import os
from torch.utils.data import Dataset
from datetime import datetime


class SafeDataset(Dataset):
    def __init__(self, base_dataset,
                 error_log_path="bad_samples.txt"):
        self.base_dataset = base_dataset
        self.error_log_path = error_log_path
        self.max_retry = 10
        self.bad_indices = set()

        with open(self.error_log_path, 'w') as f:
            f.write(f"# Bad samples logged on {datetime.now()}\n")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        retry = 0
        while retry < self.max_retry:
            try:
                return self.base_dataset[index]
            except Exception as e:
                # 记录出错样本
                sample_info = self._get_sample_info(index)
                with open(self.error_log_path, 'a') as f:
                    f.write(f"Index {index}, Sample: {sample_info}, Error: {str(e)}\n")
                self.bad_indices.add(index)

                index = (index + 1) % len(self.base_dataset)
                retry += 1

        raise RuntimeError(f"Exceeded max retries at index {index}. Last error: {str(e)}")

    def _get_sample_info(self, index):
        try:
            meta = self.base_dataset.metadata[index]
            return f"img={meta[0]}, mask={meta[1]}, file={meta[5]}"
        except Exception:
            return f"Index {index} (failed to retrieve metadata)"


class MultiTrainDataset(Dataset):
    def __init__(self,
                 cfg,
                 *,
                 images_dir_name="image",
                 masks_dir_name="annotations",
                 mode='train',
                 transform=None,
                 ref_transform=None,
                 three_chanel=True):
        dataset_cfg = cfg.data
        self.mode = mode
        self.data_name_list = dataset_cfg.dataset_list
        self.dataset_weight = dataset_cfg.dataset_weight
        self.data_dir = dataset_cfg.dataset_path
        self.num_ref = dataset_cfg.num_ref
        self.data_config = dataset_cfg.data_config
        self.dataset_path_list = [Path(self.data_dir + "/" + data_name + "/Data") for data_name in self.data_name_list]
        self.images_path_list = [dataset_path / mode / images_dir_name for dataset_path in self.dataset_path_list]
        self.masks_path_list = [dataset_path / mode / masks_dir_name for dataset_path in self.dataset_path_list]
        self.three_channel = three_chanel
        self.transform = transform
        self.ref_transform = ref_transform
        self.dataset_samples = []
        self.metadata, self.ref_metadata, self.sample_weight_list = self.load_metadata()
        # for i in range(len(self.data_name_list)):
        #     images_path = self.images_path_list[i]
        #     for x in sorted(images_path.glob('*.nii.gz')):
        #         self.dataset_samples.append(x.name)

    def load_metadata(self):
        metadata = []
        ref_metadata = []
        weight_list = []
        dataset_slice_counts = {name: 0 for name in self.data_config}

        for idx, data_name in enumerate(self.data_name_list):
            images_path = self.images_path_list[idx]
            masks_path = self.masks_path_list[idx]
            ndim = self.data_config[data_name]["ndim"]
            if ndim == 3:
                img_name_list = [x.name for x in sorted(images_path.glob('*.nii.gz'))]
                print(f"Start loading {data_name} {self.mode} metadata")

                for filename in img_name_list:
                    img_path = os.path.join(images_path, filename)
                    if data_name != "BraTS20":
                        mask_name = filename.split('.')[0] + "_gt.nii.gz"
                    else:
                        mask_name = filename.split('.')[0].rsplit('_', 1)[0] + "_gt.nii.gz"
                    mask_path = os.path.join(masks_path, mask_name)

                    itk_mask = sitk.ReadImage(mask_path)
                    array_mask = sitk.GetArrayFromImage(itk_mask)

                    unique_labels = self.data_config[data_name]["unique_labels"]
                    slices_num = array_mask.shape[0]
                    ref_metadata.append(
                                 (img_path,
                                  mask_path,
                                  filename,
                                  data_name,
                                  ndim)
                                        )
                    for slice_index in range(slices_num):
                        slice_mask = array_mask[slice_index]
                        slice_unique_labels = np.unique(slice_mask[slice_mask != 0])

                        # Filter slice_unique_labels to only keep elements present in unique_labels
                        slice_unique_labels = [label for label in slice_unique_labels if label in unique_labels]

                        if len(slice_unique_labels) > 0:
                            weight_list.append(self.dataset_weight[data_name])
                            dataset_slice_counts[data_name] += 1
                            slice_ratio = slice_index / slices_num
                            metadata.append(
                                (img_path,
                                 mask_path,
                                 slice_index,
                                 unique_labels,
                                 slice_unique_labels,
                                 filename,
                                 data_name,
                                 ndim,
                                 slice_ratio)
                            )
            elif ndim == 2:
                d2_img_name_list = [x.name for x in sorted(images_path.glob('*.png'))]
                print(f"Start loading {data_name} {self.mode} metadata")

                for filename in d2_img_name_list:
                    img_path = os.path.join(images_path, filename)
                    mask_name = filename.split('.')[0] + "_gt.png"
                    mask_path = os.path.join(masks_path, mask_name)
                    # --- Read image and convert to tensor (C, H, W) ---
                    # --- Read mask (as grayscale image) ---
                    # mask = Image.open(mask_path).convert("L")
                    ref_metadata.append(
                                 (img_path,
                                  mask_path,
                                  filename,
                                  data_name,
                                  ndim)
                                        )
                    array_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    # array_mask = np.array(mask)  # shape (H, W)
                    unique_labels = self.data_config[data_name]["unique_labels"]
                    # Since it's 2D, treat whole image as one "slice"
                    slice_index = 0
                    slice_mask = array_mask
                    slice_unique_labels = np.unique(slice_mask[slice_mask != 0])

                    if len(slice_unique_labels) > 0:
                        weight_list.append(self.dataset_weight[data_name])
                        dataset_slice_counts[data_name] += 1
                        metadata.append(
                            (img_path,
                             mask_path,
                             slice_index,
                             unique_labels,
                             slice_unique_labels,
                             filename,
                             data_name,
                             ndim,
                             slice_index)
                        )

        print(f"Load done, length of dataset: {len(metadata)}")
        print("Slice counts per dataset:")
        for dataset, count in dataset_slice_counts.items():
            print(f"  {dataset}: {count} slices")

        assert len(metadata) == len(weight_list), (
            f"Length mismatch: metadata ({len(metadata)}) and weight_list ({len(weight_list)})"
        )
        return metadata, ref_metadata, weight_list

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_path, mask_path, slice_index, labels, slice_labels, filename, data_name, ndim, slice_ratio = self.metadata[idx]
        query_mapping = self.data_config[data_name]["query_mapping"]
        # if self.mode == "train":
        unique_label = np.random.choice(slice_labels)
        unique_query = query_mapping[int(unique_label)][0]
        if ndim == 3:
            itk_img, itk_mask = sitk.ReadImage(img_path), sitk.ReadImage(mask_path)
            img, mask = sitk.GetArrayFromImage(itk_img), sitk.GetArrayFromImage(itk_mask)  #
            if itk_img is None:
                raise FileNotFoundError(f"SimpleITK failed to read image: {img_path}")
            if itk_mask is None:
                raise FileNotFoundError(f"SimpleITK failed to read mask: {mask_path}")
            # num_layer, h, w
            height, width = img.shape[1], img.shape[2]
            gray_image = torch.from_numpy(img[slice_index]).float()
            # (h, w)
            if self.three_channel:
                gray_image = gray_image.unsqueeze(0)
                image = gray_image.repeat(3, 1, 1)  # (3, h, w)
            else:
                image = gray_image
            sem_seg = torch.from_numpy(mask[slice_index]).long()
            # sem_seg: (h, w)
        else:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # shape: (H, W, 3)

            # Convert BGR -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # now (H, W, 3)

            # Transpose to (3, H, W) and convert to float32 Tensor
            image = torch.from_numpy(img.transpose(2, 0, 1)).float()

            height, width = img.shape[1], img.shape[2]

            # Read mask in grayscale
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # shape: (H, W)
            # mask = np.array(mask)
            if img is None:
                raise FileNotFoundError(f"Image file not found or unreadable: {img_path}")
            if mask is None:
                raise FileNotFoundError(f"Mask file not found or unreadable: {mask_path}")
            # (h, w)
            sem_seg = torch.from_numpy(mask).long()

        ref_images, ref_masks = self.sample_reference_batch(
            current_idx=idx,
            data_name=data_name,
            unique_label=unique_label,
            ref_transform=self.ref_transform,
            num_refs=self.num_ref,
            ndim=ndim,
        )
        # (num_ref, 3, h, w)
        # (num_ref, 1, h, w)

        sample = {
            'image': image,
            'sem_seg': sem_seg,
            'file_name': filename,
            'slice_index': slice_index,
            'width': width,
            'height': height,
            'q_index': unique_query,
            'ref_img': ref_images,
            'ref_mask': ref_masks
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

    def sample_reference_batch(self,
                               current_idx: int,
                               data_name: str,
                               unique_label: int,
                               ref_transform: bool = True,
                               num_refs: int = 3,
                               ndim: int = 3,
                               slice_index: int = None) -> (torch.Tensor, torch.Tensor):
        ref_images = []
        ref_masks = []

        if ndim == 3:
            candidate_refs = [
                i for i, data in enumerate(self.ref_metadata)
                if data[-2] == data_name and data[-1] == 3
            ]
            random.shuffle(candidate_refs)
            # selected_indices = np.random.choice(candidate_refs, size=num_refs, replace=False)
            valid_indices = []
            for ref_idx in candidate_refs:
                if len(valid_indices) >= num_refs:
                    break
                ref_img_path, ref_mask_path, filename, _, _ = self.ref_metadata[ref_idx]
                itk_ref_mask = sitk.ReadImage(ref_mask_path)
                tensor_ref_mask = sitk.GetArrayFromImage(itk_ref_mask)  # [D, H, W]

                if np.any(np.isin(tensor_ref_mask, unique_label)):
                    valid_indices.append(ref_idx)

            for selected_idx in valid_indices:
                ref_img_path, ref_mask_path, filename, _, _ = self.ref_metadata[selected_idx]
                itk_ref_img = sitk.ReadImage(ref_img_path)
                itk_ref_mask = sitk.ReadImage(ref_mask_path)
                tensor_ref_img = sitk.GetArrayFromImage(itk_ref_img)  # [D, H, W]
                tensor_ref_mask = sitk.GetArrayFromImage(itk_ref_mask)  # [D, H, W]

                num_slices = tensor_ref_img.shape[0]
                slice_ratio = self.metadata[current_idx][-1]
                center_slice = int(slice_ratio * num_slices)

                # Check slices in the range [center_slice - 2, center_slice + 2]
                offsets = list(range(-2, 3))
                slice_indices = [np.clip(center_slice + offset, 0, num_slices - 1) for offset in offsets]

                found = False
                for i in slice_indices:
                    if np.any(np.isin(tensor_ref_mask[i], unique_label)):
                        ref_image = torch.from_numpy(tensor_ref_img[i]).float()
                        ref_mask = torch.from_numpy(tensor_ref_mask[i]).long()

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
                        ref_images.append(ref_image)
                        ref_masks.append(binary_mask)
                        found = True
                        break

                if not found:
                    # Randomly select slices until a matching label is found
                    valid_slices = [i for i in range(num_slices) if np.any(np.isin(tensor_ref_mask[i], unique_label))]

                    if valid_slices:
                        sampled_slice = np.random.choice(valid_slices)
                    else:
                        # fallback: just pick a random slice if no valid one exists
                        sampled_slice = np.random.randint(0, num_slices)
                        # sampled_slice = np.random.choice(valid_slices)

                    ref_image = torch.from_numpy(tensor_ref_img[sampled_slice]).float()
                    ref_mask = torch.from_numpy(tensor_ref_mask[sampled_slice]).long()

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
                    ref_images.append(ref_image)
                    ref_masks.append(binary_mask)

        else:
            same_dataset_indices = [
                i for i, data in enumerate(self.ref_metadata)
                if data[-2] == data_name and data[-1] == 2
            ]

            # Shuffle same_dataset_indices to ensure randomness
            random.shuffle(same_dataset_indices)

            # Identify images with the unique label
            valid_indices = []
            for ref_idx in same_dataset_indices:
                if len(valid_indices) >= num_refs:
                    break
                ref_img_path, ref_mask_path, filename, _, _ = self.ref_metadata[ref_idx]
                ref_mask = Image.open(ref_mask_path).convert("L")
                ref_mask_array = np.array(ref_mask)

                if np.any(np.isin(ref_mask_array, unique_label)):
                    valid_indices.append(ref_idx)

            # Sample from identified images
            if valid_indices:
                ref_indices = np.random.choice(valid_indices, size=num_refs, replace=False)

                for ref_idx in ref_indices:
                    ref_img_path, ref_mask_path, filename, _, _ = self.ref_metadata[ref_idx]
                    ref_img = Image.open(ref_img_path).convert("RGB")
                    ref_mask = Image.open(ref_mask_path).convert("L")

                    ref_img = torch.from_numpy(np.array(ref_img).transpose(2, 0, 1)).float()
                    ref_mask = torch.from_numpy(np.array(ref_mask)).long()

                    if ref_transform:
                        ref_sample = {'ref_img': ref_img, 'sem_seg': ref_mask}
                        ref_sample = self.ref_transform(ref_sample)
                        ref_img = ref_sample['image']
                        ref_mask = ref_sample['sem_seg']

                    binary_mask = (ref_mask == unique_label).float().unsqueeze(0)
                    ref_images.append(ref_img)
                    ref_masks.append(binary_mask)

        ref_images = torch.stack(ref_images, dim=0)
        ref_masks = torch.stack(ref_masks, dim=0)

        return ref_images, ref_masks

    @staticmethod
    def preprocess(itk_img, itk_mask):
        img = sitk.GetArrayFromImage(itk_img)
        mask = sitk.GetArrayFromImage(itk_mask)
        tensor_img = torch.from_numpy(img).float()
        tensor_mask = torch.from_numpy(mask).long()
        return tensor_img, tensor_mask

