import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import SimpleITK as sitk
import yaml
import math
import random
import logging
import copy
from pathlib import Path

class TestDataset(Dataset):
    def __init__(self, cfg, data_dir, images_dir_name="image", masks_dir_name="annotations", mode='test',
                 transform=None, three_chanel=True):

        self.mode = mode
        self.dataset_path = Path(data_dir)
        self.images_path = self.dataset_path / mode / images_dir_name
        self.masks_path = self.dataset_path / mode / masks_dir_name
        self.reference_path = self.dataset_path / mode / 'reference'
        self.three_channel = three_chanel
        self.transform = transform
        self.metadata = self.load_metadata()

    def load_metadata(self):
        metadata = []
        img_name_list = [x.name for x in
                         sorted(self.images_path.glob('*.nii.gz'))]  # This is a namelist with .nii.gz
        logging.info(f"Start loading {self.mode} metadata")
        print(f"Start loading {self.mode} metadata")
        if self.mode == "train":
            print("we use the training mode")
            for filename in img_name_list:
                img_name = filename
                mask_name = img_name.split('.')[0] + "_gt.nii.gz"

                img_path = os.path.join(self.images_path, img_name)
                mask_path = os.path.join(self.masks_path, mask_name)
                itk_mask = sitk.ReadImage(mask_path)
                array_mask = sitk.GetArrayFromImage(itk_mask)
                unique_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
                slices_num = array_mask.shape[0]
                for slice_index in range(0, slices_num):
                    slice_mask = array_mask[slice_index]
                    real_unique_labels = np.unique(slice_mask[slice_mask != 0])
                    if len(real_unique_labels) != 0:
                        for real_label in unique_labels:
                        # for real_label in real_unique_labels:
                            metadata.append((img_path, mask_path, slice_index, unique_labels, real_label))
        else:
            for filename in img_name_list:
                img_name = filename
                mask_name = img_name.split('.')[0] + "_gt.nii.gz"

                img_path = os.path.join(self.images_path, img_name)
                mask_path = os.path.join(self.masks_path, mask_name)
                itk_mask = sitk.ReadImage(mask_path)
                array_mask = sitk.GetArrayFromImage(itk_mask)
                unique_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
                slices_num = array_mask.shape[0]
                for slice_index in range(0, slices_num):
                    slice_mask = array_mask[slice_index]
                    real_unique_labels = np.unique(slice_mask[slice_mask != 0])
                    if len(real_unique_labels) != 0:
                        for real_label in real_unique_labels:
                            metadata.append((img_path, mask_path, slice_index, unique_labels, real_label))
        logging.info(f"Load done, length of dataset: {len(metadata)}")
        print(f"Load done, length of dataset: {len(metadata)}")

        return metadata

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_path, mask_path, slice_index, labels, unique_label = self.metadata[idx]

        itk_img = sitk.ReadImage(img_path)
        itk_mask = sitk.ReadImage(mask_path)

        tensor_img, tensor_mask = self.preprocess(itk_img, itk_mask)

        height, width = tensor_img.shape[1], tensor_img.shape[2]

        gray_image_tensor = tensor_img[slice_index]

        if self.three_channel:
            gray_image_tensor = gray_image_tensor.unsqueeze(0)
            three_channel_image = gray_image_tensor.repeat(3, 1, 1)
            image = three_channel_image
            # image = self.get_three_channel_image(tensor_img, slice_index)
        else:
            image = gray_image_tensor

        sample = {'image': image,
                  'sem_seg': tensor_mask[slice_index]}

        if self.transform:
            sample = self.transform(sample)

        sample['file_name'] = img_path
        sample['slice_index'] = slice_index
        sample['width'] = width
        sample['height'] = height
        sample['q_index'] = unique_label


        transformed_mask = sample['sem_seg']

        single_mask = torch.stack([(transformed_mask == unique_label).bool()])
        target = {
            "labels": torch.tensor(labels).long(),
            'unique_labels': self.get_label([unique_label]).long(),
            # this place is very important to keep the label dtype long, else it will report some error
            "masks": single_mask
        }
        sample['target'] = target

        return sample

    def get_ref_pair(self, unique_label):
        ref_img_path = self.reference_path / "ref_img.nii.gz"
        ref_mask_path = self.reference_path / "ref_mask.nii.gz"
        # Load reference image and mask
        itk_ref_img = sitk.ReadImage(ref_img_path)
        itk_ref_mask = sitk.ReadImage(ref_mask_path)
        tensor_ref_img, tensor_ref_mask = self.preprocess(itk_ref_img, itk_ref_mask)

        ref_image = tensor_ref_img[ref_slice_index]
        ref_mask = tensor_ref_mask[ref_slice_index]

        if self.three_channel:
            ref_image = ref_image.unsqueeze(0).repeat(3, 1, 1)

        ref_single_mask = torch.stack([(ref_mask == unique_label).bool()])
        ref_single_mask = ref_single_mask.to(torch.float32)

    @staticmethod
    def preprocess(itk_img, itk_mask):
        img = sitk.GetArrayFromImage(itk_img)
        mask = sitk.GetArrayFromImage(itk_mask)
        tensor_img = torch.from_numpy(img).float()
        tensor_mask = torch.from_numpy(mask).long()
        return tensor_img, tensor_mask

    @staticmethod
    def get_label(unique_labels, label_set=[1, 2, 3]):
        label_mask_list = []
        for label in unique_labels:
            label_mask = [0] * (len(label_set) + 1)
            if label in label_set:
                index = label_set.index(label)
                label_mask[index] = 1
            else:
                label_mask[-1] = 1
            label_mask_list.append(label_mask)

        return torch.tensor(label_mask_list)

    @staticmethod
    def get_three_channel_image(self, tensor_img, slice_index):
        current_slice = tensor_img[slice_index]

        if slice_index == 0:
            previous_slice = current_slice
        else:
            previous_slice = tensor_img[slice_index - 1]

        if slice_index == tensor_img.shape[0] - 1:
            next_slice = current_slice
        else:
            next_slice = tensor_img[slice_index + 1]

        three_channel_image = torch.stack([previous_slice, current_slice, next_slice], dim=0)

        return three_channel_image
    




