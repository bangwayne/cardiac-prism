import logging
import math
import numpy as np
import torch
import copy
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, ConcatDataset
import os
from torchvision.io import read_image
import json
from albumentations import *
from albumentations.pytorch import ToTensorV2
from monai import data, transforms
import torch.nn.functional as F
from .datasampler import *
from .train_dataloader import MultiTrainDataset, SafeDataset

import torch
import numpy as np
from torch.utils.data import Sampler
import math

from albumentations import (
    HorizontalFlip, ShiftScaleRotate, RandomBrightnessContrast,
    GaussianBlur, Normalize, Resize, Compose
)
from albumentations.pytorch import ToTensorV2


def build_image_transform(long_size=512, final_size=512):
    return Compose([
        LongestMaxSize(max_size=long_size),
        PadIfNeeded(min_height=final_size, min_width=final_size, border_mode=0, value=0, mask_value=0),
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=45, p=0.3,
                         border_mode=0, value=0, mask_value=0),
        RandomBrightnessContrast(p=0.4),
        GaussianBlur(p=0.3),
        GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        GridDistortion(p=0.3, border_mode=0, value=0, mask_value=0),
        Normalize(mean=(0.26, 0.26, 0.26), std=(0.30, 0.30, 0.30), max_pixel_value=255.0),
        ToTensorV2(transpose_mask=True)
    ])


def build_ref_image_transform(long_size=512, final_size=512):
    return Compose([
        LongestMaxSize(max_size=long_size),
        PadIfNeeded(min_height=final_size, min_width=final_size, border_mode=0, value=0, mask_value=0),
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=45, p=0.3,
                         border_mode=0, value=0, mask_value=0),
        RandomBrightnessContrast(p=0.4),
        GaussianBlur(p=0.3),
        GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        GridDistortion(p=0.3, border_mode=0, value=0, mask_value=0),
        Normalize(mean=(0.26, 0.26, 0.26), std=(0.30, 0.30, 0.30), max_pixel_value=255.0),
        ToTensorV2(transpose_mask=True)
    ])


class AlbuTransformWrapper:
    def __init__(self, albu_transform):
        self.albu_transform = albu_transform

    def __call__(self, sample):
        image = sample['image'].numpy().transpose(1, 2, 0).astype('uint8')  # (H, W, 3)
        mask = sample['sem_seg'].numpy().astype('uint8')  # (H, W)

        augmented = self.albu_transform(image=image, mask=mask)
        sample['image'] = augmented['image']  # torch.tensor [3, H, W]
        sample['sem_seg'] = augmented['mask'].long()  # torch.tensor [H, W]
        return sample


class RefAlbuTransformWrapper:
    def __init__(self, ref_albu_transform):
        self.albu_transform = ref_albu_transform

    def __call__(self, sample):
        image = sample['ref_img'].numpy().transpose(1, 2, 0).astype('uint8')  # (H, W, 3)
        mask = sample['sem_seg'].numpy().astype('uint8')  # (H, W)

        augmented = self.albu_transform(image=image, mask=mask)
        sample['image'] = augmented['image']  # torch.tensor [3, H, W]
        sample['sem_seg'] = augmented['mask'].long()  # torch.tensor [H, W]

        return sample


def get_loader(cfg):
    train_transform = AlbuTransformWrapper(build_image_transform(long_size=512, final_size=512))
    ref_transform = RefAlbuTransformWrapper(build_ref_image_transform(long_size=512, final_size=512))

    combination_train_ds = MultiTrainDataset(cfg,
                                             images_dir_name="image",
                                             masks_dir_name="annotations",
                                             mode='val',
                                             transform=train_transform,
                                             ref_transform=ref_transform)
    samples_weight = torch.from_numpy(np.array(combination_train_ds.sample_weight_list))
    train_sampler = WeightedDistributedSampler(combination_train_ds, samples_weight)
    train_loader = data.DataLoader(
        combination_train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=(train_sampler is None),
        num_workers=cfg.training.num_workers,
        sampler=train_sampler,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=batch_collate_fn,
    )

    return train_loader
