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
from .test_dataloader import SemanticTestDataset, FewShotTestDataset

import torch
import numpy as np
from torch.utils.data import Sampler
import math

from albumentations import (
    HorizontalFlip, ShiftScaleRotate, RandomBrightnessContrast,
    GaussianBlur, Normalize, Resize, Compose
)
from albumentations.pytorch import ToTensorV2


def build_test_image_transform():
    return Compose([
        Normalize(mean=(0.26, 0.26, 0.26), std=(0.30, 0.30, 0.30), max_pixel_value=255.0),
        ToTensorV2(transpose_mask=True)
    ])

def build_ref_image_transform(long_size=512, final_size=512):
    return Compose([
        LongestMaxSize(max_size=long_size),
        PadIfNeeded(min_height=final_size, min_width=final_size, border_mode=0, value=0, mask_value=0),
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
        # sample['image'] = torch.from_numpy(augmented['image'].transpose(2, 0, 1)).float()  # (3, H, W)
        # sample['sem_seg'] = torch.from_numpy(augmented['mask']).long()  # (H, W)

        return sample


def get_loader(cfg):
    test_transform = AlbuTransformWrapper(build_test_image_transform())
    ref_transform = RefAlbuTransformWrapper(build_ref_image_transform())
    if cfg.testing.testing_click_mode[0] in ["1", "3"]:
        print("import semantic or click dataset!")
        test_ds = SemanticTestDataset(cfg,
                                      images_dir_name="image",
                                      masks_dir_name="annotations",
                                      mode='test',
                                      transform=test_transform)
    else:
        print("import fewshot dataset!")
        test_ds = FewShotTestDataset(cfg,
                                    images_dir_name="image",
                                    masks_dir_name="annotations",
                                    mode='test',
                                    transform=test_transform,
                                    ref_transform=ref_transform)

    # default to be true
    # samples_weight = torch.from_numpy(np.array(combination_train_ds.sample_weight_list))
    if cfg.testing.debug:
        test_loader = data.DataLoader(
            test_ds,
            batch_size=cfg.testing.batch_size,
            # shuffle=True,
            shuffle=False,
            num_workers=cfg.testing.num_workers,
            pin_memory=True,
            persistent_workers=False,
            collate_fn=batch_collate_fn,
        )
    else:
        test_sampler = DistributedSampler(test_ds)

        test_loader = data.DataLoader(
                                        test_ds,
                                        batch_size=cfg.testing.batch_size,
                                        # shuffle=True,
                                        shuffle=False,
                                        sampler=test_sampler,
                                        num_workers=cfg.testing.num_workers,
                                        pin_memory=True,
                                        persistent_workers=False,
                                        collate_fn=batch_collate_fn,
                                    )

    return test_loader
