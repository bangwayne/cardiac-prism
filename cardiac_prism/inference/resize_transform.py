import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F


class SegmentationPreprocessor:
    def __init__(self, long_size=512):
        self.long_size = long_size

    def resize_and_pad(self, img_tensor, mask):
        """Resize image to make long side = long_size, then pad to square."""
        c, h, w = img_tensor.shape
        scale = self.long_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        resized_img = F.interpolate(img_tensor.unsqueeze(0), size=(new_h, new_w),
                                    mode='bilinear', align_corners=False)
        resized_mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=(new_h, new_w),
                                     mode='nearest')
        pad_h = self.long_size - new_h
        pad_w = self.long_size - new_w

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        padded_image = F.pad(resized_img, (pad_left, pad_right, pad_top, pad_bottom), value=0).squeeze(0)
        padded_mask = F.pad(resized_mask, (pad_left, pad_right, pad_top, pad_bottom), value=0).squeeze(0).squeeze(
            0).long()
        return padded_image, padded_mask, (h, w), (pad_top, pad_bottom, pad_left, pad_right), scale

    def unpad_and_resize(self, tensor, original_size, pad):
        """Remove padding and resize to original size."""
        pad_top, pad_bottom, pad_left, pad_right = pad
        _, h, w = tensor.shape
        cropped = tensor[:, pad_top:h - pad_bottom, pad_left:w - pad_right]
        resized = F.interpolate(cropped.unsqueeze(0), size=original_size,
                                mode='bilinear').squeeze(0)
        return resized

    def map_valid_points_back(self, points, labels, original_size, pad, scale_factor):
        """
        Args:
            points: (N, 2)
            labels: (N,)
            original_size: (h, w)
            pad: (pad_top, pad_bottom, pad_left, pad_right)
            resized_shape: (new_h, new_w)

        Returns:
            new_points: (M, 2)
            new_labels: (M,)
        """
        pad_top, pad_bottom, pad_left, pad_right = pad
        orig_h, orig_w = original_size

        # 找到有效点（labels != -1）
        valid_mask = labels > -1
        valid_points = points[valid_mask]
        valid_labels = labels[valid_mask]

        # 去除 padding 坐标
        unpad_x = valid_points[:, 0] - pad_left
        unpad_y = valid_points[:, 1] - pad_top

        # 计算缩放因子
        scale_x = 1 / scale_factor
        scale_y = 1 / scale_factor

        # 映射回原图
        orig_x = (unpad_x * scale_x).round().long()
        orig_y = (unpad_y * scale_y).round().long()

        # 拼接回坐标
        mapped_points = torch.stack([orig_x, orig_y], dim=1)
        return mapped_points, valid_labels
