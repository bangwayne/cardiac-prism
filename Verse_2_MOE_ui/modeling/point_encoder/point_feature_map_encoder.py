import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from torch.nn import functional as F
import math


class FeatureProjectionMLP(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, output_dim=256):
        super(FeatureProjectionMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # Project to hidden dimension
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # Project back to original dimension
        )

    def forward(self, feature_query):
        # feature_query shape: (B, N, 256)
        B, N, C = feature_query.shape
        feature_query = feature_query.view(-1, C)  # Reshape to (B*N, 256)
        projected_features = self.mlp(feature_query)
        return projected_features.view(B, N, C)  # Reshape back to (B, N, 256)


def point_combine(points, labels):
    return torch.cat((points, labels.unsqueeze(1)), dim=1)


# Initialize feature maps
def get_coord_features(image, point_tuple, radius=1, mask=None):
    height, width = image.shape[1], image.shape[2]
    points = point_tuple[0].int().to(image.device)
    labels = point_tuple[1].int().to(image.device)
    combined_point = torch.cat((points, labels.unsqueeze(1)), dim=1)
    feature_map = torch.zeros(2, height, width, device=image.device)

    # Create a grid of coordinates
    x = torch.arange(height, device=image.device)
    y = torch.arange(width, device=image.device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')

    for point in combined_point:
        y_center, x_center, label = point
        # Compute the mask for points within the circle
        point_mask = (grid_y - y_center) ** 2 + (grid_x - x_center) ** 2 <= radius ** 2
        if label == 0:
            # 0 is the negative point
            feature_map[1][point_mask] = 1
        elif label == 1:
            feature_map[0][point_mask] = 1

    if mask is None:
        mask = torch.zeros(1, height, width, device=image.device)
    else:
        mask = mask.unsqueeze(0).to(image.device)

    final_feature_map = torch.cat((feature_map, mask), dim=0)
    return final_feature_map


# point_feature_batch_data = get_batch_point_feature_map(batch_data_with_next_point).to(self.device)

def get_batch_point_feature_map(batch_data):
    batch_size = len(batch_data)
    point_feature_map_list = []
    for bs in range(batch_size):
        image = batch_data[bs]['image']
        device = image.device
        point_feature_map_mask_num = []
        for mask_num in range(len(batch_data[bs]['points_list'])):
            point_tuple = batch_data[bs]['points_list'][mask_num]
            mask = None if batch_data[bs]['click_index'] == 0 else batch_data[bs]['seg_result']
            if mask is not None:
                point_feature_map = get_coord_features(image, point_tuple, radius=1, mask=mask[mask_num])
            else:
                point_feature_map = get_coord_features(image, point_tuple, radius=1, mask=None)
            point_feature_map_mask_num.append(point_feature_map.unsqueeze(0))
        point_feature_tensor = torch.cat(point_feature_map_mask_num, dim=0).to(device)
        point_feature_map_list.append(point_feature_tensor.unsqueeze(0))

    point_feature_tensor = torch.cat(point_feature_map_list, dim=0).to(device)
    return point_feature_tensor


def get_coord(point_tuple, device, target_size=(32, 32), resize_scale=8, radius=1, mask=None):
    height, width = target_size[0], target_size[1]
    points = point_tuple[0].float().to(device)
    labels = point_tuple[1].int().to(device)
    resize_points = torch.div(points, resize_scale).round().int()
    combined_point = torch.cat((resize_points, labels.unsqueeze(1)), dim=1)
    feature_map = torch.zeros(2, height, width, device=device)

    x = torch.arange(height, device=device)
    y = torch.arange(width, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')

    for point in combined_point:
        y_center, x_center, label = point
        if label < 0:
            continue
        y_center = torch.clamp(y_center, max=height - 1)
        x_center = torch.clamp(x_center, max=width - 1)

        if height > 32 or width > 32:
            point_mask = (grid_y - y_center) ** 2 + (grid_x - x_center) ** 2 <= radius ** 2
            if label == 0:
                feature_map[1][point_mask] = 1
            elif label == 1:
                feature_map[0][point_mask] = 1
        else:
            if label == 0:
                # 0 is the negative point
                feature_map[1, y_center, x_center] = 1
            elif label == 1:
                feature_map[0, y_center, x_center] = 1

    if mask is None:
        mask = torch.zeros(1, height, width, device=device)
    else:
        # mask = mask.unsqueeze(0).to(device)
        mask = mask.to(device)
    final_feature_map = torch.cat((feature_map, mask), dim=0)

    return final_feature_map


def get_resize_feature_map(point_tuple_list, resize_scale=8, target_size=(32, 32), mask=None):
    """
    :param point_tuple_list: A list of the list of the point_tuple [[(point,label)],..., [(point,label)]]
    :param resize_scale: the multiplier of the resolution
    :param target_size:  the target size of the new generated attention_mask
    :param mask: the original size mask which generated by last iteration, with size (B, H, W)
    """
    batch_size = len(point_tuple_list)
    point_feature_map_list = []
    device = point_tuple_list[0][0][0].device
    mask = F.interpolate(mask.unsqueeze(1), size=target_size, mode="bilinear",
                         align_corners=False)
    for bs in range(batch_size):
        point_tuple = point_tuple_list[bs]
        point_feature_map_mask_num = []
        if mask is not None:
            point_feature_map = get_coord(point_tuple[0], device, target_size=target_size, resize_scale=resize_scale,
                                          radius=1, mask=mask[bs])
        else:
            point_feature_map = get_coord(point_tuple[0], device, target_size=target_size, resize_scale=resize_scale,
                                          radius=1, mask=None)

        point_feature_map_mask_num.append(point_feature_map.unsqueeze(0))
        point_feature_tensor = torch.cat(point_feature_map_mask_num, dim=0).to(device)
        point_feature_map_list.append(point_feature_tensor.unsqueeze(0))

    point_feature_tensor = torch.cat(point_feature_map_list, dim=0).to(device)
    return point_feature_tensor


def get_point_feature(feature_map, point_tuple_list, resize_scale=8, radius=1, *, projection):
    """
    Get point features for each point in the list. If a point has valid features, use them;
    otherwise, use the mean feature in the surrounding 3x3 region.

    Args:
        feature_map (torch.Tensor): Input feature map of shape (HW,B,C) with C=256.
        point_tuple_list (list): List of tuples containing points and labels for each batch.
        device (torch.device): Device for computation.
        target_size (tuple): Target size for feature map after downsampling.
        resize_scale (int): Scale factor for resizing points.
        radius (int): Radius for the neighborhood if the point feature is invalid.
        projection (nn.Module): Persistent projection registered on the decoder.

    Returns:
        torch.Tensor: Projected feature tensor of shape (B, N, 256).
    """
    feature_height, feature_width = math.isqrt(feature_map.shape[0]), math.isqrt(feature_map.shape[0])
    bs, channel = feature_map.shape[1], feature_map.shape[2]
    feature_map = feature_map.view(feature_height, feature_width, bs, channel)
    device = feature_map.device
    feature_map = feature_map.permute(2, 3, 0, 1).contiguous()

    B, C, H, W = feature_map.shape
    batch_features = []

    for bs in range(B):
        points, labels = point_tuple_list[bs][0][0].float(), point_tuple_list[bs][0][1].int()
        resized_points = torch.div(points, resize_scale).round().int()  # Downsample points
        single_feature_map = feature_map[bs]
        features = []

        for i in range(resized_points.shape[0]):
            y, x = resized_points[i]
            if 0 <= y < H and 0 <= x < W:
                if H < 64 and single_feature_map[:, y, x].sum() != 0:
                    # Directly use the point feature if it's valid
                    point_feature = single_feature_map[:, y, x]
                else:
                    # Otherwise, use the mean of the surrounding 3x3 area
                    y_min, y_max = max(0, y - radius), min(H, y + radius + 1)
                    x_min, x_max = max(0, x - radius), min(W, x + radius + 1)
                    surrounding_features = single_feature_map[:, y_min:y_max, x_min:x_max]
                    point_feature = surrounding_features.mean(dim=(1, 2))
            else:
                point_feature = torch.zeros(C, device=feature_map.device)
            features.append(point_feature)
        # Concatenate all features for this batch
        batch_features.append(torch.stack(features, dim=0))

    # Stack batch features into shape (B, N, 256)
    point_feature_query = torch.stack(batch_features, dim=0)
    projected_feature_query = projection(point_feature_query)
    projected_feature_query = projected_feature_query.permute(1, 0, 2).contiguous()
    return projected_feature_query
