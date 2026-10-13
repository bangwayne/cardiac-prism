from typing import List, Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class CAResBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, residual: bool = True):
        super().__init__()
        self.residual = residual
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)

        t = int((abs(math.log2(out_dim)) + 1) // 2)
        k = t if t % 2 else t + 1
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)

        if self.residual:
            if in_dim == out_dim:
                self.downsample = nn.Identity()
            else:
                self.downsample = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = self.conv1(F.relu(x))
        x = self.conv2(F.relu(x))

        b, c = x.shape[:2]
        w = self.pool(x).view(b, 1, c)
        w = self.conv(w).transpose(-1, -2).unsqueeze(-1).sigmoid()  # B*C*1*1

        if self.residual:
            x = x * w + self.downsample(r)
        else:
            x = x * w

        return x


class PixelFFN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.conv = CAResBlock(dim, dim)

    def forward(self, pixel_flat: torch.Tensor, feature_size) -> torch.Tensor:
        h, w = feature_size
        hw, bs, embed_dim = pixel_flat.shape

        assert hw == h * w, f"Expected {h * w}, but got {hw}"

        pixel_flat = pixel_flat.view(h, w, bs, self.dim)
        pixel_flat = pixel_flat.permute(2, 3, 0, 1).contiguous()  # [Batch_size, dim, H, W]

        x = self.conv(pixel_flat)

        x = x.permute(2, 3, 0, 1).contiguous()  # [H, W, Batch_size, dim]
        x = x.view(hw, bs, self.dim)  # [(H*W), Batch_size, dim]

        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)

        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        out = F.relu(out)
        return out


class SimpleMaskDecoder(nn.Module):
    def __init__(self, in_channels: int, num_res_blocks: int = 2, out_channels: int = 32):
        super(SimpleMaskDecoder, self).__init__()

        self.initial_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.res_blocks = nn.ModuleList([ResBlock(out_channels, out_channels) for _ in range(num_res_blocks)])
        self.final_conv = nn.Conv2d(out_channels, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_queries, height, width = x.shape
        x = x.view(batch_size, num_queries, height, width)
        x = F.relu(self.initial_conv(x))
        for res_block in self.res_blocks:
            x = res_block(x)

        x = self.final_conv(x)  # [B, 1, H, W]
        return x


class MultiScaleDownsamplingModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleDownsamplingModule, self).__init__()
        # Reduce spatial dimensions by half and double the channels
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.GroupNorm(32, out_channels)
        )
        # Further reduce by half to 1/4th the original dimensions
        self.down2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.GroupNorm(32, out_channels)
        )
        # Maintain current size, apply same convolutional settings
        self.down3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.GroupNorm(32, out_channels)
        )
        # Reduce to 1/8th the original dimensions
        self.down4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.GroupNorm(32, out_channels)
        )
        # Reduce to 1/16th the original dimensions
        self.down5 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.GroupNorm(32, out_channels)
        )

    def forward(self, x):
        outputs = {}
        x = self.down1(x)
        # [B, C, H, W]
        outputs['res2'] = x  # [B, C, H, W]
        x = self.down2(x)
        outputs['res3'] = x  # [B, C, H/2, W/2]
        x = self.down3(x)
        outputs['res4'] = x  # [B, C, H/4, W/4] maintained
        x = self.down4(x)
        outputs['res5'] = x  # [B, C, H/8, W/8]
        x = self.down5(x)
        # outputs['res5'] = x  # [B, C, H/32, W/32]
        return outputs
