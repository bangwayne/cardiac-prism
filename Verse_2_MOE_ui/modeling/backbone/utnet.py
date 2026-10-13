import torch
import torch.nn as nn
from .unet_utils import up_block, down_block
from .conv_trans_utils import *
from detectron2.modeling import ShapeSpec
from omegaconf import DictConfig


class UTNet(nn.Module):

    def __init__(self, model_cfg: DictConfig):
        super().__init__()

        in_chan = model_cfg.model.backbone.in_channels
        base_chan = model_cfg.model.backbone.base_channels
        reduce_size = model_cfg.model.backbone.reduce_size
        block_list = model_cfg.model.backbone.block_list
        num_blocks = model_cfg.model.backbone.num_blocks
        projection = model_cfg.model.backbone.projection
        num_heads = model_cfg.model.backbone.num_heads
        attn_drop = model_cfg.model.backbone.attn_drop
        proj_drop = model_cfg.model.backbone.proj_drop
        bottleneck = model_cfg.model.backbone.bottleneck
        maxpool = model_cfg.model.backbone.maxpool
        rel_pos = model_cfg.model.backbone.rel_pos
        self._out_feature_strides = model_cfg.model.backbone.out_feature_strides
        self._out_feature_channels = model_cfg.model.backbone.out_feature_channels

        self.inc = [BasicBlock(in_chan, base_chan)]
        if '0' in block_list:
            for _ in range(1):
                self.inc.append(BasicTransBlock(base_chan, heads=num_heads[-5], dim_head=base_chan // num_heads[-5],
                                                attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size,
                                                projection=projection, rel_pos=rel_pos))
        else:
            self.inc.append(BasicBlock(base_chan, base_chan))
        self.inc = nn.Sequential(*self.inc)

        if '1' in block_list:
            self.down1 = down_block_trans(base_chan, base_chan, num_block=num_blocks[-4], bottleneck=bottleneck,
                                          maxpool=maxpool, heads=num_heads[-4], dim_head=base_chan // num_heads[-4],
                                          attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size,
                                          projection=projection, rel_pos=rel_pos)
        else:
            self.down1 = down_block(base_chan, base_chan, (2, 2), num_block=2)

        if '2' in block_list:
            self.down2 = down_block_trans(base_chan, 2 * base_chan, num_block=num_blocks[-3], bottleneck=bottleneck,
                                          maxpool=maxpool, heads=num_heads[-3], dim_head=2 * base_chan // num_heads[-3],
                                          attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size,
                                          projection=projection, rel_pos=rel_pos)

        else:
            self.down2 = down_block(base_chan, 2 * base_chan, (2, 2), num_block=2)

        if '3' in block_list:
            self.down3 = down_block_trans(2 * base_chan, 4 * base_chan, num_block=num_blocks[-2], bottleneck=bottleneck,
                                          maxpool=maxpool, heads=num_heads[-2], dim_head=4 * base_chan // num_heads[-2],
                                          attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size,
                                          projection=projection, rel_pos=rel_pos)
        else:
            self.down3 = down_block(2 * base_chan, 4 * base_chan, (2, 2), num_block=2)

        if '4' in block_list:
            self.down4 = down_block_trans(4 * base_chan, 8 * base_chan, num_block=num_blocks[-1],
                                          bottleneck=bottleneck, maxpool=maxpool, heads=num_heads[-1],
                                          dim_head=8 * base_chan // num_heads[-1], attn_drop=attn_drop,
                                          proj_drop=proj_drop, reduce_size=reduce_size, projection=projection,
                                          rel_pos=rel_pos)
        else:
            self.down4 = down_block(4 * base_chan, 8 * base_chan, (2, 2), num_block=2)

    def forward(self, x):
        h, w = x.size()[2:]
        output_dict = {}
        if h == 256:
            x1 = self.inc(x)
            # output_dict['res2'] = x1
            x2 = self.down1(x1)
            output_dict['res3'] = x2
            x3 = self.down2(x2)
            output_dict['res4'] = x3
            x4 = self.down3(x3)
            output_dict['res5'] = x4
        elif h == 512:
            x1 = self.inc(x)
            # output_dict['res2'] = x1
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            output_dict['res3'] = x3
            x4 = self.down3(x3)
            output_dict['res4'] = x4
            x5 = self.down4(x4)
            output_dict['res5'] = x5

        return output_dict

    def get_lowest_feat(self, x):
        h, w = x.size()[2:]
        if h == 256:
            x1 = self.inc(x)
            # output_dict['res2'] = x1
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            return x4
        elif h == 512:
            x1 = self.inc(x)
            # output_dict['res2'] = x1
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            return x5

        # return output_dict

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_feature_channels
        }
