from ..backbone.conv_trans_utils import *
from ..backbone.unet_utils import *
from detectron2.modeling import ShapeSpec
from omegaconf import DictConfig


class MaskEncoder(nn.Module):

    def __init__(self, model_cfg: DictConfig):
        super().__init__()
        in_chan = model_cfg.model.mask_encoder.in_channels
        base_chan = model_cfg.model.mask_encoder.base_channels
        self._out_feature_strides = model_cfg.model.mask_encoder.out_feature_strides
        self._out_feature_channels = model_cfg.model.mask_encoder.out_feature_channels
        self.inc = [BasicBlock(in_chan, base_chan)]
        self.inc.append(BasicBlock(base_chan, base_chan))
        self.inc = nn.Sequential(*self.inc)
        self.down1 = down_block(base_chan, base_chan, (2, 2), num_block=1)
        self.down2 = down_block(base_chan, 2 * base_chan, (2, 2), num_block=1)
        self.down3 = down_block(2 * base_chan, 4 * base_chan, (2, 2), num_block=1)
        self.down4 = down_block(4 * base_chan, 8 * base_chan, (2, 2), num_block=1)


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

