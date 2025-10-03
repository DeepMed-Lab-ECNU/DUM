from typing import Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm
from model.SwinUNETR import SwinUNETR
from model.Unet import UNet3D, OutputTransition
from model.DiNTS import TopologyInstance, DiNTS
from model.Unetpp import BasicUNetPlusPlus


class Single_Model(nn.Module):
    def __init__(self, img_size, in_channels, out_channels, backbone='swinunetr'):
        # encoding: rand_embedding or word_embedding
        super().__init__()
        self.backbone_name = backbone
        if backbone == 'swinunetr':
            self.backbone = SwinUNETR(img_size=img_size,
                                      in_channels=in_channels,
                                      out_channels=out_channels,
                                      feature_size=48,
                                      drop_rate=0.0,
                                      attn_drop_rate=0.0,
                                      dropout_path_rate=0.0,
                                      use_checkpoint=False,
                                      )
            self.out_tr = OutputTransition(48, out_channels, use_sigmoid=False)
        elif backbone == 'unet':
            self.backbone = UNet3D()
            self.out_tr = OutputTransition(64, out_channels, use_sigmoid=False)
        elif backbone == 'dints':
            ckpt = torch.load('./model/arch_code_cvpr.pth')
            node_a = ckpt["node_a"]
            arch_code_a = ckpt["arch_code_a"]
            arch_code_c = ckpt["arch_code_c"]

            dints_space = TopologyInstance(
                channel_mul=1.0,
                num_blocks=12,
                num_depths=4,
                use_downsample=True,
                arch_code=[arch_code_a, arch_code_c]
            )

            self.backbone = DiNTS(
                dints_space=dints_space,
                in_channels=1,
                num_classes=3,
                use_downsample=True,
                node_a=node_a,
            )
            self.out_tr = OutputTransition(32, out_channels, use_sigmoid=False)
        elif backbone == 'unetpp':
            self.backbone = BasicUNetPlusPlus(spatial_dims=3, features=(32, 32, 64, 128, 256, 32))
            self.out_tr = OutputTransition(32, out_channels, use_sigmoid=False)
        else:
            raise Exception('{} backbone is not implemented in curretn version'.format(backbone))


        self.class_num = out_channels

    def load_params(self, model_dict):
        if self.backbone_name == 'swinunetr':
            store_dict = self.backbone.state_dict()
            for key in model_dict.keys():
                if 'out' not in key:
                    store_dict[key] = model_dict[key]

            self.backbone.load_state_dict(store_dict)
            print('Use pretrained weights')
        elif self.backbone_name == 'unet':
            store_dict = self.backbone.state_dict()
            for key in model_dict.keys():
                if 'out_tr' not in key:
                    store_dict[key.replace("module.", "")] = model_dict[key]
            self.backbone.load_state_dict(store_dict)
            print('Use pretrained weights')

    def forward(self, x_in, **kwargs):
        dec4, out = self.backbone(x_in)
        out = self.out_tr(out)
        return out


if __name__ == "__main__":
    model = Single_Model(img_size=(32, 32, 32), in_channels=1, out_channels=32, backbone='unetpp')
    input_ = torch.rand((4, 1, 32, 32, 32))
    out = model(input_)
    print(out.shape, out.max(), out.min(), out.mean())
