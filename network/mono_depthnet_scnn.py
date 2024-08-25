# Copyright (c) 2023 42dot. All rights reserved.
import torch.nn as nn
import torch
from external.layers import ResnetEncoder, DepthDecoder
import torch.nn.functional as F


class MonoDepthNet_scnn(nn.Module):
    """
    Pytorch module for a depth network from the paper
    "Digging into Self-Supervised Monocular Depth Prediction"
    """

    def __init__(self, cfg):
        super(MonoDepthNet_scnn, self).__init__()
        num_layers = cfg['model']['num_layers']
        pretrained = cfg['model']['weights_init']
        scales = cfg['training']['scales']

        self.depth_encoder = ResnetEncoder(num_layers, pretrained, 1)
        del self.depth_encoder.encoder.fc  # For ddp training
        self.depth_decoder = DepthDecoder(self.depth_encoder.num_ch_enc, scales)
        self.scales = scales
        ms_ks=9
        self.message_passing = nn.ModuleList()
        self.message_passing.add_module('left_right',
                                        nn.Conv2d(1, 1, (ms_ks, 1), padding=(ms_ks // 2, 0), bias=False))
        self.message_passing.add_module('right_left',
                                        nn.Conv2d(1, 1, (ms_ks, 1), padding=(ms_ks // 2, 0), bias=False))

    def message_passing_forward(self, x):
        Vertical = [ False, False]
        Reverse = [False, True]
        for ms_conv, v, r in zip(self.message_passing, Vertical, Reverse):
            x = self.message_passing_once(x, ms_conv, v, r)
        return x

    def message_passing_once(self, x, conv, vertical=True, reverse=False):
        """
        Argument:
        ----------
        x: input tensor
        vertical: vertical message passing or horizontal
        reverse: False for up-down or left-right, True for down-up or right-left
        """
        nB, C, H, W = x.shape
        if vertical:
            slices = [x[:, :, i:(i + 1), :] for i in range(H)]
            dim = 2
        else:
            slices = [x[:, :, :, i:(i + 1)] for i in range(W)]
            dim = 3
        if reverse:
            slices = slices[::-1]

        out = [slices[0]]
        for i in range(1, len(slices)):
            out.append(slices[i] + F.relu(conv(out[i - 1])))
        if reverse:
            out = out[::-1]
        return torch.cat(out, dim=dim)

    def forward(self, input_images):
        depth_feature = self.depth_encoder(input_images)
        outputs = self.depth_decoder(depth_feature)
        for scale in self.scales:
            disps = outputs[('disp',scale)]
            disps_scnn = self.message_passing_forward(disps)
            outputs[('disp', scale)] = disps_scnn

        return outputs