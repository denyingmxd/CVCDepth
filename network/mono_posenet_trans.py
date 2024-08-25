# Copyright (c) 2023 42dot. All rights reserved.
import torch
import torch.nn as nn

from external.layers import ResnetEncoder,PoseDecoder

# Copyright 2020 Toyota Research Institute.  All rights reserved.

# Adapted from monodepth2
# https://github.com/nianticlabs/monodepth2/blob/master/networks/pose_decoder.py


import torch
import torch.nn as nn
from collections import OrderedDict



class MonoPoseNet_trans(nn.Module):
    """
    Pytorch module for a pose network from the paper
    "Digging into Self-Supervised Monocular Depth Prediction"
    """

    def __init__(self, cfg):
        super(MonoPoseNet_trans, self).__init__()
        num_layers = cfg['model']['num_layers']
        pretrained = cfg['model']['weights_init']

        self.pose_encoder = ResnetEncoder(num_layers, pretrained, num_input_images=2)
        del self.pose_encoder.encoder.fc  # For ddp training
        self.pose_decoder = PoseDecoder(self.pose_encoder.num_ch_enc,
                                        num_input_features=1,
                                        num_frames_to_predict_for=1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)


    def forward(self, inputs, frame_ids, cam):
        assert cam != 'fusion'
        assert  cam=='front_trans' or cam=='joint_trans'
        if inputs.get('flip_version') is not None and inputs['flip_version'] >= 3:
            ext = '_flip'
        else:
            ext = ""

        B, N, C, H, W = inputs['color_aug' + ext, 0, 0].shape
        pose_inputs = [inputs['color_aug' + ext, f_i, 0].reshape(N * B, C, H, W) for f_i in frame_ids]
        input_images = torch.cat(pose_inputs, 1)
        all_features = [self.pose_encoder(input_images)]
        BN, C, H, W = all_features[0][-1].shape
        last_features = [f[-1].reshape(B,N,C,H*W).permute((0,1,3,2)).reshape(B,N*H*W,C) for f in all_features]
        last_features = [self.transformer_encoder(f) for f in last_features]
        last_features = [f.reshape(B,N,H*W,C).permute((0,1,3,2)).reshape(B,N,C,H,W) for f in last_features]
        pose_feature = [[f.mean(1) for f in last_features]]
        axis_angle, translation = self.pose_decoder(pose_feature)
        return axis_angle, torch.clamp(translation, -4.0, 4.0)  # for DDAD da