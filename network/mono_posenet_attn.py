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



class MonoPoseNet_attn(nn.Module):
    """
    Pytorch module for a pose network from the paper
    "Digging into Self-Supervised Monocular Depth Prediction"
    """

    def __init__(self, cfg):
        super(MonoPoseNet_attn, self).__init__()
        num_layers = cfg['model']['num_layers']
        pretrained = cfg['model']['weights_init']

        self.pose_encoder = ResnetEncoder(num_layers, pretrained, num_input_images=2)
        del self.pose_encoder.encoder.fc  # For ddp training
        self.pose_decoder = PoseDecoder(self.pose_encoder.num_ch_enc,
                                        num_input_features=1,
                                        num_frames_to_predict_for=1)
        self.attn = nn.MultiheadAttention(embed_dim=512, num_heads=8,batch_first=True)


    def forward(self, inputs, frame_ids, cam):
        assert cam != 'fusion'
        assert  cam=='front_attn' or cam=='joint_attn'
        if inputs.get('flip_version') is not None and inputs['flip_version'] >= 3:
            ext = '_flip'
        else:
            ext = ""

        B, N, C, H, W = inputs['color_aug' + ext, 0, 0].shape
        pose_inputs = [inputs['color_aug' + ext, f_i, 0].reshape(N * B, C, H, W) for f_i in frame_ids]
        input_images = torch.cat(pose_inputs, 1)
        all_features = [self.pose_encoder(input_images)]
        BN, C, H, W = all_features[0][-1].shape
        last_features = [f[-1].reshape(BN,C,H*W).permute((0,2,1)) for f in all_features]
        last_features = [self.attn(f,f,f)[0] for f in last_features]
        last_features = [f.permute((0,2,1)).reshape(B,N,C,H,W) for f in last_features]
        pose_feature = [[f.mean(1) for f in last_features]]
        axis_angle, translation = self.pose_decoder(pose_feature)
        return axis_angle, torch.clamp(translation, -4.0, 4.0)  # for DDAD da