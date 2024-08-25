# Copyright (c) 2023 42dot. All rights reserved.
import torch.nn as nn

from external.layers import ResnetEncoder, DepthDecoder
from linear_attention_transformer.images import ImageLinearAttention


class MonoDepthNet_attn(nn.Module):
    """
    Pytorch module for a depth network from the paper
    "Digging into Self-Supervised Monocular Depth Prediction"
    """

    def __init__(self, cfg):
        super(MonoDepthNet_attn, self).__init__()
        num_layers = cfg['model']['num_layers']
        pretrained = cfg['model']['weights_init']
        scales = cfg['training']['scales']
        key_dim = cfg['model']['depth_attn_key_dim']
        self.depth_encoder = ResnetEncoder(num_layers, pretrained, 1)
        del self.depth_encoder.encoder.fc  # For ddp training
        self.depth_decoder = DepthDecoder(self.depth_encoder.num_ch_enc, scales)

        self.attn = ImageLinearAttention(
                      chan = 1,
                      heads = 1,
                      key_dim = key_dim       # can be decreased to 32 for more memory savings
                    )


    def forward(self, input_images):
        depth_feature = self.depth_encoder(input_images)
        outputs = self.depth_decoder(depth_feature)
        for scale in self.depth_decoder.scales:
            xxx = outputs[('disp',scale)]
            outputs[('disp',scale)] = self.attn(xxx)+xxx

        return outputs