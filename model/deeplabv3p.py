"""
Code Adapted from:
https://github.com/sthalles/deeplab_v3
Copyright 2020 Nvidia Corporation
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
import torch
from torch import nn

#from model.mynn import initialize_weights, Norm2d, Upsample
from model.utils import get_aspp, get_trunk

def initialize_weights(*models):
    """
    Initialize Model Weights
    """
    for model in models:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

class DeepV3Plus(nn.Module):
    """
    DeepLabV3+ with various trunks supported
    Always stride8
    """
    def __init__(self, num_classes, trunk='wrn38',
                 use_dpc=False, init_all=False):
        super(DeepV3Plus, self).__init__()
        self.backbone, s2_ch, _s4_ch, high_level_ch = get_trunk(trunk)
        self.aspp, aspp_out_ch = get_aspp(high_level_ch,
                                          bottleneck_ch=256,
                                          output_stride=8,
                                          dpc=use_dpc)
        self.bot_fine = nn.Conv2d(s2_ch, 48, kernel_size=1, bias=False)
        self.bot_aspp = nn.Conv2d(aspp_out_ch, 256, kernel_size=1, bias=False)
        self.final = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False))

        if init_all:
            initialize_weights(self.aspp)
            initialize_weights(self.bot_aspp)
            initialize_weights(self.bot_fine)
            initialize_weights(self.final)
        else:
            initialize_weights(self.final)

    def forward(self, x):
        x_size = x.size()
        s2_features, _, final_features = self.backbone(x)
        
        aspp = self.aspp(final_features)
        conv_aspp = self.bot_aspp(aspp)
        conv_s2 = self.bot_fine(s2_features)
        conv_aspp = Upsample(conv_aspp, s2_features.size()[2:])
        #print(conv_s2.size(), conv_aspp.size())
        cat_s4 = [conv_s2, conv_aspp]
        cat_s4 = torch.cat(cat_s4, 1)
        final = self.final(cat_s4)
        #print(final.size())
        #out = Upsample(final, x_size[2:])
        return final

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for 
        the last classification layer. Note that for each batchnorm layer, 
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return 
        any batchnorm parameter
        """
        b = []

        b.append(self.backbone)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj+=1
                    if k.requires_grad:
                        yield k        

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters of the net except for 
        the last classification layer. Note that for each batchnorm layer, 
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return 
        any batchnorm parameter
        """
        b = []

        b.append(self.aspp)
        b.append(self.bot_aspp)
        b.append(self.bot_fine)
        b.append(self.final)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj+=1
                    if k.requires_grad:
                        yield k        

    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10*args.learning_rate}]


def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear',
                                     align_corners=False)

def DeepV3PlusW38(num_classes):
    return DeepV3Plus(num_classes, trunk='wrn38')


def DeepV3PlusW38I(num_classes):
    return DeepV3Plus(num_classes, trunk='wrn38', init_all=True)

