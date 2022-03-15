"""
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
from collections import OrderedDict

import torch
import torch.nn.functional as F

from torch import nn

#from model.mynn import Norm2d, Upsample
from model.wider_resnet import wrn38

def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear',
                                     align_corners=False)

def get_trunk(trunk_name, output_stride=8):
    """
    Retrieve the network trunk and channel counts.
    """
    assert output_stride == 8, 'Only stride8 supported right now'

    if trunk_name == 'wrn38':
        #
        # FIXME: pass in output_stride once we support stride 16
        #
        backbone = wrn38(pretrained=True)
        s2_ch = 128
        s4_ch = 256
        high_level_ch = 4096
    else:
        raise 'unknown backbone {}'.format(trunk_name)

    return backbone, s2_ch, s4_ch, high_level_ch

class AtrousSpatialPyramidPoolingModule(nn.Module):
    """
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    """

    def __init__(self, in_dim, reduction_dim=256, output_stride=16,
                 rates=(6, 12, 18)):
        super(AtrousSpatialPyramidPoolingModule, self).__init__()

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1,
                                    bias=False),
                          nn.BatchNorm2d(reduction_dim), nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduction_dim), nn.ReLU(inplace=True))

    def forward(self, x):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = Upsample(img_features, x_size[2:])
        out = img_features

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out

def dpc_conv(in_dim, reduction_dim, dil, separable):
    if separable:
        groups = reduction_dim
    else:
        groups = 1

    return nn.Sequential(
        nn.Conv2d(in_dim, reduction_dim, kernel_size=3, dilation=dil,
                  padding=dil, bias=False, groups=groups),
        nn.BatchNorm2d(reduction_dim),
        nn.ReLU(inplace=True)
    )


class DPC(nn.Module):
    '''
    From: Searching for Efficient Multi-scale architectures for dense
    prediction
    '''
    def __init__(self, in_dim, reduction_dim=256, output_stride=16,
                 rates=[(1, 6), (18, 15), (6, 21), (1, 1), (6, 3)],
                 dropout=False, separable=False):
        super(DPC, self).__init__()

        self.dropout = dropout
        if output_stride == 8:
            rates = [(2 * r[0], 2 * r[1]) for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.a = dpc_conv(in_dim, reduction_dim, rates[0], separable)
        self.b = dpc_conv(reduction_dim, reduction_dim, rates[1], separable)
        self.c = dpc_conv(reduction_dim, reduction_dim, rates[2], separable)
        self.d = dpc_conv(reduction_dim, reduction_dim, rates[3], separable)
        self.e = dpc_conv(reduction_dim, reduction_dim, rates[4], separable)

        self.drop = nn.Dropout(p=0.1)

    def forward(self, x):
        a = self.a(x)
        b = self.b(a)
        c = self.c(a)
        d = self.d(a)
        e = self.e(b)
        out = torch.cat((a, b, c, d, e), 1)
        if self.dropout:
            out = self.drop(out)
        return out


def get_aspp(high_level_ch, bottleneck_ch, output_stride, dpc=False):
    """
    Create aspp block
    """
    if dpc:
        aspp = DPC(high_level_ch, bottleneck_ch, output_stride=output_stride)
    else:
        aspp = AtrousSpatialPyramidPoolingModule(high_level_ch, bottleneck_ch,
                                                 output_stride=output_stride)
    aspp_out_ch = 5 * bottleneck_ch
    return aspp, aspp_out_ch

