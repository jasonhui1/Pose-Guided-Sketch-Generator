"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.
This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as M
import torch.nn.functional as F

class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance

class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0,
                 actv=nn.LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out

class Generator(nn.Module):
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512, w_hpf=0):
        super().__init__()
        dim_in = 2**14 // img_size
        self.img_size = img_size
        self.from_rgb = nn.Conv2d(3, dim_in, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, 1, 1, 1, 0))

        # down/up-sampling blocks
        repeat_num = int(np.log2(img_size)) - 4
        if w_hpf > 0:
            repeat_num += 1
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            self.encode.append(
                self.conv_block(dim_in, dim_out, 4, 2, 1))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_in, style_dim,
                               w_hpf=w_hpf, upsample=True))  # stack-like
            dim_in = dim_out

        # bottleneck blocks
        for _ in range(2):
            self.encode.append(
                self.conv_block(dim_out, dim_out, 3, 1, 1))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_out, style_dim, w_hpf=w_hpf))

    def conv_block(self, dim_in, dim_out, k, s, p):
        block = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, dim_out, k, s, p))
        return block

    def forward(self, x, s, masks=None):
        x = self.from_rgb(x)
        for block in self.encode:
            x = block(x)
        for block in self.decode:
            x = block(x, s)
        return self.to_rgb(x)


# class Discriminator(nn.Module):
#     def __init__(self, img_size=256, num_domains=1, max_conv_dim=512):
#         super().__init__()
#         dim_in = 2**14 // img_size
#         blocks = []
#         blocks += [nn.Conv2d(1, dim_in, 3, 1, 1)]

#         repeat_num = int(np.log2(img_size)) - 2
#         for _ in range(repeat_num):
#             dim_out = min(dim_in*2, max_conv_dim)
#             blocks += [ResBlk(dim_in, dim_out, downsample=True)]
#             dim_in = dim_out

#         blocks += [nn.LeakyReLU(0.2)]
#         blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
#         blocks += [nn.LeakyReLU(0.2)]
#         blocks += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)]
#         self.main = nn.Sequential(*blocks)

#     def forward(self, x):
#         out = self.main(x)
#         out = out.view(out.size(0), -1)  # (batch, num_domains)
#         # print(out.size())
#         return out


class ResNeXtBottleneck(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, stride=1, cardinality=32, dilate=1):
        super(ResNeXtBottleneck, self).__init__()
        D = out_channels // 2
        self.out_channels = out_channels
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_conv = nn.Conv2d(D, D, kernel_size=2 + stride, stride=stride, padding=dilate, dilation=dilate,
                                   groups=cardinality,
                                   bias=False)

        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut.add_module('shortcut',
                                     nn.AvgPool2d(2, stride=2))


    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.leaky_relu(bottleneck, 0.2, True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = self.conv_expand.forward(bottleneck)

        x = self.shortcut.forward(x)
        return x + bottleneck


class NetD(nn.Module):
    def __init__(self, img_size=256, ndf=64):
        super(NetD, self).__init__()

        self.feed = nn.Sequential(nn.Conv2d(1, ndf, kernel_size=7, stride=1, padding=3, bias=False),  # 256
                                  nn.LeakyReLU(0.2, True),
                                  nn.Conv2d(ndf, ndf, kernel_size=4, stride=2, padding=1, bias=False),  # 128
                                  nn.LeakyReLU(0.2, True),

                                  ResNeXtBottleneck(ndf, ndf, cardinality=8, dilate=1),
                                  ResNeXtBottleneck(ndf, ndf, cardinality=8, dilate=1, stride=2),  # 64
                                  nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.LeakyReLU(0.2, True),

                                  ResNeXtBottleneck(ndf * 2, ndf * 2, cardinality=8, dilate=1),
                                  ResNeXtBottleneck(ndf * 2, ndf * 2, cardinality=8, dilate=1, stride=2),  # 32
                                  nn.Conv2d(ndf * 2, ndf * 4, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.LeakyReLU(0.2, True),

                                  ResNeXtBottleneck(ndf * 4, ndf * 4, cardinality=8, dilate=1),
                                  ResNeXtBottleneck(ndf * 4, ndf * 4, cardinality=8, dilate=1, stride=2),  # 16
                                  nn.Conv2d(ndf * 4, ndf * 8, kernel_size=3, stride=1, padding=1, bias=False),  # 16
                                  nn.LeakyReLU(0.2, True),
                                  )

        blocks = []

        k = img_size//128

        k = int(np.log2(img_size)) - int(np.log2(128)) + 1

        for _ in range(k):
            blocks += [ResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1)]
            blocks += [ResNeXtBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1, stride=2)]  # 8

        blocks += [nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=1, padding=0, bias=False)] # 1
        blocks += [nn.LeakyReLU(0.2, True)]
        self.feed2 = nn.Sequential(*blocks)
        self.out = nn.Linear(512, 1)

    def forward(self, sketch):
        x = self.feed(sketch)
        x = self.feed2(x)

        out = self.out(x.view(sketch.size(0), -1))
        return out


class StyleEncoder(nn.Module):
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(4, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.exit = nn.Linear(dim_out, style_dim)

    def forward(self, sketch, pose):
        x = torch.cat([sketch,pose],1)
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        s = self.exit(h)
        return s


class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=16, style_dim=64, num_domains=1):
        super().__init__()
        layers = []
        layers += [nn.Linear(latent_dim, 512)]
        layers += [nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(512, 512)]
            layers += [nn.ReLU()]
        self.shared = nn.Sequential(*layers)

        self.exit = nn.Sequential(nn.Linear(512, 512),
                                        nn.ReLU(),
                                        nn.Linear(512, 512),
                                        nn.ReLU(),
                                        nn.Linear(512, 512),
                                        nn.ReLU(),
                                        nn.Linear(512, style_dim))

    def forward(self, z):
        h = self.shared(z)
        s = self.exit(h)
        return s

class NetF(nn.Module):
    def __init__(self):
        super(NetF, self).__init__()
        vgg16 = M.vgg16(pretrained=True)
        # vgg16.load_state_dict(torch.load(VGG16_PATH))
        vgg16.features = nn.Sequential(
            *list(vgg16.features.children())[:9]
        )
        self.model = vgg16.features
        self.register_buffer('mean', torch.FloatTensor([0.485 - 0.5, 0.456 - 0.5, 0.406 - 0.5]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, images):
        return self.model((images.mul(0.5) - self.mean) / self.std)


