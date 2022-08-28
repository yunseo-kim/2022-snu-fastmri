"""
The original source code is licensed under the MIT license:

Copyright (c) 2020 Jeya Maria Jose

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
from torch import nn
from torch.nn import functional as F


class KiUnet(nn.Module):
    """
    Official Pytorch Code of KiU-Net for Image/3D Segmentation
     - MICCAI 2020 (Oral), IEEE TMI
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.ki_en1 = UpConvBlock(self.in_chans, 4)
        self.u_en1 = DownConvBlock(self.in_chans, 4)
        self.crfb1 = CRFB(4, 1)

        self.ki_en2 = UpConvBlock(4, 8)
        self.u_en2 = DownConvBlock(4, 8)
        self.crfb2 = CRFB(8, 2)

        self.ki_en3 = UpConvBlock(8, 16)
        self.u_en3 = DownConvBlock(8, 16)
        self.crfb3 = CRFB(16, 3)

        self.ki_de1 = DownConvBlock(16, 8)
        self.u_de1 = UpConvBlock(16, 8)
        self.crfb4 = CRFB(8, 2)

        self.ki_de2 = DownConvBlock(8, 4)
        self.u_de2 = UpConvBlock(8, 4)
        self.crfb5 = CRFB(4, 1)

        self.ki_de3 = DownConvBlock(4, 4)
        self.u_de3 = UpConvBlock(4, 4)

        self.final = nn.Conv2d(4, out_chans, kernel_size=1, stride=1, padding=0)

    def norm(self, x):
        b, h, w = x.shape
        x = x.view(b, h * w)
        mean = x.mean(dim=1).view(b, 1, 1)
        std = x.std(dim=1).view(b, 1, 1)
        x = x.view(b, h, w)
        return (x - mean) / std, mean, std

    def unnorm(self, x, mean, std):
        return x * std + mean

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """

        input, mean, std = self.norm(image)
        input = input.unsqueeze(1)

        ki_u1 = self.ki_en1(input)
        u_d1 = self.u_en1(input)
        del input

        (ki_u2, u_d2) = self.crfb1(ki_u1, u_d1)
        ki_u2 = self.ki_en2(ki_u2)
        u_d2 = self.u_en2(u_d2)

        (ki_m0, u_m0) = self.crfb2(ki_u2, u_d2)
        ki_m0 = self.ki_en3(ki_m0)
        u_m0 = self.u_en3(u_m0)

        (ki_m0, u_m0) = self.crfb3(ki_m0, u_m0)

        ki_d1 = self.ki_de1(ki_m0)
        u_u1 = self.u_de1(u_m0)
        del ki_m0
        del u_m0
        (ki_d1, u_u1) = self.crfb4(ki_d1, u_u1)

        ki_d2 = torch.add(ki_d1, ki_u2)
        del ki_u2
        u_u2 = torch.add(u_u1, u_d2)
        del u_d2
        ki_d2 = self.ki_de2(ki_d2)
        u_u2 = self.u_de2(u_u2)
        (ki_d2, u_u2) = self.crfb5(ki_d2, u_u2)

        ki_d3 = torch.add(ki_d2, ki_u1)
        del ki_u1
        u_u3 = torch.add(u_u2, u_d1)
        del u_d1

        ki_d3 = self.ki_de3(ki_d3)
        u_u3 = self.u_de3(u_u3)

        output = torch.add(ki_d3, u_u3)
        output = F.relu(self.final(output))
        
        output = output.squeeze(1)
        output = self.unnorm(output, mean, std)

        return output


''' 주어진 대회 서버에서 실행 불가능(out of memory)
class DenseBlock(nn.Module):

    def __init__(self, in_planes):
        super(DenseBlock, self).__init__()
        # print(int(in_planes/4))
        self.c1 = nn.Conv2d(in_planes, in_planes, 1, stride=1, padding=0)
        self.c2 = nn.Conv2d(in_planes, int(in_planes / 4), 3, stride=1, padding=1)
        self.b1 = nn.BatchNorm2d(in_planes)
        self.b2 = nn.BatchNorm2d(int(in_planes / 4))
        self.c3 = nn.Conv2d(in_planes + int(in_planes / 4), in_planes, 1, stride=1, padding=0)
        self.c4 = nn.Conv2d(in_planes, int(in_planes / 4), 3, stride=1, padding=1)

        self.c5 = nn.Conv2d(in_planes + int(in_planes / 2), in_planes, 1, stride=1, padding=0)
        self.c6 = nn.Conv2d(in_planes, int(in_planes / 4), 3, stride=1, padding=1)

        self.c7 = nn.Conv2d(in_planes + 3 * int(in_planes / 4), in_planes, 1, stride=1, padding=0)
        self.c8 = nn.Conv2d(in_planes, int(in_planes / 4), 3, stride=1, padding=1)

    def forward(self, x):
        org = x
        # print(x.shape)
        x = F.relu(self.b1(self.c1(x)))
        # print(x.shape)
        x = F.relu(self.b2(self.c2(x)))
        d1 = x
        # print(x.shape)
        x = torch.cat((org, d1), 1)
        x = F.relu(self.b1(self.c3(x)))
        x = F.relu(self.b2(self.c4(x)))
        d2 = x
        x = torch.cat((org, d1, d2), 1)
        x = F.relu(self.b1(self.c5(x)))
        x = F.relu(self.b2(self.c6(x)))
        d3 = x
        x = torch.cat((org, d1, d2, d3), 1)
        x = F.relu(self.b1(self.c7(x)))
        x = F.relu(self.b2(self.c8(x)))
        d4 = x
        x = torch.cat((d1, d2, d3, d4), 1)
        x = torch.add(org, x)
        return x
        '''


class UpConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two parallel convolution layers
    each followed by Up-sampling, Batch normalization, and ReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.conv3 = nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=1, padding=1)
        # self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=1, stride=1, padding=0)
        self.upsample = nn.Upsample(scale_factor=(2,2), mode='bilinear')
        # self.dense = DenseBlock(in_planes=out_chans)
        self.batchnorm = nn.BatchNorm2d(out_chans)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H*2, W*2)`.
        """
        x = self.conv3(image)
        # res = self.conv1(image)
        # x = torch.add(x, res)
        # x = self.dense(x)
        x = self.upsample(x)
        x = self.batchnorm(x)
        x = self.relu(x)

        return x


class DownConvBlock(nn.Module):
    """
    A Convolutional Block that consists of one convolution layer followed by
    DenseBlock, Pooling, Batch normalization, and ReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.conv3 = nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=1, padding=1)
        # self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=1, stride=1, padding=0)
        # self.dense = nn.DenseBlock(in_planes=out_chans)
        self.pool = nn.MaxPool2d(2, 2)
        self.batchnorm = nn.BatchNorm2d(out_chans)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H*2, W*2)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        x = self.conv3(image)
        # res = self.conv1(image)
        # x = torch.add(x, res)
        x = self.pool(x)
        x = self.batchnorm(x)
        x = self.relu(x)

        return x


class CRFB(nn.Module):
    """
    Cross Residual Fusion Block
    """

    def __init__(self, chans: int, depth: int):
        """
        Args:
            chans: Number of channels.
            depth:
        """
        super().__init__()

        self.chans = chans
        self.depth = depth

        self.layers = nn.Sequential(
            nn.Conv2d(chans, chans, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(chans),
            nn.ReLU(inplace=True),
        )

    def forward(self, ki_in: torch.Tensor, u_in: torch.Tensor) -> tuple:
        """
        Args:
            ki_in: Input 4D tensor of shape `(N, chans, H*(2**depth), W*(2**depth))` from Ki-Net.
            u_in: Input 4D tensor of shape `(N, chans, H/(2**depth), W/(2**depth))` from U-Net.
        Returns:
            ki_out: Output tensor of shape `(N, chans, H*(2**depth), W*(2**depth))`.
            u_out: Output tensor of shape `(N, chans, H/(2**depth), W/(2**depth))`.
        """
        ki_out = torch.add(ki_in, F.interpolate(self.layers(u_in), scale_factor=4**self.depth, mode='bilinear'))
        u_out = torch.add(u_in, F.interpolate(self.layers(ki_in), scale_factor=0.25**self.depth, mode='bilinear'))

        return ki_out, u_out
