"""
FIND-Net: Fourier-Integrated Network with Dictionary Kernels for Metal Artifact Reduction

This implementation of FIND-Net extends the DICDNet framework for metal artifact reduction in CT images.
The architecture and certain components of this model are built upon the original DICDNet work, which is cited below.

Reference:
Wang, H., Li, Y., He, N., Ma, K., Meng, D., Zheng, Y. 
"DICDNet: Deep Interpretable Convolutional Dictionary Network for Metal Artifact Reduction in CT Images."
IEEE Trans. Med. Imaging, 41(4), 869–880, 2022.
DOI: 10.1109/TMI.2021.3127074

Modifications in FIND-Net include the integration of Fourier domain processing and trainable Gaussian filtering.

This file also contains code adapted from:
Fast Fourier Convolution (FFC) for Image Classification (https://github.com/pkumivision/FFC).
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at:
http://www.apache.org/licenses/LICENSE-2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

Gaussian_filter = False

def create_gaussian_filter(freq_center=0, freq_bandwidth=0.2, dimensions=(256, 129), 
                           adjust_for_fft=True, add_extra_dim=True, binarize=False):
    """
    Generates a Gaussian bandpass filter in the frequency domain.
    
    Args:
        freq_center (float or list/tuple/tensor): The central frequency of the filter (normalized 0 to 1).
        freq_bandwidth (float or list/tuple/tensor): The bandwidth of the filter.
        dimensions (tuple): Shape of the filter (height, width).
        adjust_for_fft (bool): Whether to shift for FFT compatibility.
        add_extra_dim (bool): Whether to add an extra dimension to output.
        binarize (bool): If True, threshold the filter to binary values.
    
    Returns:
        torch.Tensor: A Gaussian bandpass filter.
    """
    
    def format_input(val):
        if not torch.is_tensor(val):
            val = torch.tensor([val] if not isinstance(val, (tuple, list)) else val)
        if val.ndim <= 3:
            val = val.view(val.numel(), 1, 1)
        return val
    
    freq_center = format_input(freq_center).clamp(0, 1)
    freq_bandwidth = format_input(freq_bandwidth).clamp(min=1e-12, max=2)
    
    if freq_center.shape != freq_bandwidth.shape:
        raise ValueError(f'Mismatch in shape: {freq_center.shape} vs {freq_bandwidth.shape}')
    
    grid_x, grid_y = torch.meshgrid(torch.arange(dimensions[0]), torch.arange(dimensions[1]), indexing='ij')
    
    grid_x = torch.repeat_interleave(grid_x.unsqueeze(0), freq_center.shape[0], dim=0).to(freq_center.device)
    grid_y = torch.repeat_interleave(grid_y.unsqueeze(0), freq_center.shape[0], dim=0).to(freq_center.device)
    
    center_x = (dimensions[0] - 1) // 2
    center_y = 0
    
    dist_sq = ((grid_x - center_x) ** 2 + (grid_y - center_y) ** 2).float()
    dist_sq /= dist_sq.max()
    
    filter_response = torch.exp(-((dist_sq - freq_center ** 2) / (dist_sq.sqrt() * freq_bandwidth + 1e-12)) ** 2)
    
    if adjust_for_fft:
        filter_response = torch.roll(filter_response, filter_response.shape[-2] // 2 + 1, -2)
    
    if add_extra_dim:
        filter_response = filter_response.unsqueeze(0)
    
    if binarize:
        threshold_value = filter_response.mean()
        filter_response[filter_response < threshold_value] = 0.0
        filter_response[filter_response >= threshold_value] = 1.0
    
    return filter_response

if Gaussian_filter:
    print('...Using FIND-Net...')
    class FourierUnit(nn.Module):
        def __init__(
            self, in_channels, out_channels, groups=1):
            # bn_layer not used
            super(). __init__()
            self.groups = groups
            transformed_channels = in_channels * 2
            self.conv = nn.Conv2d(
                in_channels=transformed_channels,
                out_channels=out_channels * 2,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                bias=False
            )
            self.norm = nn.BatchNorm2d(out_channels * 2)
            self.relu = nn.ReLU(inplace=True)
      
            self.mask = None
            self.center = nn.Parameter(torch.tensor([0.1 for _ in range(transformed_channels)], dtype=torch.float32), requires_grad=True)
            self.width = nn.Parameter(torch.tensor([1. for _ in range(transformed_channels)], dtype=torch.float32), requires_grad=True)

        def forward(self, x):
            batch = x.shape[0]
            # FFC convolution
            fft_dim = (-2, -1)  # (batch, c, h, w/2+1, 2)
            freq_data = torch.fft.rfftn(x, dim=fft_dim, norm='ortho')
            freq_data = torch.stack((freq_data.real, freq_data.imag), dim=-1)  # (batch, c, h, w/2+1, 2)
            freq_data = freq_data.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
            freq_data = freq_data.view((batch, -1,) + freq_data.size()[3:])  # (batch, 2c, h, w/2+1)

            if self.mask is not None or self.center is not None or self.width is not None:
                mask = create_gaussian_filter(self.center, self.width, dimensions=freq_data.shape[2:])
            freq_data = freq_data * mask
            freq_data = self.conv(freq_data) 
            freq_data = self.relu(self.norm(freq_data))
            freq_data = freq_data.view((batch, -1, 2,) + freq_data.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
            freq_data = torch.complex(freq_data[..., 0], freq_data[..., 1])

            ifft_shape_slice = x.shape[2:]
            output_tensor = torch.fft.irfftn(freq_data, s=ifft_shape_slice, dim=fft_dim, norm='ortho')
            return output_tensor
        

else:
    print('...Using FIND-Net without Gaussian Filtering...')
    class FourierUnit(nn.Module):

        def __init__(self, in_channels, out_channels, groups=1):
            # bn_layer not used
            super(FourierUnit, self).__init__()
            self.groups = groups
            self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                            kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
            self.bn = torch.nn.BatchNorm2d(out_channels * 2)
            self.relu = torch.nn.ReLU(inplace=True)


        def forward(self, x):
            batch, c, h, w = x.size()
            freq_data = torch.fft.rfft2(x, dim=2, norm="ortho")
            freq_data = torch.view_as_real(freq_data)
            freq_data = freq_data.permute(0, 1, 4, 2, 3).contiguous()
            freq_data = freq_data.view((batch, -1,) + freq_data.size()[3:])
            freq_data = self.conv_layer(freq_data)  # (batch, c*2, h, w/2+1)
            freq_data = self.relu(self.bn(freq_data))
            freq_data = freq_data.view((batch, -1, 2,) + freq_data.size()[2:]).permute(
                0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
            freq_data = torch.view_as_complex(freq_data)
            output_tensor = torch.fft.irfft2(freq_data, s=x.shape[2:], dim=(2, 3), norm="ortho")
            return output_tensor


class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(
            out_channels // 2, out_channels // 2, groups)
        if self.enable_lfu:
            self.lfu = FourierUnit(
                out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)
        

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s_h = h // split_no
            split_s_w = w // split_no

            # Calculate channels to use
            channels_to_use = max(1, c // (split_no ** 2))

            xs = torch.cat(torch.split(
                x[:, :channels_to_use], split_s_h, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s_w, dim=-1),
                        dim=1).contiguous()

            # Adjust xs to have c channels
            if xs.shape[1] < c:
                padding = c - xs.shape[1]
                xs = torch.cat([xs, xs.new_zeros((n, padding, xs.shape[2], xs.shape[3]))], dim=1)
            elif xs.shape[1] > c:
                xs = xs[:, :c, :, :]

            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)
        return output

class FFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        # Adjust in_cg to be the nearest lower multiple of 4
        in_cg = in_cg - (in_cg % 4) if in_cg % 4 != 0 else in_cg
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cg = out_cg - (out_cg % 4) if out_cg % 4 != 0 else out_cg
        out_cl = out_channels - out_cg

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0
        if x_g.shape[1] == 0:
            out_xl = self.convl2l(x_l)
        elif self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        if self.ratio_gout != 0:
            convl2g = self.convl2g(x_l)
            convg2g = self.convg2g(x_g)
            try:
                out_xg = convl2g + convg2g
            except:
                out_xg = convg2g

        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 enable_lfu=True):
        super(FFC_BN_ACT, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        in_cg = int(in_channels * ratio_gin)
        # Adjust in_cg to be the nearest lower multiple of 4
        in_cg = in_cg - (in_cg % 4) if in_cg % 4 != 0 else in_cg
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cg = out_cg - (out_cg % 4) if out_cg % 4 != 0 else out_cg
        out_cl = out_channels - out_cg

        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        self.bn_l = lnorm(int(out_cl))
        self.bn_g = gnorm(int(out_cg))

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        in_cg = int(self.in_channels * self.ratio_gin)
        # Adjust in_cg to be the nearest lower multiple of 4
        in_cg = in_cg - (in_cg % 4) if in_cg % 4 != 0 else in_cg
        in_cl = self.in_channels - in_cg
        out_cg = int(self.out_channels * self.ratio_gout)
        out_cg = out_cg - (out_cg % 4) if out_cg % 4 != 0 else out_cg
        out_cl = self.out_channels - out_cg

        x_l = x[:,:in_cl,:,:]
        x_g = x[:,in_cl:,:,:]

        x_l, x_g = self.ffc((x_l, x_g))

        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))

        if isinstance(x_l, int):
            if x.shape[1]%2 == 0:
                output = x_g
            else:
                output = torch.cat([x_g, x_g[:,:1,:,:]], dim=1)
        elif isinstance(x_g, int):
            if x.shape[1]%2 == 0:
                output = x_l
            else:
                output = torch.cat([x_l, x_l[:,:1,:,:]], dim=1)

        else:
            output = torch.cat([x_l, x_g], dim=1)

        return output