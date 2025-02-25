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
"""
import torch
import torch.nn as nn
import torch.nn.functional as  F
from Model.ffc import FFC_BN_ACT

# Accessing a configuration item
ratio_gin = [0.0, 0.2, 0.2, 0.3, 0.4, 0.6, 0.7, 0.7, 0.8, 0.8, 0.8]
ratio_gout = [0.0, 0.2, 0.2, 0.3, 0.4, 0.6, 0.7, 0.7, 0.8, 0.8, 0.8]

FINDNet_Mnet = True
FINDNet_Xnet = True
DICDNet_Mnet = not FINDNet_Mnet
DICDNet_Xnet = not FINDNet_Xnet

if DICDNet_Mnet:
    print("--------> Using DICDNet ProxNet Mnet")
    # Original ProxNet code
    # proxNet_M
    class Mnet(nn.Module):
        def __init__(self, args, ratio_index):
            super(Mnet, self).__init__()
            self.channels = args.num_M
            self.T = args.T  # the number of resblocks in each proxNet
            self.layer = self.make_resblock(self.T)
            self.tau0 = torch.Tensor([0.5])
            self.tau_const = self.tau0.unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0).expand(-1, self.channels, -1, -1)
            self.tau = nn.Parameter(self.tau_const, requires_grad=True)  # for sparsity

        def make_resblock(self, T):
            layers = []
            for i in range(T):
                layers.append(
                    nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                                nn.BatchNorm2d(self.channels),
                                nn.ReLU(),
                                nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                                nn.BatchNorm2d(self.channels),
                                ))
            return nn.Sequential(*layers)
        def forward(self, input):
            M = input
            counter = 0
            for i in range(self.T):
                M = F.relu(M + self.layer[i](M))
                counter += 1
            M = F.relu(M - self.tau)
            return M
        
elif FINDNet_Mnet:
    print("--------> Using Find-Net ProxNet Mnet")
    # proxNet_M
    class Mnet(nn.Module):
        def __init__(self, args, ratio_index):
            super(Mnet, self).__init__()
            self.channels = args.num_M
            self.T = args.T  # the number of resblocks in each proxNet
            self.layer = self.make_resblock(self.T, ratio_index=ratio_index)
            self.tau0 = torch.Tensor([0.5])
            self.tau_const = self.tau0.unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0).expand(-1, self.channels, -1, -1)
            self.tau = nn.Parameter(self.tau_const, requires_grad=True)  # for sparsity

        def make_resblock(self, T, ratio_index):
            layers = []
            for i in range(T):
                layers.append(
                    nn.Sequential(FFC_BN_ACT(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, ratio_gin=ratio_gin[ratio_index], ratio_gout=ratio_gout[ratio_index]),
                                nn.ReLU(),
                                FFC_BN_ACT(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, ratio_gin=ratio_gin[ratio_index], ratio_gout=ratio_gout[ratio_index]),
                                ))
            return nn.Sequential(*layers)

        def forward(self, input):
            M = input
            counter = 0
            for i in range(self.T):
                M = F.relu(M + self.layer[i](M))
                counter += 1
            M = F.relu(M - self.tau)
            return M

        

if DICDNet_Xnet:
    print("--------> Using DICDNet ProxNet Xnet")
    # proxNet_X
    class Xnet(nn.Module):
        def __init__(self, args, ratio_index):
            super(Xnet, self).__init__()
            self.channels = args.num_Q + 1
            self.T = args.T
            self.layer = self.make_resblock(self.T)

        def make_resblock(self, T):
            layers = []
            for i in range(T):
                layers.append(nn.Sequential(
                    nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                    nn.BatchNorm2d(self.channels),
                    nn.ReLU(),
                    nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                    nn.BatchNorm2d(self.channels),
                ))
            return nn.Sequential(*layers)

        def forward(self, input):
            X = input
            counter = 0
            for i in range(self.T):
                X = F.relu(X + self.layer[i](X))
                counter += 1
            return X


elif FINDNet_Xnet:
    print("--------> Using Find-Net ProxNet Xnet")
    # Original ProxNet code
    # proxNet_X
    class Xnet(nn.Module):
        def __init__(self, args, ratio_index):
            super(Xnet, self).__init__()
            self.channels = args.num_Q + 1
            self.T = args.T
            self.layer = self.make_resblock(self.T, ratio_index=ratio_index)

        def make_resblock(self, T, ratio_index):
            layers = []
            for i in range(T):
                layers.append(nn.Sequential(
                    FFC_BN_ACT(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, ratio_gin=ratio_gin[ratio_index], ratio_gout=ratio_gout[ratio_index]),
                    nn.ReLU(),
                    FFC_BN_ACT(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, ratio_gin=ratio_gin[ratio_index], ratio_gout=ratio_gout[ratio_index]),
                ))
            return nn.Sequential(*layers)

        def forward(self, input):
            X = input
            counter = 0
            for i in range(self.T):
                X = F.relu(X + self.layer[i](X))
                counter += 1
            return X