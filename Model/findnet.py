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
import scipy.io as io
from Model.ProxNet import Xnet, Mnet

kernel = io.loadmat('utils/init_kernel.mat')['C9']  # 3*32*9*9
kernel = torch.FloatTensor(kernel)
kernel = kernel[0:1, :, :, :]

# filtering on metal-affected CT image for initializing B^(0) and Z^(0), refer to supplementary material(SM)
filter = torch.FloatTensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]) / 9
filter = filter.unsqueeze(dim=0).unsqueeze(dim=0)

   
class FINDNet(nn.Module):
    def __init__(self, args):
        super(FINDNet, self).__init__()
        self.S = args.S  # Stage number S includes the initialization process
        self.iter = self.S - 1  # not include the initialization process
        self.num_M = args.num_M
        self.num_Q = args.num_Q # for concatenation channel (See Supplementary material)

        # Stepsize
        self.etaM = torch.Tensor([args.etaM])  # initialization
        self.etaX = torch.Tensor([args.etaX])  # initialization
        self.etaM_S = self.make_eta(self.iter, self.etaM)
        self.etaX_S = self.make_eta(self.S, self.etaX)

        # kernel
        self.K0 = nn.Parameter(data=kernel, requires_grad=True)  # used in initialization process
        self.K = nn.Parameter(data=kernel, requires_grad=True)  # self.K (kernel) is inter-stage sharing

        # filter for initializing X and Q
        self.K_q_const = filter.expand(self.num_Q, 1, -1, -1)  # size: self.num_Q*1*3*3
        self.K_q = nn.Parameter(self.K_q_const, requires_grad=True)

        # proxNet
        self.proxNet_X_0 = Xnet(args, ratio_index=0)  # used in initialization process
        self.proxNet_X_S = self.make_Xnet(self.S, args)
        self.proxNet_M_S = self.make_Mnet(self.S, args)
        self.proxNet_X_last_layer = Xnet(args, ratio_index=0)  # fine-tune at the last

        # for sparsity
        self.tau_const = torch.Tensor([1])
        self.tau = nn.Parameter(self.tau_const, requires_grad=True)

    def make_Xnet(self, iters, args):
        layers = []
        for i in range(iters):
            layers.append(Xnet(args, ratio_index=i))
        return nn.Sequential(*layers)

    def make_Mnet(self, iters, args):
        layers = []
        for i in range(iters):
            layers.append(Mnet(args, ratio_index=i))
        return nn.Sequential(*layers)

    def make_eta(self, iters, const):
        const_dimadd = const.unsqueeze(dim=0)
        const_f = const_dimadd.expand(iters, -1)
        eta = nn.Parameter(data=const_f, requires_grad=True)
        return eta

    def forward(self, CT_ma, LIct, Mask):
        # save mid-updating results
        ListX = []
        ListA = []
        ListM = []
        input = CT_ma
        Q00 = F.conv2d(LIct, self.K_q, stride=1, padding=1)  # dual variable Q (see supplementary material)
        input_ini = torch.cat((LIct, Q00), dim=1)
        XQ_ini = self.proxNet_X_0(input_ini)
        X0 = XQ_ini[:, :1, :, :]
        Q0 = XQ_ini[:, 1:, :, :]

        # 1st iteration：Updating X0-->M1
        A_hat =  Mask *(input -  X0)
        A_hat_cut = F.relu(A_hat - self.tau)  # for sparsity
        Epsilon = F.conv_transpose2d(A_hat_cut, self.K0 / 10, stride=1,
                                     padding=4)  # /10 for controlling the updating speed
        M1 = self.proxNet_M_S[0](Epsilon)
        A = F.conv2d(M1, self.K / 10, stride=1, padding=4)  # /10 for controlling the updating speed

        # 1st iteration: Updating M1-->X1
        A_hat = input - A
        X_mid = (1 - self.etaX_S[0] * Mask / 10) * X0 + self.etaX_S[0] * Mask/ 10 * A_hat
        input_concat = torch.cat((X_mid, Q0), dim=1)
        XQ = self.proxNet_X_S[0](input_concat)
        X1 = XQ[:, :1, :, :]
        Q1 = XQ[:, 1:, :, :]
        ListX.append(X1)
        ListA.append(A)
        ListM.append(M1)
        X = X1
        Q = Q1
        M = M1
        for i in range(self.iter):
            # M-net
            A_hat = Mask *(input -  X)
            Epsilon = self.etaM_S[i, :] / 10 * F.conv_transpose2d((Mask*A - A_hat), self.K / 10, stride=1, padding=4)
            M = self.proxNet_M_S[i + 1](M - Epsilon)

            # X-net
            A = F.conv2d(M, self.K / 10, stride=1, padding=4)
            ListA.append(A)
            X_hat = input - A
            X_mid = (1 - self.etaX_S[i + 1, :] *  Mask / 10) * X + self.etaX_S[i + 1, :] * Mask/ 10 * X_hat
            input_concat = torch.cat((X_mid, Q), dim=1)
            XQ = self.proxNet_X_S[i + 1](input_concat)
            X = XQ[:, :1, :, :]
            Q = XQ[:, 1:, :, :]
            ListX.append(X)
        XQ_adjust = self.proxNet_X_last_layer(XQ)
        X = XQ_adjust[:, :1, :, :]
        ListX.append(X)
        return X0, ListX, ListA

