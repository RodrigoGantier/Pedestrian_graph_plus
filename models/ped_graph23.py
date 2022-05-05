from functools import reduce
import math
import torch
from torch import nn
import numpy as np

    

class pedMondel(nn.Module):

    def __init__(self, frames, vel=False, seg=False, h3d=True, nodes=19, n_clss=1):
        super(pedMondel, self).__init__()

        self.h3d = h3d # bool if true 3D human keypoints data is enable otherwise 2D is only used
        self.frames = frames
        self.vel = vel
        self.seg = seg
        self.n_clss = n_clss
        self.ch = 4 if h3d else 3
        self.ch1, self.ch2 = 32, 64
        i_ch = 4 if seg else 3

        self.data_bn = nn.BatchNorm1d(self.ch * nodes)
        bn_init(self.data_bn, 1)
        self.drop = nn.Dropout(0.25)
        A = np.stack([np.eye(nodes)] * 3, axis=0)
        
        if frames:
            self.conv0 = nn.Sequential(
                nn.Conv2d(i_ch, self.ch1, kernel_size=3, stride=1, padding=0, bias=False), 
                nn.BatchNorm2d(self.ch1), nn.SiLU())
        if vel:
            self.v0 = nn.Sequential(
                nn.Conv1d(2, self.ch1, 3, bias=False), nn.BatchNorm1d(self.ch1), nn.SiLU())
        # ----------------------------------------------------------------------------------------------------
        self.l1 = TCN_GCN_unit(self.ch, self.ch1, A, residual=False)

        if frames:
            self.conv1 = nn.Sequential(
                nn.Conv2d(self.ch1, self.ch1, kernel_size=3, stride=1, padding=0, bias=False), 
                nn.BatchNorm2d(self.ch1), nn.SiLU())
        if vel:
            self.v1 = nn.Sequential(
                nn.Conv1d(self.ch1, self.ch1, 3, bias=False), 
                nn.BatchNorm1d(self.ch1), nn.SiLU())
        # ----------------------------------------------------------------------------------------------------
        self.l2 = TCN_GCN_unit(self.ch1, self.ch2, A)

        if frames:
            self.conv2 = nn.Sequential(
                nn.Conv2d(self.ch1, self.ch2, kernel_size=2, stride=1, padding=0, bias=False), 
                nn.BatchNorm2d(self.ch2), nn.SiLU())
            
        if vel:
            self.v2 = nn.Sequential(
                nn.Conv1d(self.ch1, self.ch2, kernel_size=2, bias=False), 
                nn.BatchNorm1d(self.ch2), nn.SiLU())
        # ----------------------------------------------------------------------------------------------------
        # self.l3 = TCN_GCN_unit(self.ch2, self.ch2, A)

        # if frames:
        #     self.conv3 = nn.Sequential(
        #         nn.Conv2d(self.ch2, self.ch2, kernel_size=2, stride=1, padding=0, bias=False), 
        #         nn.BatchNorm2d(self.ch2), nn.SiLU())
            
        # if vel:
        #     self.v3 = nn.Sequential(
        #         nn.Conv1d(self.ch2, self.ch2, kernel_size=2, bias=False), 
        #         nn.BatchNorm1d(self.ch2), nn.SiLU())
        # ----------------------------------------------------------------------------------------------------
        

        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.att = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.ch2, self.ch2, bias=False),
            nn.BatchNorm1d(self.ch2), 
            nn.Sigmoid()
        )

        self.linear = nn.Linear(self.ch2, self.n_clss)
        nn.init.normal_(self.linear.weight, 0, math.sqrt(2. / self.n_clss))
        # pooling sigmoid fucntion for image feature fusion
        self.pool_sigm_2d = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )
        if vel:
            self.pool_sigm_1d = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Sigmoid()
            )
        
    
    def forward(self, kp, frame=None, vel=None): 

        N, C, T, V = kp.shape
        kp = kp.permute(0, 1, 3, 2).contiguous().view(N, C * V, T)
        kp = self.data_bn(kp)
        kp = kp.view(N, C, V, T).permute(0, 1, 3, 2).contiguous()
        
        if self.frames:
            f1 = self.conv0(frame) 
        if self.vel:
            v1 = self.v0(vel)

        # --------------------------
        x1 = self.l1(kp)
        if self.frames:
            f1 = self.conv1(f1)   
            x1.mul(self.pool_sigm_2d(f1))
        if self.vel:   
            v1 = self.v1(v1)
            x1 = x1.mul(self.pool_sigm_1d(v1).unsqueeze(-1))
        # --------------------------
        
        # --------------------------
        x1 = self.l2(x1)
        if self.frames:
            f1 = self.conv2(f1) 
            x1 = x1.mul(self.pool_sigm_2d(f1))
        if self.vel:  
            v1 = self.v2(v1)
            x1 = x1.mul(self.pool_sigm_1d(v1).unsqueeze(-1))
        # --------------------------
        # x1 = self.l3(x1)
        # if self.frames:
        #     f1 = self.conv3(f1) 
        #     x1 = x1.mul(self.pool_sigm_2d(f1))
        # if self.vel:  
        #     v1 = self.v3(v1)
        #     x1 = x1.mul(self.pool_sigm_1d(v1).unsqueeze(-1))
        # --------------------------

        x1 = self.gap(x1).squeeze(-1)
        x1 = x1.squeeze(-1)
        x1 = self.att(x1).mul(x1) + x1
        x1 = self.drop(x1)
        x1 = self.linear(x1)

        return x1


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, adaptive=True):
        super(unit_gcn, self).__init__()
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_subset = A.shape[0]
        self.adaptive = adaptive
        if adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=True)
        else:
            self.A = torch.autograd.Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)

        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)
    
    def L2_norm(self, A):
        # A:N,V,V
        A_norm = torch.norm(A, 2, dim=1, keepdim=True) + 1e-4  # N,1,V
        A = A / A_norm
        return A

    def forward(self, x):
        N, C, T, V = x.size()

        y = None
        if self.adaptive:
            A = self.PA
            A = self.L2_norm(A)
        else:
            A = self.A.cuda(x.get_device())
        for i in range(self.num_subset):

            A1 = A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)

        return y


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y