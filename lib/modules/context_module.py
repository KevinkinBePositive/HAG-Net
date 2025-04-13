import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import *
from .wtconv.wtconv2d import WTConv2d_tf

class DFA_kernel_edge(nn.Module):
    #注意：
    #经过PAA_kernel图片的大小不变，通道数由in_channel->out_channel
    def __init__(self, in_channel, out_channel, receptive_size=3):
        super(DFA_kernel_edge, self).__init__()
        self.conv0 = conv(in_channel, out_channel, 1)
        self.conv1 = conv(out_channel, out_channel, kernel_size=(1, receptive_size))
        self.conv2 = conv(out_channel, out_channel, kernel_size=(receptive_size, 1))
        self.conv3 = conv(out_channel, out_channel, 3, dilation=receptive_size)
        self.att = DFA(out_channel, 'B')

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.att(x)
        x = self.conv3(x)

        return x

class DFA_e_edge(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DFA_e_edge, self).__init__()
        self.relu = nn.ReLU(True)

        self.branch0 = conv(in_channel, out_channel, 1)  #通道数改变，大小不变
        #PAA_kernel就是水平+垂直分别操作
        self.branch1 = DFA_kernel_edge(in_channel, out_channel, 3)  #通道数改变，大小不变
        self.branch2 = DFA_kernel_edge(in_channel, out_channel, 5)
        self.branch3 = DFA_kernel_edge(in_channel, out_channel, 7)

        self.conv_cat = conv(4 * out_channel, out_channel, 3)
        self.conv_res = conv(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = self.relu(x_cat + self.conv_res(x))

        return x

class WCAttention(nn.Module):
    def __init__(self, in_channels,out_channels, rate=4):
        super(WCAttention, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),  # 线性层，将通道数缩减为 1/rate
            nn.ReLU(inplace=True),  # 激活函数，ReLU
            nn.Linear(int(in_channels / rate), in_channels)  # 线性层，将通道数恢复为原始通道数
        )
        
        self.spatial_attention = nn.Sequential(
            WTConv2d_tf(in_channels, int(in_channels / rate)), 
            nn.BatchNorm2d(int(in_channels / rate)), 
            nn.ReLU(inplace=True),
            WTConv2d_tf(int(in_channels / rate), in_channels),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        # 前向传播函数，定义了数据通过网络时的计算过程
        b, c, h, w = x.shape  # 获取输入 x 的形状 (batch_size, channels, height, width)
        
        # 计算通道注意力
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)  # 将输入 x 的维度从 (b, c, h, w) 变换为 (b, h*w, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)  # 通过通道注意力模块，输出形状为 (b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2).sigmoid()  # 变换回原始形状 (b, c, h, w)，并通过 Sigmoid 函数
        
        # 应用通道注意力
        x = x * x_channel_att  # 将原始输入 x 与通道注意力权重相乘
        
        # 计算空间注意力
        x_spatial_att = self.spatial_attention(x).sigmoid()  # 通过空间注意力模块，输出形状为 (b, c, h, w)，并通过 Sigmoid 函数
        out = x * x_spatial_att  # 将通道加权后的输入与空间注意力权重相乘
        out = self.conv1x1(out)
        
        return out  # 返回最终的注意力加权输出