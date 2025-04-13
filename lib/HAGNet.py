import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .optim.losses import *
from .modules.layers import *
from .modules.context_module import *
from .modules.attention_module import *
from .modules.decoder_module import *

from .backbones.Res2Net_v1b import res2net50_v1b_26w_4s

class HAGNet(nn.Module):
    def __init__(self, channels=256, output_stride=16, pretrained=False):
        super(HAGNet, self).__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=pretrained, output_stride=output_stride)

        self.context0 = DFA_e_edge(64, channels)
        self.context1 = DFA_e_edge(256, channels)
        self.context2 = DFA_e_edge(512, channels)
        self.context3 = WCAttention(1024, channels)
        self.context4 = WCAttention(2048, channels)

        self.decoder = DFA_d3(channels)
        self.decoder2 = DFA_d2(channels)


        self.attention2 = HAG(channels * 2, channels)
        self.attention3 = HAG(channels * 2, channels)
        self.attention4 = HAG(channels * 2, channels)

        self.loss_fn1 = bce_cel_loss
        self.loss_fn2 = boundary_gradient_loss
        

        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, sample):
        x = sample['image']
        if 'gt' in sample.keys():
            y = sample['gt']
        else:
            y = None
        base_size = x.shape[-2:]
        
        x = self.resnet.conv1(x)

        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        x1 = self.context1(x1)
        x2 = self.context2(x2)
        x3 = self.context3(x3)
        x4 = self.context4(x4)


        f51, a51 = self.decoder(x4, x3, x2) 
        
        f52, a52 = self.decoder2(x3, x2)
        out5 = self.res(a51, base_size)

        f4, a4 = self.attention4(torch.cat([x4, self.ret(f51, x4)], dim=1), a51)
        out4 = self.res(a4, base_size)

        f3, a3 = self.attention3(torch.cat([x3, self.ret(f52, x3)], dim=1), a52)
        out3 = self.res(a3, base_size)

        _, a2 = self.attention2(torch.cat([x2, self.ret((f3+f4), x2)], dim=1), (a3+a4))
        out2 = self.res(a2, base_size)
        #print('out2',out2)


        if y is not None:
            loss5 = self.loss_fn1(out5, y)
            loss4 = self.loss_fn1(out4, y)
            loss3 = self.loss_fn1(out3, y)
            loss2 = self.loss_fn1(out2, y)

            loss = loss2 + loss3 + loss4 + loss5
            debug = [out5, out4, out3]
        else:
            loss = 0
            debug = []

        return {'pred': out2, 'loss': loss, 'debug': debug}
        #return out2

if __name__ == '__main__':
    a = torch.randn(1, 3, 224, 224)
    b = {'image':a}
    uaca = HAGNet(b)