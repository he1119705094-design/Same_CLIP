'''                                        
Copyright 2024 Image Processing Research Group of University Federico
II of Naples ('GRIP-UNINA'). All rights reserved.
                        
Licensed under the Apache License, Version 2.0 (the "License");       
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at                    
                                           
    http://www.apache.org/licenses/LICENSE-2.0
                                                      
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,    
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.                         
See the License for the specific language governing permissions and
limitations under the License.
'''

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo # model_zoo 是 PyTorch 提供的预训练模型下载工具，比如用来加载官方的 resnet50 参数。

__all__ = ["ResNet", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}

class ChannelLinear(nn.Linear):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, pool=None
    ) -> None:
        super(ChannelLinear, self).__init__(in_features, out_features, bias)
        self.compute_axis = 1
        self.pool = pool

    def forward(self, x):
        axis_ref = len(x.shape) - 1
        x = torch.transpose(x, self.compute_axis, axis_ref)
        out_shape = list(x.shape)
        out_shape[-1] = self.out_features
        x = x.reshape(-1, x.shape[-1])
        x = x.matmul(self.weight.t())
        if self.bias is not None:
            x = x + self.bias[None, :]
        x = torch.transpose(x.view(out_shape), axis_ref, self.compute_axis)
        if self.pool is not None:
            x = self.pool(x)
        return x


def conv3x3(in_planes, out_planes, stride=1, padding=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, padding=1, downsample=None, leaky=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(planes)# 批量归一化（Batch Normalization）假设你有一批图片通过卷积层后，输出了一些特征图。这些特征图有的大有的小，BN变为标准分布分布更平稳
                                        # 通常的顺序就是：卷积->批量归一化->relu
        if leaky:
            self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, padding=padding)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.padding == 0:
            identity = identity[..., 1:-1, 1:-1]
        if self.downsample is not None:
            identity = self.downsample(identity)
        if self.padding == 0:
            identity = identity[..., 1:-1, 1:-1]

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, padding=1, downsample=None, leaky=False):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)              # 降维
        self.bn1 = nn.BatchNorm2d(planes)        #
        self.conv2 = conv3x3(planes, planes, stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)#升维
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        if leaky:
            self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        identity = x    # identity表示原始的输入，便于最后把原始的输入与经过某个网络块的输出进行相加

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.padding == 0:
            identity = identity[..., 1:-1, 1:-1]
        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out

import numpy as np
import torch
from torch import nn
from torch.nn import init
from collections import OrderedDict

"Selective Kernel Networks"

# 这个是新加的模块
class SKAttention(nn.Module):

    def __init__(self, channel=512,kernels=[1,3,5,7],reduction=8,group=1,L=32):
        super().__init__()
        self.d=max(L,channel//reduction)
        self.convs=nn.ModuleList([])
        # 有几个卷积核,就有几个尺度, 每个尺度对应的卷积层由Conv-bn-relu实现
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv',nn.Conv2d(channel,channel,kernel_size=k,padding=k//2,groups=group)),
                    ('bn',nn.BatchNorm2d(channel)),
                    ('relu',nn.ReLU())
                ]))
            )
        # 将全局向量降维
        self.fc=nn.Linear(channel,self.d)
        self.fcs=nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d,channel))
        self.softmax=nn.Softmax(dim=0)



    def forward(self, x):
        # (B, C, H, W)
        B, C, H, W = x.size()
        # 存放多尺度的输出
        conv_outs=[]
        # Split: 执行K个尺度对应的卷积操作
        for conv in self.convs:
            scale = conv(x)  #每一个尺度的输出shape都是: (B, C, H, W),是因为使用了padding操作
            conv_outs.append(scale)
        feats=torch.stack(conv_outs,0) # 将K个尺度的输出在第0个维度上拼接: (K,B,C,H,W)

        # Fuse: 首先将多尺度的信息进行相加,sum()默认在第一个维度进行求和
        U=sum(conv_outs) #(K,B,C,H,W)-->(B,C,H,W)
        # 全局平均池化操作: (B,C,H,W)-->mean-->(B,C,H)-->mean-->(B,C)  【mean操作等价于全局平均池化的操作】
        S=U.mean(-1).mean(-1)
        # 降低通道数,提高计算效率: (B,C)-->(B,d)
        Z=self.fc(S)

        # 将紧凑特征Z通过K个全连接层得到K个尺度对应的通道描述符表示, 然后基于K个通道描述符计算注意力权重
        weights=[]
        for fc in self.fcs:
            weight=fc(Z) #恢复预输入相同的通道数: (B,d)-->(B,C)
            weights.append(weight.view(B,C,1,1)) # (B,C)-->(B,C,1,1)
        scale_weight=torch.stack(weights,0) #将K个通道描述符在0个维度上拼接: (K,B,C,1,1)
        scale_weight=self.softmax(scale_weight) #在第0个维度上执行softmax,获得每个尺度的权重: (K,B,C,1,1)

        # Select
        V=(scale_weight*feats).sum(0) # 将每个尺度的权重与对应的特征进行加权求和,第一步是加权，第二步是求和：(K,B,C,1,1) * (K,B,C,H,W) = (K,B,C,H,W)-->sum-->(B,C,H,W)
        return V

class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        stride0=2,
        padding=1,
        dropout=0.0,
        gap_size=None,
        leaky=False,
        use_proj=False,
        proj_ratio=None
    ):
        super(ResNet, self).__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=stride0, padding=3 * padding, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        if leaky:
            self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=stride0, padding=padding)
        self.layer1 = self._make_layer(block, 64, layers[0], padding=padding, leaky=leaky)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, padding=padding, leaky=leaky)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, padding=padding, leaky=leaky)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, padding=padding, leaky=leaky)
        self.use_proj = use_proj
        self.proj_ratio = proj_ratio
        if gap_size is None:
            self.gap_size = None
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        elif gap_size < 0:
            with torch.no_grad():
                y = self.forward_features(
                    torch.zeros((1, 3, -gap_size, -gap_size), dtype=torch.float32)
                ).shape
            print("gap_size:", -gap_size, ">>", y[-1])
            self.gap_size = y[-1]
            self.avgpool = nn.AvgPool2d(kernel_size=self.gap_size, stride=1, padding=0)
        elif gap_size == 1:
            self.gap_size = gap_size
            self.avgpool = None
        else:
            self.gap_size = gap_size
            self.avgpool = nn.AvgPool2d(kernel_size=self.gap_size, stride=1, padding=0)
        self.num_features = 512 * block.expansion
        if use_proj:
            int_dim = int(self.num_features//proj_ratio)
            self.proj = ChannelLinear(self.num_features, int_dim)
            self.fc = ChannelLinear(int_dim, num_classes)
        else:
            self.fc = ChannelLinear(self.num_features, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
        else:
            print('Custom initialization')
            self._initialize_weights()
        self.ska = SKAttention(channel=3)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # block:用的残差模块类（如 BasicBlock 或 Bottleneck）。planes:当前层输出的“中间通道数”
    # blocks:这一层要堆叠几个 block
    def _make_layer(self, block, planes, blocks, stride=1, padding=1, leaky=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                downsample=downsample,
                padding=padding,
                leaky=leaky
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, padding=padding, leaky=leaky))

        return nn.Sequential(*layers)

    def change_output(self, num_classes, use_proj=False):
        if use_proj:
            self.fc = ChannelLinear(int(self.num_features//self.proj_ratio), num_classes)
        else:
            self.fc = ChannelLinear(self.num_features, num_classes)
        torch.nn.init.normal_(self.fc.weight.data, 0.0, 0.02)
        return self

    def change_input(self, num_inputs):
        data = self.conv1.weight.data
        old_num_inputs = int(data.shape[1])
        if num_inputs > old_num_inputs:
            times = num_inputs // old_num_inputs
            if (times * old_num_inputs) < num_inputs:
                times = times + 1
            data = data.repeat(1, times, 1, 1) / times
        elif num_inputs == old_num_inputs:
            return self

        data = data[:, :num_inputs, :, :]
        print(self.conv1.weight.data.shape, "->", data.shape)
        self.conv1.weight.data = data

        return self

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward_head(self, x, return_feats=False):
        if self.use_proj:
            x = self.proj(x)
        if self.avgpool is not None:
            x = self.avgpool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        y = self.fc(x)
        if self.gap_size is None:
            y = torch.squeeze(torch.squeeze(y, -1), -1)
        if return_feats:
            return y,x
        else:
            return y,None

    def forward(self, x, return_feats=False):
        #x = self.ska(x)
        x = self.forward_features(x)                      # 第一步：提取特征
        x = self.forward_head(x,return_feats=return_feats)# 第二步：分类或特征输出
        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet18"]))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet34"]))
    return model


def resnet50(pretrained=False, leaky=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs, leaky=leaky)
    if pretrained:
        #model.load_state_dict(model_zoo.load_url(model_urls["resnet50"]))
    
        try:
            model.load_state_dict(model_zoo.load_url(model_urls["resnet50"]))
        except RuntimeError as e:
            print(f"Standard loading failed: {e}")
            try:
                # Attempt to load flexibly
                state_dict = model_zoo.load_url(model_urls["resnet50"])
                filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and v.size() == model.state_dict()[k].size()}
                missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
                # Print the missing and unexpected keys
                if missing_keys:
                    print(f"Missing keys: {missing_keys}")
                if unexpected_keys:
                    print(f"Unexpected keys: {unexpected_keys}")
        
                print("Model loaded successfully with the flexible method.")
            except Exception as e:
                print(f"Flexible loading also failed: {e}")
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet101"]))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet152"]))
    return model
