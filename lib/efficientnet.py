import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchsummary import summary
from typing import Tuple
import math


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return x * self.sigmoid(x)


def Round_Channels(ch, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_ch = max(min_value, int(ch + divisor / 2) // divisor * divisor)
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


def Round_Repeats(r):
    return int(math.ceil(r))


def Drop_Path(x, drop_prob, training):
    if drop_prob > 0 and training:
        keep_prob = 1 - drop_prob
        if x.is_cuda:
            mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        else:
            mask = Variable(torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)

    return x


def BatchNorm(channels, eps=1e-3, momentum=0.01):
    return nn.BatchNorm2d(channels, eps=eps, momentum=momentum)


def Conv3x3Bn(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
        BatchNorm(out_channels),
        Swish())


def Conv1x1Bn(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        BatchNorm(out_channels),
        Swish())


class SqueezeAndExcite(nn.Module):
    def __init__(self, channels, squeeze_channels, se_ratio):
        super(SqueezeAndExcite, self).__init__()

        squeeze_channels = squeeze_channels * se_ratio
        if not squeeze_channels.is_integer():
            raise ValueError('Channels must be divisible by 1/ratio')

        squeeze_channels = int(squeeze_channels)
        self.se_reduce = nn.Conv2d(channels, squeeze_channels, 1, 1, 0, bias=True)
        self.swish = Swish()
        self.se_expand = nn.Conv2d(squeeze_channels, channels, 1, 1, 0, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = torch.mean(x, (2, 3), keepdim=True)
        y = self.se_reduce(y)
        y = self.swish(y)
        y = self.se_expand(y)
        y = self.sigmoid(y)
        y = x * y

        return y


class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, drop_path_rate):
        super(MBConvBlock, self).__init__()

        expand = (expand_ratio != 1)
        expand_channels = in_channels * expand_ratio
        se = (se_ratio != 0.0)
        self.residual_connection = (stride == 1 and in_channels == out_channels)
        self.drop_path_rate = drop_path_rate

        conv = []

        if expand:
            # expansion phase
            pw_expansion = nn.Sequential(
                nn.Conv2d(in_channels, expand_channels, 1, 1, 0, bias=False),
                BatchNorm(expand_channels),
                Swish()
            )
            conv.append(pw_expansion)

        # depthwise convolution phase
        dw = nn.Sequential(
            nn.Conv2d(
                expand_channels,
                expand_channels,
                kernel_size,
                stride,
                kernel_size//2,
                groups=expand_channels,
                bias=False
            ),
            BatchNorm(expand_channels),
            Swish()
        )
        conv.append(dw)

        if se:
            # squeeze and excite
            squeeze_excite = SqueezeAndExcite(expand_channels, in_channels, se_ratio)
            conv.append(squeeze_excite)

        # projection phase
        pw_projection = nn.Sequential(
            nn.Conv2d(expand_channels, out_channels, 1, 1, 0, bias=False),
            BatchNorm(out_channels)
        )
        conv.append(pw_projection)

        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        if self.residual_connection:
            y = self.conv(x)
            y = Drop_Path(y, self.drop_path_rate, self.training)
            return x + y
        else:
            return self.conv(x)


class EfficientNet(nn.Module):
    config = [
        # (in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, repeats)
        [32,  16,  3, 1, 1, 0.25, 1],
        [16,  24,  3, 2, 6, 0.25, 2],
        [24,  40,  5, 2, 6, 0.25, 2],
        [40,  80,  3, 2, 6, 0.25, 3],
        [80,  112, 5, 1, 6, 0.25, 3],
        [112, 192, 5, 2, 6, 0.25, 4],
        [192, 320, 3, 1, 6, 0.25, 1]
    ]

    def __init__(self, model_type: str = 'efficientnet-b0', 
                 num_classes=1000, 
                 stem_channels=32, 
                 feature_size=1280, 
                 drop_connect_rate=0.2):
        
        super(EfficientNet, self).__init__()


        param = self._model_selection(model_type)
        # scaling width
        width_coefficient = param[0]
        if width_coefficient != 1.0:
            stem_channels = Round_Channels(stem_channels * width_coefficient)
            for conf in self.config:
                conf[0] = Round_Channels(conf[0] * width_coefficient)
                conf[1] = Round_Channels(conf[1] * width_coefficient)

        # scaling depth
        depth_coefficient = param[1]
        if depth_coefficient != 1.0:
            for conf in self.config:
                conf[6] = Round_Repeats(conf[6] * depth_coefficient)

        # scaling resolution
        input_size = param[2]

        # stem convolution
        self.stem_conv = Conv3x3Bn(3, stem_channels, 2)

        # total #blocks
        total_blocks = 0
        for conf in self.config:
            total_blocks += conf[6]

        # mobile inverted bottleneck
        blocks = []
        for in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, repeats in self.config:
            # drop connect rate based on block index
            drop_rate = drop_connect_rate * (len(blocks) / total_blocks)
            blocks.append(MBConvBlock(in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, drop_rate))
            for _ in range(repeats - 1):
                drop_rate = drop_connect_rate * (len(blocks) / total_blocks)
                blocks.append(MBConvBlock(out_channels, out_channels, kernel_size, 1, expand_ratio, se_ratio, drop_rate))
        self.blocks = nn.Sequential(*blocks)

        # last several layers
        self.head_conv = Conv1x1Bn(self.config[-1][1], feature_size)
        # self.avgpool = nn.AvgPool2d(input_size // 32, stride=1)
        self.dropout = nn.Dropout(param[3])
        self.classifier = nn.Linear(feature_size, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.blocks(x)
        x = self.head_conv(x)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        x = torch.mean(x, (2, 3))
        x = self.dropout(x)
        x = self.classifier(x)

        return x

    def _model_selection(self, model_type) -> Tuple:
        net_param = {
            # 'efficientnet type': (width_coef, depth_coef, resolution, dropout_rate)
            'efficientnet-b0': (1.0, 1.0, 224, 0.2),
            'efficientnet-b1': (1.0, 1.1, 240, 0.2),
            'efficientnet-b2': (1.1, 1.2, 260, 0.3),
            'efficientnet-b3': (1.2, 1.4, 300, 0.3),
            'efficientnet-b4': (1.4, 1.8, 380, 0.4),
            'efficientnet-b5': (1.6, 2.2, 456, 0.4),
            'efficientnet-b6': (1.8, 2.6, 528, 0.5),
            'efficientnet-b7': (2.0, 3.1, 600, 0.5)}
        return net_param[model_type]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _summary(self, input_size, device='cpu'):
        return summary(self, input_size, device=device)


if __name__ == "__main__":
    model = EfficientNet('efficientnet-b0')
    x = Variable(torch.randn(1, 3, 224, 224))
    y = model(x)
    assert y.size() == torch.randn(1, 1000).size()

    model._summary((3, 224, 224))