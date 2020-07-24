import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch.autograd import Function
from config.config import cfg
import torch
import ipdb

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}



class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class ZeroLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = 0.0

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output * 0.0

        return output, None

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        # self.fc = nn.Linear(512 * block.expansion, num_classes*2)  ## for classification

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # out = self.fc(x)

        return x


def resnet18():
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=cfg.DATASET.NUM_CLASSES)
    if cfg.MODEL.PRETRAINED:
        pretrained_dict = model_zoo.load_url(model_urls['resnet18'])
        model_dict = model.state_dict()
        pretrained_dict_temp = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict_temp)
        model.load_state_dict(model_dict)
    classifier = nn.Linear(512 * BasicBlock.expansion, cfg.DATASET.NUM_CLASSES*2)
    return model, classifier


def resnet34():
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=cfg.DATASET.NUM_CLASSES)
    if cfg.MODEL.PRETRAINED:
        pretrained_dict = model_zoo.load_url(model_urls['resnet34'])
        model_dict = model.state_dict()
        pretrained_dict_temp = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict_temp)
        model.load_state_dict(model_dict)
    classifier = nn.Linear(512 * BasicBlock.expansion, cfg.DATASET.NUM_CLASSES * 2)
    return model, classifier


def resnet50():
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=cfg.DATASET.NUM_CLASSES)
    if cfg.MODEL.PRETRAINED:
        print('load the ImageNet pretrained parameters')
        pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
        model_dict = model.state_dict()
        pretrained_dict_temp = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict_temp)
        model.load_state_dict(model_dict)
    classifier = nn.Linear(512 * Bottleneck.expansion, cfg.DATASET.NUM_CLASSES * 2)  ## the concatenation of two task classifiers
    return model, classifier


def resnet101():
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=cfg.DATASET.NUM_CLASSES)
    if cfg.MODEL.PRETRAINED:
        pretrained_dict = model_zoo.load_url(model_urls['resnet101'])
        model_dict = model.state_dict()
        pretrained_dict_temp = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict_temp)
        model.load_state_dict(model_dict)
    # classifier = nn.Linear(512 * Bottleneck.expansion, cfg.DATASET.NUM_CLASSES * 2)
    classifier = nn.Sequential(nn.Linear(512 * Bottleneck.expansion, 512),
                            nn.BatchNorm1d(512),
                            nn.ReLU(inplace=True),
                            nn.Linear(512, cfg.DATASET.NUM_CLASSES * 2))
    return model, classifier


def resnet152():
    """Constructs a ResNet-152 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes=cfg.DATASET.NUM_CLASSES)
    if cfg.MODEL.PRETRAINED:
        pretrained_dict = model_zoo.load_url(model_urls['resnet152'])
        model_dict = model.state_dict()
        pretrained_dict_temp = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict_temp)
        model.load_state_dict(model_dict)
    classifier = nn.Linear(512 * Bottleneck.expansion, cfg.DATASET.NUM_CLASSES * 2)
    return model, classifier


def resnet():
    print("==> creating model '{}' ".format(cfg.MODEL.FEATURE_EXTRACTOR))
    if cfg.MODEL.FEATURE_EXTRACTOR == 'resnet18':
        return resnet18()
    elif cfg.MODEL.FEATURE_EXTRACTOR == 'resnet34':
        return resnet34()
    elif cfg.MODEL.FEATURE_EXTRACTOR == 'resnet50':
        return resnet50()
    elif cfg.MODEL.FEATURE_EXTRACTOR == 'resnet101':
        return resnet101()
    elif cfg.MODEL.FEATURE_EXTRACTOR == 'resnet152':
        return resnet152()
    else:
        raise ValueError('Unrecognized model architecture', cfg.MODEL.FEATURE_EXTRACTOR)