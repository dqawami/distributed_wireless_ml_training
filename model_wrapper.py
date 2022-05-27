import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('./pytorch-cifar/')
from models import *

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model_dict = {
    "DenseNet121": DenseNet121,
    "DenseNet169": DenseNet169,
    "DenseNet201": DenseNet201,
    "DenseNet161": DenseNet161,
    "SimpleDLA": SimpleDLA,
    "DLA": DLA,
    "DPN26": DPN26,
    "DPN92": DPN92,
    "EfficientNetB0": EfficientNetB0,
    "GoogLeNet": GoogLeNet,
    "LeNet": LeNet,
    "MobileNet": MobileNet,
    "MobileNetV2": MobileNetV2,
    "PNASNetA": PNASNetA,
    "PNASNetB": PNASNetB,
    "PreActResNet18": PreActResNet18,
    "PreActResNet34": PreActResNet34,
    "PreActResNet50": PreActResNet50,
    "PreActResNet101": PreActResNet101,
    "PreActResNet152": PreActResNet152,
    "RegNetX_200MF": RegNetX_200MF,
    "RegNetX_400MF": RegNetX_400MF,
    "RegNetY_400MF": RegNetY_400MF,
    "ResNet18": ResNet18,
    "ResNet34": ResNet34,
    "ResNet50": ResNet50,
    "ResNet101": ResNet101,
    "ResNet152": ResNet152,
    "ResNeXt29_2x64d": ResNeXt29_2x64d,
    "ResNeXt29_4x64d": ResNeXt29_4x64d,
    "ResNeXt29_8x64d": ResNeXt29_8x64d,
    "ResNeXt29_32x4d": ResNeXt29_32x4d,
    "SENet18": SENet18,
    "ShuffleNetG2": ShuffleNetG2,
    "ShuffleNetG3": ShuffleNetG3,
    "ShuffleNetV2_0_5": ShuffleNetV2_0_5,
    "ShuffleNetV2_1": ShuffleNetV2_1,
    "ShuffleNetV2_1_5": ShuffleNetV2_1_5,
    "ShuffleNetV2_2": ShuffleNetV2_2,
    "ToyModel": ToyModel,
}

def get_model(model_name: str):
    if model_name not in model_dict:
        raise ValueError("Model does not exist in project")
    return model_dict[model_name]()