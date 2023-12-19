import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

from utils.tools import pyout


class TowelCornerResNet(nn.Module):
    def __init__(self, num_classes: int, dropout_rate: float = 0.5):
        super(TowelCornerResNet, self).__init__()
        # Load pre-trained ResNet
        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])
        self.drop_ou = nn.Dropout(p=dropout_rate)
        self.line_ou = nn.Linear(2048, num_classes)

    def forward(self, x):
        h = self.features(x)
        h = torch.flatten(h, start_dim=1)
        h = self.line_ou(h)
        return torch.sigmoid(h)
