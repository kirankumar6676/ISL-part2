import torch.nn as nn
import torchvision.models as models

class WordCNN(nn.Module):
    """
    CNN model for ISL word classification using transfer learning.
    Uses ResNet18 pretrained on ImageNet.
    """

    def __init__(self, num_classes):
        super().__init__()
        # Load pretrained ResNet18 (updated API to avoid deprecation warning)
        self.base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Replace final FC layer for our number of classes
        self.base.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.base(x)
