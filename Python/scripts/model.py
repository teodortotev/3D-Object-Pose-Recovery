from torch import nn
from torchvision.models.segmentation import fcn_resnet50, fcn_resnet101

class FTModel(nn.Module):

    def __init__(self, backbone, num_classes=9):
        super(FTModel, self).__init__()
        self.backbone = backbone
        self.add_module('297', nn.Conv2d(256, num_classes, 1))

    def forward(self, x):
        x = self.backbone(x)
        x = x["out"]
        x = self.final_layer(x)
        return x