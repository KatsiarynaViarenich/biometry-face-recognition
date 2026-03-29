import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1

class PretrainedBaselineModel(nn.Module):
    def __init__(self, pretrained='vggface2'):
        super(PretrainedBaselineModel, self).__init__()
        self.model = InceptionResnetV1(pretrained=pretrained)
        self.model.eval()

    def forward(self, x):
        return self.model(x)
