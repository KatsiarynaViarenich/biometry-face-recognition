import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ArcMarginProduct(nn.Module):
    """
    Kod pochodzi z oficjalnego repozytorium ArcFace:
    https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output

from torchvision.models import resnet18, ResNet18_Weights
from facenet_pytorch import InceptionResnetV1

class FaceEmbeddingModel(nn.Module):
    def __init__(self, embedding_size=512, backbone_type='resnet18'):
        super(FaceEmbeddingModel, self).__init__()
        self.backbone_type = backbone_type
        
        if backbone_type == 'resnet18':
            self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, embedding_size)
            
        elif backbone_type == 'inception_resnet':
            self.backbone = InceptionResnetV1(pretrained='vggface2')
            
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        if self.backbone_type == 'resnet18':
            x = self.backbone(x)
        else:
            x = self.backbone(x)
        x = self.bn(x)
        return x

class ArcFaceModel(nn.Module):
    def __init__(self, num_classes, embedding_size=512, backbone_type='inception_resnet'):
        super(ArcFaceModel, self).__init__()
        self.feature_extractor = FaceEmbeddingModel(embedding_size, backbone_type)
        self.arcface = ArcMarginProduct(in_features=embedding_size, out_features=num_classes)
        
    def forward(self, x, labels=None):
        features = self.feature_extractor(x)
        if labels is not None:
            out = self.arcface(features, labels)
            return out, features
        return features
