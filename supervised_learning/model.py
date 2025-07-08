import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50
class ResNet_SimCLR_SimSiam_no_meteo(nn.Module):
    def __init__(self, ssl_path, backbone='resnet18', resnet_trainable=False):
        super(ResNet_SimCLR_SimSiam_no_meteo, self).__init__()
        if backbone == 'resnet18':
            resnet = resnet18(pretrained=False)
            in_features = 512
        elif backbone == 'resnet50':
            resnet = resnet50(pretrained=False)
            in_features = 2048
        else:
            raise ValueError("Backbone must be 'resnet18' or 'resnet50'")
        resnet.fc = nn.Identity()
        checkpoint_resnet = torch.load(ssl_path, map_location=torch.device('cpu'))
        resnet.load_state_dict(checkpoint_resnet, strict=False)
        for param in resnet.parameters():
            param.requires_grad = resnet_trainable
        
        self.resnet_pretrained = resnet
        self.fc1 = nn.Linear(in_features, 512)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)
    
    def forward(self, image, epoch):
        img_features = self.resnet_pretrained(image)
        x = self.fc1(img_features)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x.float())
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x