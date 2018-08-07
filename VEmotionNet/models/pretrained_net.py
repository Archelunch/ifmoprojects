# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch.common.losses import *
from collections import OrderedDict

from resnet import resnet34, get_vec

class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, data):
        x = data
        for name, module in self.submodule._modules.items():
            if len(module._modules.items()) != 0:
                for name2, module2 in module._modules.items():
                    x = module2(x)
            else:
                x = module(x)
        return x

class CNNNet(nn.Module):
    def __init__(self, num_classes, depth, data_size, emb_name=[], pretrain_weight=None):
        super(CNNNet, self).__init__()
        # sample_size = data_size['width']
        # sample_duration = data_size['depth']
        self.pretrained = resnet34(num_classes=8631)
        self.pretrained.load_state_dict(torch.load("model_13.pt"))
        num_ftrs = self.pretrained.fc.in_features
        self.pretrained.fc = nn.Linear(num_ftrs, 2)

        # # TODO: Реализуйте архитектуру нейронной сети

        # net = []

        # self.net = FeatureExtractor(net, emb_name)

    def forward(self, data):
        output = self.pretrained(data)
        return output

