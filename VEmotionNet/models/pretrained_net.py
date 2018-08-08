# -*- coding: utf-8 -*-
import sys
import os

sys.path.append('../../')
sys.path.append('/media/stc_ml_school/team1')
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch.common.losses import *
from collections import OrderedDict

from pytorch.VEmotionNet.models.resnet import resnet34, get_vec
from pytorch.VEmotionNet.models.VGG_gru import FERANet
from pytorch.VEmotionNet.models.utils import Initial

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
        self.pretrained.load_state_dict(torch.load("/media/stc_ml_school/team1/pytorch/VEmotionNet/models/model_13.pt"))

        for param in self.pretrained.parameters():
            param.requires_grad = False

        num_ftrs = self.pretrained.fc.in_features
        self.pretrained.fc = nn.Sequential(nn.Linear(num_ftrs, 64),  nn.ReLU(inplace=True), nn.Dropout())
        self.output_layer = nn.Linear(64, num_classes)

        # # TODO: Реализуйте архитектуру нейронной сети

        # net = []

        # self.net = FeatureExtractor(net, emb_name)

    def forward(self, data):
        output = self.pretrained(data[:,:,0,:,:])
        output = self.output_layer(output)
        return output

class GRUNet(nn.Module):
    def __init__(self,depth,data_size):
        super(GRUNet, self).__init__()
        self.depth = depth
        self.data_size = data_size
        self.pretrained = resnet34(num_classes=8631)
        self.pretrained.load_state_dict(torch.load("/media/stc_ml_school/team1/pytorch/VEmotionNet/models/model_13.pt"))
        self.num_ftrs = self.pretrained.fc.in_features


        self.gru = nn.GRU(self.num_ftrs, 128, batch_first=True)
        self.classify = nn.Sequential(nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Dropout(inplace=True), nn.Linear(64, 2))

    def forward(self, x):
        x = x.permute(0,2,1,3,4)
        print(x.size())
        x = x.reshape(-1,self.data_size['width'],self.data_size['height'],3)
        x = self.pretrained(x)
        x = x.reshape(-1, self.depth, self.self.num_ftrs)  #batchsize,sequence_length,data_dim
        x, hn = self.gru(x)
        x = self.dropout(x)
        x = x[:,-1,:]
        x = self.classify(x)

        return x

