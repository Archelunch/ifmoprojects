# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def get_var(x, mean):
    return ((x.pow(2) - mean.pow(2)) / (x.size()[0])).sum()


def calc_l(x, y):
    mean_x = x.mean()
    mean_y = y.mean()

    var_x = get_var(x, mean_x)
    var_y = get_var(y, mean_y)

    cov = torch.mean((x - mean_x) * (y - mean_y))

    return 1 - 2*cov / (var_x + var_y + (mean_x - mean_y) ** 2)


class TotalLoss(nn.Module):
    def __init__(self, loss_param, num_samples_per_classes, cuda_id):
        super(TotalLoss, self).__init__()
        self.loss_param = loss_param
        self.loss_types = list(loss_param.keys())

    def forward(self, logits, targets, emb=None, emb_norm=None, step=None, summary_writer=None):
        total_loss = 0

        if 'MSE' in self.loss_types:
            total_loss += self.loss_param['MSE']['w'] * nn.MSELoss()(logits, targets)

        if "ObjectiveFunction" in self.loss_types:
            vx = logits[:, 0]
            ax = logits[:, 1]

            vy = targets[:, 0]
            ay = targets[:, 1]

            total_loss += (calc_l(vx, vy) + calc_l(ax, ay)) / 2

        # TODO: реализуйте objective function из статьи https://arxiv.org/pdf/1704.08619.pdf

        return total_loss