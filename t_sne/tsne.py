# -*- coding: utf-8 -*-
import torch
import torch.autograd
from torch import nn
import numpy as np

class TSNE(nn.Module):
    def __init__(self, n_points, n_dim):
        self.n_points = n_points
        self.n_dim = n_dim
        super(TSNE, self).__init__()
        # Logit of datapoint-to-topic weight
        self.logits = nn.Embedding(n_points, n_dim)

    def forward(self, pij, i, j):
        # TODO: реализуйте вычисление матрицы сходства для точек отображения и расстояние Кульбака-Лейблера
        # pij - значения сходства между точками данных
        # i, j - индексы точек  
        i = i.long()
        j = j.long()
        print(i, j)
        embd_points = self.logits(torch.from_numpy(np.arange(self.n_points)).long())
        q_exp = torch.exp(-((embd_points[i] - embd_points[j])**2))
        t_i = embd_points[i]
        try:
            embd_points = torch.hstack((embd_points[:i],embd_points[i+1:]))
        except:
            embd_points = embd_points[:i]
        q_sum = torch.sum(torch.exp(-((embd_points - t_i)**2)))
        qij = q_exp / q_sum
        loss_kld = pij*torch.log(pij/qij)
        return loss_kld.sum()

    def __call__(self, *args):
        return self.forward(*args)
