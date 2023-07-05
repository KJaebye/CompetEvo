# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Class Value
#   @author: Modified from khrylib by Ye Yuan, modified by Kangyao Huang
#   @created date: 28.Nov.2022
# ------------------------------------------------------------------------------------------------------------------- #

import torch.nn as nn
import torch
from lib.core.running_norm import RunningNorm

class Value(nn.Module):
    def __init__(self, state_dim, hidden_size=(128, 128), activation='tanh'):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.value_norm = RunningNorm(state_dim)
        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.value_head = nn.Linear(last_dim, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = self.value_norm(x)
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        value = self.value_head(x)
        return value
