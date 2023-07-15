import torch.nn as nn
import torch
import numpy as np
from lib.models.mlp import MLP
from lib.rl.core.running_norm import RunningNorm
from custom.utils.tools import *
from custom.models.gnn import GNNSimple

class NormalValue(nn.Module):
    def __init__(self, cfg, env):
        super().__init__()
        self.cfg = cfg
        self.env = env
        self.state_dim = env.observation_space[0].shape[0]
        self.norm = RunningNorm(self.state_dim)
        cur_dim = self.state_dim
        if 'pre_mlp' in cfg:
            self.pre_mlp = MLP(cur_dim, cfg['pre_mlp'], cfg['htype'])
            cur_dim = self.pre_mlp.out_dim
        else:
            self.pre_mlp = None
        if 'mlp' in cfg:
            self.mlp = MLP(cur_dim, cfg['mlp'], cfg['htype'])
            cur_dim = self.mlp.out_dim
        else:
            self.mlp = None
        self.value_head = nn.Linear(cur_dim, 1)
        init_fc_weights(self.value_head)
    
    def forward(self, x):
        x = self.norm(x)
        if self.pre_mlp is not None:
            x = self.pre_mlp(x)
        if self.mlp is not None:
            x = self.mlp(x)
        value = self.value_head(x)
        return value