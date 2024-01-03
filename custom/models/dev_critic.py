"""
    This uses only one policy to learn scale vector and control.
"""
import numpy as np
import torch.nn as nn
import torch
from lib.rl.core.running_norm import RunningNorm
from lib.models.mlp import MLP

class DevValue(nn.Module):
    def __init__(self, cfg, agent):
        super(DevValue, self).__init__()
        self.cfg = cfg
        self.agent = agent
        # dimension define
        self.scale_state_dim = agent.scale_state_dim
        self.sim_action_dim = agent.sim_action_dim
        self.sim_obs_dim = agent.sim_obs_dim
        self.action_dim = agent.action_dim
        self.state_dim = agent.state_dim

        self.norm = RunningNorm(self.state_dim)
        cur_dim = self.state_dim
        self.mlp = MLP(cur_dim,
                       hidden_dims=self.cfg.dev_value_specs['mlp'],
                       activation=self.cfg.dev_value_specs['htype'])
        cur_dim = self.mlp.out_dim
        self.value_head = nn.Linear(cur_dim, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def batch_data(self, x):
        stage_ind, scale_state, sim_obs = zip(*x)
        scale_state = torch.stack(scale_state, 0)
        stage_ind = torch.stack(stage_ind, 0)
        sim_obs = torch.stack(sim_obs, 0)
        return stage_ind, scale_state, sim_obs

    def forward(self, x):
        stage_ind, scale_state, sim_obs = self.batch_data(x)
        x = torch.cat((stage_ind, scale_state, sim_obs), -1)
        x = self.norm(x)
        x = self.mlp(x)
        value = self.value_head(x)
        return value