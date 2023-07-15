from collections import defaultdict
from lib.utils.torch import LongTensor
import torch.nn as nn
from lib.rl.core.distributions import Categorical, DiagGaussian
from lib.rl.core.policy import Policy
from lib.rl.core.running_norm import RunningNorm
from lib.models.mlp import MLP
from lib.utils.math import *

from custom.utils.tools import *

class NormalPolicy(Policy):
    def __init__(self, cfg, env):
        super().__init__()
        self.type = 'gaussian'
        self.cfg = cfg
        self.env = env
        self.control_state_dim = env.observation_space[0].shape[0]
        self.control_action_dim = env.action_space[0].shape[0]

        self.control_norm = RunningNorm(self.control_state_dim)
        cur_dim = self.control_state_dim
        if 'control_pre_mlp' in cfg:
            self.control_pre_mlp = MLP(cur_dim, cfg['control_pre_mlp']['hdims'], cfg['htype'])
            cur_dim = self.control_pre_mlp.out_dim
        else:
            self.control_pre_mlp = None

        self.control_mlp = MLP(cur_dim, cfg['control_mlp']['hdims'], cfg['htype'])
        cur_dim = self.control_mlp.out_dim

        self.control_action_mean = nn.Linear(cur_dim, self.control_action_dim)
        init_fc_weights(self.control_action_mean)
        self.control_action_log_std = nn.Parameter(
            torch.ones(1, self.control_action_dim) * cfg['control_log_std'], 
            requires_grad=not cfg['fix_control_std'])
    
    def forward(self, x):
        x = self.control_norm(x)
        if self.control_pre_mlp is not None:
            x = self.control_pre_mlp(x)
        x = self.control_mlp(x)
        control_action_mean = self.control_action_mean(x)
        control_action_std = self.control_action_log_std.exp()
        control_dist = DiagGaussian(control_action_mean, control_action_std)
        return control_dist
    
    def select_action(self, x, mean_action=False):
        dist = self.forward(x)
        action = dist.mean_sample() if mean_action else dist.sample()
        return action