from custom.models.transform2act_actor import Transform2ActPolicy
from custom.models.transform2act_critic import Transform2ActValue
from lib.utils.torch import *
from lib.utils.tools import *
from lib.rl.core import estimate_advantages
import math

class EvoSampler:
    def __init__(self, cfg, dtype, device, agent, is_shadow=False) -> None:
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.agent = agent

        self.flag = "evo"
        self.is_shadow = is_shadow

        self.setup_policy()

        self.sample_modules = [self.policy_net]
        self.running_state = None # running_state is running_mean_std

    def setup_policy(self):
        self.policy_net = Transform2ActPolicy(self.cfg.evo_policy_specs, self.agent)
        to_device(self.device, self.policy_net)
    
    def load_ckpt(self, model):
        self.policy_net.load_state_dict(model['policy_dict'])
        self.running_state = model['running_state']
        self.loss_iter = model['loss_iter']
