from custom.models.transform2act_actor import Transform2ActPolicy
from custom.models.transform2act_critic import Transform2ActValue
from lib.utils.torch import *
from lib.utils.tools import *
from lib.rl.core import estimate_advantages
import math

def tensorfy(np_list, device=torch.device('cpu')):
    if isinstance(np_list[0], list):
        return [[torch.tensor(x).to(device) if i < 2 else x for i, x in enumerate(y)] for y in np_list]
    else:
        return [torch.tensor(y).to(device) for y in np_list]

class EvoSampler:
    def __init__(self, cfg, dtype, device, agent) -> None:
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.agent = agent
        self.setup_policy()

        self.sample_modules = [self.policy_net]
        self.running_state = None # running_state is running_mean_std

    def setup_policy(self):
        self.policy_net = Transform2ActPolicy(self.cfg.policy_specs, self.agent)
        to_device(self.device, self.policy_net)