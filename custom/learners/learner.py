from lib.utils.torch import *

class Learner:
    def __init__(self, cfg, dtype, device) -> None:
        self.cfg = cfg
        self.device = device
        self.dtype = dtype

    def setup_policy(self):
        cfg = self.cfg
        self.policy_net = Transform2ActPolicy(cfg.policy_specs, self)
        to_device(self.device, self.policy_net)