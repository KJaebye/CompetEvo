from custom.models.normal_actor import NormalPolicy
from lib.utils.torch import *
from lib.utils.tools import *
    
class Sampler:
    """
        Sampler is only for loading models and sampling.
        Thus, there are only policy network definition and no optimizer.
    """
    def __init__(self, cfg, dtype, device, agent) -> None:
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.agent = agent

        self.__setup_policy()
        self.__setup_reward_scaling()

        self.sample_modules = [self.policy_net]
        self.running_state = None # running_state is running_mean_std
    
    def load_ckpt(self, model):
        self.policy_net.load_state_dict(model['policy_dict'])
        self.running_state = model['running_state']
        self.reward_scaling = model['reward_scaling']

    ###################################################################################
    ############################## Setup part #########################################
    def __setup_policy(self):
        cfg = self.cfg
        self.policy_net = NormalPolicy(cfg.policy_specs, self.agent)
        to_device(self.device, self.policy_net)
    
    def __setup_reward_scaling(self):
        if self.cfg.use_reward_scaling:
            from lib.rl.core.running_norm import RewardScaling
            self.reward_scaling = RewardScaling(shape=1, gamma=self.cfg.gamma)
            self.reward_scaling.reset()
        else:
            self.reward_scaling = None