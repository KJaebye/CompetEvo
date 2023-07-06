import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

import competevo
import gym_compete

class BaseRunner:
    def __init__(self, cfg, logger, dtype, device, num_threads=1, training=True) -> None:
        self.cfg = cfg
        self.logger = logger
        self.dtype = dtype
        self.device = device
        self.num_threads = num_threads
        self.training = training
        self.env_name = cfg.env_name
        # dirs
        self.run_dir = logger.run_dir
        self.model_dir = logger.model_dir
        self.log_dir = logger.log_dir
        self.tb_dir = logger.tb_dir

        self.setup_env(self.env_name)

    def setup_env(self, env_name):
        self.env = gym.make(env_name, render_mode="human", rundir=self.run_dir, cfg=self.cfg)

    def setup_writer(self):
        self.writer = SummaryWriter(log_dir=self.tb_dir) if self.training else None

    def setup_learner(self):
        raise NotImplementedError
    
    def optimize(self, epoch):
        raise NotImplementedError
    
    def sample(self):
        raise NotImplementedError
    
    
