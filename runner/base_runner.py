import gymnasium as gym
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import competevo
import gym_compete

class BaseRunner:
    def __init__(self, cfg, logger, dtype, device, num_threads=1, training=True, ckpt=0) -> None:
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

        self.t_start = time.time()

        self.noise_rate = 1.0

        self.setup_env(self.env_name)
        self.setup_writer()
        self.setup_learner()

        if ckpt != 0 or not training:
            self.load_checkpoint(ckpt)

    def setup_env(self, env_name):
        if self.training:
            # self.env = gym.make(env_name, render_mode="human")
            self.env = gym.make(env_name)
        else:
            self.env = gym.make(env_name, render_mode="human")

    def setup_writer(self):
        self.writer = SummaryWriter(log_dir=self.tb_dir) if self.training else None

    def setup_learner(self):
        raise NotImplementedError
    
    def optimize(self, epoch):
        raise NotImplementedError
    
    def sample(self):
        raise NotImplementedError
    
    def load_checkpoint(self, ckpt):
        raise NotImplementedError
    
    def seed_worker(self, pid):
        if pid > 0:
            torch.manual_seed(torch.randint(0, 5000, (1,)) * pid)
            # if hasattr(self.env, 'np_random'):
            #     self.env.np_random.seed(self.env.np_random.randint(5000) * pid)
    