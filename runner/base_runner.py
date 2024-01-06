import gymnasium as gym
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import competevo
import gym_compete
import os
import shutil

class BaseRunner:
    def __init__(self, cfg, logger, dtype, device, num_threads=1, training=True, ckpt_dir=None, ckpt=0) -> None:
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

        if (ckpt != 0 and ckpt[0] != 0) or not training:
            self.load_checkpoint(ckpt_dir, ckpt)

            def copy_checkpoint(ckpt_dir, ckpt, model_dir, new_filename='epoch_0000.p'):
                # 确保目标文件夹存在，如果不存在则创建
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)

                # 构建源文件路径
                source_file = os.path.join(ckpt_dir, ckpt)

                # 构建目标文件路径，使用指定的新文件名
                target_file = os.path.join(model_dir, new_filename)

                try:
                    # 复制文件
                    shutil.copyfile(source_file, target_file)
                    self.logger.info(f"Checkpoint {source_file} copied successfully to {target_file}")
                except FileNotFoundError:
                    self.logger.critical(f"Source file {source_file} not found.")
                except Exception as e:
                    self.logger.critical(f"Error copying checkpoint: {e}")

            if training: # consider the load ckpt is epoch_0
                ckpt0 = ckpt[0]
                if isinstance(ckpt0, str):
                    ckpt0 = ckpt0 + '.p'
                elif isinstance(ckpt0, int):
                    ckpt0 = 'epoch_%04d.p' % (ckpt0)

                ckpt1 = ckpt[1]
                if isinstance(ckpt1, str):
                    ckpt1 = ckpt1 + '.p'
                elif isinstance(ckpt1, int):
                    ckpt1 = 'epoch_%04d.p' % (ckpt1)

                copy_checkpoint(ckpt_dir+'/agent_0', ckpt0, self.model_dir+'/agent_0')
                copy_checkpoint(ckpt_dir+'/agent_1', ckpt1, self.model_dir+'/agent_1')

    def setup_env(self, env_name):
        if self.training:
            # self.env = gym.make(env_name, cfg=self.cfg, render_mode="human")
            self.env = gym.make(env_name, cfg=self.cfg)
        else:
            # self.env = gym.make(env_name, cfg=self.cfg, render_mode="human")
            self.env = gym.make(env_name, cfg=self.cfg)

    def setup_writer(self):
        self.writer = SummaryWriter(log_dir=self.tb_dir) if self.training else None

    def setup_learner(self):
        raise NotImplementedError
    
    def optimize(self, epoch):
        raise NotImplementedError
    
    def sample(self):
        raise NotImplementedError
    
    def load_checkpoint(self, ckpt_dir, ckpt):
        raise NotImplementedError
    
    def seed_worker(self, pid):
        if pid > 0:
            torch.manual_seed(torch.randint(0, 5000, (1,)) * pid)
            # if hasattr(self.env, 'np_random'):
            #     self.env.np_random.seed(self.env.np_random.randint(5000) * pid)
    