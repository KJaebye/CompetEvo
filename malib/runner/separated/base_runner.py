    
import time
import os
import numpy as np
from itertools import chain
import torch
from tensorboardX import SummaryWriter

from malib.utils.separated_buffer import SeparatedReplayBuffer
from malib.utils.util import update_linear_schedule

from config.config import cfg

def _t2n(x):
    return x.detach().cpu().numpy()

class Runner(object):
    def __init__(self, config):

        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']

        # parameters
        self.env_name = cfg.ENV_NAME
        self.algo = cfg.ALGO
        self.use_centralized_V = cfg.NETWORK.USE_CENTRALIZED_V
        self.use_obs_instead_of_state = cfg.USE_OBS_INSTEAD_OF_STATE
        self.num_env_steps = cfg.EMAT.NUM_ENV_STEPS
        self.episode_length = cfg.ENV.EPISODE_LENGTH
        self.n_rollout_threads = cfg.EMAT.N_ROLLOUT_THREADS
        self.n_eval_rollout_threads = cfg.EMAT.N_EVAL_ROLLOUT_THREADS
        self.use_linear_lr_decay = cfg.MAPPO.USE_LINEAR_LR_DECAY
        self.hidden_size = cfg.NETWORK.HIDDEN_SIZE
        self.use_render = cfg.USE_RENDER
        self.recurrent_N = cfg.NETWORK.RECURRENT_N

        # interval
        self.save_interval = cfg.CHECKPOINT_PERIOD
        self.use_eval = cfg.USE_EVAL
        self.eval_interval = cfg.EVAL_PERIOD
        self.log_interval = cfg.LOG_PERIOD

        # dir
        self.model_dir = config["model_dir"]

        # by default, use_render is False so that /logs, /models dirs can be created.
        # use_render is set to True in display.py manually.
        if self.use_render:
            self.run_dir = config["run_dir"]
            path = os.path.join(self.run_dir, 'video')
            path = os.path.abspath(path)
            if not os.path.exists(path):
                os.makedirs(path)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir + '/logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir, exist_ok=False)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir + '/models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir, exist_ok=False)

        from malib.algorithms.algorithm.r_mappo import RMAPPO as TrainAlgo
        from malib.algorithms.algorithm.rMAPPOPolicy import RMAPPOPolicy as Policy


        self.policy = []
        for agent_id in range(self.num_agents):
            if self.use_centralized_V:
                share_observation_space = self.envs.share_observation_space[agent_id]
            else:
                share_observation_space = self.envs.observation_space[agent_id]

            # policy network
            po = Policy(self.envs.observation_space[agent_id],
                        share_observation_space,
                        self.envs.action_space[agent_id],
                        device=self.device)
            self.policy.append(po)

        if self.model_dir is not None:
            self.restore()

        self.trainer = []
        self.buffer = []
        for agent_id in range(self.num_agents):
            # algorithm
            tr = TrainAlgo(self.policy[agent_id], device=self.device)
            # buffer
            share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else self.envs.observation_space[agent_id]
            bu = SeparatedReplayBuffer(self.envs.observation_space[agent_id],
                                       share_observation_space,
                                       self.envs.action_space[agent_id])
            self.buffer.append(bu)
            self.trainer.append(tr)
            
    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self):
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            next_value = self.trainer[agent_id].policy.get_values(self.buffer[agent_id].share_obs[-1], 
                                                                self.buffer[agent_id].rnn_states_critic[-1],
                                                                self.buffer[agent_id].masks[-1])
            next_value = _t2n(next_value)
            self.buffer[agent_id].compute_returns(next_value, self.trainer[agent_id].value_normalizer)

    def train(self):
        train_infos = []
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_training()
            train_info = self.trainer[agent_id].train(self.buffer[agent_id])
            train_infos.append(train_info)       
            self.buffer[agent_id].after_update()

        return train_infos

    def save(self):
        for agent_id in range(self.num_agents):
            policy_actor = self.trainer[agent_id].policy.actor
            torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_agent" + str(agent_id) + ".pt")
            policy_critic = self.trainer[agent_id].policy.critic
            torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_agent" + str(agent_id) + ".pt")

    def restore(self):
        for agent_id in range(self.num_agents):
            policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_agent' + str(agent_id) + '.pt')
            self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
            policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_agent' + str(agent_id) + '.pt')
            self.policy[agent_id].critic.load_state_dict(policy_critic_state_dict)

    def log_train(self, train_infos, total_num_steps): 
        for agent_id in range(self.num_agents):
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for agent_id in range(self.num_agents):
            for k, v in env_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)
