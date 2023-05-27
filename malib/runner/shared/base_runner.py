import os
import numpy as np
import torch
from tensorboardX import SummaryWriter
from malib.utils.shared_buffer import SharedReplayBuffer

from config.config import cfg

def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()

class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """
    def __init__(self, config):

        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']       

        # parameters
        self.env_name = cfg.ENV_NAME
        self.algo = cfg.ALGO
        self.use_centralized_V = cfg.NETWORK.USE_CENTRALIZED_V
        self.use_obs_instead_of_state = cfg.USE_OBS_INSTEAD_OF_STATE
        self.num_env_steps = cfg.EMAT.NUM_ENV_STEPS
        self.episode_length = cfg.ENV.EPISODE_LENGTH
        self.n_rollout_threads = cfg.EMAT.N_ROLLOUT_THREADS
        self.n_eval_rollout_threads = cfg.EMAT.N_EVAL_ROLLOUT_THREADS
        self.n_render_rollout_threads = cfg.EMAT.N_RENDER_ROLLOUT_THREADS
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

        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]

        # policy network
        self.policy = Policy(self.envs.observation_space[0],
                            share_observation_space,
                            self.envs.action_space[0],
                            device=self.device)

        if self.model_dir is not None:
            self.restore()

        # algorithm
        self.trainer = TrainAlgo(self.policy, device=self.device)
        
        # buffer
        self.buffer = SharedReplayBuffer(self.num_agents,
                                        self.envs.observation_space[0],
                                        share_observation_space,
                                        self.envs.action_space[0])

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                np.concatenate(self.buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)
    
    def train(self):
        """Train policies with data in buffer. """
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)      
        self.buffer.after_update()
        return train_infos

    def save(self):
        """Save policy's actor and critic networks."""
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor.pt")
        policy_critic = self.trainer.policy.critic
        torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic.pt")

    def restore(self):
        """Restore policy's networks from a saved model."""
        policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor.pt')
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        if not self.use_render:
            policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic.pt')
            self.policy.critic.load_state_dict(policy_critic_state_dict)
 
    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training envs steps.
        """
        for k, v in train_infos.items():
            self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """
        Log envs info.
        :param env_infos: (dict) information about envs state.
        :param total_num_steps: (int) total number of training envs steps.
        """
        for k, v in env_infos.items():
            if len(v)>0:
                self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
