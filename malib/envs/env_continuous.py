import gymnasium as gym
from gymnasium import spaces
import numpy as np


class ContinuousActionEnv(object):
    """对于连续动作环境的封装"""
    def __init__(self, num_agent, env):
        self.env = env
        self.num_agent = num_agent

        self.sa_obs_dim = self.env.sa_obs_dim
        self.sa_action_dim = self.env.sa_action_dim

        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False

        self.movable = True

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        share_obs_dim = 0
        total_action_space = []
        for agent in range(self.num_agent):
            # physical action space
            u_action_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(self.sa_action_dim,), dtype=np.float32)

            if self.movable:
                total_action_space.append(u_action_space)

            # total action space
            self.action_space.append(total_action_space[0])

            # observation space
            share_obs_dim += self.sa_obs_dim
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(self.sa_obs_dim,),
                                                     dtype=np.float32))  # [-inf,inf]

        self.share_observation_space = [spaces.Box(low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32)
                                        for _ in range(self.num_agent)]
        

    def step(self, actions):
        """
        输入actions纬度假设：
        # actions shape = (5, 2, 5)
        # 5个线程的环境，里面有2个智能体，每个智能体的动作是一个one_hot的5维编码
        """

        results = self.env.step(actions)
        obs, rews, terminateds, truncated, infos = results
        return np.stack(obs), np.stack(rews), np.stack(terminateds), truncated, infos

    def reset(self):
        obs, _ = self.env.reset()
        return np.stack(obs)

    def close(self):
        pass

    def render(self, mode="rgb_array"):
        self.env.render()
        pass

    def seed(self, seed):
        pass