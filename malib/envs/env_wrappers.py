"""
# @Time    : 2021/7/1 8:44 上午
# @Author  : hezhiqiang01
# @Email   : hezhiqiang01@baidu.com
# @File    : evo_env_wrappers.py
Modified from OpenAI Baselines code to work with multi-agent envs
"""

import numpy as np

from config.config import cfg

# single envs
class DummyVecEnv():
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.env = self.envs[0]
        self.num_envs = len(env_fns)
        self.observation_space = self.env.observation_space
        self.share_observation_space = self.env.share_observation_space
        self.action_space = self.env.action_space
        self.actions = None

    def step(self, actions):
        """
        Step the environments synchronously.
        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, rews, terminateds, truncateds, infos = map(np.array, zip(*results))

        for (i, terminated) in enumerate(terminateds):
            if 'bool' in terminated.__class__.__name__:
                if terminated:
                    obs[i] = self.envs[i].reset()
            else:
                if np.all(terminated):
                    obs[i] = self.envs[i].reset()

        self.actions = None
        return obs, rews, terminateds, truncateds, infos

    def reset(self):
        obs = [env.reset() for env in self.envs]
        return np.array(obs)

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode="human"):
        if mode == "rgb_array":
            return np.array([env.render(mode=mode) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError