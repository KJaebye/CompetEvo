from .agent import Agent
from .spider import Spider
from gymnasium.spaces import Box
import numpy as np
import six


def mass_center(mass, xpos):
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]


class SpiderFighter(Spider):
    CFRC_CLIP = 100.

    COST_COEFS = {
        'ctrl': 1e-1,
        # 'pain': 1e-4,
        # 'attack': 1e-1,
    }

    JNT_NPOS = {
        0: 7,
        1: 4,
        2: 1,
        3: 1,
    }

    def __init__(self, agent_id, xml_path=None, team='spider'):
        super(SpiderFighter, self).__init__(agent_id, xml_path)
        self.team = team
    
    def before_step(self):
        self.posbefore = self.get_qpos()[:2].copy()

    def set_env(self, env):
        super(SpiderFighter, self).set_env(env)
        self.arena_id = self.env.geom_names.index('arena')
        self.arena_height = self.env.model.geom_size[self.arena_id][1] * 2

    def after_step(self, action):
        """ RoboSumo design.
        """
        self.posafter = self.get_qpos()[:2].copy()
        # Control cost
        control_reward = - self.COST_COEFS['ctrl'] * np.square(action).sum()

        alive_reward = 2.0

        return control_reward, alive_reward

    def _get_obs(self):
        '''
        Return agent's observations
        '''
        # Observe self
        self_forces = np.abs(np.clip(
            self.get_cfrc_ext(), -self.CFRC_CLIP, self.CFRC_CLIP))
        obs  = [
            self.get_qpos().flat,           # self all positions
            self.get_qvel().flat,           # self all velocities
            self_forces.flat,               # self all forces
        ]
        
        # Observe opponents
        other_qpos = self.get_other_qpos()
        obs.extend([
            other_qpos[:7].flat,    # opponent torso position
        ])

        torso_xmat = self.get_torso_xmat()
        obs.extend([
            torso_xmat.flat,
        ])

        obs = np.concatenate(obs)
        assert np.isfinite(obs).all(), "Spider observation is not finite!!"
        return obs