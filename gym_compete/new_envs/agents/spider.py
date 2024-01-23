from .agent import Agent
from gymnasium.spaces import Box
import numpy as np
import os

from lxml.etree import XMLParser, parse, ElementTree, Element, SubElement
from lxml import etree
from io import BytesIO

class Spider(Agent):

    def __init__(self, agent_id, xml_path=None, n_agents=2):
        if xml_path is None:
            xml_path = os.path.join(os.path.dirname(__file__), "assets", "spider_body.xml")
        super(Spider, self).__init__(agent_id, xml_path, n_agents)

        parser = XMLParser(remove_blank_text=True)
        self.tree = parse(xml_path, parser=parser)
        self.cur_xml_str = etree.tostring(self.tree, pretty_print=True).decode('utf-8')

    def set_goal(self, goal):
        self.GOAL = goal
        self.move_left = False
        if self.get_qpos()[0] > 0:
            self.move_left = True

    def before_step(self):
        self._xposbefore = self.get_body_com("torso")[0]

    def after_step(self, action):
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - self._xposbefore) / self.env.dt
        if self.move_left:
            forward_reward *= -1
        ctrl_cost = .08 * np.square(action).sum()
        cfrc_ext = self.get_cfrc_ext()
        contact_cost = 0

        qpos = self.get_qpos()
        agent_standing = qpos[2] >= 0.3
        # agent_standing = qpos[2] >= 0.3 and qpos[2] <= 1.2
        survive = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive

        reward_info = dict()
        reward_info['reward_forward'] = forward_reward
        reward_info['reward_ctrl'] = ctrl_cost
        reward_info['reward_contact'] = contact_cost
        reward_info['reward_survive'] = survive
        reward_info['reward_dense'] = reward

        terminated = not agent_standing

        return reward, terminated, reward_info


    def _get_obs(self, stage=None):
        '''
        Return agent's observations
        '''
        my_pos = self.get_qpos()

        # other_pos = self.get_other_qpos()
        other_pos = self.get_other_qpos()[:2]
        if other_pos.shape == (0,):
            # other_pos = np.zeros(2) # x and y
            other_pos = np.random.uniform(-5, 5, 2)
        
        my_vel = self.get_qvel()

        # for multiagent play
        obs = np.concatenate(
            [my_pos.flat, my_vel.flat,
             other_pos.flat]
        )

        return obs

    def set_observation_space(self):
        obs = self._get_obs()
        self.obs_dim = obs.size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = Box(low, high)

    def reached_goal(self):
        if self.n_agents == 1: return False
        xpos = self.get_body_com('torso')[0]
        if self.GOAL > 0 and xpos > self.GOAL:
            return True
        elif self.GOAL < 0 and xpos < self.GOAL:
            return True
        return False

    def reset_agent(self):
        xpos = self.get_qpos()[0]
        if xpos * self.GOAL > 0 :
            self.set_goal(-self.GOAL)
        if xpos > 0:
            self.move_left = True
        else:
            self.move_left = False
