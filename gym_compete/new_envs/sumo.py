from .multi_agent_env import MultiAgentEnv
import numpy as np
from gymnasium import spaces
import gymnasium as gym

try:
    import mujoco
except ImportError as e:
    MUJOCO_IMPORT_ERROR = e
else:
    MUJOCO_IMPORT_ERROR = None

class SumoEnv(MultiAgentEnv):

    def __init__(self, max_episode_steps=500, min_radius=None, max_radius=None, **kwargs):
        super(SumoEnv, self).__init__(**kwargs)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0
        self.GOAL_REWARD = 1000
        self.RADIUS = self.MAX_RADIUS = self.current_max_radius = max_radius
        self.MIN_RADIUS = min_radius
        self.LIM_X = [(-2, 0), (0, 2)]
        self.LIM_Y = [(-2, 2), (-2, 2)]
        self.RANGE_X = self.LIM_X.copy()
        self.RANGE_Y = self.LIM_Y.copy()
        self.arena_id = self.env_scene.geom_names.index('arena')
        self.arena_height = self.env_scene.model.geom_size[self.arena_id][1] * 2
        self._set_geom_radius()
        self.agent_contacts = False

    def _past_limit(self):
        if self._max_episode_steps <= self._elapsed_steps:
            return True
        return False

    def _past_arena(self, agent_id):
        xy = self.agents[agent_id].get_qpos()[:2]
        r = np.sum(xy ** 2) ** 0.5
        # print("Agent", agent_id, "at", r)
        if r > self.RADIUS:
            return True
        return False

    def _is_fallen(self, agent_id, limit=0.5):
        if self.agents[agent_id].team == 'ant':
            limit = 0.3
        limit = limit + self.arena_height
        return bool(self.agents[agent_id].get_qpos()[2] <= limit)

    def _is_standing(self, agent_id, limit=0.9):
        limit = limit + self.arena_height
        return bool(self.agents[agent_id].get_qpos()[2] > limit)

    def get_agent_contacts(self):
        mjcontacts = self.env_scene.data.contact
        ncon = self.env_scene.data.ncon
        contacts = []
        for i in range(ncon):
            ct = mjcontacts[i]
            g1 , g2 = ct.geom1, ct.geom2
            g1 = self.env_scene.geom_names[g1]
            g2 = self.env_scene.geom_names[g2]
            if g1.find('agent') >= 0 and g2.find('agent') >= 0:
                if g1.find('agent0') >= 0:
                    if g2.find('agent1') >= 0 and ct.dist < 0:
                        contacts.append((g1, g2, ct.dist))
                elif g1.find('agent1') >= 0:
                    if g2.find('agent0') >= 0 and ct.dist < 0:
                        contacts.append((g1, g2, ct.dist))
        return contacts

    def goal_rewards(self, infos=None, agent_dones=None):
        self._elapsed_steps += 1
        goal_rews = [0. for _ in range(self.n_agents)]

        fallen = [self._is_fallen(i) for i in range(self.n_agents)]
        timeup = self._past_limit()
        past_arena = [self._past_arena(i) for i in range(self.n_agents)]
        done = False

        agent_contacts = self.get_agent_contacts()
        if len(agent_contacts) > 0:
            # print('Detected contacts:', agent_contacts)
            self.agent_contacts = True

        if any(fallen):
            done = True
            for j in range(self.n_agents):
                if fallen[j]:
                    # print('Agent', j, 'fallen')
                    goal_rews[j] -= self.GOAL_REWARD
                elif self.agent_contacts:
                    goal_rews[j] += self.GOAL_REWARD
                    infos[j]['winner'] = True
                
        elif any(past_arena):
            done = True
            for j in range(self.n_agents):
                if past_arena[j]:
                    # print('Agent', j, 'past arena')
                    goal_rews[j] -= self.GOAL_REWARD
                elif self.agent_contacts:
                    goal_rews[j] += self.GOAL_REWARD
                    infos[j]['winner'] = True
        elif timeup:
            for j in range(self.n_agents):
                goal_rews[j] -= self.GOAL_REWARD
        done = timeup or done

        return goal_rews, done

    def _set_observation_space(self):
        ob_spaces_limits = []
        # nextra = 3 + self.n_agents - 1
        nextra = 4
        for i in range(self.n_agents):
            s = self.agents[i].observation_space.shape[0]
            h = np.ones(s+nextra) * np.inf
            l = -h
            ob_spaces_limits.append((l, h))
        self.observation_space = spaces.Tuple(
            [spaces.Box(l, h) for l,h in ob_spaces_limits]
        )

    def _get_obs(self):
        obs = []
        dists = []
        for i in range(self.n_agents):
            xy = self.agents[i].get_qpos()[:2]
            r = np.sqrt(np.sum(xy**2))
            d = self.RADIUS - r
            dists.append(d)
        for i in range(self.n_agents):
            ob = self.agents[i]._get_obs()
            mydist = np.asarray(dists[i])
            if self.n_agents == 1:
                other_dist = np.asarray(self.RADIUS)
            elif self.n_agents == 2:
                other_dist = np.asarray(dists[1-i])
            else:
                other_dist = np.asarray([dists[j] for j in range(self.n_agents) if j != i])
            ob = np.concatenate(
                [ob.flat, np.asarray(self.RADIUS).flat,
                 mydist.flat, other_dist.flat,
                np.asarray(self._max_episode_steps - self._elapsed_steps).flat
                ]
            )
            obs.append(ob)
        return tuple(obs)

    def _reset_max_radius(self, version):
        decay_func_r = lambda x: 0.1 * np.exp(0.001 * x)
        vr = decay_func_r(version)
        self.current_max_radius = min(self.MAX_RADIUS, self.MIN_RADIUS + vr)

    def _reset_radius(self):
        self.RADIUS = np.random.uniform(self.MIN_RADIUS, self.current_max_radius)

    def _set_geom_radius(self):
        gs = self.env_scene.model.geom_size.copy()
        gs[self.arena_id][0] = self.RADIUS
        self.env_scene.model.__setattr__('geom_size', gs)
        mujoco.mj_forward(self.env_scene.model, self.env_scene.data)

    # def _reset_agents(self):
    #     min_gap = 0.3 + self.MIN_RADIUS / 2
    #     ###########################################################################
    #     # random_pos_flag is a flag that determine which side should agent rebirth
    #     # This is very important because one agent trained at a fixed side might 
    #     # become confused when put it to another side. Therefore, the best solution
    #     # is to sample more balanced data, especially in a decentralised training.
    #     idx = np.random.randint(0, 2)
    #     random_pos_flag = [idx, 1-idx]
    #     ###########################################################################
    #     for i in range(self.n_agents):
    #         if random_pos_flag[i] % 2 == 0:
    #             x = np.random.uniform(-self.RADIUS + min_gap, -0.3)
    #             y_lim = np.sqrt(self.RADIUS**2 - x**2)
    #             y = np.random.uniform(-y_lim + min_gap, y_lim - min_gap)
    #         else:
    #             x = np.random.uniform(0.3, self.RADIUS - min_gap)
    #             y_lim = np.sqrt(self.RADIUS**2 - x**2)
    #             y = np.random.uniform(-y_lim + min_gap, y_lim - min_gap)
    #         self.agents[i].set_xyz((x,y,None))

    def _reset_agents(self):
        min_gap = 0.3 + self.MIN_RADIUS / 2
        for i in range(self.n_agents):
            if i % 2 == 0:
                x = np.random.uniform(-self.RADIUS + min_gap, -0.3)
                y_lim = np.sqrt(self.RADIUS**2 - x**2)
                y = np.random.uniform(-y_lim + min_gap, y_lim - min_gap)
            else:
                x = np.random.uniform(0.3, self.RADIUS - min_gap)
                y_lim = np.sqrt(self.RADIUS**2 - x**2)
                y = np.random.uniform(-y_lim + min_gap, y_lim - min_gap)
            self.agents[i].set_xyz((x,y,None))
            # print('setting agent', i, 'at', self.agents[i].get_qpos()[:3])

    def _reset(self, version=None):
        self._elapsed_steps = 0
        self.agent_contacts = False
        # self.RADIUS = self.START_RADIUS
        if version is not None:
            self._reset_max_radius(version)
        self._reset_radius()
        self._set_geom_radius()
        self.env_scene.reset()
        self._reset_agents()
        ob = self._get_obs()
        return ob, {}

    def reset(self, margins=None, version=None):
        ob, info = self._reset(version=version)
        if margins:
            for i in range(self.n_agents):
                self.agents[i].set_margin(margins[i])
        return ob, info
    
    def step(self, actions):
        obses, rews, terminateds, truncated, infos = self._step(actions)
        if self._past_limit():
            return obses, rews, terminateds, True, infos
        
        return obses, rews, terminateds, truncated, infos
