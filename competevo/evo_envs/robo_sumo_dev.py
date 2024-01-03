from .multi_dev_agent_env import MultiDevAgentEnv
import numpy as np
from gymnasium import spaces
import gymnasium as gym

try:
    import mujoco
except ImportError as e:
    MUJOCO_IMPORT_ERROR = e
else:
    MUJOCO_IMPORT_ERROR = None

class RoboSumoDevEnv(MultiDevAgentEnv):
    WIN_REWARD = 2000.
    DRAW_PENALTY = -1000.
    STAY_IN_CENTER_COEF = 0.1
    # MOVE_TO_CENTER_COEF = 0.1
    MOVE_TO_OPP_COEF = 10. # 0.1
    PUSH_OUT_COEF = 10.0

    def __init__(self, max_episode_steps=500, min_radius=None, max_radius=None, **kwargs):
        super(RoboSumoDevEnv, self).__init__(**kwargs)
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
        if self.agents[agent_id].team == 'bug':
            limit = 0.3
        if self.agents[agent_id].team == 'spider':
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

    def _set_observation_space(self):
        ob_spaces_limits = []
        # nextra = 3 + self.n_agents - 1
        nextra = 0
        for i in range(self.n_agents):
            s = self.agents[i].observation_space.shape[0]
            h = np.ones(s+nextra) * np.inf
            l = -h
            ob_spaces_limits.append((l, h))
        self.observation_space = spaces.Tuple(
            [spaces.Box(l, h) for l,h in ob_spaces_limits]
        )

    def _get_obs(self):
        return tuple([agent._get_obs(self.stage) for agent in self.agents.values()])

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

    def _reset_agents(self):
        min_gap = 0.3 + self.MIN_RADIUS / 2
        ###########################################################################
        # random_pos_flag is a flag that determine which side should agent rebirth
        # This is very important because one agent trained at a fixed side might 
        # become confused when put it to another side. Therefore, the best solution
        # is to sample more balanced data, especially in a decentralised training.
        idx = np.random.randint(0, 2)
        random_pos_flag = [idx, 1-idx]
        ###########################################################################
        for i in range(self.n_agents):
            if random_pos_flag[i] % 2 == 0:
                x = np.random.uniform(-self.RADIUS + min_gap, -0.3)
                y_lim = np.sqrt(self.RADIUS**2 - x**2)
                y = np.random.uniform(-y_lim + min_gap, y_lim - min_gap)
            else:
                x = np.random.uniform(0.3, self.RADIUS - min_gap)
                y_lim = np.sqrt(self.RADIUS**2 - x**2)
                y = np.random.uniform(-y_lim + min_gap, y_lim - min_gap)
            self.agents[i].set_xyz((x,y,None))


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
        if self.agents is not None:
            del self.agents
        if hasattr(self, "env_scene"):
            self.env_scene.close()
            del self.env_scene
        self.stage = 'attribute_transform'

        # reload from init files, to avoid multiprocessing w/r issue
        self.setup_agents(self.cfg, self.agent_names, self.agent_map, self.agent_args)
        self.reload_init_mujoco_env()
        if margins:
            for i in range(self.n_agents):
                self.agents[i].set_margin(margins[i])
        return self._reset()

    def _step(self, actions):
        self._elapsed_steps += 1
        for i in range(self.n_agents):
            self.agents[i].before_step()
        
        self.env_scene.simulate(actions)

        dones = [False for _ in range(self.n_agents)]
        rewards = [0. for _ in range(self.n_agents)]
        infos = [{} for _ in range(self.n_agents)]
        game_done = False

        for i in range(self.n_agents):
            infos[i]['ctrl_reward'], infos[i]['alive_reward'] = self.agents[i].after_step(actions[i])

        for i, agent in self.agents.items():
            self_xyz = agent.get_qpos()[:3]
            # Loose penalty
            infos[i]['lose_penalty'] = 0.
            if (self_xyz[2] < 0.29 + self.arena_height or 
                np.sqrt(np.sum(self_xyz[:2]**2)) > self.RADIUS):
                infos[i]['lose_penalty'] = - self.WIN_REWARD
                dones[i] = True
                game_done = True
            # Win reward
            infos[i]['win_reward'] = 0.
            for j, opp in self.agents.items():
                if i == j: continue
                opp_xyz = opp.get_qpos()[:3]
                if (opp_xyz[2] < 0.29 + self.arena_height or 
                    np.sqrt(np.sum(opp_xyz[:2]**2)) > self.RADIUS):
                    infos[i]['win_reward'] += self.WIN_REWARD
                    infos[i]['winner'] = True
                    dones[i] = True
                    game_done = True
            infos[i]['reward_parse'] = \
                infos[i]['win_reward'] + infos[i]['lose_penalty']
            # Draw penalty
            if self._max_episode_steps <= self._elapsed_steps:
                infos[i]['reward_parse'] += self.DRAW_PENALTY
                dones[i] = True
            # Move to opponent(s) and push them out of center
            infos[i]['move_to_opp_reward'] = 0.
            infos[i]['push_opp_reward'] = 0.
            for j, opp in self.agents.items():
                if i == j: continue
                infos[i]['move_to_opp_reward'] += \
                    self._comp_move_reward(agent, opp.posafter)
                infos[i]['push_opp_reward'] += \
                    self._comp_push_reward(agent, opp.posafter)
            # Stay in center reward (unused)
            # infos[i]['stay_in_center'] = self._comp_stay_in_center_reward(agent)
            # Contact rewards and penalties (unused)
            # infos[i]['contact_reward'] = self._comp_contact_reward(agent)
            # Reward shaping
            infos[i]['reward_dense'] = infos[i]['alive_reward'] + \
                infos[i]['ctrl_reward'] + \
                infos[i]['push_opp_reward'] + \
                infos[i]['move_to_opp_reward']
            # Add up rewards
            rewards[i] = infos[i]['reward_parse'] + infos[i]['reward_dense']
            # print(i, infos[i]['ctrl_reward'], infos[i]['push_opp_reward'], infos[i]['move_to_opp_reward'])

        rewards = tuple(rewards)
        terminateds = self._get_done(dones, game_done)
        infos = tuple(infos)
        obses = self._get_obs()

        return obses, rewards, terminateds, False, infos
    

    def step(self, actions):
        self.cur_t += 1
        # scale transform stage
        if self.stage == 'attribute_transform':
            terminateds = tuple([False, False])
            infos = []
            cur_xml_strs = []
            for i in range(self.n_agents):
                design_params = actions[i]
                self.agents[i].set_design_params(design_params)

                info = {'use_transform_action': True, 
                        'stage': 'attribute_transform',
                        'reward_parse': 0,
                        'reward_dense': 0,
                        'design_params': design_params[:self.agents[i].scale_state_dim],
                        }
                infos.append(info)

                cur_xml_str = self.agents[i].cur_xml_str
                cur_xml_strs.append(cur_xml_str)

            try:
                self.load_tmp_mujoco_env(self.world_xml_path, cur_xml_strs, \
                                     self.agent_scopes, self.ini_pos, self.ini_euler, self.rgb, **self.kwargs)
                # print(self._env_xml_str)
            except:
                print("Warning: Errors occur when loading xml files.")
                terminateds = tuple([True, True])
            
            self.transit_execution()

            obses = self._get_obs()
            rews = tuple([0., 0.])

            self._reset()
            return obses, rews, terminateds, False, infos
        # execution
        else:
            actions = [actions[i][-agent.sim_action_dim:] for i, agent in self.agents.items()]
            obses, rews, terminateds, truncated, infos = self._step(actions)
        if self._past_limit():
            return obses, rews, terminateds, True, infos
        
        return obses, rews, terminateds, truncated, infos
    

    def _comp_move_reward(self, agent, target):
        move_vec = (agent.posafter - agent.posbefore) / self.dt
        direction = target - agent.posbefore
        direction /= np.linalg.norm(direction)
        return max(np.sum(move_vec * direction), 0.) * self.MOVE_TO_OPP_COEF
    
    def _comp_push_reward(self, agent, target):
        dist_to_center = np.linalg.norm(target)
        return - self.PUSH_OUT_COEF * np.exp(-dist_to_center)