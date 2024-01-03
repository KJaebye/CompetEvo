import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from gym_compete.new_envs.agents import *
from gym_compete.new_envs.multi_agent_scene import MultiAgentScene
from competevo.evo_envs.evo_utils import create_multiagent_xml, create_multiagent_xml_str
from competevo.evo_envs.agents import *
from competevo.evo_envs.multi_evo_agent_scene import MultiEvoAgentScene
import os
import six


class MultiAnimalEnv(MujocoEnv):
    '''
    A multi-agent environment supporting morph EVO that consists of some number of EVO Agent and
    a MultiAgentScene
    '''
    AGENT_MAP = {
        'ant': (
            os.path.join(os.path.dirname(__file__), "assets", "ant_body.xml"),
            Ant
        ),
        'dev_ant': (
            os.path.join(os.path.dirname(__file__), "assets", "evo_ant_body_base2.xml"),
            DevAnt
        ),
        'evo_ant': (
            os.path.join(os.path.dirname(__file__), "assets", "evo_ant_body_base1.xml"),
            EvoAnt
        ),
        'ant_fighter': (
            os.path.join(os.path.dirname(__file__), "assets", "ant_body.xml"),
            AntFighter
        ),
        'robo_ant_fighter': (
            os.path.join(os.path.dirname(__file__), "assets", "ant_body.xml"),
            RoboAntFighter
        ),
        'bug': (
            os.path.join(os.path.dirname(__file__), "assets", "bug_body.xml"),
            Bug
        ),
        'dev_bug':(
            os.path.join(os.path.dirname(__file__), "assets", "dev_bug_body.xml"),
            DevBug
        ),
        'dev_bug_fighter':(
            os.path.join(os.path.dirname(__file__), "assets", "dev_bug_body.xml"),
            DevBugFighter
        ),
        'spider': (
            os.path.join(os.path.dirname(__file__), "assets", "spider_body.xml"),
            Spider
        ),
        'dev_spider':(
            os.path.join(os.path.dirname(__file__), "assets", "dev_spider_body.xml"),
            DevSpider
        ),
        'dev_spider_fighter':(
            os.path.join(os.path.dirname(__file__), "assets", "dev_spider_body.xml"),
            DevSpiderFighter
        ),
    }
    WORLD_XML = os.path.join(os.path.dirname(__file__), "assets", "world_body.xml")
    GOAL_REWARD = 1000

    def __init__(
        self, 
        cfg,
        agent_names,
        rundir=os.path.join(os.path.dirname(__file__), "assets"),
        world_xml_path=WORLD_XML, agent_map=AGENT_MAP, move_reward_weight=1.0,
        init_pos=None, ini_euler=None, rgb=None, agent_args=None,
        max_episode_steps=500,
        **kwargs,
    ):
        '''
            agent_args is a list of kwargs for each agent
        '''
        self.agent_names = agent_names
        self.rundir = rundir
        self.world_xml_path = world_xml_path
        self.agent_map = agent_map
        self.move_reward_weight = move_reward_weight
        self.ini_pos = init_pos
        self.ini_euler = ini_euler
        self.rgb = rgb
        self.agent_args = agent_args
        self.kwargs = kwargs

        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

        self.cfg = cfg
        self.n_agents = len(agent_names)
        
        # create agents and corresponding robots
        all_agent_xml_paths = self.setup_agents(cfg, agent_names, agent_map, agent_args)
        
        agent_scopes = ['agent' + str(i) for i in range(self.n_agents)]
        self.agent_scopes = agent_scopes

        # reset time
        self.cur_t = 0
        
        # build environment from init xml path
        self.setup_init_mujoco_env(world_xml_path, all_agent_xml_paths, agent_scopes, rundir, init_pos, ini_euler, rgb, **kwargs)
    
    def setup_agents(self, cfg, agent_names, agent_map, agent_args):
        self.agents = {}
        all_agent_xml_paths = []
        if not agent_args:
            agent_args = [{} for _ in range(self.n_agents)]
        assert len(agent_args) == self.n_agents, "Incorrect length of agent_args"
        for i, name in enumerate(agent_names):
            # print("Creating agent", name)
            agent_xml_path, agent_class = agent_map[name]
            if hasattr(agent_class, 'flag'):
                self.agents[i] = agent_class(i, cfg, agent_xml_path, self.n_agents, **agent_args[i])
            else:
                self.agents[i] = agent_class(i, agent_xml_path, self.n_agents, **agent_args[i])
            all_agent_xml_paths.append(agent_xml_path)
        return all_agent_xml_paths

    def setup_init_mujoco_env(
            self, 
            world_xml_path, 
            all_agent_xml_paths,
            agent_scopes, 
            rundir, 
            init_pos, 
            ini_euler, 
            rgb, 
            **kwargs
            ):

        # the initial xml path is None
        self._env_xml_path = None
        # create multiagent env xml
        # the xml file will be saved at "scene_xml_path"
        _, self._env_xml_path = create_multiagent_xml(
            world_xml_path, all_agent_xml_paths, agent_scopes, outdir=rundir,
            ini_pos=init_pos, ini_euler=ini_euler, rgb=rgb
        )
        self.env_scene = MultiAgentScene(self._env_xml_path, self.n_agents, **kwargs,)

        for i, agent in self.agents.items():
            agent.set_env(self.env_scene)
        self._set_observation_space()
        self._set_action_space()
        self.metadata = self.env_scene.metadata
        
        gid = self.env_scene.geom_names.index('rightgoal')
        self.RIGHT_GOAL = self.env_scene.model.geom_pos[gid][0]
        gid = self.env_scene.geom_names.index('leftgoal')
        self.LEFT_GOAL = self.env_scene.model.geom_pos[gid][0]
        for i in range(self.n_agents):
            if self.agents[i].get_qpos()[0] > 0:
                self.agents[i].set_goal(self.LEFT_GOAL)
            else:
                self.agents[i].set_goal(self.RIGHT_GOAL)

    def reload_init_mujoco_env(self, **kwargs):
        if hasattr(self, "env_scene"):
            self.env_scene.close()
            del self.env_scene
        self.env_scene = MultiAgentScene(self._env_xml_path, self.n_agents, **kwargs,)

        for i, agent in self.agents.items():
            agent.set_env(self.env_scene)
        self._set_observation_space()
        self._set_action_space()
        self.metadata = self.env_scene.metadata
        
        gid = self.env_scene.geom_names.index('rightgoal')
        self.RIGHT_GOAL = self.env_scene.model.geom_pos[gid][0]
        gid = self.env_scene.geom_names.index('leftgoal')
        self.LEFT_GOAL = self.env_scene.model.geom_pos[gid][0]
        for i in range(self.n_agents):
            if self.agents[i].get_qpos()[0] > 0:
                self.agents[i].set_goal(self.LEFT_GOAL)
            else:
                self.agents[i].set_goal(self.RIGHT_GOAL)

    def load_tmp_mujoco_env(
            self, 
            world_xml_path, 
            all_agent_xml_strs,
            agent_scopes, 
            init_pos, 
            ini_euler, 
            rgb, 
            **kwargs
            ):
        
        if hasattr(self, "env_scene"):
            self.env_scene.close()
            del self.env_scene

        self._env_xml_str = create_multiagent_xml_str(
            world_xml_path, all_agent_xml_strs, agent_scopes,
            ini_pos=init_pos, ini_euler=ini_euler, rgb=rgb
        )
        # print(self._env_xml_str)
        self.env_scene = MultiEvoAgentScene(self._env_xml_str, self.n_agents, **kwargs,)

        for i, agent in self.agents.items():
            agent.set_env(self.env_scene)
        self._set_observation_space()
        self._set_action_space()
        self.metadata = self.env_scene.metadata
        
        gid = self.env_scene.geom_names.index('rightgoal')
        self.RIGHT_GOAL = self.env_scene.model.geom_pos[gid][0]
        gid = self.env_scene.geom_names.index('leftgoal')
        self.LEFT_GOAL = self.env_scene.model.geom_pos[gid][0]
        for i in range(self.n_agents):
            if self.agents[i].get_qpos()[0] > 0:
                self.agents[i].set_goal(self.LEFT_GOAL)
            else:
                self.agents[i].set_goal(self.RIGHT_GOAL)

    def _past_limit(self):
        if self._max_episode_steps <= self._elapsed_steps:
            return True
        return False

    def _set_observation_space(self):
        pass

    def _set_action_space(self):
        pass

    # def _set_action_space(self):
    #     """ Only for env testing.
    #     """
    #     self.action_space = spaces.Tuple(
    #         [self.agents[i].action_space for i in range(self.n_agents)]
    #     )

    def goal_rewards(self, infos=None, agent_dones=None):
        touchdowns = [self.agents[i].reached_goal()
                      for i in range(self.n_agents)]
        num_reached_goal = sum(touchdowns)
        goal_rews = [0. for _ in range(self.n_agents)]

        if num_reached_goal != 1:
            return goal_rews, num_reached_goal > 0
        
        for i in range(self.n_agents):
            if touchdowns[i]:
                goal_rews[i] = self.GOAL_REWARD
                if infos:
                    infos[i]['winner'] = True
            else:
                goal_rews[i] = - self.GOAL_REWARD
        return goal_rews, True

    def _get_done(self, dones, game_done):
        done = np.all(dones)
        done = game_done or not np.isfinite(self.state_vector()).all() or done
        dones = tuple(done for _ in range(self.n_agents))
        return dones

    def _step(self, actions):
        self._elapsed_steps += 1
        for i in range(self.n_agents):
            self.agents[i].before_step()
        
        self.env_scene.simulate(actions)
        move_rews = []
        infos = []
        dones = []
        for i in range(self.n_agents):
            move_r, agent_done, rinfo = self.agents[i].after_step(actions[i])
            move_rews.append(move_r)
            dones.append(agent_done)
            rinfo['agent_done'] = agent_done
            infos.append(rinfo)
        if self.cfg.use_parse_reward:
            goal_rews, game_done = self.goal_rewards(infos=infos, agent_dones=dones)
        else:
            goal_rews = [0, 0]
            game_done = False
        rews = []
        for i, info in enumerate(infos):
            info['reward_parse'] = float(goal_rews[i])
            rews.append(float(goal_rews[i] + self.move_reward_weight * move_rews[i]))
        rews = tuple(rews)
        terminateds = self._get_done(dones, game_done)
        infos = tuple(infos)
        obses = self._get_obs()
        
        return obses, rews, terminateds, False, infos

    def step(self, actions):
        self.cur_t += 1
        # skeleton transform stage
        if self.stage == 'skeleton_transform':
            terminateds = tuple([False, False])
            infos = []
            cur_xml_strs = []
            for i in range(self.n_agents):
                if hasattr(self.agents[i], 'flag') and self.agents[i].flag == "evo":
                    skel_a = actions[i][:, -1]
                    self.agents[i].apply_skel_action(skel_a)
                    info = {'use_transform_action': True, 
                            'stage': 'skeleton_transform',
                            'reward_parse': 0,
                            'reward_dense': 0,
                            }
                else:
                    info = {'use_transform_action': True, 
                            'stage': 'skeleton_transform',
                            'reward_parse': 0,
                            'reward_dense': 0,
                            }
                    
                infos.append(info)
                cur_xml_str = self.agents[i].cur_xml_str
                cur_xml_strs.append(cur_xml_str)
            
            try:
                self.load_tmp_mujoco_env(self.world_xml_path, cur_xml_strs, \
                                     self.agent_scopes, self.ini_pos, self.ini_euler, self.rgb, **self.kwargs)
            except:
                print("Warning: Errors occur when loading xml files.")
                terminateds = tuple([True, True])
            
            if self.cur_t == self.cfg.skel_transform_nsteps:
                self.transit_attribute_transform()

            obses = self._get_obs()
            rews = tuple([0., 0.])
            return obses, rews, terminateds, False, infos
        # attribute transform stage
        elif self.stage == 'attribute_transform':
            terminateds = tuple([False, False])
            infos = []
            cur_xml_strs = []
            for i in range(self.n_agents):
                if hasattr(self.agents[i], 'flag') and self.agents[i].flag == "evo":
                    design_a = actions[i][:, self.agents[i].control_action_dim:-1] 
                    if self.agents[i].abs_design:
                        design_params = design_a * self.cfg.robot_param_scale
                    else:
                        design_params = self.agents[i].design_cur_params + design_a * self.cfg.robot_param_scale
                    self.agents[i].set_design_params(design_params)

                    info = {'use_transform_action': True, 
                            'stage': 'attribute_transform',
                            'reward_parse': 0,
                            'reward_dense': 0,
                            }
                elif hasattr(self.agents[i], 'flag') and self.agents[i].flag == "dev":
                    design_params = actions[i]
                    self.agents[i].set_design_params(design_params)

                    info = {'use_transform_action': True, 
                            'stage': 'attribute_transform',
                            'reward_parse': 0,
                            'reward_dense': 0,
                            'design_params': design_params[:self.agents[i].scale_state_dim],
                            }
                else:
                    info = {'use_transform_action': False, 
                            'stage': 'attribute_transform',
                            'reward_parse': 0,
                            'reward_dense': 0,
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
            
            if self.cur_t == self.cfg.skel_transform_nsteps + 1:
                self.transit_execution()

            obses = self._get_obs()
            rews = tuple([0., 0.])
            return obses, rews, terminateds, False, infos
        # execution
        else:
            self.control_nsteps += 1
            flatten_actions = []
            for i in range(self.n_agents):
                if hasattr(self.agents[i], 'flag') and self.agents[i].flag == 'evo':
                    action = actions[i]
                    assert np.all(action[:, self.agents[i].control_action_dim:] == 0)
                    control_a = action[1:, :self.agents[i].control_action_dim] # rm the torso node
                    flatten_actions.append(control_a.flatten())
                elif hasattr(self.agents[i], 'flag') and self.agents[i].flag == 'dev':
                    action = actions[i][-self.agents[i].sim_action_dim:]
                    flatten_actions.append(action.flatten())
                else:
                    action = actions[i]
                    flatten_actions.append(action.flatten())
            obses, rews, terminateds, truncated, infos = self._step(flatten_actions)
        if self._past_limit():
            return obses, rews, terminateds, True, infos
        
        return obses, rews, terminateds, truncated, infos

    def _get_obs(self):
        return tuple([self.agents[i]._get_obs(self.stage) for i in range(self.n_agents)])

    def transit_attribute_transform(self):
        self.stage = 'attribute_transform'

    def transit_execution(self):
        self.stage = 'execution'
        self.control_nsteps = 0

    '''
    Following remaps all mujoco-env calls to the scene
    '''
    def _seed(self, seed=None):
        return self.env_scene._seed(seed)

    def _reset(self):
        self.cur_t = 0
        self._elapsed_steps = 0
        self.env_scene.reset()
        self.reset_model()
        # reset agent position
        for i in range(self.n_agents):
            self.agents[i].set_xyz((None,None,None))
        ob = self._get_obs()
        return ob, {}
    
    def reset(self):
        if self.agents is not None:
            del self.agents
        if hasattr(self, "env_scene"):
            self.env_scene.close()
            del self.env_scene
        self.stage = 'skeleton_transform'

        # reload from init files, to avoid multiprocessing w/r issue
        self.setup_agents(self.cfg, self.agent_names, self.agent_map, self.agent_args)
        self.reload_init_mujoco_env()
        return self._reset()

    def set_state(self, qpos, qvel):
        self.env_scene.set_state(qpos, qvel)

    @property
    def dt(self):
        return self.env_scene.dt

    def state_vector(self):
        return self.env_scene.state_vector()
    
    def reset_model(self):
        _ = self.env_scene.reset()
        for i in range(self.n_agents):
            self.agents[i].reset_agent()
        return self._get_obs(), {}