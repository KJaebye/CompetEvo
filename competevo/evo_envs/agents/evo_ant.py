from gym_compete.new_envs.agents import Ant
from gymnasium.spaces import Box
import numpy as np
import os

try:
    import mujoco
except ImportError as e:
    MUJOCO_IMPORT_ERROR = e
else:
    MUJOCO_IMPORT_ERROR = None

from competevo.evo_envs.robot.xml_robot import Robot
from lib.utils import get_single_body_qposaddr, get_graph_fc_edges
from custom.utils.transformation import quaternion_matrix

class EvoAnt(Ant):

    def __init__(self, agent_id, cfg, xml_path=None, n_agents=2):
        super(EvoAnt, self).__init__(agent_id, xml_path, n_agents)
        self.cfg = cfg
        self.xml_folder = os.path.dirname(xml_path)
        self.evo_flag = True

        # robot xml
        self.robot = Robot(cfg.robot_cfg, xml=xml_path)
        self.init_xml_str = self.robot.export_xml_string()
        self.cur_xml_str = self.init_xml_str.decode('utf-8')
        # design options
        self.clip_qvel = cfg.obs_specs.get('clip_qvel', False)
        self.use_projected_params = cfg.obs_specs.get('use_projected_params', True)
        self.abs_design = cfg.obs_specs.get('abs_design', False)
        self.use_body_ind = cfg.obs_specs.get('use_body_ind', False)
        self.design_ref_params = self.get_attr_design()
        self.design_cur_params = self.design_ref_params.copy()
        self.design_param_names = self.robot.get_params(get_name=True)
        self.attr_design_dim = self.design_ref_params.shape[-1]
        self.index_base = 5
        self.stage = 'skeleton_transform'    # transform or execute
        self.control_nsteps = 0
        self.sim_specs = set(cfg.obs_specs.get('sim', []))
        self.attr_specs = set(cfg.obs_specs.get('attr', []))
        self.control_action_dim = 1
        self.skel_num_action = 3 if cfg.enable_remove else 2
        self.sim_obs_dim = 15 #13
        self.attr_fixed_dim = self.get_attr_fixed().shape[-1]

        self.state_dim = self.attr_fixed_dim + self.sim_obs_dim + self.attr_design_dim
        self.action_dim = self.control_action_dim + self.attr_design_dim

    def set_goal(self, goal):
        self.GOAL = goal
        self.move_left = False
        if self.get_qpos()[0] > 0:
            self.move_left = True

    def before_step(self):
        self._xposbefore = self.get_body_com("0")[0]

    # def after_step(self, action):
    #     xposafter = self.get_body_com("0")[0]
    #     forward_reward = (xposafter - self._xposbefore) / self.env.dt
    #     if self.move_left:
    #         forward_reward *= -1
    #     ctrl_cost = .5 * np.square(action).sum()
    #     cfrc_ext = self.get_cfrc_ext()
    #     contact_cost = 0.5 * 1e-3 * np.sum(
    #         np.square(np.clip(cfrc_ext, -1, 1))
    #     )
    #     qpos = self.get_qpos()
    #     agent_standing = qpos[2] >= 0.28
    #     survive = 1.0
    #     reward = forward_reward - ctrl_cost - contact_cost + survive

    #     reward_info = dict()
    #     reward_info['reward_forward'] = forward_reward
    #     reward_info['reward_ctrl'] = ctrl_cost
    #     reward_info['reward_contact'] = contact_cost
    #     reward_info['reward_survive'] = survive
    #     reward_info['reward_dense'] = reward

    #     terminated = not agent_standing

    #     return reward, terminated, reward_info

    def after_step(self, action):
        xposafter = self.get_body_com("0")[0]
        forward_reward = (xposafter - self._xposbefore) / self.env.dt
        if self.move_left:
            forward_reward *= -1

        
        # ctrl_cost = .5 * np.square(action).sum()
        # cfrc_ext = self.get_cfrc_ext()
        # contact_cost = 0.5 * 1e-3 * np.sum(
        #     np.square(np.clip(cfrc_ext, -1, 1))
        # )
    
        ctrl_cost = 1e-4 * np.square(action).mean()
        contact_cost = 0

        survive = 0.0
        reward = forward_reward - ctrl_cost - contact_cost + survive

        reward_info = dict()
        reward_info['reward_forward'] = forward_reward
        reward_info['reward_ctrl'] = ctrl_cost
        reward_info['reward_contact'] = contact_cost
        reward_info['reward_survive'] = survive
        reward_info['reward_dense'] = reward

        info = reward_info
        info['use_transform_action'] = False
        info['stage'] = 'execution'

        # terminate condition
        qpos = self.get_qpos()
        height = qpos[2]
        zdir = quaternion_matrix(qpos[3:7])[:3, 2]
        ang = np.arccos(zdir[2])
        done_condition = self.cfg.done_condition
        min_height = done_condition.get('min_height', 0.28)
        max_height = done_condition.get('max_height', 1.2)
        max_ang = done_condition.get('max_ang', 3600)

        terminated = not (np.isfinite(self.get_qpos()).all() and np.isfinite(self.get_qvel()).all() and (height > min_height) and (height < max_height) and (abs(ang) < np.deg2rad(max_ang)))
        # terminated = not (np.isfinite(self.get_qpos()).all() and np.isfinite(self.get_qvel()).all() and (height > min_height) and (height < max_height))
        
        return reward, terminated, info


    # def _get_obs(self):
    #     '''
    #     Return agent's observations
    #     '''
    #     my_pos = self.get_qpos()
    #     other_pos = self.get_other_qpos()
        
    #     my_vel = self.get_qvel()
    #     cfrc_ext = np.clip(self.get_cfrc_ext(), -1, 1)
        
    #     # for multiagent play
    #     obs = np.concatenate(
    #         [my_pos.flat, my_vel.flat, cfrc_ext.flat,
    #          other_pos.flat]
    #     )

    #     return obs

    def set_env(self, env):
        self.env = env
        self._env_init = True
        self._set_body()
        self._set_joint()
        if self.n_agents > 1:
            self._set_other_joint()
        # self.set_action_space() # testing only


    def reached_goal(self):
        if self.n_agents == 1: return False
        xpos = self.get_body_com('0')[0]
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

    ############################################################################
    ############################# robot xml ####################################

    def allow_add_body(self, body):
        add_body_condition = self.cfg.add_body_condition
        max_nchild = add_body_condition.get('max_nchild', 3)
        min_nchild = add_body_condition.get('min_nchild', 0)
        return body.depth >= self.cfg.min_body_depth \
                and body.depth < self.cfg.max_body_depth - 1 \
                and len(body.child) < max_nchild and len(body.child) >= min_nchild
    
    def allow_remove_body(self, body):
        if body.depth >= self.cfg.min_body_depth + 1 and len(body.child) == 0:
            if body.depth == 1:
                return body.parent.child.index(body) > 0
            else:
                return True
        return False
    
    def apply_skel_action(self, skel_action):
        bodies = list(self.robot.bodies)
        for body, a in zip(bodies, skel_action):
            if a == 1 and self.allow_add_body(body):
                self.robot.add_child_to_body(body)
            if a == 2 and self.allow_remove_body(body):
                self.robot.remove_body(body)

        xml_str = self.robot.export_xml_string()
        self.cur_xml_str = xml_str.decode('utf-8')
        self.design_cur_params = self.get_attr_design()

    
    def set_design_params(self, in_design_params):
        design_params = in_design_params
        for params, body in zip(design_params, self.robot.bodies):
            body.set_params(params, pad_zeros=True, map_params=True)
            body.sync_node()

        xml_str = self.robot.export_xml_string()
        self.cur_xml_str = xml_str.decode('utf-8')
        
        if self.use_projected_params:
            self.design_cur_params = self.get_attr_design()
        else:
            self.design_cur_params = in_design_params.copy()

    def action_to_control(self, a):
        ctrl = np.zeros_like(self.data.ctrl)
        assert a.shape[0] == len(self.robot.bodies)
        for body, body_a in zip(self.robot.bodies[1:], a[1:]):
            aname = body.get_actuator_name()
            if aname in self.model.actuator_names:
                aind = self.model.actuator_names.index(aname)
                ctrl[aind] = body_a
        return ctrl
    
    def if_use_transform_action(self):
        return ['skeleton_transform', 'attribute_transform', 'execution'].index(self.stage)
    
    def get_sim_obs(self):
        obs = []
        if 'root_offset' in self.sim_specs:
            root_pos = self.env.data.body_xpos[self.env.model._body_name2id[self.robot.bodies[0].name]]

        # body_names = []
        # for body in self.robot.bodies:
        #     body_names.append(body.name)
        # print(body_names)

        other_pos = self.get_other_qpos()[:2]
        if other_pos.shape == (0,):
            other_pos = np.zeros(2) # x and y

        for i, body in enumerate(self.robot.bodies):
            qpos = self.get_qpos()
            qvel = self.get_qvel()
            if self.clip_qvel:
                qvel = np.clip(qvel, -10, 10)
            # print(self.id, self.joint_names[i])
            # print(self.id, self.qvel_start_idx, self.qvel_end_idx)
            if i == 0:
                obs_i = [qpos[2:7], qvel[:6], np.zeros(2), other_pos]
            else:
                # print(self.id, i, body.name)
                qs, qe = get_single_body_qposaddr(self.env.model, self.scope + "/" + body.name)
                # if self.id == 1:
                #     print(qs-1-self.id, qe-1-self.id)
                #     print(self.id, i, qvel[:])
                #     print(self.id, i, self.env.data.qvel[10:])
                if qe - qs >= 1:
                    assert qe - qs == 1
                    # print(qs, qe)
                    obs_i = [np.zeros(11), self.env.data.qpos[qs:qe], self.env.data.qvel[qs-1-self.id:qe-1-self.id], other_pos]
                else:
                    obs_i = [np.zeros(13), other_pos]
            if 'root_offset' in self.sim_specs:
                offset = self.data.body_xpos[self.model._body_name2id[body.name]][[0, 2]] - root_pos[[0, 2]]
                obs_i.append(offset)
            
            obs_i = np.concatenate(obs_i)
            obs.append(obs_i)
            # print(i, obs_i.shape)
        obs = np.stack(obs)
        return obs

    def get_attr_fixed(self):
        obs = []
        for i, body in enumerate(self.robot.bodies):
            obs_i = []
            if 'depth' in self.attr_specs:
                obs_depth = np.zeros(self.cfg.max_body_depth)
                obs_depth[body.depth] = 1.0
                obs_i.append(obs_depth)
            if 'jrange' in self.attr_specs:
                obs_jrange = body.get_joint_range()
                obs_i.append(obs_jrange)
            if 'skel' in self.attr_specs:
                obs_add = self.allow_add_body(body)
                obs_rm = self.allow_remove_body(body)
                obs_i.append(np.array([float(obs_add), float(obs_rm)]))
            if len(obs_i) > 0:
                obs_i = np.concatenate(obs_i)
                obs.append(obs_i)
        
        if len(obs) == 0:
            return None
        obs = np.stack(obs)
        return obs

    def get_attr_design(self):
        obs = []
        for i, body in enumerate(self.robot.bodies):
            obs_i = body.get_params([], pad_zeros=True, demap_params=True)
            obs.append(obs_i)
        obs = np.stack(obs)
        return obs
    
    def get_body_index(self):
        index = []
        for i, body in enumerate(self.robot.bodies):
            ind = int(body.name.split("/")[-1], base=self.index_base)
            index.append(ind)
        index = np.array(index)
        return index

    def _get_obs(self, stage):
        # update stage tag from env
        self.stage = stage
        obs = []
        attr_fixed_obs = self.get_attr_fixed()
        sim_obs = self.get_sim_obs()
        design_obs = self.design_cur_params
        obs = np.concatenate(list(filter(lambda x: x is not None, [attr_fixed_obs, sim_obs, design_obs])), axis=-1)
        if self.cfg.obs_specs.get('fc_graph', False):
            edges = get_graph_fc_edges(len(self.robot.bodies))
        else:
            edges = self.robot.get_gnn_edges()
        use_transform_action = np.array([self.if_use_transform_action()])
        num_nodes = np.array([sim_obs.shape[0]])
        all_obs = [obs, edges, use_transform_action, num_nodes]
        if self.use_body_ind:
            body_index = self.get_body_index()
            all_obs.append(body_index)
        return all_obs
    
    def reset_state(self, add_noise):
        if add_noise:
            qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
            qvel = self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        else:
            qpos = self.init_qpos
            qvel = self.init_qvel
        if self.env_specs.get('init_height', True):
            qpos[2] = 0.4
        self.set_state(qpos, qvel)

    def reset_robot(self):
        del self.robot
        self.robot = Robot(self.cfg.robot_cfg, xml=self.init_xml_str, is_xml_str=True)
        self.cur_xml_str = self.init_xml_str.decode('utf-8')
        self.design_ref_params = self.get_attr_design()
        self.design_cur_params = self.design_ref_params.copy()

    def reset_model(self):
        self.reset_robot()
        self.control_nsteps = 0
        self.stage = 'skeleton_transform'
        self.cur_t = 0
        self.reset_state(False)
        return self._get_obs()