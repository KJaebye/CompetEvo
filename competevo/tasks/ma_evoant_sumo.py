from calendar import c
from typing import Tuple
import attr
import numpy as np
import os
import math
from sympy import N
import torch
import random
import torch.nn.functional as F

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.gymtorch import *
from competevo.utils.reformat import omegaconf_to_dict, omegaconflist_to_list
# from torch.tensor import Tensor

from competevo.utils.torch_jit_utils import *
from competevo.robot.xml_robot import Robot
from competevo.robot.robo_utils import *

from .base.ma_evo_vec_task import MA_Evo_VecTask


# todo critic_state full obs
class MA_EvoAnt_Sumo(MA_Evo_VecTask):

    def __init__(self, cfg, sim_device, rl_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.termination_height = self.cfg["env"]["terminationHeight"]
        self.borderline_space = cfg["env"]["borderlineSpace"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]
        self.joints_at_limit_cost_scale = self.cfg["env"]["jointsAtLimitCost"]
        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]

        self.draw_penalty_scale = -1000
        self.win_reward_scale = 2000
        self.move_to_op_reward_scale = 1.
        self.stay_in_center_reward_scale = 0.2
        self.action_cost_scale = -0.000025
        self.push_scale = 1.
        self.dense_reward_scale = 1.
        self.hp_decay_scale = 1.

        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]

        # see func: compute_ant_observations() for details
        
        # define transform2act evo obs, action dimensions
        # robot config
        robo_config = self.cfg['robot']

        # design options
        self.robot_param_scale = robo_config['robot_param_scale']
        self.skel_transform_nsteps = robo_config["skel_transform_nsteps"]
        self.clip_qvel = robo_config["obs_specs"]["clip_qvel"]
        self.use_projected_params = robo_config["obs_specs"]["use_projected_params"]
        self.abs_design = robo_config["obs_specs"].get("abs_design", False)
        self.use_body_ind = robo_config["obs_specs"]["use_body_ind"]

        self.sim_specs = robo_config["obs_specs"]["sim"]
        self.attr_specs = robo_config["obs_specs"]["attr"]

        super().__init__(config=self.cfg, sim_device=sim_device, rl_device=rl_device,
                         graphics_device_id=graphics_device_id,
                         headless=headless, virtual_screen_capture=virtual_screen_capture,
                         force_render=force_render)

        self.reset_robot(robo_config)
        
        # constant variables
        self.attr_design_dim = 5
        self.attr_fixed_dim = 4
        self.gym_obs_dim = 26 # 13(root states) + 7(op root states) + 2(dist) + 1(height termination) + 1(op height termination) + 2(dof pos/vel)
        self.index_base = 5

        # actions dim: (num_nodes, action_dim)
        ###############################################################
        # action for every node:
        #                 control_action      attr_action        skel_action 
        #  node0(root):  #-------------##--------------------##---------------#
        #  node1      :  #-------------##--------------------##---------------#
        #  node2      :  #-------------##--------------------##---------------#
        #  .....      :  #-------------##--------------------##---------------#
        #  nodeN      :  #-------------##--------------------##---------------#

        self.skel_num_action = 3 if robo_config["enable_remove"] else 2 # skel action dimension is 1
        self.attr_action_dim = self.attr_design_dim
        self.control_action_dim = 1 
        self.action_dim = self.control_action_dim + self.attr_action_dim + 1
        
        # states dim and construction:
        # {obses, edges, stage, num_nodes, body_index}

        # obses dim (num_nodes, obs_dim)
        ###############################################################
        # observation for every node:
        #                   attr_fixed           gym_obs          attr_design 
        #  node0(root): #---------------##--------------------##---------------#
        #  node1      : #---------------##--------------------##---------------#
        #  node2      : #---------------##--------------------##---------------#
        #  node3      : #---------------##--------------------##---------------#
        #  .....      : #---------------##--------------------##---------------#
        #  nodeN      : #---------------##--------------------##---------------#
        # 
        # The gym_obs has structures as follow:
        #                  24 (states)       2 (pos, vel)
        #  node0(root): xxxxxxxxxxxxxxxxxxxxx - oooo
        #  node1      : ooooooooooooooooooooo - xxxx
        #  node2      : ooooooooooooooooooooo - xxxx
        #  node3      : ooooooooooooooooooooo - xxxx
        #  .....      : ooooooooooooooooooooo - xxxx
        #  nodeN      : ooooooooooooooooooooo - xxxx
        # Only the first node (root/agent torso) has the root state, other nodes use zero-padding as instead.
        # 
        self.skel_state_dim = self.attr_fixed_dim + self.attr_design_dim
        self.attr_state_dim = self.attr_fixed_dim + self.attr_design_dim
        self.obs_dim = self.attr_fixed_dim + self.gym_obs_dim + self.attr_design_dim

        self.use_central_value = False
        
        self.allocate_buffers() # init buffers

    def reset_robot(self, robo_config):
        # define robots: 2
        self.base_ant_path = '/home/kjaebye/ws/competevo/assets/mjcf/ant.xml'
        self.robots = {}
        # xml tmp dir
        self.out_dir = 'out/evo_ant'
        os.makedirs(self.out_dir, exist_ok=True)
        name = "evo_ant"
        name_op = "evo_ant_op"

        # reformate robo_config
        robo_config = omegaconf_to_dict(robo_config)
        robo_config = omegaconflist_to_list(robo_config)

        self.robot = Robot(robo_config, self.base_ant_path, is_xml_str=False)
        self.robot_op = Robot(robo_config, self.base_ant_path, is_xml_str=False)

        # ant
        self.design_ref_params = torch.from_numpy(get_attr_design(self.robot)).to(device=self.device, dtype=torch.float32)
        self.design_cur_params = torch.from_numpy(get_attr_design(self.robot)).to(device=self.device, dtype=torch.float32)
        self.design_param_names = self.robot.get_params(get_name=True)
        self.num_nodes = len(list(self.robot.bodies))
        # ant op
        self.design_ref_params_op = torch.from_numpy(get_attr_design(self.robot_op)).to(device=self.device, dtype=torch.float32)
        self.design_cur_params_op = torch.from_numpy(get_attr_design(self.robot_op)).to(device=self.device, dtype=torch.float32)
        self.design_param_names_op = self.robot_op.get_params(get_name=True)
        if robo_config["obs_specs"].get('fc_graph', False):
            self.edges_op = get_graph_fc_edges(len(self.robot_op.bodies))
        else:
            self.edges_op = torch.from_numpy(self.robot_op.get_gnn_edges()).to(device=self.device, dtype=torch.int)
        self.num_nodes_op = len(list(self.robot_op.bodies))

    def reset(self, env_ids=None, gym_only=False):
        """
            Reset the environment.
        """
        if not gym_only:
            # destroy last sim
            if self.sim is not None:
                self.gym.destroy_sim(self.sim)
                self.viewer.close()
            
            self.isaacgym_initialized = False
            self.stage = "skel_trans"
            self.cur_t = 0
            # reset buffer
            self.allocate_buffers()
            # reset robots
            self.reset_robot(self.cfg['robot'])
            self.compute_observations()
        else:
            if self.sim is not None: # reset only isaacgym sim is running
                self.gym_reset(env_ids)

    def step(self, all_actions: torch.tensor):
        """ all action shape: (num_envs, num_nodes + num_nodes_op, action_dim)
        """
        print("#--------------------------------------------", self.stage, "-------------------------------------------------#")

        assert all_actions.shape[1] == self.num_nodes + self.num_nodes_op
        self.cur_t += 1
        actions = all_actions[:, :self.num_nodes, :]
        actions_op = all_actions[:, self.num_nodes:, :]
        # skeleton transform stage
        if self.stage == 'skel_trans':
            # check data in a tensor are definitely equal along the first dimension
            def check_equal(tensor):
                return (tensor == tensor[0]).all()
            assert (check_equal(actions) and check_equal(actions_op)).item() is True, \
                "Skeleton transform stage needs all agents to have the same actions!"
            # ant
            skel_a = actions[0][:, -1:]
            apply_skel_action(self.robot, skel_a)
            self.design_cur_params = torch.from_numpy(get_attr_design(self.robot)).to(device=self.device, dtype=torch.float32)
            # ant op
            skel_a_op = actions_op[0][:, -1:]
            apply_skel_action(self.robot_op, skel_a_op)
            self.design_cur_params_op = torch.from_numpy(get_attr_design(self.robot_op)).to(device=self.device, dtype=torch.float32)

            # transit to attribute transform stage
            if self.cur_t == self.skel_transform_nsteps:
                self.transit_attribute_transform()
            self.compute_observations()
            self.compute_reward()
            return
        
        # attribute transform stage
        elif self.stage == 'attr_trans':
            # check data in a tensor are definitely equal along the first dimension
            def check_equal(tensor):
                return (tensor == tensor[0]).all()

            try:
                assert (check_equal(actions) and check_equal(actions_op)).item() is True, \
                "Attribute transform stage needs all agents to have the same actions!"
            except AssertionError:
                cosine_similar_per_sample = F.cosine_similarity(actions[0, :, :], actions[1, :, :])
                average_similar = cosine_similar_per_sample.mean()
                print("Attribute transform actions maight be slightly different, average cosine similarity between env0 and env1 is: ", \
                      average_similar.item())
                assert average_similar == 1.
            
            # ant
            design_a = actions[0][:, self.control_action_dim:-1]
            if self.abs_design:
                design_params = design_a * self.robot_param_scale
            else:
                design_params = self.design_cur_params \
                                + design_a * self.robot_param_scale
            set_design_params(self.robot, design_params.cpu().numpy(), self.out_dir, "evo_ant")
            if self.use_projected_params:
                self.design_cur_params = torch.from_numpy(get_attr_design(self.robot)).to(device=self.device, dtype=torch.float32)
            else:
                self.design_cur_params = design_params.copy()
            # ant op
            design_a_op = actions_op[0][:, self.control_action_dim:-1]
            if self.abs_design:
                design_params_op = design_a_op * self.robot_param_scale
            else:
                design_params_op = self.design_cur_params_op \
                                   + design_a_op * self.robot_param_scale
            set_design_params(self.robot_op, design_params_op.cpu().numpy(), self.out_dir, "evo_ant_op")
            if self.use_projected_params:
                self.design_cur_params_op = torch.from_numpy(get_attr_design(self.robot_op)).to(device=self.device, dtype=torch.float32)
            else:
                self.design_cur_params_op = design_params_op.copy()

            # transit to execution stage
            self.transit_execution()

            self.compute_observations()
            self.compute_reward()

            ###############################################################################
            ######## Create IsaacGym sim and envs for Execution ####################
            ############################################################################### 
            self.create_sim() # create sim and envs
            self.gym.prepare_sim(self.sim)
            self.isaacgym_initialized = True
            self.set_viewer()

            # add border, add camera
            self.add_border_cam()
            # init gym state
            self.gym_state_init()
            self.gym_reset()

        # execution stage
        else:
            # check zero becuase in execution stage, only control action is computated
            assert torch.all(all_actions[:, :, self.control_action_dim:] == 0)
            self.control_nsteps += 1
            self.gym_step(all_actions)

    def gym_reset(self, env_ids=None) -> torch.Tensor:
        """Reset the gym environment.
        """
        if (env_ids is None):
            # zero_actions = self.zero_actions()
            # self.step(zero_actions)
            env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
            self.reset_idx(env_ids)
            self.compute_observations()
            self.pos_before = self.obs_buf[:, 0, :2].clone()
        else:
            self._reset_envs(env_ids=env_ids)
        return

    def reset_idx(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # ant
        positions = torch_rand_float(-0.2, 0.2, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids] = tensor_clamp(self.initial_dof_pos[env_ids] + positions, self.dof_limits_lower,
                                             self.dof_limits_upper)
        self.dof_vel[env_ids] = velocities
        # ant op
        positions_op = torch_rand_float(-0.2, 0.2, (len(env_ids), self.num_dof_op), device=self.device)
        velocities_op = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof_op), device=self.device)

        self.dof_pos_op[env_ids] = tensor_clamp(self.initial_dof_pos[env_ids] + positions_op, self.dof_limits_lower_op,
                                                self.dof_limits_upper_op)
        self.dof_vel_op[env_ids] = velocities_op

        env_ids_int32 = (torch.cat((self.actor_indices[env_ids], self.actor_indices_op[env_ids]))).to(dtype=torch.int32)
        agent_env_ids = expand_env_ids(env_ids, 2)

        rand_angle = torch.rand((len(env_ids),), device=self.device) * torch.pi * 2

        rand_pos = torch.ones((len(agent_env_ids), 2), device=self.device) * (
                self.borderline_space * torch.ones((len(agent_env_ids), 2), device=self.device) - torch.rand(
            (len(agent_env_ids), 2), device=self.device) * 2)
        rand_pos[0::2, 0] *= torch.cos(rand_angle)
        rand_pos[0::2, 1] *= torch.sin(rand_angle)
        rand_pos[1::2, 0] *= torch.cos(rand_angle + torch.pi)
        rand_pos[1::2, 1] *= torch.sin(rand_angle + torch.pi)
        rand_floats = torch_rand_float(-1.0, 1.0, (len(agent_env_ids), 3), device=self.device)
        rand_rotation = quat_from_angle_axis(rand_floats[:, 1] * np.pi, self.z_unit_tensor[agent_env_ids])
        rand_rotation2 = quat_from_angle_axis(rand_floats[:, 2] * np.pi, self.z_unit_tensor[agent_env_ids])
        self.root_states[agent_env_ids] = self.initial_root_states[agent_env_ids]
        self.root_states[agent_env_ids, :2] = rand_pos
        self.root_states[agent_env_ids[1::2], 3:7] = rand_rotation[1::2]
        self.root_states[agent_env_ids[0::2], 3:7] = rand_rotation2[0::2]
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.pos_before = self.root_states[0::2, :2].clone()

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def transit_attribute_transform(self):
        self.stage = 'attr_trans'

    def transit_execution(self):
        self.stage = 'execution'
        self.control_nsteps = 0
    
    def add_border_cam(self):
        if self.viewer is not None:
            for env in self.envs:
                self._add_circle_borderline(env)
            cam_pos = gymapi.Vec3(15.0, 0.0, 3.0)
            cam_target = gymapi.Vec3(10.0, 0.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def gym_state_init(self):
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim) # all actors root state
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim) # all actors dof state

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        # reshape actor properties
        self.dof_limits_lower = self.dof_limits_lower.view(self.num_envs, -1)
        self.dof_limits_upper = self.dof_limits_upper.view(self.num_envs, -1)
        self.dof_limits_lower_op = self.dof_limits_lower_op.view(self.num_envs, -1)
        self.dof_limits_upper_op = self.dof_limits_upper_op.view(self.num_envs, -1)

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        print(f'root states:{self.root_states.shape}')
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:, 7:13] = 0  # set lin_vel and ang_vel to 0

        # create some wrapper tensors for different slices
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        print(f"dof states shape: {self.dof_state.shape}")
        self.dof_pos = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_dof, 0]
        self.dof_pos_op = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_dof:, 0]
        self.dof_vel = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_dof, 1]
        self.dof_vel_op = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_dof:, 1]

        self.initial_dof_pos = torch.zeros_like(self.dof_pos, device=self.device, dtype=torch.float)
        zero_tensor = torch.tensor([0.0], device=self.device)
        self.initial_dof_pos = torch.where(self.dof_limits_lower > zero_tensor, self.dof_limits_lower,
                                           torch.where(self.dof_limits_upper < zero_tensor, self.dof_limits_upper, self.initial_dof_pos))
        self.initial_dof_vel = torch.zeros_like(self.dof_vel, device=self.device, dtype=torch.float)
        self.dt = self.cfg["sim"]["dt"]

        torques = self.gym.acquire_dof_force_tensor(self.sim)
        self.torques = gymtorch.wrap_tensor(torques).view(self.num_envs, self.num_dof + self.num_dof_op)

        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((2 * self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((2 * self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((2 * self.num_envs, 1))

        self.hp = torch.ones((self.num_envs,), device=self.device, dtype=torch.float32) * 100
        self.hp_op = torch.ones((self.num_envs,), device=self.device, dtype=torch.float32) * 100

    def allocate_buffers(self):
        """Allocate the observation, states, etc. buffers.

        These are what is used to set observations and states in the environment classes which
        inherit from this one, and are read in `step` and other related functions.

        """
        # allocate buffers
        # transform2act state data
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_nodes + self.num_nodes_op, self.obs_dim), device=self.device, dtype=torch.float)
        if self.cfg['robot']['obs_specs'].get('fc_graph', False):
            pass
        else: # use gnn_edges
            self.edge_buf = torch.zeros(
                (self.num_envs, 2, (self.num_nodes-1)*2 + (self.num_nodes_op-1)*2), device=self.device, dtype=torch.int64)
        self.stage_buf = torch.ones(
            (self.num_envs, 1), device=self.device, dtype=torch.int)
        self.num_nodes_buf = torch.zeros(
            (self.num_envs, 2), device=self.device, dtype=torch.int)
        if self.use_body_ind:
            self.body_ind_buf = torch.zeros(
                (self.num_envs, self.num_nodes + self.num_nodes_op), device=self.device, dtype=torch.int64)
        
        self.rew_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long)
        self.timeout_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long) # progress buffer only for isaacgym simulation, before isaacgym, it is always zero
        self.randomize_buf = torch.zeros(
            self.num_envs * self.num_agents, device=self.device, dtype=torch.long)
        self.extras = {}

    def create_sim(self):
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, 'z')
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        print(f'num envs {self.num_envs} env spacing {self.cfg["env"]["envSpacing"]}')
        self._create_envs(self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)
    
    def _add_circle_borderline(self, env):
        lines = []
        borderline_height = 0.01
        for height in range(20):
            for angle in range(360):
                begin_point = [np.cos(np.radians(angle)), np.sin(np.radians(angle)), borderline_height * height]
                end_point = [np.cos(np.radians(angle + 1)), np.sin(np.radians(angle + 1)), borderline_height * height]
                lines.append(begin_point)
                lines.append(end_point)
        lines = np.array(lines, dtype=np.float32) * self.borderline_space
        colors = np.array([[1, 0, 0]] * int(len(lines) / 2), dtype=np.float32)
        self.gym.add_lines(self.viewer, env, int(len(lines) / 2), lines, colors)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_options = gymapi.AssetOptions()
        # Note - DOF mode is set in the MJCF file and loaded by Isaac Gym
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.angular_damping = 0.0

        # load assets
        name = "evo_ant"
        name_op = "evo_ant_op"
        file_name = name + ".xml"
        file_name_op = name_op + ".xml"
        ant_asset = self.gym.load_mjcf(self.sim, self.out_dir, file_name, asset_options)
        ant_asset_op = self.gym.load_mjcf(self.sim, self.out_dir, file_name_op, asset_options)

        # create envs and actors
        # set agent and opponent start pose
        box_pose = gymapi.Transform()
        box_pose.p = gymapi.Vec3(0, 0, 0)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(-self.borderline_space + 1, -self.borderline_space + 1, 1.)
        start_pose_op = gymapi.Transform()
        start_pose_op.p = gymapi.Vec3(self.borderline_space - 1, self.borderline_space - 1, 1.)

        print(start_pose.p, start_pose_op.p)
        self.start_rotation = torch.tensor([start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w],
                                           device=self.device)

        self.ant_handles = []
        self.actor_indices = []
        self.actor_handles_op = []
        self.actor_indices_op = []

        self.envs = []
        self.pos_before = torch.zeros(2, device=self.device)

        self.dof_limits_lower = []
        self.dof_limits_upper = []
        self.dof_limits_lower_op = []
        self.dof_limits_upper_op = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            name = f"evo_ant_{i}"
            name_op = f"evo_ant_op_{i}"

            #------------------------- set ant props -----------------------#
            ant_handle = self.gym.create_actor(env_ptr, ant_asset, start_pose, name, i, -1, 0)
            actor_index = self.gym.get_actor_index(env_ptr, ant_handle, gymapi.DOMAIN_SIM)
            # set def_props
            dof_props = self.gym.get_actor_dof_properties(env_ptr, ant_handle)
            self.num_dof = self.gym.get_actor_dof_count(env_ptr, ant_handle)
            self.num_bodies = self.gym.get_actor_rigid_body_count(env_ptr, ant_handle)  # num_dof + 1(torso)
            assert self.num_bodies == self.num_nodes
            for i in range(self.num_dof):
                dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
                dof_props['stiffness'][i] = self.Kp
                dof_props['damping'][i] = self.Kd
            for j in range(self.num_dof):
                if dof_props['lower'][j] > dof_props['upper'][j]:
                    self.dof_limits_lower.append(dof_props['upper'][j])
                    self.dof_limits_upper.append(dof_props['lower'][j])
                else:
                    self.dof_limits_lower.append(dof_props['lower'][j])
                    self.dof_limits_upper.append(dof_props['upper'][j])

            self.gym.set_actor_dof_properties(env_ptr, ant_handle, dof_props)
            self.actor_indices.append(actor_index)
            self.gym.enable_actor_dof_force_sensors(env_ptr, ant_handle)
            # set color
            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr, ant_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.71, 0.49, 0.01))

            #------------------------- set ant_op props -----------------------#
            ant_handle_op = self.gym.create_actor(env_ptr, ant_asset_op, start_pose_op, name_op, i, -1, 0)
            actor_index_op = self.gym.get_actor_index(env_ptr, ant_handle_op, gymapi.DOMAIN_SIM)
            # set def_props
            dof_props_op = self.gym.get_actor_dof_properties(env_ptr, ant_handle_op)
            self.num_dof_op = self.gym.get_actor_dof_count(env_ptr, ant_handle_op)
            self.num_bodies_op = self.gym.get_actor_rigid_body_count(env_ptr, ant_handle_op)  # num_dof + 1(torso)
            assert self.num_bodies_op == self.num_nodes_op
            for i in range(self.num_dof_op):
                dof_props_op['driveMode'][i] = gymapi.DOF_MODE_POS
                dof_props_op['stiffness'][i] = self.Kp
                dof_props_op['damping'][i] = self.Kd
            for j in range(self.num_dof_op):
                if dof_props_op['lower'][j] > dof_props_op['upper'][j]:
                    self.dof_limits_lower_op.append(dof_props_op['upper'][j])
                    self.dof_limits_upper_op.append(dof_props_op['lower'][j])
                else:
                    self.dof_limits_lower_op.append(dof_props_op['lower'][j])
                    self.dof_limits_upper_op.append(dof_props_op['upper'][j])

            self.gym.set_actor_dof_properties(env_ptr, ant_handle_op, dof_props_op)
            self.actor_indices_op.append(actor_index_op)
            self.gym.enable_actor_dof_force_sensors(env_ptr, ant_handle_op)
            # set color
            for j in range(self.num_bodies_op):
                self.gym.set_rigid_body_color(
                    env_ptr, ant_handle_op, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.15, 0.21, 0.42))

            self.envs.append(env_ptr)
            self.ant_handles.append(ant_handle)
            self.actor_handles_op.append(ant_handle_op)

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_lower_op = to_torch(self.dof_limits_lower_op, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)
        self.dof_limits_upper_op = to_torch(self.dof_limits_upper_op, device=self.device)
        self.actor_indices = to_torch(self.actor_indices, dtype=torch.long, device=self.device)
        self.actor_indices_op = to_torch(self.actor_indices_op, dtype=torch.long, device=self.device)

    def pre_physics_step(self, actions):
        # actions.shape = [num_envs, num_dof + num_dof_op, num_actions]
        self.actions = actions.clone().to(self.device)
        targets = self.actions

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(targets))

    def post_physics_step(self):
        """
            IsaacGym post step.
        """
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward()
        self.pos_before = self.obs_buf[:, 0, :2].clone() # first node is ant torso

    def get_sim_obs(self):
        # obs:
        #  {obses, edges, stage, num_nodes, body_index}
        #  obses: (:, num_nodes, attr_fixed_dim + gym_obs_dim + attr_design_dim)
        #  edges: (:, 2, num_dof * 2)
        #  stage: (:, 1)
        #  num_nodes: (:, 1)
        #  body_index: (:, num_nodes)
        obs = {}
        obs['obses'] = self.obs_buf.to(self.rl_device)
        obs['edges'] = self.edge_buf.to(self.rl_device)
        obs['stage'] = self.stage_buf.to(self.rl_device)
        obs['num_nodes'] = self.num_nodes_buf.to(self.rl_device)
        obs['body_index'] = self.body_ind_buf.to(self.rl_device)
        return obs

    def compute_observations(self):
        """
            Calculate ant observations.
        """
        if self.stage == "skel_trans" or "attr_trans":
            if self.stage == 'skel_trans':
                self.num_nodes = len(list(self.robot.bodies))
                self.num_nodes_op = len(list(self.robot_op.bodies))
                self.allocate_buffers()
            #--------------------------- agent -----------------------------#
            attr_fixed_obs = torch.from_numpy(get_attr_fixed(self.cfg['robot']['obs_specs'], self.robot)).to(device=self.device, dtype=torch.float32).unsqueeze(0).repeat(self.num_envs, 1, 1)
            gym_obs = torch.zeros((self.num_envs, self.num_nodes, self.gym_obs_dim), device=self.device, dtype=torch.float32)
            design_obs = self.design_cur_params.unsqueeze(0).repeat(self.num_envs, 1, 1)
            # obs
            self.obs_buf[:, :self.num_nodes] = torch.cat((attr_fixed_obs, gym_obs, design_obs), dim=-1)
            # edges
            if self.cfg['robot']['obs_specs'].get('fc_graph', False):
                self.edges = torch.from_numpy(get_graph_fc_edges(len(self.robot.bodies))).to(device=self.device, dtype=torch.int)
            else:
                self.edges = torch.from_numpy(self.robot.get_gnn_edges()).to(device=self.device, dtype=torch.int)
                self.edge_buf[:, :, :(self.num_nodes-1)*2] = self.edges.unsqueeze(0).repeat(self.num_envs, 1, 1) # create vector of edges
            # stage flag
            self.stage_buf = torch.zeros((self.num_envs, 1)) if self.stage == "skel_trans" else torch.ones((self.num_envs, 1))# 0 for skel_trans
            # num_nodes
            self.num_nodes_buf[:, 0] = torch.tensor(self.num_nodes)
            # body index
            if self.use_body_ind:
                self.body_index = torch.from_numpy(get_body_index(self.robot, self.index_base)).to(device=self.device, dtype=torch.float32)
                self.body_ind_buf[:, :self.num_nodes] = self.body_index.unsqueeze(0).repeat(self.num_envs, 1)

            #-------------------------- agent opponent -----------------------#
            attr_fixed_obs_op = torch.from_numpy(get_attr_fixed(self.cfg['robot']['obs_specs'], self.robot)).to(device=self.device, dtype=torch.float32).unsqueeze(0).repeat(self.num_envs, 1, 1)
            self.num_nodes_op = len(list(self.robot_op.bodies))
            gym_obs_op = torch.zeros((self.num_envs, self.num_nodes_op, self.gym_obs_dim), device=self.device, dtype=torch.float32)
            design_obs_op = self.design_cur_params_op.unsqueeze(0).repeat(self.num_envs, 1, 1)
            # obs
            self.obs_buf[:, self.num_nodes:] = torch.cat((attr_fixed_obs_op, gym_obs_op, design_obs_op), dim=-1)
            # edges
            if self.cfg['robot']['obs_specs'].get('fc_graph', False):
                self.edges_op = torch.from_numpy(get_graph_fc_edges(len(self.robot_op.bodies))).to(device=self.device, dtype=torch.int)
            else:
                self.edges_op = torch.from_numpy(self.robot_op.get_gnn_edges()).to(device=self.device, dtype=torch.int)
                self.edge_buf[:, :, (self.num_nodes-1)*2:] = self.edges_op.unsqueeze(0).repeat(self.num_envs, 1, 1) # create vector of edges
            # num_nodes
            self.num_nodes_buf[:, 1] = torch.tensor(self.num_nodes_op)
            if self.use_body_ind:
                self.body_index_op = torch.from_numpy(get_body_index(self.robot_op, self.index_base)).to(device=self.device, dtype=torch.float32)
                self.body_ind_buf[:, self.num_nodes:] = self.body_index_op.unsqueeze(0).repeat(self.num_envs, 1)
            
        else: # execution stage: isaacgym simulation
            assert self.isaacgym_initialized is True 
            # these refresh functions are necessary to get the latest state of the simulation
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_force_sensor_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)
            # obs
            self.obs_buf[:, :self.num_nodes] = \
                compute_ant_observations(
                    self.root_states[0::2], # ant torse states
                    self.root_states[1::2], # ant op torse states
                    self.num_nodes,
                    self.dof_pos,
                    self.dof_vel,
                    self.dof_limits_lower,
                    self.dof_limits_upper,
                    self.dof_vel_scale,
                    self.termination_height
                )
            self.obs_buf[:, self.num_nodes:] = compute_ant_observations(
                self.root_states[1::2], # ant op torse states
                self.root_states[0::2], # ant torse states
                self.num_nodes_op,
                self.dof_pos_op,
                self.dof_vel_op,
                self.dof_limits_lower_op,
                self.dof_limits_upper_op,
                self.dof_vel_scale,
                self.termination_height
            )
            # edges
            self.edge_buf[:, :, :(self.num_nodes-1)*2] = self.edges.repeat(self.num_envs, 1)
            self.edge_buf[:, :, (self.num_nodes-1)*2:] = self.edges_op.repeat(self.num_envs, 1)
            # stage flag
            self.stage_buf = torch.ones((self.num_envs, 1)) * 2 # 2 means execution stage
            # num_nodes
            self.num_nodes_buf[:, 0] = torch.tensor(self.num_nodes)
            self.num_nodes_buf[:, 1] = torch.tensor(self.num_nodes_op)
            # body index
            if self.use_body_ind:
                self.body_ind_buf[:, :self.num_nodes] = self.body_index.repeat(self.num_envs, 1)
                self.body_ind_buf[:, self.num_nodes:] = self.body_index_op.repeat(self.num_envs, 1)

    def compute_reward(self):
        if self.stage == "skel_trans" or "attr_trans":
            self.extras['win'] = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.int)
            self.extras['lose'] = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.int)
            self.extras['draw'] = torch.ones((self.num_envs, 1), device=self.device, dtype=torch.int)
        else: # execution stage
            self.rew_buf[:], self.reset_buf[:], self.hp[:], self.hp_op[:], \
            self.extras['win'], self.extras['lose'], self.extras['draw'] = compute_ant_reward(
                self.obs_buf[:, :self.num_nodes],
                self.obs_buf[:, self.num_nodes:],
                self.reset_buf,
                self.progress_buf,
                self.pos_before,
                self.torques[:, :self.num_dof],
                self.hp,
                self.hp_op,
                self.termination_height,
                self.max_episode_length,
                self.borderline_space,
                self.draw_penalty_scale,
                self.win_reward_scale,
                self.move_to_op_reward_scale,
                self.stay_in_center_reward_scale,
                self.action_cost_scale,
                self.push_scale,
                self.joints_at_limit_cost_scale,
                self.dense_reward_scale,
                self.hp_decay_scale,
                self.dt,
            )

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def expand_env_ids(env_ids, n_agents):
    # type: (Tensor, int) -> Tensor
    device = env_ids.device
    agent_env_ids = torch.zeros((n_agents * len(env_ids)), device=device, dtype=torch.long)
    for idx in range(n_agents):
        agent_env_ids[idx::n_agents] = env_ids * n_agents + idx
    return agent_env_ids


@torch.jit.script
def compute_move_reward(
        pos,
        pos_before,
        target,
        dt,
        move_to_op_reward_scale
):
    # type: (Tensor,Tensor,Tensor,float,float) -> Tensor
    move_vec = (pos - pos_before) / dt
    direction = target - pos_before
    direction = torch.div(direction, torch.linalg.norm(direction, dim=-1).view(-1, 1))
    s = torch.sum(move_vec * direction, dim=-1)
    return torch.maximum(s, torch.zeros_like(s)) * move_to_op_reward_scale


@torch.jit.script
def compute_ant_reward(
        obs_buf,
        obs_buf_op,
        reset_buf,
        progress_buf,
        pos_before,
        torques,
        hp,
        hp_op,
        termination_height,
        max_episode_length,
        borderline_space,
        draw_penalty_scale,
        win_reward_scale,
        move_to_op_reward_scale,
        stay_in_center_reward_scale,
        action_cost_scale,
        push_scale,
        joints_at_limit_cost_scale,
        dense_reward_scale,
        hp_decay_scale,
        dt,
):
    # type: (Tensor, Tensor, Tensor, Tensor,Tensor,Tensor,Tensor,Tensor,float, float,float, float,float,float,float,float,float,float,float,float,float) -> Tuple[Tensor, Tensor,Tensor,Tensor,Tensor,Tensor,Tensor]

    hp -= (obs_buf[:, 0, 2] < termination_height) * hp_decay_scale # the first node is torso
    hp_op -= (obs_buf_op[:, 0, 2] < termination_height) * hp_decay_scale # the first node is torso
    is_out = torch.sum(torch.square(obs_buf[:, 0, 0:2]), dim=-1) >= borderline_space ** 2
    is_out_op = torch.sum(torch.square(obs_buf_op[:, 0, 0:2]), dim=-1) >= borderline_space ** 2
    is_out = is_out | (hp <= 0)
    is_out_op = is_out_op | (hp_op <= 0)
    # reset agents
    tmp_ones = torch.ones_like(reset_buf)
    reset = torch.where(is_out, tmp_ones, reset_buf)
    reset = torch.where(is_out_op, tmp_ones, reset)
    reset = torch.where(progress_buf >= max_episode_length - 1, tmp_ones, reset)

    hp = torch.where(reset > 0, tmp_ones * 100., hp)
    hp_op = torch.where(reset > 0, tmp_ones * 100., hp_op)

    win_reward = win_reward_scale * is_out_op
    lose_penalty = -win_reward_scale * is_out
    draw_penalty = torch.where(progress_buf >= max_episode_length - 1, tmp_ones * draw_penalty_scale,
                               torch.zeros_like(reset, dtype=torch.float))
    move_reward = compute_move_reward(obs_buf[:, 0, 0:2], pos_before,
                                      obs_buf_op[:, 0, 0:2], dt,
                                      move_to_op_reward_scale)
    # stay_in_center_reward = stay_in_center_reward_scale * torch.exp(-torch.linalg.norm(obs_buf[:, :2], dim=-1))
    dof_at_limit_cost = torch.sum(obs_buf[:, 1:, -2] > 0.99, dim=-1) * joints_at_limit_cost_scale
    push_reward = -push_scale * torch.exp(-torch.linalg.norm(obs_buf_op[:, 0, :2], dim=-1))
    action_cost_penalty = torch.sum(torch.square(torques), dim=1) * action_cost_scale
    not_move_penalty = -10 * torch.exp(-torch.sum(torch.abs(torques), dim=1))
    dense_reward = move_reward + dof_at_limit_cost + push_reward + action_cost_penalty + not_move_penalty
    total_reward = win_reward + lose_penalty + draw_penalty + dense_reward * dense_reward_scale

    return total_reward, reset, hp, hp_op, is_out_op, is_out, progress_buf >= max_episode_length - 1


@torch.jit.script
def compute_ant_observations(
        root_states,
        root_states_op,
        num_nodes,
        dof_pos,
        dof_vel,
        dof_limits_lower,
        dof_limits_upper,
        dof_vel_scale,
        termination_height
):
    # type: (Tensor,Tensor,int,Tensor,Tensor,Tensor,Tensor,float,float)->Tensor
    dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper)
    obs_root = torch.cat((
        torch.unsqueeze(root_states[:, :13], dim=1), # 13, (num_envs, 1, 13)
        torch.unsqueeze(root_states_op[:, :7], dim=1), # 7, (num_envs, 1, 7)
        torch.unsqueeze(root_states[:, :2] - root_states_op[:, :2], dim=1), # 2, (num_envs, 1, 2)
        torch.unsqueeze(root_states[:, 2] < termination_height, dim=1), # 1, (num_envs, 1, 1)
        torch.unsqueeze(root_states_op[:, 2] < termination_height, dim=1), # 1, (num_envs, 1, 1)
        torch.zeros((root_states.shape[0], 1, 1), device=root_states.device, dtype=torch.float), # 1, root pos=0
        torch.zeros((root_states.shape[0], 1, 1), device=root_states.device, dtype=torch.float)), # 1, root vel=0
        -1
    ) # shape should be (num_envs, 1, 24)
    obs_nodes = torch.cat((
        torch.zeros((root_states.shape[0], 1, 22), device=root_states.device, dtype=torch.float), # 22, all 0
        dof_pos_scaled,
        dof_vel * dof_vel_scale),
        -1
    ) # shape should be (num_envs, num_dof, 24)
    obs = torch.cat((obs_root, obs_nodes), dim=1) # (num_envs, num_nodes, gym_obs_dim=24)
    assert obs.shape[1] == num_nodes
    return obs


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))
