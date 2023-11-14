from typing import Tuple
import numpy as np
import os
import math
import torch
import random

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.gymtorch import *
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
        self.abs_design = robo_config["obs_specs"]["abs_design"]
        self.use_body_ind = robo_config["obs_specs"]["use_body_ind"]

        self.sim_specs = robo_config["obs_specs"]["sim"]
        self.attr_specs = robo_config["obs_specs"]["attr"]

        # define all robots: num_envs * 2
        self.base_ant_path = f'/home/kjaebye/ws/competevo/assets/mjcf/ant.xml'
        self.robots = {}
        self.asset_files = []
        # xml tmp dir
        self.out_dir = 'out/evo_ant'
        os.makedirs(self.out_dir, exist_ok=True)
        for i in range(self.num_envs):
            name = f"evo_ant_{i}"
            name_op = f"evo_ant_op_{i}"
            self.robots[name] = Robot(robo_config, self.base_ant_path, is_xml_str=False)
            self.robots[name_op] = Robot(robo_config, self.base_ant_path, is_xml_str=False)
            self.asset_files.append(f"{name}.xml")
            self.asset_files.append(f"{name_op}.xml")

        # data saved by names
        self.design_ref_params = {}
        self.design_cur_params = {}
        self.design_param_names = {}
        self.edges = {}
        self.use_transform_action = {}
        self.num_nodes = {}
        
        for name, robot in self.robots.items():
            self.design_ref_params[name] = get_attr_design(robot)
            self.design_cur_params[name] = get_attr_design(robot)
            self.design_param_names = robot.get_params(get_name=True)
            if robo_config["obs_specs"].get('fc_graph', False):
                self.edges[name] = get_graph_fc_edges(len(robot.bodies))
            else:
                self.edges[name] = robot.get_gnn_edges()
            self.num_nodes[name] = len(list(robot.bodies))
        
        # constant variables
        self.attr_design_dim = 5
        self.attr_fixed_dim = 4
        self.gym_obs_dim = ... # 13?
        self.index_base = 5

        # actions dim: (num_nodes, action_dim)
        ###############################################################
        # action for every node:
        #    control_action      attr_action        skel_action 
        #   #-------------##--------------------##---------------#
        self.skel_num_action = 3 if robo_config["enable_remove"] else 2 # it is not an action dimension
        self.attr_action_dim = self.attr_design_dim
        self.control_action_dim = 1 
        self.action_dim = self.control_action_dim + self.attr_action_dim + 1
        
        # states dim and construction:
        # [obses, edges, stage, num_nodes, body_index]

        # obses dim (num_nodes, obs_dim)
        ###############################################################
        # observation for every node:
        #      attr_fixed            gym_obs          attr_design 
        #   #---------------##--------------------##---------------#
        self.skel_state_dim = self.attr_fixed_dim + self.attr_design_dim
        self.attr_state_dim = self.attr_fixed_dim + self.attr_design_dim
        self.obs_dim = self.attr_fixed_dim + self.gym_obs_dim + self.attr_design_dim

        self.use_central_value = False

        super().__init__(config=self.cfg, sim_device=sim_device, rl_device=rl_device,
                         graphics_device_id=graphics_device_id,
                         headless=headless, virtual_screen_capture=virtual_screen_capture,
                         force_render=force_render)

    def gym_reset(self, assets: list):
        super().gym_reset(assets)

        if self.viewer is not None:
            for env in self.envs:
                self._add_circle_borderline(env)
            cam_pos = gymapi.Vec3(15.0, 0.0, 3.0)
            cam_target = gymapi.Vec3(10.0, 0.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)

        sensors_per_env = 4
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs * self.num_agents,
                                                                          sensors_per_env * 6)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        print(f'root_states:{self.root_states.shape}')
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:, 7:13] = 0  # set lin_vel and ang_vel to 0

        # create some wrapper tensors for different slices
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        print(f"dof state shape: {self.dof_state.shape}")
        self.dof_pos = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_dof, 0]
        self.dof_pos_op = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_dof:2 * self.num_dof, 0]
        self.dof_vel = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_dof, 1]
        self.dof_vel_op = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_dof:2 * self.num_dof, 1]

        self.initial_dof_pos = torch.zeros_like(self.dof_pos, device=self.device, dtype=torch.float)
        zero_tensor = torch.tensor([0.0], device=self.device)
        self.initial_dof_pos = torch.where(self.dof_limits_lower > zero_tensor, self.dof_limits_lower,
                                           torch.where(self.dof_limits_upper < zero_tensor, self.dof_limits_upper,
                                                       self.initial_dof_pos))
        self.initial_dof_vel = torch.zeros_like(self.dof_vel, device=self.device, dtype=torch.float)
        self.dt = self.cfg["sim"]["dt"]

        torques = self.gym.acquire_dof_force_tensor(self.sim)
        self.torques = gymtorch.wrap_tensor(torques).view(self.num_envs, 2 * self.num_dof)

        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((2 * self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((2 * self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((2 * self.num_envs, 1))

        self.hp = torch.ones((self.num_envs,), device=self.device, dtype=torch.float32) * 100
        self.hp_op = torch.ones((self.num_envs,), device=self.device, dtype=torch.float32) * 100


    def step(self, actions):
        """ action shape: (num_envs, num_agents, action_dim)
        """
        self.cur_t += 1
        # skeleton transform stage
        skel_a = actions[:, :, -1]
        if self.stage == 'skel_trans':
            def skel_trans(idx, actions, op=False):
                name = f"evo_ant_{idx}" if op else f"evo_ant_op_{idx}"
                apply_skel_action(self.robots[name], actions[op, :])
                self.design_cur_params[name] = get_attr_design(self.robots[name])

            # apply skel transform actions on robots
            for idx in range(self.num_envs):
                skel_trans(idx, skel_a[idx], op=False)
                skel_trans(idx, skel_a[idx], op=True)

            if self.cur_t == self.skel_transform_nsteps:
                self.transit_attribute_transform()
            self.compute_observations(skel_a)
            self.compute_rewards()
            return
        
        # attribute transform stage
        elif self.stage == 'attr_trans':
            design_a = actions[:, :, self.control_action_dim:-1]
            
            def attr_trans(idx, design_a, op=False):
                name = f"evo_ant_{idx}" if op else f"evo_ant_op_{idx}"
                if self.abs_design:
                    design_params = design_a[op, :] * self.robot_param_scale
                else:
                    design_params = self.design_cur_params[name] \
                                    + design_a[op, :] * self.robot_param_scale
                set_design_params(self.robots[name], design_params, self.out_dir, name)

                if self.use_projected_params:
                    self.design_cur_params[name] = get_attr_design(self.robots[name])
                else:
                    self.design_cur_params[name] = design_params[op, :].copy()
            
            # apply attr transform actions on robots
            for idx in range(self.num_envs):
                attr_trans(idx, design_a[idx], op=False)
                attr_trans(idx, design_a[idx], op=True)

            self.transit_execution()

            self.compute_observations(design_a)
            self.compute_rewards()

            ###############################################################################
            ######## Create IsaacGym sim and envs for Control Training ####################
            # create assets list
            assets = ...
            self.gym_reset(assets)


        # execution stage
        else:
            self.control_nsteps += 1
            self.gym_step(actions)
            
    def transit_attribute_transform(self):
        self.stage = 'attribute_transform'

    def transit_execution(self):
        self.stage = 'execution'
        self.control_nsteps = 0

    def pre_physics_step(self, actions):
        # actions.shape = [num_envs * num_agents, num_actions], stacked as followed:
        # {[(agent1_act_1, agent1_act2)|(agent2_act1, agent2_act2)|...]_(env0),
        #  [(agent1_act_1, agent1_act2)|(agent2_act1, agent2_act2)|...]_(env1),
        #  ... }

        self.actions = actions.clone().to(self.device)
        self.actions = torch.cat((self.actions[:self.num_envs], self.actions[self.num_envs:]), dim=-1)

        # reshape [num_envs * num_agents, num_actions] to [num_envs, num_agents * num_actions]
        targets = self.actions

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(targets))

    def post_physics_step(self):
        """
            IsaacGym post step.
        """
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions)
        self.pos_before = self.obs_buf[:self.num_envs, :2].clone()

    def compute_observations(self, actions: None):
        """
            Calculate ant observations.
        """
        if self.stage == "skel_trans":
            assert actions is not None, "Skeleton transform needs actions to get obs, edge, node, and body_ind infos!"
            # agent
            self.obs_buf[:self.num_envs] = ...
            self.edge_buf[:self.num_envs] = ...
            self.stage_buf[:self.num_envs] = 0 # 0 for skel_trans
            self.num_nodes_buf[:self.num_envs] = ...
            self.body_ind[:self.num_envs] = ...
            # agent opponent
            self.obs_buf[self.num_envs:] = ...
            self.edge_buf[self.num_envs:] = ...
            self.stage_buf[self.num_envs:] = 0 # 0 for skel_trans
            self.num_nodes_buf[self.num_envs:] = ...
            self.body_ind[self.num_envs:] = ...
        elif self.stage == "attr_trans":
            assert actions is not None, "Attribute transform needs actions to get obs, edge, node, and body_ind infos!"
            # agent
            self.obs_buf[:self.num_envs] = ...
            self.edge_buf[:self.num_envs] = ...
            self.stage_buf[:self.num_envs] = 0 # 0 for skel_trans
            self.num_nodes_buf[:self.num_envs] = ...
            self.body_ind[:self.num_envs] = ...
            # agent opponent
            self.obs_buf[self.num_envs:] = ...
            self.edge_buf[self.num_envs:] = ...
            self.stage_buf[self.num_envs:] = 0 # 0 for skel_trans
            self.num_nodes_buf[self.num_envs:] = ...
            self.body_ind[self.num_envs:] = ...
        else: # execution stage
            assert self.sim_initialized is True
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_force_sensor_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)
            self.obs_buf[:self.num_envs] = \
                compute_ant_observations(
                    self.root_states[0::2],
                    self.root_states[1::2],
                    self.dof_pos,
                    self.dof_vel,
                    self.dof_limits_lower,
                    self.dof_limits_upper,
                    self.dof_vel_scale,
                    self.termination_height
                )

            self.obs_buf[self.num_envs:] = compute_ant_observations(
                self.root_states[1::2],
                self.root_states[0::2],
                self.dof_pos_op,
                self.dof_vel_op,
                self.dof_limits_lower,
                self.dof_limits_upper,
                self.dof_vel_scale,
                self.termination_height
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

    hp -= (obs_buf[:, 2] < termination_height) * hp_decay_scale
    hp_op -= (obs_buf_op[:, 2] < termination_height) * hp_decay_scale
    is_out = torch.sum(torch.square(obs_buf[:, 0:2]), dim=-1) >= borderline_space ** 2
    is_out_op = torch.sum(torch.square(obs_buf_op[:, 0:2]), dim=-1) >= borderline_space ** 2
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
    move_reward = compute_move_reward(obs_buf[:, 0:2], pos_before,
                                      obs_buf_op[:, 0:2], dt,
                                      move_to_op_reward_scale)
    # stay_in_center_reward = stay_in_center_reward_scale * torch.exp(-torch.linalg.norm(obs_buf[:, :2], dim=-1))
    dof_at_limit_cost = torch.sum(obs_buf[:, 13:21] > 0.99, dim=-1) * joints_at_limit_cost_scale
    push_reward = -push_scale * torch.exp(-torch.linalg.norm(obs_buf_op[:, :2], dim=-1))
    action_cost_penalty = torch.sum(torch.square(torques), dim=1) * action_cost_scale
    not_move_penalty = -10 * torch.exp(-torch.sum(torch.abs(torques), dim=1))
    dense_reward = move_reward + dof_at_limit_cost + push_reward + action_cost_penalty + not_move_penalty
    total_reward = win_reward + lose_penalty + draw_penalty + dense_reward * dense_reward_scale

    return total_reward, reset, hp, hp_op, is_out_op, is_out, progress_buf >= max_episode_length - 1


@torch.jit.script
def compute_ant_observations(
        root_states,
        root_states_op,
        dof_pos,
        dof_vel,
        dof_limits_lower,
        dof_limits_upper,
        dof_vel_scale,
        termination_height
):
    # type: (Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,float,float)->Tensor
    dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper)
    obs = torch.cat(
        (root_states[:, :13], dof_pos_scaled, dof_vel * dof_vel_scale, root_states_op[:, :7],
         root_states[:, :2] - root_states_op[:, :2], torch.unsqueeze(root_states[:, 2] < termination_height, -1),
         torch.unsqueeze(root_states_op[:, 2] < termination_height, -1)), dim=-1)

    return obs


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))
