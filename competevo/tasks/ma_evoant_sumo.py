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
        # self.cfg["env"]["numObservations"] = 48 # dof pos(2) + dof vel(2) + dof action(2) + feet force sensor(force&torque, 6)
        self.cfg["env"][
            "numObservations"] = 40
        self.cfg["env"]["numActions"] = 8
        self.cfg["env"]["numAgents"] = 2


        self.use_central_value = False

        super().__init__(config=self.cfg, sim_device=sim_device, rl_device=rl_device,
                         graphics_device_id=graphics_device_id,
                         headless=headless, virtual_screen_capture=virtual_screen_capture,
                         force_render=force_render)

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
            Gym post step.
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