# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from typing import Dict, Any, Tuple
import gym
from gym import spaces
import torch
import numpy as np
import abc
from abc import ABC


class EvoEnv(ABC):
    def __init__(self, config: Dict[str, Any], rl_device: str, sim_device: str, graphics_device_id: int, headless: bool):
        """Initialise the evo env.

        Args:
            config: the configuration dictionary.
            sim_device: the device to simulate physics on. eg. 'cuda:0' or 'cpu'
            graphics_device_id: the device ID to render with.
            headless: Set to False to disable viewer rendering.
        """

        split_device = sim_device.split(":")
        self.device_type = split_device[0]
        self.device_id = int(split_device[1]) if len(split_device) > 1 else 0

        self.device = "cpu"
        if config["sim"]["use_gpu_pipeline"]:
            if self.device_type.lower() == "cuda" or self.device_type.lower() == "gpu":
                self.device = "cuda" + ":" + str(self.device_id)
            else:
                print("GPU Pipeline can only be used with GPU simulation. Forcing CPU Pipeline.")
                config["sim"]["use_gpu_pipeline"] = False

        self.rl_device = rl_device

        # Rendering
        # if training in a headless mode
        self.headless = headless

        enable_camera_sensors = config.get("enableCameraSensors", False)
        self.graphics_device_id = graphics_device_id
        if enable_camera_sensors == False and self.headless == True:
            self.graphics_device_id = -1

        self.num_environments = config["env"]["numEnvs"]
        self.num_agents = config["env"].get("numAgents", 1)  # used for multi-agent environments

        # define transform2act evo obs, action dimensions
        # self.num_observations = config["env"]["numObservations"]
        # self.num_states = config["env"].get("numStates", 0)
        # self.num_actions = config["env"]["numActions"]

        # design options
        self.skel_transform_nsteps = config["robot"]["skel_transform_nsteps"]
        self.clip_qvel = config["robot"]["obs_specs"]["clip_qvel"]
        self.use_projected_params = config["robot"]["obs_specs"]["use_projected_params"]
        self.abs_design = config["robot"]["obs_specs"]["abs_design"]
        self.use_body_ind = config["robot"]["obs_specs"]["use_body_ind"]

        self.sim_specs = config["robot"]["obs_specs"]["sim"]
        self.attr_specs = config["robot"]["obs_specs"]["attr"]

        # dims
        self.design_ref_params = 0
        self.design_cur_params = 0
        self.design_param_names = 0
        self.attr_design_dim = 0
        self.attr_fixed_dim = 0
        self.gym_obs_dim = 0
        self.index_base = 5

        # actions dim
        ###############################################################
        #      control_action       attr_action        skel_action 
        #   #-----------------##--------------------##---------------#
        ###############################################################
        self.skel_action_dim = 3 if config["robot"]["enable_remove"] else 2
        self.attr_action_dim = self.attr_design_dim
        self.control_action_dim = 1
        self.action_dim = self.control_action_dim + self.attr_action_dim + 1
        
        # obses dim
        ###############################################################
        #      attr_fixed            gym_obs          attr_design 
        #   #---------------##--------------------##---------------#
        ###############################################################
        self.skel_state_dim = self.attr_fixed_dim + self.attr_design_dim
        self.attr_state_dim = self.attr_fixed_dim + self.attr_design_dim
        self.obs_dim = self.attr_fixed_dim + self.gym_obs_dim + self.attr_design_dim

        self.control_freq_inv = config["env"].get("controlFrequencyInv", 1)
        self.clip_obs = config["env"].get("clipObservations", np.Inf)
        self.clip_actions = config["env"].get("clipActions", np.Inf)

    @abc.abstractmethod 
    def allocate_buffers(self):
        """Create torch buffers for observations, rewards, actions dones and any additional data."""

    @abc.abstractmethod
    def step(self, actions: torch.Tensor):
        """Step the physics of the environment.
        """

    @abc.abstractmethod
    def reset(self):
        """Reset the environment.
        """
    
    @abc.abstractmethod
    def gym_step(self, actions: torch.Tensor):
        """Step only for Isaac Gym.
        """
    
    @abc.abstractmethod
    def gym_reset(self):
        """Reset only for Isaac Gym.
        """

    @property
    def num_envs(self) -> int:
        """Get the number of environments."""
        return self.num_environments