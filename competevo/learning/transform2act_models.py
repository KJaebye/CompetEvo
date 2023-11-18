from rl_games.algos_torch.models import BaseModelNetwork

import torch
import numpy as np

class ModelTransform2Act():
    def __init__(self, network):
        self.model_class = 'transform2act'
        self.network_builder = network
    
    def is_rnn(self):
        return False

    def is_separate_critic(self):
        return False

    def build(self, config):
        obs_shape = config['input_shape']
        normalize_value = config.get('normalize_value', False)
        normalize_input = config.get('normalize_input', False)
        value_size = config.get('value_size', 1)
        return self.Network(self.network_builder.build(self.model_class, **config), obs_shape=obs_shape,
            normalize_value=normalize_value, normalize_input=normalize_input, value_size=value_size)

    class Network(BaseModelNetwork):
        def __init__(self, network, **kwargs):
            self.transform2act_network = network
            self.action_dim = self.transform2act_network.action_dim
            self.control_action_dim = self.transform2act_network.control_action_dim
            
        def forward(self, input_dict):
            return self.transform2act_network(input_dict)

        def select_action(self, x, mean_action=False):
            control_dist, attr_dist, skel_dist, node_design_mask, _, total_num_nodes, _, _, _, device, _ = self.forward(x)
            if control_dist is not None:
                control_action = control_dist.mean_sample() if mean_action else control_dist.sample()
            else:
                control_action = None

            if attr_dist is not None:
                attr_action = attr_dist.mean_sample() if mean_action else attr_dist.sample()
            else:
                attr_action = None

            if skel_dist is not None:
                skel_action = skel_dist.mean_sample() if mean_action else skel_dist.sample()
            else:
                skel_action = None

            action = torch.zeros(total_num_nodes, self.action_dim).to(device)
            if control_action is not None:
                action[node_design_mask['execution'], :self.control_action_dim] = control_action
            if attr_action is not None:
                action[node_design_mask['attr_trans'], self.control_action_dim:-1] = attr_action
            if skel_action is not None:
                action[node_design_mask['skel_trans'], [-1]] = skel_action
            return action

