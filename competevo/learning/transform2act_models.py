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
            BaseModelNetwork.__init__(self, **kwargs)
            self.transform2act_network = network
            self.action_dim = self.transform2act_network.action_dim
            self.control_action_dim = self.transform2act_network.control_action_dim
        
        def is_rnn(self):
            return self.transform2act_network.is_rnn()
            
        def forward(self, input_dict):
            """
                Input dict: {
                    'is_train' : bool,
                    'prev_actions' : torch.Tensor / None,
                    'obs' : dict,
                }
            """
            if type(input_dict['obs']) is list:
                input_dict_list = input_dict['obs']
                assert len(input_dict_list) >= 1
                res = []
                for input_dict in input_dict_list:
                    input_dict = {'obs' : input_dict, 'is_train' : True}
                    num_envs = input_dict['obs']['obses'].shape[0]
                    control_dist, attr_dist, skel_dist, \
                        node_design_mask, design_mask, \
                            total_num_nodes, num_nodes_cum_control, num_nodes_cum_design, num_nodes_cum_skel, \
                                device, values \
                        =  self.transform2act_network(input_dict)
                    
                    is_train = input_dict.get('is_train', True)
                    if is_train:
                        prev_action_log_prob = self.get_log_prob(control_dist, attr_dist, skel_dist, node_design_mask, design_mask, \
                                                            num_nodes_cum_control, num_nodes_cum_design, num_nodes_cum_skel, device, actions)
                        result = {
                            'prev_neglogp' : torch.squeeze(prev_action_log_prob),
                            'value' : values,
                        }
                    else:
                        actions = self.select_actions(control_dist, attr_dist, skel_dist, node_design_mask, total_num_nodes, device)
                        action_log_prob = self.get_log_prob(control_dist, attr_dist, skel_dist, node_design_mask, design_mask, \
                                                            num_nodes_cum_control, num_nodes_cum_design, num_nodes_cum_skel, device, actions)
                        actions = actions.view(num_envs, -1, self.action_dim)
                        result = {
                            'neglogpacs' : torch.squeeze(action_log_prob),
                            'values' : values,
                            'actions' : actions,
                        }
                    
                    print(result['value'].shape)
                    return 









            else:
                num_envs = input_dict['obs']['obses'].shape[0]
                control_dist, attr_dist, skel_dist, \
                    node_design_mask, design_mask, \
                        total_num_nodes, num_nodes_cum_control, num_nodes_cum_design, num_nodes_cum_skel, \
                            device, values \
                    =  self.transform2act_network(input_dict)
                
                is_train = input_dict.get('is_train', True)
                if is_train:
                    prev_action_log_prob = self.get_log_prob(control_dist, attr_dist, skel_dist, node_design_mask, design_mask, \
                                                        num_nodes_cum_control, num_nodes_cum_design, num_nodes_cum_skel, device, actions)
                    result = {
                        'prev_neglogp' : torch.squeeze(prev_action_log_prob),
                        'value' : values,
                    }
                    return result
                else:
                    actions = self.select_actions(control_dist, attr_dist, skel_dist, node_design_mask, total_num_nodes, device)
                    action_log_prob = self.get_log_prob(control_dist, attr_dist, skel_dist, node_design_mask, design_mask, \
                                                        num_nodes_cum_control, num_nodes_cum_design, num_nodes_cum_skel, device, actions)
                    actions = actions.view(num_envs, -1, self.action_dim)
                    result = {
                        'neglogpacs' : torch.squeeze(action_log_prob),
                        'values' : values,
                        'actions' : actions,
                    }
                    return  result

        def select_actions(
                self, 
                control_dist, 
                attr_dist, 
                skel_dist, 
                node_design_mask, 
                total_num_nodes, 
                device, 
                mean_action=False
            ):
            #---------------- select actions ---------------------------#
            if control_dist is not None:
                control_action = control_dist.mean_sample() if mean_action else control_dist.sample()
            else:
                control_action = None

            if attr_dist is not None:
                # attr_action = attr_dist.mean_sample() if mean_action else attr_dist.sample()
                attr_action = attr_dist.mean_sample()
            else:
                attr_action = None

            if skel_dist is not None:
                # skel_action = skel_dist.mean_sample() if mean_action else skel_dist.sample()
                skel_action = skel_dist.mean_sample()
            else:
                skel_action = None

            actions = torch.zeros(total_num_nodes, self.action_dim).to(device)
            if control_action is not None:
                actions[node_design_mask['execution'], :self.control_action_dim] = control_action
            if attr_action is not None:
                actions[node_design_mask['attr_trans'], self.control_action_dim:-1] = attr_action
            if skel_action is not None:
                actions[node_design_mask['skel_trans'], [-1]] = skel_action.to(dtype=torch.float32)
            return actions

        def get_log_prob(
                self, 
                control_dist, 
                attr_dist, 
                skel_dist, 
                node_design_mask, 
                design_mask, 
                num_nodes_cum_control, 
                num_nodes_cum_design, 
                num_nodes_cum_skel, 
                device, 
                action
            ):
            action_log_prob = torch.zeros(design_mask['execution'].shape[0], 1).to(device)
            # execution log prob
            if control_dist is not None:
                control_action = action[node_design_mask['execution'], :self.control_action_dim]
                control_action_log_prob_nodes = control_dist.log_prob(control_action)
                control_action_log_prob_cum = torch.cumsum(control_action_log_prob_nodes, dim=0)
                control_action_log_prob_cum = control_action_log_prob_cum[num_nodes_cum_control - 1]
                control_action_log_prob = torch.cat([control_action_log_prob_cum[[0]], control_action_log_prob_cum[1:] - control_action_log_prob_cum[:-1]])
                action_log_prob[design_mask['execution']] = control_action_log_prob
            # attribute transform log prob
            if attr_dist is not None:
                attr_action = action[node_design_mask['attr_trans'], self.control_action_dim:-1]
                attr_action_log_prob_nodes = attr_dist.log_prob(attr_action)
                attr_action_log_prob_cum = torch.cumsum(attr_action_log_prob_nodes, dim=0)
                attr_action_log_prob_cum = attr_action_log_prob_cum[num_nodes_cum_design - 1]
                attr_action_log_prob = torch.cat([attr_action_log_prob_cum[[0]], attr_action_log_prob_cum[1:] - attr_action_log_prob_cum[:-1]])
                action_log_prob[design_mask['attr_trans']] = attr_action_log_prob
            # skeleton transform log prob
            if skel_dist is not None:
                skel_action = action[node_design_mask['skel_trans'], [-1]]
                skel_action_log_prob_nodes = skel_dist.log_prob(skel_action)
                skel_action_log_prob_cum = torch.cumsum(skel_action_log_prob_nodes, dim=0)
                skel_action_log_prob_cum = skel_action_log_prob_cum[num_nodes_cum_skel - 1]
                skel_action_log_prob = torch.cat([skel_action_log_prob_cum[[0]], skel_action_log_prob_cum[1:] - skel_action_log_prob_cum[:-1]])
                action_log_prob[design_mask['skel_trans']] = skel_action_log_prob
            return action_log_prob

