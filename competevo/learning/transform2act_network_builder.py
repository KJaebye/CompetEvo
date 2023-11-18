from matplotlib.pyplot import flag
import torch
import torch.nn as nn
from collections import defaultdict

from competevo.rl.core.utils import *
from competevo.rl.core.running_norm import RunningNorm
from competevo.rl.modules.mlp import MLP
from competevo.rl.modules.gnn import GNNSimple
from competevo.rl.modules.jsmlp import JSMLP
from competevo.rl.core.distributions import Categorical, DiagGaussian

class Transform2ActBuilder():
    def __init__(self, **kwargs):
        pass

    def load(self, params):
        self.params = params

    class Network():
        def __init__(self, params, **kwargs):
            self.load(params)
            
            #######################################################################
            ####### define policy #################################################
            ########################################################################
            # skeleton transform
            self.skel_norm = RunningNorm(self.skel_state_dim)
            cur_dim = self.skel_state_dim
            if 'skel_pre_mlp' in self.actor_cfg:
                self.skel_pre_mlp = MLP(cur_dim, self.actor_cfg['skel_pre_mlp'], self.actor_cfg['htype'])
                cur_dim = self.skel_pre_mlp.out_dim
            else:
                self.skel_pre_mlp = None
            if 'skel_gnn_specs' in self.actor_cfg:
                self.skel_gnn = GNNSimple(cur_dim, self.actor_cfg['skel_gnn_specs'])
                cur_dim = self.skel_gnn.out_dim
            else:
                self.skel_gnn = None
            if 'skel_mlp' in self.actor_cfg:
                self.skel_mlp = MLP(cur_dim, self.actor_cfg['skel_mlp'], self.actor_cfg['htype'])
                cur_dim = self.skel_mlp.out_dim
            else:
                self.skel_mlp = None
            if 'skel_index_mlp' in self.actor_cfg:
                imlp_cfg = self.actor_cfg['skel_index_mlp']
                self.skel_ind_mlp = JSMLP(cur_dim, imlp_cfg['hdims'], self.skel_action_dim, imlp_cfg.get('max_index', 256), imlp_cfg.get('htype', 'tanh'), imlp_cfg['rescale_linear'], imlp_cfg.get('zero_init', False))
                cur_dim = self.skel_ind_mlp.out_dim
            else:
                self.skel_ind_mlp = None
                self.skel_action_logits = nn.Linear(cur_dim, self.skel_action_dim)

            # attribute transform
            self.attr_norm = RunningNorm(self.attr_state_dim) if self.actor_cfg.get('attr_norm', True) else None
            cur_dim = self.attr_state_dim
            if 'attr_pre_mlp' in self.actor_cfg:
                self.attr_pre_mlp = MLP(cur_dim, self.actor_cfg['attr_pre_mlp'], self.actor_cfg['htype'])
                cur_dim = self.attr_pre_mlp.out_dim
            else:
                self.attr_pre_mlp = None
            if 'attr_gnn_specs' in self.actor_cfg:
                self.attr_gnn = GNNSimple(cur_dim, self.actor_cfg['attr_gnn_specs'])
                cur_dim = self.attr_gnn.out_dim
            else:
                self.attr_gnn = None
            if 'attr_mlp' in self.actor_cfg:
                self.attr_mlp = MLP(cur_dim, self.actor_cfg['attr_mlp'], self.actor_cfg['htype'])
                cur_dim = self.attr_mlp.out_dim
            else:
                self.attr_mlp = None
            if 'attr_index_mlp' in self.actor_cfg:
                imlp_cfg = self.actor_cfg['attr_index_mlp']
                self.attr_ind_mlp = JSMLP(cur_dim, imlp_cfg['hdims'], self.attr_action_dim, imlp_cfg.get('max_index', 256), imlp_cfg.get('htype', 'tanh'), imlp_cfg['rescale_linear'], imlp_cfg.get('zero_init', False))
                cur_dim = self.attr_ind_mlp.out_dim
            else:
                self.attr_ind_mlp = None
                self.attr_action_mean = nn.Linear(cur_dim, self.attr_action_dim)
                init_fc_weights(self.attr_action_mean)
            self.attr_action_log_std = nn.Parameter(torch.ones(1, self.attr_action_dim) * self.actor_cfg['attr_log_std'], requires_grad=not self.actor_cfg['fix_attr_std'])
            
            # execution
            self.control_norm = RunningNorm(self.control_state_dim)
            cur_dim = self.control_state_dim
            if 'control_pre_mlp' in self.actor_cfg:
                self.control_pre_mlp = MLP(cur_dim, self.actor_cfg['control_pre_mlp'], self.actor_cfg['htype'])
                cur_dim = self.control_pre_mlp.out_dim
            else:
                self.control_pre_mlp = None
            if 'control_gnn_specs' in self.actor_cfg:
                self.control_gnn = GNNSimple(cur_dim, self.actor_cfg['control_gnn_specs'])
                cur_dim = self.control_gnn.out_dim
            else:
                self.control_gnn = None
            if 'control_mlp' in self.actor_cfg:
                self.control_mlp = MLP(cur_dim, self.actor_cfg['control_mlp'], self.actor_cfg['htype'])
                cur_dim = self.control_mlp.out_dim
            else:
                self.control_mlp = None
            if 'control_index_mlp' in self.actor_cfg:
                imlp_cfg = self.actor_cfg['control_index_mlp']
                self.control_ind_mlp = JSMLP(cur_dim, imlp_cfg['hdims'], self.control_action_dim, imlp_cfg.get('max_index', 256), imlp_cfg.get('htype', 'tanh'), imlp_cfg['rescale_linear'], imlp_cfg.get('zero_init', False))
                cur_dim = self.control_ind_mlp.out_dim
            else:
                self.control_ind_mlp = None
                self.control_action_mean = nn.Linear(cur_dim, self.control_action_dim)
                init_fc_weights(self.control_action_mean)
            self.control_action_log_std = nn.Parameter(torch.ones(1, self.control_action_dim) * self.actor_cfg['control_log_std'], requires_grad=not self.actor_cfg['fix_control_std'])

            #######################################################################
            ####### define value #################################################
            ########################################################################
            self.critic_norm = RunningNorm(self.value_state_dim)
            cur_dim = self.value_state_dim
            if 'pre_mlp' in self.critic_cfg:
                self.critic_pre_mlp = MLP(cur_dim, self.critic_cfg['pre_mlp'], self.critic_cfg['htype'])
                cur_dim = self.critic_pre_mlp.out_dim
            else:
                self.critic_pre_mlp = None
            if 'gnn_specs' in self.critic_cfg:
                self.critic_gnn = GNNSimple(cur_dim, self.critic_cfg['gnn_specs'])
                cur_dim = self.critic_gnn.out_dim
            else:
                self.critic_gnn = None
            if 'mlp' in self.critic_cfg:
                self.critic_mlp = MLP(cur_dim, self.critic_cfg['mlp'], self.critic_cfg['htype'])
                cur_dim = self.critic_mlp.out_dim
            else:
                self.critic_mlp = None
            self.value_head = nn.Linear(cur_dim, 1)
            init_fc_weights(self.value_head)


        def forward(self, states_dict: dict):
            """
                Input:
                    states_dict: dict of states:
                    {obses, edges, stage, num_nodes, body_index}
                    obses: (:, num_nodes, attr_fixed_dim + gym_obs_dim + attr_design_dim)
                    edges: (:, 2, num_dof * 2)
                    stage: (:, 1)
                    num_nodes: (:, 1)
                    body_index: (:, num_nodes)
            """
            stages = ['skel_trans', 'attr_trans', 'execution']
            obses = states_dict['obses']
            edges = states_dict['edges']
            stage = states_dict['stage']
            num_nodes = states_dict['num_nodes']
            body_ind = states_dict['body_index'] if "body_index" in states_dict else None

            ########################################################################
            ############ policy forward #############################################
            ########################################################################
            # re-construct states by stage
            def filter_state(x: torch.tensor, flag: str):
                if flag not in stages:
                    raise ValueError(f"stage_flag should be one of {stages}")
                return x[stage == stages.index(flag)]
                
            node_design_mask = defaultdict(torch.tensor)
            design_mask = defaultdict(torch.tensor)
            # expand stage according to num_ndoes
            stage_flat = stage.flatten()
            num_nodes_flat = num_nodes.flatten()
            expanded_stage = torch.cat([stage_flat[i].repeat(num_nodes_flat[i]) for i in range(stage_flat.size(0))])
            total_num_nodes = num_nodes.sum(0)
            expanded_stage = expanded_stage.view(-1, 1)
            assert expanded_stage.shape[0] == total_num_nodes
            # get node design mask
            for flag in stages:
                node_design_mask[flag] = (expanded_stage == stages.index(flag)).nonzero(as_tuple=True)[0]
                design_mask[flag] = (stage == stages.index(flag)).nonzero(as_tuple=True)[0]

            # skeleton trans
            skel_obs = filter_state(obses, "skel_trans")
            if skel_obs.shape[0] != 0:
                skel_obs = torch.cat((skel_obs[:, :, :self.attr_fixed_dim], skel_obs[:, :, -self.attr_design_dim:]), dim=-1)
                x = self.skel_norm(skel_obs)
                if self.skel_pre_mlp is not None:
                    x = self.skel_pre_mlp(x)
                if self.skel_gnn is not None:
                    x = self.skel_gnn(x, filter_state(edges, "skel_trans"))

                if self.skel_mlp is not None:
                    x = self.skel_mlp(x)
                if self.skel_ind_mlp is not None:
                    skel_logits = self.skel_ind_mlp(x, filter_state(body_ind, "skel_trans"))
                else:
                    skel_logits = self.skel_action_logits(x)
                skel_dist = Categorical(logits=skel_logits, uniform_prob=self.skel_uniform_prob)
            else:
                num_nodes_cum_skel = None
                skel_dist = None
            
            # attribute trans
            attr_obs = filter_state(obses, "attr_trans")
            if attr_obs.shape[0] != 0:
                attr_obs = torch.cat((attr_obs[:, :, :self.attr_fixed_dim], attr_obs[:, :, -self.attr_design_dim:]), dim=-1)
                if self.attr_norm is not None:
                    x = self.attr_norm(attr_obs)
                else:
                    x = attr_obs
                if self.attr_pre_mlp is not None:
                    x = self.attr_pre_mlp(x)
                if self.attr_gnn is not None:
                    x = self.attr_gnn(x, filter_state(edges, "attr_trans"))
                if self.attr_mlp is not None:
                    x = self.attr_mlp(x)
                if self.attr_ind_mlp is not None:
                    attr_action_mean = self.attr_ind_mlp(x, filter_state(body_ind, "attr_trans"))
                else:
                    attr_action_mean = self.attr_action_mean(x)
                attr_action_std = self.attr_action_log_std.expand_as(attr_action_mean).exp()
                attr_dist = DiagGaussian(attr_action_mean, attr_action_std)
            else:
                num_nodes_cum_design = None
                attr_dist = None
            
            # execution
            control_obs = filter_state(obses, "execution")
            if control_obs.shape[0] != 0:
                x = self.control_norm(control_obs)
                if self.control_pre_mlp is not None:
                    x = self.control_pre_mlp(x)
                if self.control_gnn is not None:
                    x = self.control_gnn(x, filter_state(edges, "execution"))
                if self.control_mlp is not None:
                    x = self.control_mlp(x)
                if self.control_ind_mlp is not None:
                    control_action_mean = self.control_ind_mlp(x, filter_state(body_ind, "execution"))
                else:
                    control_action_mean = self.control_action_mean(x)
                control_action_std = self.control_action_log_std.expand_as(control_action_mean).exp()
                control_dist = DiagGaussian(control_action_mean, control_action_std)
            else:
                num_nodes_cum_control = None
                control_dist = None
            
            ########################################################################
            ############ value forward #############################################
            ########################################################################
            num_nodes_cum = torch.cumsum(num_nodes) if num_nodes.shape[0] > 1 else None
            if self.design_flag_in_state:
                if self.onehot_design_flag:
                    design_flag_onehot = torch.zeros(expanded_stage.shape[0], 3).to(obses.device)
                    design_flag_onehot.scatter_(1, expanded_stage.unsqueeze(1), 1)
                    x = torch.cat([obses, design_flag_onehot], dim=-1)
                else:
                    x = torch.cat([obses, expanded_stage.unsqueeze(1)], dim=-1)
            else:
                x = obses
            x = self.critic_norm(x)
            if self.critic_pre_mlp is not None:
                x = self.critic_pre_mlp(x)
            if self.critic_gnn is not None:
                x = self.critic_gnn(x, edges)
            if self.critic_mlp is not None:
                x = self.critic_mlp(x)
            value_nodes = self.value_head(x)
            if num_nodes_cum is None:
                value = value_nodes[[0]]
            else:
                value = value_nodes[torch.LongTensor(torch.cat([torch.zeros(1), num_nodes_cum[:-1]]))]

            return control_dist, attr_dist, skel_dist, \
                node_design_mask, design_mask, \
                total_num_nodes, num_nodes_cum_control, num_nodes_cum_design, num_nodes_cum_skel, \
                x.device, value
                    
        def is_separate_critic(self):
            return self.separate

        def load(self, params):
            self.actor_cfg = params["policy_specs"]
            self.critic_cfg = params["value_specs"]

            # input dim related params
            self.attr_fixed_dim = params["attr_fixed_dim"]
            self.attr_design_dim = params["attr_design_dim"]
            self.gym_obs_dim = params["gym_obs_dim"]

            self.skel_state_dim = self.attr_fixed_dim + self.attr_design_dim
            self.attr_state_dim = self.attr_fixed_dim + self.attr_design_dim
            self.control_state_dim = self.attr_fixed_dim + self.gym_obs_dim + self.attr_design_dim

            # output dim related params
            self.skel_action_dim = 3 if params["enable_remove"] else 2 # skel action dimension is 1
            self.attr_action_dim = self.attr_design_dim
            self.control_action_dim = params["control_action_dim"]
            self.action_dim = self.control_action_dim + self.attr_action_dim + 1
            self.skel_uniform_prob = params.get('skel_uniform_prob', 0.0)

            # critic params
            self.design_flag_in_state = params["value_specs"].get('design_flag_in_state', False)
            self.onehot_design_flag = params["value_specs"].get('onehot_design_flag', False)
            self.value_state_dim = self.attr_fixed_dim + self.gym_obs_dim + self.attr_design_dim
            self.value_state_dim = self.value_state_dim + self.design_flag_in_state * (3 if self.onehot_design_flag else 1)



    def build(self, name, **kwargs):
        net = Transform2ActBuilder.Network(self.params, **kwargs)
        return net