from custom.models.transform2act_actor import Transform2ActPolicy
from custom.models.transform2act_critic import Transform2ActValue
from lib.utils.torch import *
from lib.utils.tools import *
from lib.rl.core import estimate_advantages
import math

def tensorfy(np_list, device=torch.device('cpu')):
    if isinstance(np_list[0], list):
        return [[torch.tensor(x).to(device) if i < 2 else x for i, x in enumerate(y)] for y in np_list]
    else:
        return [torch.tensor(y).to(device) for y in np_list]

class EvoLearner:
    def __init__(self, cfg, dtype, device, agent, is_shadow=False) -> None:
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.agent = agent
        self.flag = "evo"
        
        self.loss_iter = 0
        self.is_shadow = is_shadow
        
        self.setup_policy()
        self.setup_value()
        self.setup_optimizer()
        self.setup_param_scheduler()

        self.sample_modules = [self.policy_net]
        self.update_modules = [self.policy_net, self.value_net]

        self.clip_epsilon = cfg.clip_epsilon
        self.tau = cfg.tau
        self.gamma = cfg.gamma
        self.running_state = None # running_state is running_mean_std

        self.num_optim_epoch = cfg.num_optim_epoch
        self.min_batch_size = cfg.min_batch_size
        self.use_mini_batch = cfg.mini_batch_size < cfg.min_batch_size
        self.mini_batch_size = cfg.mini_batch_size

        self.value_opt_niter = 1
        self.policy_grad_clip = [(self.policy_net.parameters(), 40)]

        # initialize reward and save flag
        self.best_reward = -1000.0
        self.best_win_rate = 0.
        self.save_best_flag = False

    def setup_policy(self):
        self.policy_net = Transform2ActPolicy(self.cfg.evo_policy_specs, self.agent)
        to_device(self.device, self.policy_net)

    def setup_value(self):
        self.value_net = Transform2ActValue(self.cfg.evo_value_specs, self.agent)
        to_device(self.device, self.value_net)
    
    def save_ckpt(self, epoch):
        with to_cpu(self.policy_net, self.value_net):
            model = {'policy_dict': self.policy_net.state_dict(),
                    'value_dict': self.value_net.state_dict(),
                    'running_state': self.running_state,
                    'loss_iter': self.loss_iter,
                    'best_reward': self.best_reward,
                    'epoch': epoch}
        return model
    
    def load_ckpt(self, model):
        self.policy_net.load_state_dict(model['policy_dict'])
        self.value_net.load_state_dict(model['value_dict'])
        self.running_state = model['running_state']
        self.loss_iter = model['loss_iter']

        if 'epoch' in model:
            epoch = model['epoch']
        
        if self.is_shadow:
            # should not input averaged reward because the shadow
            # agent's averaged reward is always less than best_reward
            return
        self.best_reward = model.get('best_reward', self.best_reward)

    def setup_optimizer(self):
        cfg = self.cfg
        # policy optimizer
        if cfg.policy_optimizer == 'Adam':
            self.optimizer_policy = \
                torch.optim.Adam(self.policy_net.parameters(), 
                                 lr=cfg.policy_lr, 
                                 weight_decay=cfg.policy_weightdecay)
        else:
            self.optimizer_policy = \
                torch.optim.SGD(self.policy_net.parameters(), 
                                lr=cfg.policy_lr, 
                                momentum=cfg.policy_momentum, 
                                weight_decay=cfg.policy_weightdecay)
        # value optimizer
        if cfg.value_optimizer == 'Adam':
            self.optimizer_value = \
                torch.optim.Adam(self.value_net.parameters(), 
                                 lr=cfg.value_lr, 
                                 weight_decay=cfg.value_weightdecay)
        else:
            self.optimizer_value = \
                torch.optim.SGD(self.value_net.parameters(), 
                                lr=cfg.value_lr, 
                                momentum=cfg.value_momentum, 
                                weight_decay=cfg.value_weightdecay)

    def setup_param_scheduler(self):
        self.scheduled_params = {}
        for name, specs in self.cfg.scheduled_params.items():
            if specs['type'] == 'step':
                self.scheduled_params[name] = \
                    StepParamScheduler(specs['start_val'], 
                                       specs['step_size'], 
                                       specs['gamma'], 
                                       specs.get('smooth', False))
            elif specs['type'] == 'linear':
                self.scheduled_params[name] = \
                    LinearParamScheduler(specs['start_val'], 
                                         specs['end_val'], 
                                         specs['start_epoch'], 
                                         specs['end_epoch'])

    def pre_epoch_update(self, epoch):
        for param in self.scheduled_params.values():
            param.set_epoch(epoch)

    def update_params(self, batch):
        to_train(*self.update_modules)
        states = tensorfy(batch.states, self.device)
        actions = tensorfy(batch.actions, self.device)
        rewards = torch.from_numpy(batch.rewards).to(self.dtype).to(self.device)
        masks = torch.from_numpy(batch.masks).to(self.dtype).to(self.device)
        exps = torch.from_numpy(batch.exps).to(self.dtype).to(self.device)
        with to_test(*self.update_modules):
            with torch.no_grad():
                values = []
                chunk = 10000
                for i in range(0, len(states), chunk):
                    states_i = states[i:min(i + chunk, len(states))]
                    values_i = self.value_net(self.trans_value(states_i))
                    values.append(values_i)
                values = torch.cat(values)

        """get advantage estimation from the trajectories"""
        advantages, returns = estimate_advantages(rewards, masks, values, self.gamma, self.tau)

        if self.cfg.agent_specs.get('reinforce', False):
            advantages = returns.clone()

        return self.update_policy(states, actions, returns, advantages, exps)

    def get_perm_batch_design(self, states):
        inds = [[], [], []]
        for i, x in enumerate(states):
            use_transform_action = x[2]
            inds[use_transform_action.item()].append(i)
        perm = np.array(inds[0] + inds[1] + inds[2])
        return perm, LongTensor(perm).to(self.device)

    def update_policy(self, states, actions, returns, advantages, exps):
        """update policy"""
        with to_test(*self.update_modules):
            with torch.no_grad():
                fixed_log_probs = []
                chunk = 10000
                for i in range(0, len(states), chunk):
                    states_i = states[i:min(i + chunk, len(states))]
                    actions_i = actions[i:min(i + chunk, len(states))]
                    fixed_log_probs_i = self.policy_net.get_log_prob(self.trans_policy(states_i), actions_i)
                    fixed_log_probs.append(fixed_log_probs_i)
                fixed_log_probs = torch.cat(fixed_log_probs)
        num_state = len(states)

        policy_losses = []
        value_losses = []
        for _ in range(self.num_optim_epoch):
            if self.use_mini_batch:
                perm_np = np.arange(num_state)
                np.random.shuffle(perm_np)
                perm = LongTensor(perm_np).to(self.device)

                states, actions, returns, advantages, fixed_log_probs, exps = \
                    index_select_list(states, perm_np), index_select_list(actions, perm_np), returns[perm].clone(), advantages[perm].clone(), \
                    fixed_log_probs[perm].clone(), exps[perm].clone()

                if self.cfg.agent_specs.get('batch_design', False):
                    perm_design_np, perm_design = self.get_perm_batch_design(states)
                    states, actions, returns, advantages, fixed_log_probs, exps = \
                        index_select_list(states, perm_design_np), index_select_list(actions, perm_design_np), returns[perm_design].clone(), advantages[perm_design].clone(), \
                        fixed_log_probs[perm_design].clone(), exps[perm_design].clone()

                optim_iter_num = int(math.floor(num_state / self.mini_batch_size))
                for i in range(optim_iter_num):
                    ind = slice(i * self.mini_batch_size, min((i + 1) * self.mini_batch_size, num_state))
                    states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b, exps_b = \
                        states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind], exps[ind]
                    value_loss = self.update_value(states_b, returns_b)
                    value_losses.append(value_loss.item())
                    surr_loss = self.ppo_loss(states_b, actions_b, advantages_b, fixed_log_probs_b)
                    self.optimizer_policy.zero_grad()
                    surr_loss.backward()
                    policy_losses.append(surr_loss.item())
                    self.clip_policy_grad()
                    self.optimizer_policy.step()
            else:
                ind = exps.nonzero(as_tuple=False).squeeze(1)
                self.update_value(states, returns)
                surr_loss = self.ppo_loss(states, actions, advantages, fixed_log_probs)
                self.optimizer_policy.zero_grad()
                surr_loss.backward()
                policy_losses.append(surr_loss.item())
                self.clip_policy_grad()
                self.optimizer_policy.step()
        return np.mean(policy_losses), np.mean(value_losses)

    def ppo_loss(self, states, actions, advantages, fixed_log_probs):
        log_probs = self.policy_net.get_log_prob(self.trans_policy(states), actions)
        ratio = torch.exp(log_probs - fixed_log_probs)
        advantages = advantages
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        surr_loss = -torch.min(surr1, surr2).mean()
        return surr_loss

    def clip_policy_grad(self):
        if self.policy_grad_clip is not None:
            for params, max_norm in self.policy_grad_clip:
                torch.nn.utils.clip_grad_norm_(params, max_norm)

    def update_value(self, states, returns):
        value_losses = []
        """update critic"""
        for _ in range(self.value_opt_niter):
            values_pred = self.value_net(self.trans_value(states))
            value_loss = (values_pred - returns).pow(2).mean()
            self.optimizer_value.zero_grad()
            value_loss.backward()
            value_losses.append(value_loss.item())
            self.optimizer_value.step()
        return np.mean(value_losses)

    def trans_policy(self, states):
        """transform states before going into policy net"""
        return states

    def trans_value(self, states):
        """transform states before going into value net"""
        return states