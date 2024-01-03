from custom.models.dev_actor import DevPolicy
from custom.models.dev_critic import DevValue
from lib.utils.torch import *
from lib.utils.tools import *
from lib.rl.core import estimate_advantages
import math

def tensorfy(np_list, device=torch.device('cpu')):
    if isinstance(np_list[0], list):
        return [[torch.tensor(x).to(device) for i, x in enumerate(y)] for y in np_list]
    else:
        return [torch.tensor(y).to(device) for y in np_list]

class DevLearner:
    def __init__(self, cfg, dtype, device, agent, is_shadow=False) -> None:
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.agent = agent
        self.flag = "dev"
        
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
        self.policy_net = DevPolicy(self.cfg, self.agent)
        to_device(self.device, self.policy_net)

    def setup_value(self):
        self.value_net = DevValue(self.cfg, self.agent)
        to_device(self.device, self.value_net)
    
    def save_ckpt(self, epoch):
        with to_cpu(self.policy_net, self.value_net):
            model = {'policy_dict': self.policy_net.state_dict(),
                    'value_dict': self.value_net.state_dict(),
                    'running_state': self.running_state,
                    'best_reward': self.best_reward,
                    'epoch': epoch}
        return model
    
    def load_ckpt(self, model):
        self.policy_net.load_state_dict(model['policy_dict'])
        self.value_net.load_state_dict(model['value_dict'])
        self.running_state = model['running_state']

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

        with torch.no_grad():
            values = self.value_net(states)
            fixed_log_probs = self.policy_net.get_log_prob(states, actions)

        """get advantage estimation from the trajectories"""
        advantages, returns = estimate_advantages(rewards, masks, values, self.cfg.gamma, self.cfg.tau)

        """perform mini-batch PPO update"""
        optim_iter_num = int(math.ceil(len(states) / self.mini_batch_size))
        for _ in range(self.num_optim_epoch):
            perm_np = np.arange(len(states))
            np.random.shuffle(perm_np)
            perm = torch.LongTensor(perm_np).to(self.device)

            def index_select_list(x, ind):
                return [x[i] for i in ind]

            states, actions, returns, advantages, fixed_log_probs = \
                index_select_list(states, perm_np), index_select_list(actions, perm_np), \
                returns[perm].clone(), advantages[perm].clone(), \
                fixed_log_probs[perm].clone()

            for i in range(optim_iter_num):
                ind = slice(i * self.mini_batch_size, min((i + 1) * self.mini_batch_size, len(states)))
                states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                    states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind]

                policy_loss_i, value_loss_i, entropy_i = \
                    self.ppo_step(self.policy_net, self.value_net, self.optimizer_policy, self.optimizer_value,
                                  1, states_b, actions_b, returns_b, advantages_b, fixed_log_probs_b,
                                  self.clip_epsilon, self.cfg.l2_reg)


    def ppo_step(self, policy_net, value_net, optimizer_policy, optimizer_value, optim_value_iternum, states, actions,
                 returns, advantages, fixed_log_probs, clip_epsilon, l2_reg):

        """update critic"""
        for _ in range(optim_value_iternum):
            values_pred = value_net(states)
            value_loss = (values_pred - returns).pow(2).mean()
            # weight decay
            for param in value_net.parameters():
                value_loss += param.pow(2).sum() * float(l2_reg)
            optimizer_value.zero_grad()
            value_loss.backward()
            optimizer_value.step()

        """update policy"""
        log_probs = policy_net.get_log_prob(states, actions)
        probs = torch.exp(log_probs)
        entropy = torch.sum(-(log_probs * probs))

        ratio = torch.exp(log_probs - fixed_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        policy_surr = -torch.min(surr1, surr2).mean()
        # policy_surr = -torch.min(surr1, surr2).mean() - self.cfg.entropy_coeff * entropy
        optimizer_policy.zero_grad()
        policy_surr.backward()

        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 40)
        optimizer_policy.step()

        return policy_surr, value_loss, entropy
