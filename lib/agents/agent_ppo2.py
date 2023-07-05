# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Class PendulumAgent
#   @author: by Kangyao Huang
#   @created date: 24.Nov.2022
# ------------------------------------------------------------------------------------------------------------------- #
"""
    This agent is an example for training a Pendulum.
"""

import math
import time
import pickle
import numpy as np
import torch

from lib.agents.agent import Agent
from lib.core.policy import Policy
from lib.core.critic import Value
from lib.core.zfilter import ZFilter
from lib.core.common import estimate_advantages
from torch.utils.tensorboard import SummaryWriter
from lib.core.utils import *


class AgentPPO2(Agent):
    def __init__(self, cfg, env, logger, dtype, device, num_threads, training=True, checkpoint=0, mean_action=False):
        self.cfg = cfg
        self.env = env
        self.logger = logger
        self.dtype = dtype
        self.device = device
        self.num_threads = num_threads
        self.training = training
        self.checkpoint = checkpoint
        self.total_steps = 0
        self.t_start = time.time()
        self.mean_action = mean_action

        self.setup_networks()

        if training:
            # setup tensorboard logger
            self.setup_tb_logger()
        self.save_best_flag = False

        super().__init__(self.env, self.policy_net, self.device, running_state=self.running_state, num_threads=self.num_threads)

        if checkpoint != 0 or not training:
            self.load_checkpoint(checkpoint)

        self.last = torch.tensor([0, 0, 0, 0])

    def setup_networks(self):

        self.running_state = ZFilter((self.env.observation_space.shape[0]), clip=5)

        """define actor and critic"""
        self.policy_net = Policy(self.env.observation_space.shape[0],
                                 self.env.action_space.shape[0],
                                 hidden_sizes=self.cfg.policy_spec['mlp'],
                                 activation=self.cfg.policy_spec['htype'],
                                 log_std=self.cfg.policy_spec['log_std'])
        self.value_net = Value(self.env.observation_space.shape[0],
                               hidden_size=self.cfg.value_spec['mlp'],
                               activation=self.cfg.value_spec['htype'])

        self.policy_net.to(self.device)
        self.value_net.to(self.device)

        self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=self.cfg.policy_lr)
        self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=self.cfg.value_lr)

    def setup_tb_logger(self):
        self.tb_logger = SummaryWriter(self.cfg.tb_dir)
        self.best_reward = - 1000
        self.save_best_flag = False

    def load_checkpoint(self, checkpoint):
        if isinstance(checkpoint, int):
            checkpoint_path = './tmp/%s/%s/%s/models/iter_%04d.p' % (self.cfg.domain, self.cfg.task, self.cfg.rec, checkpoint)
        else:
            assert isinstance(checkpoint, str)
            checkpoint_path = './tmp/%s/%s/%s/models/%s.p' % (self.cfg.domain, self.cfg.task, self.cfg.rec, checkpoint)

        model_checkpoint = pickle.load(open(checkpoint_path, "rb"))
        self.logger.critical('Loading model from checkpoint: %s' % checkpoint_path)

        self.policy_net.load_state_dict(model_checkpoint['policy_dict'])
        self.value_net.load_state_dict(model_checkpoint['value_dict'])
        self.running_state = model_checkpoint['running_state']

    def save_checkpoint(self, iter, log, log_eval):
        def save(checkpoint_path):
            to_device(torch.device('cpu'), self.policy_net, self.value_net)
            model_checkpoint = \
                {
                    'policy_dict': self.policy_net.state_dict(),
                    'value_dict': self.value_net.state_dict(),
                    'running_state': self.running_state,
                    'best_reward': self.best_reward,
                    'iter': iter
                }
            pickle.dump(model_checkpoint, open(checkpoint_path, 'wb'))
            to_device(self.device, self.policy_net, self.value_net)

        cfg = self.cfg

        if cfg.save_model_interval > 0 and (iter + 1) % cfg.save_model_interval == 0:
            self.tb_logger.flush()
            self.logger.critical(f'Saving the interval checkpoint with rewards {self.best_reward:.2f}')
            save('%s/iter_%04d.p' % (cfg.model_dir, iter + 1))

        if log_eval['avg_reward'] > self.best_reward:
            self.best_reward = log_eval['avg_reward']
            self.save_best_flag = True
            self.logger.critical('Get the best episode reward: {:.2f}'.format(self.best_reward))

        if self.save_best_flag:
            self.tb_logger.flush()
            self.logger.critical(f'Saving the best checkpoint with rewards {self.best_reward:.2f}')
            save('%s/best.p' % self.cfg.model_dir)
            self.save_best_flag = False

    def test(self):
        _, log_eval = self.sample(10000, mean_action=True, training=False)

    def optimize(self, iter):
        """
        Optimize and main part of logging.
        """
        self.logger.info('#-------------------------------- Iteration {} ----------------------------------#'.format(iter))

        """ generate multiple trajectories that reach the minimum batch_size """
        t0 = time.time()
        batch, log = self.sample(self.cfg.batch_size)
        t1 = time.time()
        self.logger.info('Sampling time: {:.2f} s by {} slaves'.format(t1 - t0, self.num_threads))
        self.update_params(batch, iter)
        t2 = time.time()
        self.logger.info('Policy update time: {:.2f} s'.format(t2 - t1))

        """ evaluate with determinstic action (remove noise for exploration) """
        _, log_eval = self.sample(self.cfg.eval_batch_size, mean_action=self.mean_action)

        self.tb_logger.add_scalar('train_R_avg', log['avg_reward'], iter)
        self.tb_logger.add_scalar('eval_R_eps_avg', log_eval['avg_reward'], iter)

        self.logger.info('Average TRAINING episode reward: {:.2f}'.format(log['avg_reward']))
        self.logger.info('Maximum TRAINING episode reward: {:.2f}'.format(log['max_reward']))
        self.logger.info('Average EVALUATION episode reward: {:.2f}'.format(log_eval['avg_reward']))
        self.save_checkpoint(iter, log, log_eval)
        t_cur = time.time()
        self.logger.info('Total time: {:10.2f} min'.format((t_cur - self.t_start)/60))
        self.total_steps += self.cfg.batch_size
        self.logger.info('{} total steps have happened'.format(self.total_steps))

    def update_params(self, batch, iter):
        states = torch.from_numpy(np.stack(batch.state)).to(self.dtype).to(self.device)
        actions = torch.from_numpy(np.stack(batch.action)).to(self.dtype).to(self.device)
        rewards = torch.from_numpy(np.stack(batch.reward)).to(self.dtype).to(self.device)
        masks = torch.from_numpy(np.stack(batch.mask)).to(self.dtype).to(self.device)

        with torch.no_grad():
            values = self.value_net(states)
            fixed_log_probs = self.policy_net.get_log_prob(states, actions)

        """get advantage estimation from the trajectories"""
        advantages, returns = estimate_advantages(rewards, masks, values, self.cfg.gamma, self.cfg.tau, self.device)

        """perform mini-batch PPO update"""
        self.logger.info('| %16s | %16s | %16s |' % ('policy_loss', 'value_loss', 'entropy'))
        optim_iter_num = int(math.ceil(states.shape[0] / self.cfg.mini_batch_size))
        for _ in range(self.cfg.optim_num_epoch):
            perm = np.arange(states.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm).to(self.device)

            states, actions, returns, advantages, fixed_log_probs = \
                states[perm].clone(), actions[perm].clone(), returns[perm].clone(), advantages[perm].clone(), \
                fixed_log_probs[perm].clone()

            policy_loss, value_loss, entropy = [], [], []
            for i in range(optim_iter_num):
                ind = slice(i * self.cfg.mini_batch_size, min((i + 1) * self.cfg.mini_batch_size, states.shape[0]))
                states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                    states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind]

                policy_loss_i, value_loss_i, entropy_i = \
                    self.ppo_step(self.policy_net, self.value_net, self.optimizer_policy, self.optimizer_value,
                                  1, states_b, actions_b, returns_b, advantages_b, fixed_log_probs_b,
                                  self.cfg.clip_epsilon, self.cfg.l2_reg, iter)

                policy_loss.append(policy_loss_i.detach().numpy())
                value_loss.append(value_loss_i.detach().numpy())
                entropy.append(entropy_i.detach().numpy())
            self.logger.info('| %16.4f | %16.4f | %16.4f |' %
                             (np.mean(policy_loss), np.mean(value_loss), np.mean(entropy)))

    def ppo_step(self, policy_net, value_net, optimizer_policy, optimizer_value, optim_value_iternum, states, actions,
                 returns, advantages, fixed_log_probs, clip_epsilon, l2_reg, iter):

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

        self.tb_logger.add_scalar('policy_surr', policy_surr, iter)
        self.tb_logger.add_scalar('value_loss', value_loss, iter)

        return policy_surr, value_loss, entropy
