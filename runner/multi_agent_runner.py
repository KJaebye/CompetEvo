from runner.base_runner import BaseRunner
from custom.learners.learner import Learner
from custom.learners.evo_learner import EvoLearner
from custom.utils.logger import MaLoggerRL
from lib.rl.core.trajbatch import MaTrajBatch
from lib.utils.torch import *
from lib.utils.memory import Memory

import time
import math
import os
import platform
import pickle
import multiprocessing
import gymnasium as gym
if platform.system() != "Linux":
    from multiprocessing import set_start_method
    set_start_method("fork")
os.environ["OMP_NUM_THREADS"] = "1"

from operator import add
from functools import reduce
import collections


def tensorfy(np_list, device=torch.device('cpu')):
    if isinstance(np_list[0], list):
        return [[torch.tensor(x).to(device) if i < 2 else x for i, x in enumerate(y)] for y in np_list]
    else:
        return [torch.tensor(y).to(device) for y in np_list]

class MultiAgentRunner(BaseRunner):
    def __init__(self, cfg, logger, dtype, device, num_threads=1, training=True, ckpt=0) -> None:
        super().__init__(cfg, logger, dtype, device, num_threads=num_threads, training=training, ckpt=ckpt)
        self.agent_num = self.learners.__len__()

        self.logger_cls = MaLoggerRL
        self.traj_cls = MaTrajBatch
        self.logger_kwargs = dict()

        self.end_reward = False

    def setup_learner(self):
        """ Learners are corresponding to agents. """
        self.learners = {}
        if self.cfg.use_shadow_sample:
            """ If use opponent sampling strategy, one should fight with its old self which we call it shadow. """
            # Shadow agent owns unique policy that is loaded before every epoch but needn't an optimizer.
            # Always set idx '0' learner to shadow agent.
            self.learners[0] = Learner(self.cfg, self.dtype, self.device, self.env, is_shadow=True)
            self.learners[1] = Learner(self.cfg, self.dtype, self.device, self.env)
        else:
            for i, agent in self.env.agents.items():
                self.learners[i] = Learner(self.cfg, self.dtype, self.device, self.env)

    def optimize_policy(self):
        epoch = self.epoch
        """generate multiple trajectories that reach the minimum batch_size"""
        self.logger.info('#------------------------ Iteration {} --------------------------#'.format(epoch)) # actually this is iteration
        t0 = time.time()
        batches, logs, total_scores = self.sample(self.cfg.min_batch_size)
        t1 = time.time()
        self.logger.info(
            "Sampling {} steps by {} slaves from environment, spending {:.2f} s.".format(
            self.cfg.min_batch_size, self.num_threads, t1-t0)
        )

        """update networks"""
        use_shadow_data = False
        if use_shadow_data:
            for i in self.learners:
                for j in self.learners:
                    self.learners[i].update_params(batches[j])
        else:
            for i in self.learners:
                self.learners[i].update_params(batches[i])
        t2 = time.time()
        self.logger.info("Policy update time: {:.2f} s.".format(t2-t1))

        """evaluate policy"""
        _, log_evals, total_scores = self.sample(self.cfg.eval_batch_size, mean_action=True)
        t3 = time.time()
        self.logger.info("Evaluation time: {:.2f} s.".format(t3-t2))

        info = {
            'logs': logs, 'log_evals': log_evals, 
            'T_sample': t1 - t0, 'T_update': t2 - t1, 'T_eval': t3 - t2, 'T_total': t3 - t0, 
            'total_scores': total_scores
        }
        self.logger.info('Total time: {:10.2f} min'.format((t3 - self.t_start)/60))
        return info
    
    def log_optimize_policy(self, info):
        epoch = self.epoch
        cfg = self.cfg
        logs, log_evals, total_scores = info['logs'], info['log_evals'], info['total_scores']
        logger, writer = self.logger, self.writer
        best_reward = max([learner.best_reward for i, learner in self.learners.items()])
        for i in self.learners:
            logger.info("Agent_{} gets eval reward: {}.".format(i, log_evals[i].avg_episode_reward))
            if log_evals[i].avg_episode_reward > best_reward:
                best_reward = log_evals[i].avg_episode_reward
                self.learners[i].save_best_flag = True
            else:
                self.learners[i].save_best_flag = False
            self.learners[i].best_reward = best_reward

            # writer.add_scalar('train_R_avg_{}'.format(i), logs[i].avg_reward, epoch)
            writer.add_scalar('train_R_eps_avg_{}'.format(i), logs[i].avg_episode_reward, epoch)
            writer.add_scalar('eval_R_eps_avg_{}'.format(i), log_evals[i].avg_episode_reward, epoch)
            # writer.add_scalar('eval_R_avg_{}'.format(i), log_evals[i].avg_reward, epoch)
    
    def optimize(self, epoch):
        self.epoch = epoch
        # load shadow agent model
        if self.cfg.use_shadow_sample and epoch > 0:
            self.load_agent_params(delta=self.cfg.delta, idx=1)
        if self.cfg.use_opponent_sample and epoch > 0:
            self.load_agent_params(delta=self.cfg.delta, idx=0)
        # set annealing params
        for i in self.learners:
            self.learners[i].pre_epoch_update(epoch)
        info = self.optimize_policy()
        self.log_optimize_policy(info)

    def push_memory(self, memories, states, actions, masks, next_states, rewards, exps):
        for i, memory in enumerate(memories):
            memory.push(states[i], actions[i], masks[i], next_states[i], rewards[i], exps[i])
    
    def custom_reward(self, infos):
        """ Exploration curriculum: 
            set linear annealing factor alpha to balance parse and dense rewards. 
        """
        if hasattr(self, "epoch"):
            epoch = self.epoch
        else:
            # only for display
            epoch = 1000
        termination_epoch = self.cfg.termination_epoch
        # alpha = max((termination_epoch - epoch) / termination_epoch, 0)
        alpha = 1
        c_rew = []
        for info in infos:
            goal_rew = info['reward_remaining'] # goal_rew is parse rewarding
            move_rew = info['reward_move'] # move_rew is dense rewarding
            rew = alpha * move_rew + (1-alpha) * goal_rew
            c_rew.append(rew)
        return tuple(c_rew), infos

    def sample_worker(self, pid, queue, min_batch_size, mean_action, render):
        self.seed_worker(pid)
        
        # define multi-agent logger
        ma_logger = self.logger_cls(self.agent_num, **self.logger_kwargs).loggers
        total_score = [0 for _ in self.learners]
        # define multi-agent memories
        ma_memory = []
        for i in self.learners: ma_memory.append(Memory())
                
        while ma_logger[0].num_steps < min_batch_size:
            states, info = self.env.reset()
            # normalize states
            for i, learner in self.learners.items():
                if learner.running_state is not None:
                    states[i] = learner.running_state(states[i])
                ma_logger[i].start_episode(self.env)
            
            for t in range(10000):
                state_var = tensorfy(states)
                use_mean_action = mean_action or torch.bernoulli(torch.tensor([1 - self.noise_rate])).item()
                # select actions
                actions = []
                
                for i, learner in self.learners.items():
                    actions.append(learner.policy_net.select_action(state_var[i], use_mean_action).squeeze().numpy().astype(np.float64))
                
                next_states, env_rewards, terminateds, truncated, infos = self.env.step(actions)
                
                # normalize states
                for i, learner in self.learners.items():
                    if learner.running_state is not None:
                        next_states[i] = learner.running_state(next_states[i])

                # use custom or env reward
                if self.cfg.use_exploration_curriculum:
                    assert hasattr(self, 'custom_reward')
                    c_rewards, c_infos = self.custom_reward(infos)
                    rewards = c_rewards
                else:
                    c_rewards, c_infos = [0.0 for _ in self.learners], [np.array([0.0]) for _ in self.learners]
                    rewards = env_rewards
                # add end reward
                if self.end_reward and infos.get('end', False):
                    rewards += self.env.end_rewards
                
                # logging (logging the original rewards)
                for i, logger in enumerate(ma_logger):
                    logger.step(self.env, rewards[i], c_rewards[i], c_infos[i], infos[i])

                # normalize rewards and train use this value
                if self.cfg.use_reward_scaling:
                    assert self.learners[0].reward_scaling
                    rewards = list(rewards)
                    for i, learner in self.learners.items():
                        rewards[i] = learner.reward_scaling(rewards[i])
                    rewards = tuple(rewards)

                masks = [1 for _ in self.learners]
                if truncated:
                    draw = True
                elif terminateds[0]:
                    for i in self.learners:
                        if "winner" in infos[i]:
                            draw = False
                            total_score[i] += 1
                    masks = [0 for _ in self.learners]

                exps = [(1-use_mean_action) for _ in self.learners]
                self.push_memory(ma_memory, states, actions, masks, next_states, rewards, exps)

                if terminateds[0] or truncated:
                    break
                states = next_states

            for logger in ma_logger: logger.end_episode(self.env)
        for logger in ma_logger: logger.end_sampling()
        
        if queue is not None:
            queue.put([pid, ma_memory, ma_logger, total_score])
        else:
            return ma_memory, ma_logger, total_score

    def sample(self, min_batch_size, mean_action=False, render=False, nthreads=None):
        if nthreads is None:
            nthreads = self.num_threads

        t_start = time.time()
        for i, learner in self.learners.items():
            to_test(*learner.sample_modules)

        with to_cpu(*reduce(add, (learner.sample_modules for i, learner in self.learners.items()))):
            with torch.no_grad():
                thread_batch_size = int(math.floor(min_batch_size / nthreads))
                torch.set_num_threads(1) # bug occurs due to thread pools copy while forking
                queue = multiprocessing.Queue()
                memories = [None] * nthreads
                loggers = [None] * nthreads
                total_scores = [None] * nthreads
                for i in range(nthreads-1):
                    worker_args = (i+1, queue, thread_batch_size, mean_action, render)
                    worker = multiprocessing.Process(target=self.sample_worker, args=worker_args)
                    worker.start()
                memories[0], loggers[0], total_scores[0] = self.sample_worker(0, None, thread_batch_size, mean_action, render)

                for i in range(nthreads - 1):
                    pid, worker_memory, worker_logger, total_score = queue.get()
                    memories[pid] = worker_memory
                    loggers[pid] = worker_logger
                    total_scores[pid] = total_score

                # merge batch data and log data from multiprocessings
                ma_buffer = self.traj_cls(memories).buffers
                ma_logger = self.logger_cls.merge(loggers, **self.logger_kwargs)

                # merge total scores
                total_scores = list(map(list, zip(*total_scores)))
                total_scores = [sum(scores) for scores in total_scores]
        
        for logger in ma_logger: logger.sample_time = time.time() - t_start
        return ma_buffer, ma_logger, total_scores
    
    def load_checkpoint(self, checkpoint):
        if isinstance(checkpoint, int):
            cp_path = '%s/epoch_%04d.p' % (self.model_dir, checkpoint)
            epoch = checkpoint
        else:
            assert isinstance(checkpoint, str)
            cp_path = '%s/%s.p' % (self.model_dir, checkpoint)
        self.logger.info('loading model from checkpoint: %s' % cp_path)
        model_cp = pickle.load(open(cp_path, "rb"))

        # load epoch
        if 'epoch' in model_cp:
            epoch = model_cp['epoch']
        else:
            epoch = model_cp['0']['epoch']
        # load model
        for i, learner in self.learners.items():
            learner.load_ckpt(model_cp[str(i)])
            learner.pre_epoch_update(epoch) # set anneal params
    
    def load_agent_params(self, delta=0., idx=None):
        epoch = self.epoch
        start = max(math.floor(epoch * delta), 1)
        end = epoch
        ckpt = np.random.randint(start, end) if start!=end else end
        ckpt_path = '%s/epoch_%04d.p' % (self.model_dir, ckpt)
        shadow_model = pickle.load(open(ckpt_path, "rb"))[str(idx)]
        self.learners[idx].load_ckpt(shadow_model)

    def save_checkpoint(self, epoch):
        
        def save(cp_path, idx=None):
            model_cp = {}
            for i, learner in self.learners.items():
                if idx:
                    model_cp[str(i)] = self.learners[idx].save_ckpt(epoch)
                else:
                    model_cp[str(i)] = learner.save_ckpt(epoch)
            pickle.dump(model_cp, open(cp_path, 'wb'))

        cfg = self.cfg
        additional_saves = cfg.agent_specs.get('additional_saves', None)
        if (cfg.save_model_interval > 0 and (epoch+1) % cfg.save_model_interval == 0) or \
           (additional_saves is not None and (epoch+1) % additional_saves[0] == 0 and epoch+1 <= additional_saves[1]):
            self.writer.flush()
            save('%s/epoch_%04d.p' % (self.model_dir, epoch + 1))
        for i in self.learners:
            if self.learners[i].save_best_flag:
                self.writer.flush()
                self.logger.critical(f"save best checkpoint with agent_{i}'s rewards {self.learners[i].best_reward:.2f}!")
                save('%s/best.p' % self.model_dir, idx=i)
    
    def display(self, num_episode=3, mean_action=True):
        total_score = [0 for _ in self.learners]
        total_reward = []
        for _ in range(num_episode):
            episode_reward = [0 for _ in self.learners]
            states, info = self.env.reset()
            # normalize states
            for i, learner in self.learners.items():
                if learner.running_state is not None:
                    states[i] = learner.running_state(states[i])

            for t in range(10000):
                state_var = tensorfy(states)
                use_mean_action = mean_action or torch.bernoulli(torch.tensor([1 - self.noise_rate])).item()
                # select actions
                with torch.no_grad():
                    actions = []
                    for i, learner in self.learners.items():
                        actions.append(learner.policy_net.select_action(state_var[i], use_mean_action).squeeze().numpy().astype(np.float64))
                next_states, env_rewards, terminateds, truncated, infos = self.env.step(actions)

                # normalize states
                for i, learner in self.learners.items():
                    if learner.running_state is not None:
                        next_states[i] = learner.running_state(next_states[i])
                
                # use custom or env reward
                if self.cfg.use_exploration_curriculum:
                    assert hasattr(self, 'custom_reward')
                    c_rewards, c_infos = self.custom_reward(infos)
                    rewards = c_rewards
                else:
                    c_rewards, c_infos = [0.0 for _ in self.learners], [np.array([0.0]) for _ in self.learners]
                    rewards = env_rewards
                
                for i in self.learners:
                    episode_reward[i] += rewards[i]

                if truncated:
                    draw = True
                elif terminateds[0]:
                    for i in self.learners:
                        if "winner" in infos[i]:
                            draw = False
                            total_score[i] += 1
                    masks = [0 for _ in self.learners]
                
                if terminateds[0] or truncated:
                    break
                states = next_states
            
            total_reward.append(episode_reward)
        
        def average(list):
            total = sum(list)
            length = len(list)
            return total / length

        self.logger.info("Agent_0 gets averaged episode reward: {:.2f}".format(average(list(zip(*total_reward))[0])))
        self.logger.info("Agent_1 gets averaged episode reward: {:.2f}".format(average(list(zip(*total_reward))[1])))
        