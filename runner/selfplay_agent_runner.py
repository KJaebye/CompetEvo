from runner.base_runner import BaseRunner
from custom.learners.learner import Learner
from custom.learners.sampler import Sampler
from custom.learners.evo_learner import EvoLearner
from custom.utils.logger import LoggerRL
from lib.rl.core.trajbatch import TrajBatch
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

class SPAgentRunner(BaseRunner):
    def __init__(self, cfg, logger, dtype, device, num_threads=1, training=True, ckpt_dir=None, ckpt=0) -> None:
        super().__init__(cfg, logger, dtype, device, num_threads=num_threads, training=training, ckpt_dir=ckpt_dir, ckpt=ckpt)
        self.agent_num = self.learners.__len__()

        self.logger_cls = LoggerRL
        self.traj_cls = TrajBatch
        self.logger_kwargs = dict()

        self.end_reward = False

    def setup_learner(self):
        """ Set a selfplay learner and shadow agents, which load historical policy before every rollout. """
        self.learners = {}
        # Always set idx '0' learner to ego agent.
        self.learners[0] = Learner(self.cfg, self.dtype, self.device, self.env)
        """ If use opponent sampling strategy, one should fight with its old self which we call it shadow. """
        # Shadow agent owns unique policy that is loaded before every epoch but needn't an optimizer.
        for i in range(1, self.env.n_agents):
            self.learners[i] = Learner(self.cfg, self.dtype, self.device, self.env, is_shadow=True)

    def optimize_policy(self):
        epoch = self.epoch
        """generate multiple trajectories that reach the minimum batch_size"""
        self.logger.info('#------------------------ Iteration {} --------------------------#'.format(epoch)) # actually this is iteration
        t0 = time.time()

        """sampling data"""
        batch, log, _ = self.sample(self.cfg.min_batch_size)
        t1 = time.time()
        self.logger.info(
            "Sampling {} steps by {} slaves, spending {:.2f} s.".format(
            self.cfg.min_batch_size, self.num_threads, t1-t0)
        )
        
        """updating policy"""
        self.learners[0].update_params(batch)
        t2 = time.time()
        self.logger.info("Policy update, spending: {:.2f} s.".format(t2-t1))

        """evaluate policy"""
        _, log_eval, win_rate = self.sample(self.cfg.eval_batch_size, mean_action=True, nthreads=10)
        t3 = time.time()
        self.logger.info("Evaluation time: {:.2f} s.".format(t3-t2))

        info = {
            'log': log, 'log_eval': log_eval, 
            'win_rate': win_rate
        }
        self.logger.info('Total time: {:10.2f} min'.format((t3 - self.t_start)/60))
        return info
    
    def log_optimize_policy(self, info):
        epoch = self.epoch
        cfg = self.cfg
        log, log_eval, win_rate = info['log'], info['log_eval'], info['win_rate']
        logger, writer = self.logger, self.writer

        logger.info("Agent gets eval reward: {:.2f}.".format(log_eval.avg_episode_reward))
        logger.info("Agent gets win rate: {:.2f}.".format(win_rate))
        if log_eval.avg_episode_reward > self.learners[0].best_reward or win_rate > self.learners[0].best_win_rate:
            self.learners[0].best_reward = log_eval.avg_episode_reward
            self.learners[0].best_win_rate = win_rate
            self.learners[0].save_best_flag = True
        else:
            self.learners[0].save_best_flag = False

        # writer.add_scalar('train_R_avg_{}'.format(i), logs[i].avg_reward, epoch)
        writer.add_scalar('train_R_eps_avg', log.avg_episode_reward, epoch)
        writer.add_scalar('eval_R_eps_avg', log_eval.avg_episode_reward, epoch)
        # writer.add_scalar('eval_R_avg_{}'.format(i), log_evals[i].avg_reward, epoch)
        # logging win rate
        writer.add_scalar("eval_win_rate", win_rate, epoch)
        # eps len
        writer.add_scalar("episode_length", log_eval.avg_episode_len, epoch)
    
    def optimize(self, epoch):
        self.epoch = epoch
        # set annealing params
        for i in self.learners:
            self.learners[i].pre_epoch_update(epoch)
        info = self.optimize_policy()
        self.log_optimize_policy(info)

    def push_memory(self, memory, state, action, mask, next_state, reward, exp):
        memory.push(state, action, mask, next_state, reward, exp)
    
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
        alpha = max((termination_epoch - epoch) / termination_epoch, 0)
        c_rew = []
        for i, info in enumerate(infos):
            goal_rew = info['reward_remaining'] # goal_rew is parse rewarding
            move_rew = info['reward_move'] # move_rew is dense rewarding
            rew = alpha * move_rew + (1-alpha) * goal_rew
            c_rew.append(rew)
        return tuple(c_rew), infos

    def sample_worker(self, pid, queue, min_batch_size, mean_action, render, randomstate):
        self.seed_worker(pid)
        
        # define logger
        logger = self.logger_cls(**self.logger_kwargs)
        # total score record: [agent_0_win_times, agent_1_win_times, games_num]
        total_score = [0 for _ in range(self.agent_num)]
        total_score.append(0)
        # define agent memory
        memory = Memory()

        while logger.num_steps < min_batch_size:
            # shadows load random historical policies before every rollout
            # the first learner is always ego agent
            """set sampling policy for shadows"""
            if mean_action or self.epoch == 0:
                pass
            else:
                """set sampling policy for opponent"""
                start = math.floor(self.epoch * self.cfg.delta)
                end = self.epoch
                ckpt = randomstate.randint(start, end) if start!=end else end
                ckpt = 1 if ckpt==0 else ckpt # avoid first data
                # get ckpt modeal
                cp_path = '%s/epoch_%04d.p' % (self.model_dir, ckpt)
                model_cp = pickle.load(open(cp_path, "rb"))
                self.learners[1].load_ckpt(model_cp)

            states, info = self.env.reset()
            # normalize states
            for i, learner in self.learners.items():
                if learner.running_state is not None:
                    states[i] = learner.running_state(states[i])
            logger.start_episode(self.env)
            
            for t in range(10000):
                state_var = tensorfy(states)
                
                # select actions
                actions = []
                
                for i, learner in self.learners.items():
                    use_mean_action = mean_action or torch.bernoulli(torch.tensor([1 - self.noise_rate])).item() or i == 1
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
                logger.step(self.env, rewards[0], c_rewards[0], c_infos[0], infos[0])

                # normalize rewards and train use this value
                if self.cfg.use_reward_scaling:
                    assert self.learners[0].reward_scaling
                    rewards = list(rewards)
                    rewards = self.learners[0].reward_scaling(rewards)
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

                exps = [0 for _ in self.learners]
                exps[0] = 1
                # only push ego memory
                self.push_memory(memory, states[0], actions[0], masks[0], next_states[0], rewards[0], exps[0])

                if terminateds[0] or truncated:
                    total_score[-1] += 1
                    break
                states = next_states

            logger.end_episode(self.env)
        logger.end_sampling()
        
        if queue is not None:
            queue.put([pid, memory, logger, total_score])
        else:
            return memory, logger, total_score

    def sample(self, min_batch_size, mean_action=False, render=False, nthreads=None):
        if nthreads is None:
            nthreads = self.num_threads

        t_start = time.time()
        for i, learner in self.learners.items():
            to_test(*learner.sample_modules)

        with to_cpu(*reduce(add, (learner.update_modules for i, learner in self.learners.items()))):
            with torch.no_grad():
                thread_batch_size = int(math.floor(min_batch_size / nthreads))
                torch.set_num_threads(1) # bug occurs due to thread pools copy while forking
                queue = multiprocessing.Queue()
                memories = [None] * nthreads
                loggers = [None] * nthreads
                total_scores = [None] * nthreads
                for i in range(nthreads-1):
                    worker_args = (i+1, queue, thread_batch_size, mean_action, render, np.random.RandomState())
                    worker = multiprocessing.Process(target=self.sample_worker, args=worker_args)
                    worker.start()
                memories[0], loggers[0], total_scores[0] = self.sample_worker(0, None, thread_batch_size, mean_action, render, np.random.RandomState())

                for i in range(nthreads - 1):
                    pid, worker_memory, worker_logger, total_score = queue.get()
                    memories[pid] = worker_memory
                    loggers[pid] = worker_logger
                    total_scores[pid] = total_score

                # merge batch data and log data from multiprocessings
                traj_batch = self.traj_cls(memories)
                logger = self.logger_cls.merge(loggers, **self.logger_kwargs)

                # win rate
                total_scores = list(zip(*total_scores))
                total_scores = [sum(scores) for scores in total_scores]
                win_rate = total_scores[0]/total_scores[-1]
                
            logger.sample_time = time.time() - t_start
            return traj_batch, logger, win_rate
    
    def load_checkpoint(self, ckpt_dir, checkpoint):
        assert isinstance(checkpoint, list) or isinstance(checkpoint, tuple)
        for i, learner in self.learners.items():
            self.load_agent_checkpoint(checkpoint[i], i, ckpt_dir)
    
    def load_agent_checkpoint(self, ckpt, idx, ckpt_dir=None):
        ckpt_dir = self.model_dir if not ckpt_dir else ckpt_dir
        if isinstance(ckpt, int):
            cp_path = '%s/epoch_%04d.p' % (ckpt_dir, ckpt)
        else:
            assert isinstance(ckpt, str)
            cp_path = '%s/%s.p' % (ckpt_dir, ckpt)
        self.logger.info('loading agent model from checkpoint: %s' % (cp_path))
        model_cp = pickle.load(open(cp_path, "rb"))

        # load model
        self.learners[idx].load_ckpt(model_cp)

    def save_checkpoint(self, epoch):
        def save(cp_path, idx):
            try:
                pickle.dump(self.learners[idx].save_ckpt(epoch), open(cp_path, 'wb'))
            except FileNotFoundError:
                folder_path = os.path.dirname(cp_path)
                os.makedirs(folder_path, exist_ok=True)
                pickle.dump(self.learners[idx].save_ckpt(epoch), open(cp_path, 'wb'))
            except Exception as e:
                print("An error occurred while saving the model:", e)

        cfg = self.cfg
        additional_saves = cfg.agent_specs.get('additional_saves', None)
        if (cfg.save_model_interval > 0 and (epoch+1) % cfg.save_model_interval == 0) or \
           (additional_saves is not None and (epoch+1) % additional_saves[0] == 0 and epoch+1 <= additional_saves[1]):
            self.writer.flush()
            save('%s/epoch_%04d.p' % (self.model_dir, epoch + 1), 0)
        if self.learners[0].save_best_flag:
            self.writer.flush()
            self.logger.critical(f"save best checkpoint with agent's rewards {self.learners[0].best_reward:.2f}!")
            save('%s/best.p' % (self.model_dir), 0)
    
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

        self.logger.info("Agent gets averaged episode reward: {:.2f}".format(average(list(zip(*total_reward))[0])))
        