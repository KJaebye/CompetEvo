from runner.base_runner import BaseRunner
from custom.learners.learner import Learner
from custom.learners.evo_learner import EvoLearner
from custom.utils.logger import MaLoggerRLV1
from lib.rl.core.trajbatch import MaTrajBatch
from lib.utils.torch import *
from lib.utils.memory import Memory

import time
import math
import os
import platform
import pickle
import multiprocessing
if platform.system() != "Linux":
    from multiprocessing import set_start_method
    set_start_method("fork")
os.environ["OMP_NUM_THREADS"] = "1"


def tensorfy(np_list, device=torch.device('cpu')):
    if isinstance(np_list[0], list):
        return [[torch.tensor(x).to(device) if i < 2 else x for i, x in enumerate(y)] for y in np_list]
    else:
        return [torch.tensor(y).to(device) for y in np_list]

class MultiEvoAgentRunner(BaseRunner):
    def __init__(self, cfg, logger, dtype, device, num_threads=1, training=True) -> None:
        super().__init__(cfg, logger, dtype, device, num_threads=num_threads, training=True)

        self.logger_cls = MaLoggerRLV1
        self.traj_cls = MaTrajBatch
        self.logger_kwargs = dict()

        self.end_reward = False
        self.custom_reward = None

    def setup_learner(self):
        """ Learners are corresponding to agents. """
        self.learners = {}
        for i, agent in self.env.agents.items():
            if "evo" in agent.team:
                self.learners[i] = EvoLearner(self.cfg, self.dtype, self.device, agent)
            else:
                self.learners[i] = Learner(self.cfg, self.dtype, self.device, agent)

    def optimize_policy(self, epoch):
        """generate multiple trajectories that reach the minimum batch_size"""
        t0 = time.time()
        batches, logs, total_scores = self.sample(self.cfg.min_batch_size)

        """update networks"""
        t1 = time.time()
        for i in self.learners:
            self.learners[i].update_params(batches[i])
        t2 = time.time()

        """evaluate policy"""
        _, log_evals, total_scores = self.sample(self.cfg.eval_batch_size, mean_action=True)
        t3 = time.time() 

        info = {
            'logs': logs, 'log_evals': log_evals, 
            'T_sample': t1 - t0, 'T_update': t2 - t1, 'T_eval': t3 - t2, 'T_total': t3 - t0, 
            'total_scores': total_scores
        }
        return info
    
    def log_optimize_policy(self, epoch, info):
        cfg = self.cfg
        logs, log_evals = info['logs'], info['log_evals']
        logger, writer = self.logger, self.writer
        for i in self.learners:
            logger.info("Agent{} gets reward: {}.".format(i, log_evals[i].avg_exec_episode_reward))
            if log_evals[i].avg_exec_episode_reward > self.best_rewards[i]:
                self.best_rewards[i] = log_evals[i].avg_exec_episode_reward
                self.save_best_flag[i] = True
            else:
                self.save_best_flag[i] = False

            writer.add_scalar('train_R_avg_{}'.format(i), logs[i].avg_reward, epoch)
            writer.add_scalar('train_R_eps_avg_{}'.format(i), logs[i].avg_episode_reward, epoch)
            writer.add_scalar('eval_R_eps_avg_{}'.format(i), log_evals[i].avg_episode_reward, epoch)
            writer.add_scalar('exec_R_avg_{}'.format(i), log_evals[i].avg_exec_reward, epoch)
            writer.add_scalar('exec_R_eps_avg_{}'.format(i), log_evals[i].avg_exec_episode_reward, epoch)
    
    def optimize(self, epoch):
        for learner in self.learners:
            learner.pre_epoch_update(epoch)
        info = self.optimize_policy(epoch)
        self.log_optimize_policy(epoch, info)

    def push_memory(self, memories, states, actions, masks, next_states, rewards, exps):
        for i, memory in enumerate(memories):
            memory.push(states[i], actions[i], masks[i], next_states[i], rewards[i], exps[i])

    def sample_worker(self, pid, queue, min_batch_size, mean_action, render):
        self.seed_worker(pid)
        # define multi-agent logger
        ma_logger = self.logger_cls(**self.logger_kwargs).loggers
        total_score = [0 for _ in self.learners]
        # define multi-agent memories
        ma_memory = []
        for i in self.learners: ma_memory.append(Memory())

        while ma_logger[0].num_steps < min_batch_size:
            states, info = self.env.reset()
            # normalize states
            for i in self.learners:
                if self.learners[i].running_state is not None:
                    states[i] = self.learners[i].running_state(states[i])
                ma_logger[i].start_episode(self.env)

            for t in range(10000):
                state_var = tensorfy([states])
                use_mean_action = mean_action or torch.bernoulli(torch.tensor([1 - self.noise_rate])).item()
                # select actions
                actions = []
                for i in self.learners:
                    actions.append(self.learners[i].policy_net.select_action(state_var, use_mean_action).numpy().astype(np.float64))
                actions = np.hstack(actions)
                next_states, env_rewards, terminateds, truncated, infos = self.env.step(actions)

                # normalize states
                for i in self.learners:
                    if self.learners[i].running_state is not None:
                        next_states[i] = self.learners[i].running_state(next_states[i])

                # use custom or env reward
                if self.custom_reward is not None:
                    c_rewards, c_infos = self.custom_reward(self.env, states, actions, env_rewards, infos)
                    rewards = c_rewards
                else:
                    c_rewards, c_infos = [0.0 for _ in self.learners], [np.array([0.0]) for _ in self.learners]
                    rewards = env_rewards
                # add end reward
                if self.end_reward and infos.get('end', False):
                    rewards += self.env.end_rewards
                # logging
                for i, logger in enumerate(ma_logger):
                    logger.step(self.env, env_rewards[i], c_rewards[i], c_infos[i], infos[i])

                masks = [1 for _ in self.learners]
                if terminateds[0]:
                    draw = True
                    for i in self.learners:
                        if "winner" in infos[i]:
                            draw = False
                            total_score[i] += 1
                    masks = [0 for _ in self.learners]

                exps = [(1-use_mean_action) for _ in self.learners]
                self.push_memory(ma_memory, states, actions, masks, next_states, rewards, exps)

                if pid == 0 and render:
                    self.env.render()
                if terminateds[0]:
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
        for learner in self.learners:
            to_test(*learner.sample_modules)
        try:
            for learner in self.learners:
                to_cpu(*learner.sample_modules)
        finally:
            with torch.no_grad():
                thread_batch_size = int(math.floor(min_batch_size / nthreads))
                queue = multiprocessing.Queue()
                memories = [None for i in range(nthreads)]
                loggers = [None for i in range(nthreads)]
                total_scores = [None for i in range(nthreads)]
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
        cfg = self.cfg
        if isinstance(checkpoint, int):
            cp_path = '%s/epoch_%04d.p' % (cfg.model_dir, checkpoint)
            epoch = checkpoint
        else:
            assert isinstance(checkpoint, str)
            cp_path = '%s/%s.p' % (cfg.model_dir, checkpoint)
        self.logger.info('loading model from checkpoint: %s' % cp_path)
        model_cp = pickle.load(open(cp_path, "rb"))
        self.policy_net.load_state_dict(model_cp['policy_dict'])
        self.value_net.load_state_dict(model_cp['value_dict'])
        self.running_state = model_cp['running_state']
        self.loss_iter = model_cp['loss_iter']
        self.best_rewards = model_cp.get('best_rewards', self.best_rewards)
        if 'epoch' in model_cp:
            epoch = model_cp['epoch']
        self.pre_epoch_update(epoch)

    def save_checkpoint(self, epoch):
        def save(cp_path):
            with to_cpu(self.policy_net, self.value_net):
                model_cp = {'policy_dict': self.policy_net.state_dict(),
                            'value_dict': self.value_net.state_dict(),
                            'running_state': self.running_state,
                            'loss_iter': self.loss_iter,
                            'best_rewards': self.best_rewards,
                            'epoch': epoch}
                pickle.dump(model_cp, open(cp_path, 'wb'))

        cfg = self.cfg
        additional_saves = self.cfg.agent_specs.get('additional_saves', None)
        if (cfg.save_model_interval > 0 and (epoch+1) % cfg.save_model_interval == 0) or \
           (additional_saves is not None and (epoch+1) % additional_saves[0] == 0 and epoch+1 <= additional_saves[1]):
            self.tb_logger.flush()
            save('%s/epoch_%04d.p' % (cfg.model_dir, epoch + 1))
        if self.save_best_flag:
            self.tb_logger.flush()
            self.logger.info(f'save best checkpoint with rewards {self.best_rewards:.2f}!')
            save('%s/best.p' % cfg.model_dir)