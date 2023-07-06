from runner.base_runner import BaseRunner
from custom.learners.learner import Learner
from custom.learners.evo_learner import EvoLearner
from custom.utils.logger import LoggerRLV1
from lib.utils.torch import *
from lib.utils.memory import Memory

import time
import math
import os
import platform
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

        self.logger_cls = LoggerRLV1
        self.logger_kwargs = dict()

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
        batches, log = self.sample(self.cfg.min_batch_size)

        """update networks"""
        t1 = time.time()
        for i, learner in enumerate(self.learners):
            learner.update_params(batches[i])
        t2 = time.time()

        """evaluate policy"""
        _, log_eval = self.sample(self.cfg.eval_batch_size, mean_action=True)
        t3 = time.time() 

        info = {
            'log': log, 'log_eval': log_eval, 'T_sample': t1 - t0, 'T_update': t2 - t1, 'T_eval': t3 - t2, 'T_total': t3 - t0
        }
        return info
    
    def optimize(self, epoch):
        for learner in self.learners:
            learner.pre_epoch_update(epoch)
        info = self.optimize_policy(epoch)
        self.log_optimize_policy(epoch, info)

    def sample_worker(self, pid, queue, min_batch_size, mean_action, render):
        self.seed_worker(pid)
        ma_memory = {}
        ma_logger = {}
        for i in self.learners:
            ma_memory[i] = Memory()
            ma_logger[i] = self.logger_cls(**self.logger_kwargs)

        while ma_logger[0].num_steps < min_batch_size:
            state, info = self.env.reset()
            for i in self.learners:
                if self.learners[i].running_state is not None:
                    state[i] = self.learners[i].running_state(state[i])
                ma_logger[i].start_episode(self.env)

            for t in range(10000):
                state_var = tensorfy([state])
                use_mean_action = mean_action or torch.bernoulli(torch.tensor([1 - self.noise_rate])).item()
                actions = []
                for i in self.learners:
                    actions.append(self.learners[i].policy_net.select_action(state_var, use_mean_action).numpy().astype(np.float64))
                actions = np.hstack(actions)
                next_state, env_reward, done, info = self.env.step(actions)



                
                if self.running_state is not None:
                    next_state = self.running_state(next_state)
                # use custom or env reward
                if self.custom_reward is not None:
                    c_reward, c_info = self.custom_reward(self.env, state, action, env_reward, info)
                    reward = c_reward
                else:
                    c_reward, c_info = 0.0, np.array([0.0])
                    reward = env_reward
                # add end reward
                if self.end_reward and info.get('end', False):
                    reward += self.env.end_reward
                # logging
                logger.step(self.env, env_reward, c_reward, c_info, info)

                mask = 0 if done else 1
                exp = 1 - use_mean_action
                self.push_memory(memory, state, action, mask, next_state, reward, exp)

                if pid == 0 and render:
                    if t < 10:
                        self.env._get_viewer('human')._paused = True
                    self.env.render()
                if done:
                    break
                state = next_state

            logger.end_episode(self.env)
        logger.end_sampling()

        if queue is not None:
            queue.put([pid, memory, logger])
        else:
            return memory, logger

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
                memories = [None] * nthreads
                loggers = [None] * nthreads
                for i in range(nthreads-1):
                    worker_args = (i+1, queue, thread_batch_size, mean_action, render)
                    worker = multiprocessing.Process(target=self.sample_worker, args=worker_args)
                    worker.start()
                memories[0], loggers[0] = self.sample_worker(0, None, thread_batch_size, mean_action, render)

                for i in range(nthreads - 1):
                    pid, worker_memory, worker_logger = queue.get()
                    memories[pid] = worker_memory
                    loggers[pid] = worker_logger
                traj_batch = self.traj_cls(memories)
                logger = self.logger_cls.merge(loggers, **self.logger_kwargs)

        logger.sample_time = time.time() - t_start
        return traj_batch, logger