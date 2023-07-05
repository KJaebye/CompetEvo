# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Class Agent
#   @author: Modified from khrylib by Ye Yuan, modified by Kangyao Huang
#   @created date: 27.Oct.2022
# ------------------------------------------------------------------------------------------------------------------- #

import multiprocessing
from lib.core.memory import Memory
from lib.core.utils import to_device
import torch
import gym
import numpy as np
import math
import time
import os
import platform

os.environ["OMP_NUM_THREADS"] = "1"

if platform.system() != "Linux":
    from multiprocessing import set_start_method
    set_start_method("fork")


class Agent:
    def __init__(self, env, policy_net, device, custom_reward=None, running_state=None, num_threads=1):
        self.env = env
        self.policy_net = policy_net
        self.device = device
        self.custom_reward = custom_reward
        self.running_state = running_state
        self.num_threads = num_threads
        self.noise_rate = 1.0

    def sample(self, min_batch_size, mean_action=False, render=False, training=True):
        t_start = time.time()
        to_device(torch.device('cpu'), self.policy_net)
        thread_batch_size = int(math.floor(min_batch_size / self.num_threads))
        queue = multiprocessing.Queue()
        slaves = []

        for i in range(self.num_threads - 1):
            slave_args = (i + 1, queue, self.env, self.policy_net, self.custom_reward, mean_action,
                           False, self.running_state, thread_batch_size)
            slaves.append(multiprocessing.Process(target=self.collect_samples, args=slave_args))
        for slave in slaves:
            slave.start()

        memory, log = self.collect_samples(0, None, self.env, self.policy_net, self.custom_reward, mean_action,
                                      render, self.running_state, thread_batch_size)

        slave_logs = [None] * len(slaves)
        slave_memories = [None] * len(slaves)

        for _ in slaves:
            pid, slave_memory, slave_log = queue.get()
            slave_memories[pid - 1] = slave_memory
            slave_logs[pid - 1] = slave_log

        for slave_memory in slave_memories:
            memory.append(slave_memory)

        batch = memory.sample()
        if self.num_threads > 1:
            log_list = [log] + slave_logs
            log = self.merge_log(log_list)
        to_device(self.device, self.policy_net)
        t_end = time.time()
        log['sample_time'] = t_end - t_start
        log['action_mean'] = np.mean(np.vstack(batch.action), axis=0)
        log['action_min'] = np.min(np.vstack(batch.action), axis=0)
        log['action_max'] = np.max(np.vstack(batch.action), axis=0)
        return batch, log

    def collect_samples(self, pid, queue, env, policy_net, custom_reward, mean_action, render, running_state, min_batch_size):
        if pid > 0:
            torch.manual_seed(torch.randint(0, 5000, (1,)) * pid)
            gym.utils.seeding.np_random(pid)
            # if hasattr(envs, 'np_random'):
            #     envs.np_random.integers(5000) * pid
            # envs.np_random.seed(envs.np_random.randint(5000) * pid)
            # if hasattr(envs, 'envs') and hasattr(envs.envs, 'np_random'):
            #     envs.envs.np_random.seed(envs.envs.np_random.randint(5000) * pid)
        log = dict()
        memory = Memory()
        num_steps = 0
        total_reward = 0
        min_reward = 1e6
        max_reward = -1e6
        total_c_reward = 0
        min_c_reward = 1e6
        max_c_reward = -1e6
        num_episodes = 0

        while num_steps < min_batch_size:
            observation, info = env.reset()
            state = observation

            if running_state is not None:
                state = running_state(state)
            reward_episode = 0

            for t in range(10000):
                state_var = torch.tensor(state).unsqueeze(0)
                # with torch.no_grad():
                #     if mean_action:
                #         action = policy(state_var)[0][0].numpy()
                #     else:
                #         action = policy.select_action(state_var)[0].numpy()

                if mean_action:
                    action = policy_net(state_var)[0][0].detach().numpy()
                else:
                    action = policy_net.select_action(state_var)[0].detach().numpy()

                action = int(action) if policy_net.is_disc_action else action.astype(np.float64)

                observation, reward, terminated, truncated, info = env.step(action)
                next_state = observation

                reward_episode += reward
                if running_state is not None:
                    next_state = running_state(next_state)

                if custom_reward is not None:
                    reward = custom_reward(state, action)
                    total_c_reward += reward
                    min_c_reward = min(min_c_reward, reward)
                    max_c_reward = max(max_c_reward, reward)

                mask = 0 if terminated else 1

                memory.push(state, action, mask, next_state, reward)

                # rendering
                # envs.render()

                if terminated or truncated:
                    break

                state = next_state

            env.close()

            # log stats
            num_steps += (t + 1)
            num_episodes += 1
            total_reward += reward_episode
            min_reward = min(min_reward, reward_episode)
            max_reward = max(max_reward, reward_episode)

        log['num_steps'] = num_steps
        log['num_episodes'] = num_episodes
        log['total_reward'] = total_reward
        log['avg_reward'] = total_reward / num_episodes
        log['max_reward'] = max_reward
        log['min_reward'] = min_reward
        if custom_reward is not None:
            log['total_c_reward'] = total_c_reward
            log['avg_c_reward'] = total_c_reward / num_steps
            log['max_c_reward'] = max_c_reward
            log['min_c_reward'] = min_c_reward

        if queue is not None:
            queue.put([pid, memory, log])
        else:
            return memory, log

    def merge_log(self, log_list):
        log = dict()
        log['total_reward'] = sum([x['total_reward'] for x in log_list])
        log['num_episodes'] = sum([x['num_episodes'] for x in log_list])
        log['num_steps'] = sum([x['num_steps'] for x in log_list])
        log['avg_reward'] = log['total_reward'] / log['num_episodes']
        log['max_reward'] = max([x['max_reward'] for x in log_list])
        log['min_reward'] = min([x['min_reward'] for x in log_list])
        if 'total_c_reward' in log_list[0]:
            log['total_c_reward'] = sum([x['total_c_reward'] for x in log_list])
            log['avg_c_reward'] = log['total_c_reward'] / log['num_steps']
            log['max_c_reward'] = max([x['max_c_reward'] for x in log_list])
            log['min_c_reward'] = min([x['min_c_reward'] for x in log_list])

        return log