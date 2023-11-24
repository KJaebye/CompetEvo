# License: see [LICENSE, LICENSES/isaacgymenvs/LICENSE]

import copy
from datetime import datetime
from multiprocessing import process
from gym import spaces
import numpy as np
import os
import time
from .pfsp_player_pool import PFSPPlayerPool, SinglePlayer, PFSPPlayerThreadPool, PFSPPlayerProcessPool, \
    PFSPPlayerVectorizedPool, EvoPFSPPlayerPool
from competevo.utils.utils import load_checkpoint
from rl_games.algos_torch import a2c_continuous
from rl_games.common.a2c_common import swap_and_flatten01
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import central_value
from rl_games.common import common_losses

import torch
from torch import optim
from tensorboardX import SummaryWriter
import torch.distributed as dist


class T2A_SPAgent(a2c_continuous.A2CAgent):
    def __init__(self, base_name, params):
        params['config']['device'] = params['device']
        super().__init__(base_name, params)
        self.bound_loss_type = self.config.get('bound_loss_type', None)  # None for evo

        self.player_pool_type = params['player_pool_type']
        self.base_model_config = {
            'actions_num': self.actions_num,
            'input_shape': self.obs_shape,
            'num_seqs': self.num_agents,
            'value_size': self.env_info.get('value_size', 1),
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
        }
        self.max_his_player_num = params['player_pool_length']

        if params['op_load_path']:
            self.init_op_model = self.create_model()
            self.restore_op(params['op_load_path'])
        else:
            self.init_op_model = self.model
        self.players_dir = os.path.join(self.experiment_dir, 'policy_dir')
        os.makedirs(self.players_dir, exist_ok=True)
        self.update_win_rate = params['update_win_rate']
        self.num_opponent_agents = params['num_agents'] - 1
        self.player_pool = self._build_player_pool(params)

        self.games_to_check = params['games_to_check']
        self.now_update_steps = 0
        self.max_update_steps = params['max_update_steps']
        self.update_op_num = 0
        self.update_player_pool(self.init_op_model, player_idx=self.update_op_num)
        self.resample_op(torch.arange(end=self.num_actors, device=self.device, dtype=torch.long))

        assert self.num_actors % self.max_his_player_num == 0

    def _build_player_pool(self, params):
        if self.player_pool_type == 'multi_thread':
            return PFSPPlayerProcessPool(max_length=self.max_his_player_num,
                                         device=self.device)
        elif self.player_pool_type == 'multi_process':
            return PFSPPlayerThreadPool(max_length=self.max_his_player_num,
                                        device=self.device)
        elif self.player_pool_type == 'vectorized':
            vector_model_config = self.base_model_config
            vector_model_config['num_envs'] = self.num_actors * self.num_opponent_agents
            vector_model_config['population_size'] = self.max_his_player_num

            return PFSPPlayerVectorizedPool(max_length=self.max_his_player_num, device=self.device,
                                            vector_model_config=vector_model_config, params=params)
        elif self.player_pool_type == 'evo':
            return EvoPFSPPlayerPool(max_length=self.max_his_player_num, device=self.device)
        else:
            return PFSPPlayerPool(max_length=self.max_his_player_num, device=self.device)

    def init_tensors(self):
        batch_size = self.num_agents * self.num_actors
        algo_info = {
            'num_actors' : self.num_actors,
            'horizon_length' : self.horizon_length,
            'has_central_value' : self.has_central_value,
            'use_action_masks' : self.use_action_masks
        }

        # define experience buffer for varying nodes number EVO data
        class EvoExperienceBuffer():
            """
                This buffer records data including: obses, actions, rewards, dones, values.
            """
            def __init__(self, algo_info, device) -> None:
                self.num_actors = algo_info['num_actors']
                self.horizon_length = algo_info['horizon_length']
                self.has_central_value = algo_info['has_central_value']
                self.use_action_masks = algo_info['use_action_masks']

                self.tensor_dict = {}
                self.tensor_dict['obs'] = []
                self.tensor_dict['actions'] = []

                self.tensor_dict['rewards'] = torch.zeros((self.horizon_length, self.num_actors, 1), dtype=torch.float32, device=device)
                self.tensor_dict['dones'] = torch.zeros((self.horizon_length, self.num_actors,), dtype=torch.uint8, device=device)
                self.tensor_dict['values'] = torch.zeros((self.horizon_length, self.num_actors, 1), dtype=torch.float32, device=device)
                self.tensor_dict['neglogpacs'] = torch.zeros((self.horizon_length, self.num_actors, 1), dtype=torch.float32, device=device)

            def update_data(self, key, idx, val):
                self.tensor_dict[key][idx] = val

            def update_list_data(self, key, val):
                self.tensor_dict[key].append(val)
            
            def get_transformed_list(self, transform_op, tensor_list):
                res_dict = {}
                for k in tensor_list:
                    v = self.tensor_dict.get(k)
                    if v is None:
                        continue
                    if type(v) is dict:
                        transformed_dict = {}
                        for kd,vd in v.items():
                            transformed_dict[kd] = transform_op(vd)
                        res_dict[k] = transformed_dict
                    else:
                        res_dict[k] = transform_op(v)
                
                return res_dict

        self.experience_buffer = EvoExperienceBuffer(algo_info, self.device)

        val_shape = (self.horizon_length, batch_size, self.value_size)
        current_rewards_shape = (batch_size, self.value_size)
        self.current_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.ppo_device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.ppo_device)
        self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.ppo_device)

    def play_steps(self):
        step_time = 0.0
        env_done_indices = torch.tensor([], device=self.device, dtype=torch.long)
        flag = []

        for n in range(self.horizon_length):
            # logging
            if self.vec_env.stage not in flag:
                print("#---------------------------------------------------", self.vec_env.stage, "-------------------------------------------------------#")
                flag.append(self.vec_env.stage)

            self.obs = self.env_reset(env_done_indices, gym_only=True)
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict_op = self.get_action_values(self.obs, is_op=True)
                res_dict = self.get_action_values(self.obs)

                # print(self.obs['ego']['stage'])
                # print(res_dict['actions'])
            
            self.experience_buffer.update_list_data('obs', self.obs['ego'])
            self.experience_buffer.update_list_data('actions', res_dict['actions'])

            self.experience_buffer.update_data('dones', n, self.dones)
            for k in ["values", "neglogpacs"]:
                self.experience_buffer.update_data(k, n, res_dict[k])
            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            if self.player_pool_type == 'multi_thread':
                self.player_pool.thread_pool.shutdown()
            step_time_start = time.time()
            self.obs, rewards, self.dones, infos = self.env_step(
                torch.cat((res_dict['actions'], res_dict_op['actions']), dim=1))
            step_time_end = time.time()
            step_time += (step_time_end - step_time_start)

            shaped_rewards = self.rewards_shaper(rewards)
            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * self.cast_obs(infos['time_outs']).unsqueeze(
                    1).float()

            self.experience_buffer.update_data('rewards', n, shaped_rewards)

            self.current_rewards += rewards # (num_envs, 1)
            self.current_lengths += 1 # (num_envs, 1)

            all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = self.dones.view(self.num_actors, self.num_agents).all(dim=1).nonzero(as_tuple=False)
            # print(f"env done indices: {env_done_indices}")
            # print(f"self.dones {self.dones}")
            self.game_rewards.update(self.current_rewards[env_done_indices])
            self.game_lengths.update(self.current_lengths[env_done_indices])
            self.algo_observer.process_infos(infos, env_done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

            self.player_pool.update_player_metric(infos=infos)
            self.resample_op(all_done_indices.flatten())

            env_done_indices = env_done_indices[:, 0]

        last_values = self.get_values(self.obs['obs']['ego']) # (num_envs, 1)

        fdones = self.dones.float() # (num_envs, 1)
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float() # (horizon_length, num_envs, 1)
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values

        def process_list_data(data: list):
            res = []
            for i in range(self.num_actors):
                for j in range(self.horizon_length):
                    sub_data = {}
                    for k,v in data[j].items():
                        sub_data[k] = v[i]
                    res.append(sub_data)
            return res

        batch_dict = {}
        batch_dict['obs'] = process_list_data(self.experience_buffer.tensor_dict['obs'])
        # print(batch_dict['obs'][:10])
        batch_dict['actions'] = process_list_data(self.experience_buffer.tensor_dict['actions'])
        print(batch_dict['actions'][:10])

        batch_dict['rewards'] = self.experience_buffer.tensor_dict['rewards']
        batch_dict['dones'] = self.experience_buffer.tensor_dict['dones']
        batch_dict['neglogpacs'] = self.experience_buffer.tensor_dict['neglogpacs']
        batch_dict['values'] = self.experience_buffer.tensor_dict['values']
        batch_dict['returns'] = mb_returns
        batch_dict['advs'] = mb_advs

        print("rewards:\n", batch_dict['rewards'].shape)
        print("dones:\n", batch_dict['dones'].shape)
        print("neglogpacs:\n", batch_dict['neglogpacs'].shape)
        print("values:\n", batch_dict['values'].shape)
        print("returns:\n", batch_dict['returns'].shape)
        print("advs:\n", batch_dict['advs'].shape)

        batch_dict['played_frames'] = self.batch_size
        batch_dict['step_time'] = step_time
        return batch_dict

    def env_step(self, actions):
        actions = self.preprocess_actions(actions)
        obs, rewards, dones, infos = self.vec_env.step(actions)
        splited_obs = self.split_obs(obs)
        if self.is_tensor_obses:
            if self.value_size == 1:
                rewards = rewards.unsqueeze(1)
            return self.obs_to_tensors(splited_obs), rewards.to(self.ppo_device), dones.to(self.ppo_device), infos
        else:
            if self.value_size == 1:
                rewards = np.expand_dims(rewards, axis=1)
            return self.obs_to_tensors(obs), torch.from_numpy(rewards).to(self.ppo_device).float(), torch.from_numpy(
                dones).to(self.ppo_device), infos
    
    def env_reset(self, env_ids=None, gym_only=False):
        obs = self.vec_env.reset(env_ids, gym_only)
        obs = self.obs_to_tensors(obs)
        splited_obs = self.split_obs(obs)
        return splited_obs
        
    def split_obs(self, obs:dict):
        splited_obs = {}
        obs = obs['obs']

        ego = {}
        num_nodes = obs['num_nodes'][:, 0][0].item()
        ego['obses'] = obs['obses'][:, :num_nodes]
        ego['edges'] = obs['edges'][:, :, :(num_nodes-1)*2]
        ego['stage'] = obs['stage'] # every agent has the same stage
        ego['num_nodes'] = obs['num_nodes'][:, :1]
        ego['body_ind'] = obs['body_ind'][:, :num_nodes]
        splited_obs['ego'] = ego

        op = {}
        op['obses'] = obs['obses'][:, num_nodes:]
        op['edges'] = obs['edges'][:, :, (num_nodes-1)*2:]
        op['stage'] = obs['stage'] # every agent has the same stage
        op['num_nodes'] = obs['num_nodes'][:, 1:]
        op['body_ind'] = obs['body_ind'][:, num_nodes:]
        splited_obs['op'] = op

        return splited_obs


    
    def calc_gradients(self, input_dict):
        value_preds_batch = input_dict['old_values'] # (64, 4, 1)
        old_action_log_probs_batch = input_dict['old_logp_actions'] # (64, 4)
        advantage = input_dict['advantages'] # (64, 1)
        return_batch = input_dict['returns'] # (64, 4, 1)

        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']

        # obs_batch = self._preproc_obs(obs_batch)

        lr_mul = 1.0
        curr_e_clip = self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch, 
            'obs' : obs_batch,
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_len
            
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']

            a_loss = self.actor_loss_func(old_action_log_probs_batch, action_log_probs, advantage, self.ppo, curr_e_clip)

            if self.has_value_loss:
                c_loss = common_losses.critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            else:
                c_loss = torch.zeros(1, device=self.ppo_device)
            if self.bound_loss_type == 'regularisation':
                b_loss = self.reg_loss(mu)
            elif self.bound_loss_type == 'bound':
                b_loss = self.bound_loss(mu)
            else:
                b_loss = torch.zeros(1, device=self.ppo_device)
            losses, sum_mask = torch_ext.apply_masks([a_loss.unsqueeze(1), c_loss, b_loss.unsqueeze(1)], rnn_masks)
            a_loss, c_loss, b_loss = losses[0], losses[1], losses[2]

            loss = a_loss + 0.5 * c_loss * self.critic_coef + b_loss * self.bounds_loss_coef
            
            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        #TODO: Refactor this ugliest code of they year
        self.trancate_gradients_and_step()

        self.diagnostics.mini_batch(self,
        {
            'values' : value_preds_batch,
            'returns' : return_batch,
            'new_neglogp' : action_log_probs,
            'old_neglogp' : old_action_log_probs_batch,
            'masks' : rnn_masks
        }, curr_e_clip, 0)

        self.train_result = (a_loss, c_loss, self.last_lr, lr_mul, b_loss)

    def train_epoch(self):
        super().train_epoch()

        self.set_eval()
        play_time_start = time.time()
        with torch.no_grad():
            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
            else:
                batch_dict = self.play_steps()

        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks = batch_dict.get('rnn_masks', None)

        self.set_train()
        self.curr_frames = batch_dict.pop('played_frames')
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()
        if self.has_central_value:
            self.train_central_value()

        a_losses = []
        c_losses = []
        b_losses = []

        for mini_ep in range(0, self.mini_epochs_num):
            for i in range(len(self.dataset)):
                a_loss, c_loss, last_lr, lr_mul, b_loss = self.train_actor_critic(self.dataset[i])
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss)

            if self.schedule_type == 'standard':
                self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
                self.update_lr(self.last_lr)

            self.diagnostics.mini_epoch(self, mini_ep)

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        return batch_dict['step_time'], play_time, update_time, total_time, a_losses, c_losses, b_losses, last_lr, lr_mul
    
    def train(self):
        self.init_tensors()
        self.mean_rewards = self.last_mean_rewards = -100500
        start_time = time.time()
        total_time = 0
        rep_count = 0
        # self.frame = 0  # loading from checkpoint
        self.obs = self.env_reset()

        if self.multi_gpu:
            torch.cuda.set_device(self.rank)
            print("====================broadcasting parameters")
            model_params = [self.model.state_dict()]
            dist.broadcast_object_list(model_params, 0)
            self.model.load_state_dict(model_params[0])

        while True:
            epoch_num = self.update_epoch()
            step_time, play_time, update_time, sum_time, a_losses, c_losses, b_losses, last_lr, lr_mul = self.train_epoch()
            # cleaning memory to optimize space
            self.dataset.update_values_dict(None)
            total_time += sum_time
            curr_frames = self.curr_frames * self.rank_size if self.multi_gpu else self.curr_frames
            self.frame += curr_frames
            should_exit = False

            if self.rank == 0:
                self.diagnostics.epoch(self, current_epoch=epoch_num)
                scaled_time = self.num_agents * sum_time
                scaled_play_time = self.num_agents * play_time

                frame = self.frame // self.num_agents

                if self.print_stats:
                    step_time = max(step_time, 1e-6)
                    fps_step = curr_frames / step_time
                    fps_step_inference = curr_frames / scaled_play_time
                    fps_total = curr_frames / scaled_time
                    print(
                        f'fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num}/{self.max_epochs}')

                self.write_stats(total_time, epoch_num, step_time, play_time, update_time, a_losses, c_losses,
                                 last_lr, lr_mul, frame, scaled_time, scaled_play_time, curr_frames)

                self.algo_observer.after_print_stats(frame, epoch_num, total_time)

                if self.game_rewards.current_size > 0:
                    mean_rewards = self.game_rewards.get_mean()
                    mean_lengths = self.game_lengths.get_mean()
                    self.mean_rewards = mean_rewards[0]

                    for i in range(self.value_size):
                        rewards_name = 'rewards' if i == 0 else 'rewards{0}'.format(i)
                        self.writer.add_scalar(rewards_name + '/step'.format(i), mean_rewards[i], frame)
                        self.writer.add_scalar(rewards_name + '/iter'.format(i), mean_rewards[i], epoch_num)
                        self.writer.add_scalar(rewards_name + '/time'.format(i), mean_rewards[i], total_time)

                    self.writer.add_scalar('episode_lengths/step', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)
                    self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)

                    # removed equal signs (i.e. "rew=") from the checkpoint name since it messes with hydra CLI parsing
                    checkpoint_name = self.config['name'] + '_ep_' + str(epoch_num) + '_rew_' + str(mean_rewards[0])

                    if self.save_freq > 0:
                        if (epoch_num % self.save_freq == 0) and (mean_rewards <= self.last_mean_rewards):
                            self.save(os.path.join(self.nn_dir, 'last_' + checkpoint_name))

                    if mean_rewards[0] > self.last_mean_rewards and epoch_num >= self.save_best_after:
                        print('saving next best rewards: ', mean_rewards)
                        self.last_mean_rewards = mean_rewards[0]
                        self.save(os.path.join(self.nn_dir, self.config['name']))

                        if 'score_to_win' in self.config:
                            if self.last_mean_rewards > self.config['score_to_win']:
                                print('Network won!')
                                self.save(os.path.join(self.nn_dir, checkpoint_name))
                                should_exit = True

                if epoch_num >= self.max_epochs:
                    if self.game_rewards.current_size == 0:
                        print('WARNING: Max epochs reached before any env terminated at least once')
                        mean_rewards = -np.inf

                    self.save(os.path.join(self.nn_dir,
                                           'last_' + self.config['name'] + 'ep' + str(epoch_num) + 'rew' + str(
                                               mean_rewards)))
                    print('MAX EPOCHS NUM!')
                    should_exit = True
                self.update_metric()
                update_time = 0

            if self.multi_gpu:
                should_exit_t = torch.tensor(should_exit, device=self.device).float()
                dist.broadcast(should_exit_t, 0)
                should_exit = should_exit_t.bool().item()
            if should_exit:
                return self.last_mean_rewards, epoch_num

    def write_stats(self, total_time, epoch_num, step_time, play_time, update_time, a_losses, c_losses, last_lr, lr_mul, frame, scaled_time, scaled_play_time, curr_frames):
        # do we need scaled time?
        self.diagnostics.send_info(self.writer)
        self.writer.add_scalar('performance/step_inference_rl_update_fps', curr_frames / scaled_time, frame)
        self.writer.add_scalar('performance/step_inference_fps', curr_frames / scaled_play_time, frame)
        self.writer.add_scalar('performance/step_fps', curr_frames / step_time, frame)
        self.writer.add_scalar('performance/rl_update_time', update_time, frame)
        self.writer.add_scalar('performance/step_inference_time', play_time, frame)
        self.writer.add_scalar('performance/step_time', step_time, frame)
        self.writer.add_scalar('losses/a_loss', torch_ext.mean_list(a_losses).item(), frame)
        self.writer.add_scalar('losses/c_loss', torch_ext.mean_list(c_losses).item(), frame)
                
        self.writer.add_scalar('info/last_lr', last_lr * lr_mul, frame)
        self.writer.add_scalar('info/lr_mul', lr_mul, frame)
        self.writer.add_scalar('info/e_clip', self.e_clip * lr_mul, frame)
        self.writer.add_scalar('info/epochs', epoch_num, frame)
        self.algo_observer.after_print_stats(frame, epoch_num, total_time)

    def update_metric(self):
        tot_win_rate = 0
        tot_games_num = 0
        self.now_update_steps += 1
        # self_player process
        for player in self.player_pool.players:
            win_rate = player.win_rate()
            games = player.games_num()
            self.writer.add_scalar(f'rate/win_rate_player_{player.player_idx}', win_rate, self.epoch_num)
            tot_win_rate += win_rate * games
            tot_games_num += games
        win_rate = tot_win_rate / tot_games_num
        if tot_games_num > self.games_to_check:
            self.check_update_opponent(win_rate)
        self.writer.add_scalar('rate/win_rate', win_rate, self.epoch_num)

    def get_action_values(self, obs, is_op=False):
        processed_obs = self._preproc_obs(obs['op'] if is_op else obs['ego'])
        if not is_op:
            self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': processed_obs,
        }
        with torch.no_grad():
            if is_op:
                res_dict = self.player_pool.inference(input_dict, processed_obs)
            else:
                res_dict = self.model(input_dict)
            if self.has_central_value:
                states = obs['states']
                input_dict = {
                    'is_train': False,
                    'states': states,
                }
                value = self.get_central_value(input_dict)
                res_dict['values'] = value
        return res_dict

    def get_values(self, obs):
        with torch.no_grad():
            if self.has_central_value:
                states = obs['states']
                self.central_value_net.eval()
                input_dict = {
                    'is_train': False,
                    'states' : states,
                    'actions' : None,
                    'is_done': self.dones,
                }
                value = self.get_central_value(input_dict)
            else:
                self.model.eval()
                processed_obs = self._preproc_obs(obs)
                input_dict = {
                    'is_train': False,
                    'prev_actions': None, 
                    'obs' : processed_obs,
                }
                result = self.model(input_dict)
                value = result['values']
            return value

    def resample_op(self, resample_indices):
        for op_idx in range(self.num_opponent_agents):
            for player in self.player_pool.players:
                player.remove_envs(resample_indices + op_idx * self.num_actors)
        for op_idx in range(self.num_opponent_agents):
            for env_idx in resample_indices:
                player = self.player_pool.sample_player()
                player.add_envs(env_idx + op_idx * self.num_actors)
        for player in self.player_pool.players:
            player.reset_envs()

    def resample_batch(self):
        env_indices = torch.arange(end=self.num_actors * self.num_opponent_agents,
                                   device=self.device, dtype=torch.long,
                                   requires_grad=False)
        step = self.num_actors // 32
        for player in self.player_pool.players:
            player.clear_envs()
        for i in range(0, self.num_actors, step):
            player = self.player_pool.sample_player()
            player.add_envs(env_indices[i:i + step])
        print("resample done")

    def restore_op(self, fn):
        checkpoint = load_checkpoint(fn, device=self.device)
        self.init_op_model.load_state_dict(checkpoint['model'])
        if self.normalize_input and 'running_mean_std' in checkpoint:
            self.init_op_model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

    def check_update_opponent(self, win_rate):
        if win_rate > self.update_win_rate or self.now_update_steps > self.max_update_steps:
            print(f'winrate:{win_rate},add opponent to player pool')
            self.update_op_num += 1
            self.now_update_steps = 0
            self.update_player_pool(self.model, player_idx=self.update_op_num)
            self.player_pool.clear_player_metric()
            self.resample_op(torch.arange(end=self.num_actors, device=self.device, dtype=torch.long))
            self.save(os.path.join(self.players_dir, f'policy_{self.update_op_num}'))

    def create_model(self):
        model = self.network.build(self.base_model_config)
        model.to(self.device)
        return model

    def update_player_pool(self, model, player_idx):
        new_model = self.create_model()
        new_model.load_state_dict(copy.deepcopy(model.state_dict()))
        if hasattr(model, 'running_mean_std'):
            new_model.running_mean_std.load_state_dict(copy.deepcopy(model.running_mean_std.state_dict()))
        player = SinglePlayer(player_idx, new_model, self.device, self.num_actors * self.num_opponent_agents)
        self.player_pool.add_player(player)
