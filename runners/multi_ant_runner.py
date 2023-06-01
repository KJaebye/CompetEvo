import torch
import numpy as np
from malib.runner.separated.env_runner import EnvRunner as SeparatedEnvRunner
from malib.runner.shared.env_runner import EnvRunner as SharedEnvRunner

def _t2n(x):
    return x.detach().cpu().numpy()

class MultiAntSeparatedRunner(SeparatedEnvRunner):
    def __init__(self, config):
        super(SeparatedEnvRunner, self).__init__(config)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_x_velocity = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                                   dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            eval_temp_actions_env = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(np.array(list(eval_obs[:, agent_id])),
                                                                                eval_rnn_states[:, agent_id],
                                                                                eval_masks[:, agent_id],
                                                                                deterministic=True)

                eval_action = eval_action.detach().cpu().numpy()
                # rearrange action
                if self.eval_envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                    for i in range(self.eval_envs.action_space[agent_id].shape):
                        eval_uc_action_env = np.eye(self.eval_envs.action_space[agent_id].high[i] + 1)[eval_action[:, i]]
                        if i == 0:
                            eval_action_env = eval_uc_action_env
                        else:
                            eval_action_env = np.concatenate((eval_action_env, eval_uc_action_env), axis=1)
                elif self.eval_envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                    eval_action_env = np.squeeze(np.eye(self.eval_envs.action_space[agent_id].n)[eval_action], 1)
                else:
                    eval_action_env = eval_action
                    # raise NotImplementedError

                eval_temp_actions_env.append(eval_action_env)
                eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)

            # [envs, agents, dim]
            eval_actions_env = []
            for i in range(self.n_eval_rollout_threads):
                eval_one_hot_action_env = []
                for eval_temp_action_env in eval_temp_actions_env:
                    eval_one_hot_action_env.append(eval_temp_action_env[i])
                eval_actions_env.append(eval_one_hot_action_env)

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_terminateds, eval_truncateds, eval_infos = \
                self.eval_envs.step(eval_actions_env)
            
            eval_episode_rewards.append(eval_rewards)
            eval_x_velocity.append(np.array([eval_infos[i]['ma_x_velocity'] for i in range(self.n_eval_rollout_threads)]))

            eval_rnn_states[eval_terminateds == True] = \
                np.zeros(((eval_terminateds == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_terminateds == True] = np.zeros(((eval_terminateds == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_x_velocity = np.array(eval_x_velocity)

        eval_train_infos = []
        for agent_id in range(self.num_agents):
            eval_average_episode_rewards = np.mean(np.sum(eval_episode_rewards[:, :, agent_id], axis=0))
            eval_train_infos.append({'eval_average_episode_rewards': eval_average_episode_rewards})
            print("eval average episode rewards of agent%i: " % agent_id + str(eval_average_episode_rewards))

        eval_env_infos = []
        for agent_id in range(self.num_agents):
            eval_average_episode_x_velocity = np.mean(np.mean(eval_x_velocity[:, :, agent_id], axis=0))
            eval_env_infos.append({'eval_average_episode_x_velocity': eval_average_episode_x_velocity})
            print("eval average episode x velocity of agent%i: " % agent_id + str(eval_average_episode_x_velocity))

        self.log_train(eval_train_infos, total_num_steps)
        self.log_env(eval_env_infos, total_num_steps)


class MultiAntSharedRunner(SharedEnvRunner):
    def __init__(self, config):
        super(SharedEnvRunner, self).__init__(config)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_x_velocity = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(np.concatenate(eval_obs),
                                                                   np.concatenate(eval_rnn_states),
                                                                   np.concatenate(eval_masks),
                                                                   deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            if self.eval_envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.eval_envs.action_space[0].shape):
                    eval_uc_actions_env = np.eye(self.eval_envs.action_space[0].high[i] + 1)[eval_actions[:, :, i]]
                    if i == 0:
                        eval_actions_env = eval_uc_actions_env
                    else:
                        eval_actions_env = np.concatenate((eval_actions_env, eval_uc_actions_env), axis=2)
            elif self.eval_envs.action_space[0].__class__.__name__ == 'Discrete':
                eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
            else:
                # raise NotImplementedError
                eval_actions_env = eval_actions

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_terminateds, eval_truncateds, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)
            eval_x_velocity.append(np.array([eval_infos[i]['ma_x_velocity'] for i in range(self.n_eval_rollout_threads)]))

            eval_rnn_states[eval_terminateds == True] = np.zeros(
                ((eval_terminateds == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_terminateds == True] = np.zeros(((eval_terminateds == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_x_velocity = np.array(eval_x_velocity)

        eval_env_infos = {}
        eval_env_infos['eval_average_episode_rewards'] = np.sum(np.array(eval_episode_rewards), axis=0)
        eval_average_episode_rewards = np.mean(eval_env_infos['eval_average_episode_rewards'])
        print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))

        eval_env_infos['eval_average_episode_x_velocity'] = np.mean(np.mean(eval_x_velocity, axis=0), axis=0)
        eval_average_episode_x_velocity = np.mean(eval_env_infos['eval_average_episode_x_velocity'])
        print('eval average x velocity of agent: ' + str(eval_average_episode_x_velocity))

        self.log_env(eval_env_infos, total_num_steps)