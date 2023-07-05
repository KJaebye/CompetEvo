import yaml
import os
import glob
import numpy as np


class Config:

    def __init__(self, cfg_path):
        self.cfg_path = cfg_path
        files = glob.glob(cfg_path, recursive=True)
        assert(len(files) == 1)
        cfg = yaml.safe_load(open(files[0], 'r'))
        self.cfg = cfg
        # create dirs
        self.out_dir = '/root/ws/competevo/tmp'

        # main config
        self.env_name = cfg.get('env_name')
        self.use_gpu = cfg.get('use_gpu', bool)
        self.device = cfg.get('device', str)
        self.cuda_deterministic = cfg.get('cuda_deterministic', bool)

        # load hyperparams.
        self.algo = cfg.get('algo', str)

        # prepare params
        self.seed = cfg.get('seed', int)
        self.n_training_threads = cfg.get('n_training_threads', int)
        self.n_rollout_threads = cfg.get('n_rollout_threads', int)
        self.n_eval_rollout_threads = cfg.get('n_eval_rollout_threads', int)
        self.n_render_rollout_threads = cfg.get('n_render_rollout_threads', int)
        self.num_env_steps = cfg.get('num_env_steps', int)
        self.user_name = cfg.get('user_name', str)
        self.use_wandb = cfg.get('use_wandb', bool)

        # envs params
        self.use_obs_instead_of_state = cfg.get('use_obs_instead_of_state', bool)

        # simulation params
        self.episode_length = cfg.get('episode_length', int)

        # network params
        self.share_policy = cfg.get('share_policy', bool)
        self.use_centralized_V = cfg.get('use_centralized_V', bool)
        self.stacked_frames = cfg.get('stacked_frames', int)
        self.use_stacked_frames = cfg.get('use_stacked_frames', bool)
        self.hidden_size = cfg.get('hidden_size', 64)
        self.layer_N = cfg.get('layer_N', 2)
        self.use_ReLU = cfg.get('use_ReLU', bool)
        self.use_popart = cfg.get('use_popart', bool)
        self.use_valuenorm = cfg.get('use_valuenorm', bool)
        self.use_feature_normalization = cfg.get('use_feature_normalization', bool)
        self.use_orthogonal = cfg.get('use_orthogonal', bool)
        self.gain = cfg.get('gain', float)

        # recurrent params
        self.use_naive_recurrent_policy = cfg.get('use_naive_recurrent_policy', bool)
        self.use_recurrent_policy = cfg.get('use_recurrent_policy', bool)
        self.recurrent_N = cfg.get('recurrent_N', int)
        self.data_chunk_length = cfg.get('data_chunk_length', int)

        # optimizer params
        self.lr = cfg.get('lr', float)
        self.critic_lr = cfg.get('critic_lr', float)
        self.opti_eps = cfg.get('opti_eps', float)
        self.weight_decay = cfg.get('weight_decay', float)

        # PPO params
        self.ppo_epoch = cfg.get('ppo_epoch', int)
        self.use_clipped_value_loss = cfg.get('use_clipped_value_loss', bool)
        self.clip_param = cfg.get('clip_param', float)
        self.num_mini_batch = cfg.get('num_mini_batch', int)
        self.entropy_coef = cfg.get('entropy_coef', float)
        self.value_loss_coef = cfg.get('value_loss_coef', float)
        self.use_max_grad_norm = cfg.get('use_max_grad_norm', bool)
        self.max_grad_norm = cfg.get('max_grad_norm', float)
        self.use_gae = cfg.get('use_gae', bool)
        self.gamma = cfg.get('gamma', float)
        self.gae_lambda = cfg.get('gae_lambda',float)
        self.use_proper_time_limits = cfg.get('use_proper_time_limits', bool)
        self.use_huber_loss = cfg.get('use_huber_loss', bool)
        self.use_value_active_masks = cfg.get('use_value_active_masks', bool)
        self.use_policy_active_masks = cfg.get('use_policy_active_masks', bool)
        self.huber_delta = cfg.get('huber_delta', int)

        # run params
        self.use_linear_lr_decay = cfg.get('use_linear_lr_decay', bool)

        # save model inteval
        self.save_interval = cfg.get('save_interval', int)

        # log params
        self.log_interval = cfg.get('log_interval', int)

        # eval params
        self.use_eval = cfg.get('use_eval', bool)
        self.eval_interval = cfg.get('eval_interval', int)
        self.eval_episodes = cfg.get('eval_episodes', int)

        # render params
        self.save_video = cfg.get('save_video', bool)
        self.use_render = cfg.get('use_render', bool)
        self.render_episodes = cfg.get('render_episodes', int)
        self.ifi = cfg.get('ifi', float)

        # robot config
        self.robot_param_scale = cfg.get('robot_param_scale', 0.1)
        self.robot_cfg = cfg.get('robot', dict())

    def save_config(self, directory_path):
        # Create the YAML file path
        file_path = os.path.join(directory_path, 'config.yml')
        # Write the configuration data to the YAML file
        with open(file_path, 'w') as f:
            yaml.dump(self.cfg, f)
        print(f"Config file is saved at {file_path}")
