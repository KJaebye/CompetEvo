# !/usr/bin/envs python
import sys
import os

import setproctitle
import numpy as np
import torch
import argparse

import gymnasium as gym
from config.config import cfg

from custom.runner.multi_ant_runner import MultiAntSeparatedRunner, MultiAntSharedRunner
from malib.runner.separated.env_runner import EnvRunner as SeparatedRunner
from malib.runner.shared.env_runner import EnvRunner as SharedRunner

def make_train_env():
    from envs import CUSTOM_ENVS
    def get_env_fn(rank):
        def init_env():
            assert cfg.ENV_NAME in CUSTOM_ENVS, "Alert: Undefined environment!"
            env = gym.make(cfg.ENV_NAME, agent_num=cfg.EMAT.AGENT_NUM, render_mode='human')

            from malib.envs.env_continuous import ContinuousActionEnv
            env = ContinuousActionEnv(cfg.EMAT.AGENT_NUM, env)
            return env
        return init_env
    from malib.envs.env_wrappers import DummyVecEnv
    return DummyVecEnv([get_env_fn(i) for i in range(cfg.EMAT.N_ROLLOUT_THREADS)])

def make_eval_env():
    from envs import CUSTOM_ENVS
    def get_env_fn(rank):
        def init_env():
            assert cfg.ENV_NAME in CUSTOM_ENVS, "Alert: Undefined environment!"
            env = gym.make(cfg.ENV_NAME, agent_num=cfg.EMAT.AGENT_NUM, render_mode='human')

            from malib.envs.env_continuous import ContinuousActionEnv
            env = ContinuousActionEnv(cfg.EMAT.AGENT_NUM, env)
            return env
        return init_env
    from malib.envs.env_wrappers import DummyVecEnv
    return DummyVecEnv([get_env_fn(i) for i in range(cfg.EMAT.N_EVAL_ROLLOUT_THREADS)])


def main(args):
    # ----------------------------------------------------------------------------#
    # Load CFG file
    # ----------------------------------------------------------------------------#
    parser = argparse.ArgumentParser(description="User's arguments from terminal.")
    parser.add_argument(
        '--run_dir', 
        type=str, 
        help='run directory name',
        required=True)
    args = parser.parse_args()

    # Load config options
    run_dir = args.run_dir
    cfg_file= run_dir + "config.yaml"
    cfg.merge_from_file(cfg_file)

    # set threads to 1 when using display
    cfg.EMAT.N_ROLLOUT_THREADS = 1
    cfg.EMAT.NUM_ENV_STEPS = cfg.ENV.EPISODE_LENGTH * cfg.EMAT.AGENT_NUM
    cfg.USE_RENDER = True

    # ----------------------------------------------------------------------------#
    # Check model_dir
    # ----------------------------------------------------------------------------#
    # model dir. If training, model directory is None.
    model_dir = run_dir + "models/"

    # ----------------------------------------------------------------------------#
    # Check for training
    # ----------------------------------------------------------------------------#
    if cfg.ALGO == "rmappo":
        assert (cfg.NETWORK.USE_RECURRENT_POLICY or cfg.USE_NAIVE_RECURRENT_POLICY), \
        ("check recurrent policy!")
    elif cfg.ALGO == "mappo":
        assert \
            (cfg.NETWORK.USE_RECURRENT_POLICY == False and 
             cfg.NETWORK.USE_NAIVE_RECURRENT_POLICY == False), \
            ("check recurrent policy!")
    else:
        raise NotImplementedError

    # ----------------------------------------------------------------------------#
    # CUDA and set torch multiple threads
    # ----------------------------------------------------------------------------#
    if cfg.USE_GPU and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device(cfg.DEVICE) if cfg.DEVICE else torch.device("cuda:0")
        torch.set_num_threads(cfg.EMAT.N_TRAINING_THREADS)

        if cfg.CUDNN.DETERMINISTIC:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(cfg.EMAT.N_TRAINING_THREADS)

    setproctitle.setproctitle(str(cfg.ALGO) + "-" + \
                              str(cfg.ENV_NAME) + "-" + "@" +
                              str(cfg.USER_NAME))

    # ----------------------------------------------------------------------------#
    # Seeding
    # ----------------------------------------------------------------------------#
    torch.manual_seed(cfg.RNG_SEED)
    torch.cuda.manual_seed_all(cfg.RNG_SEED)
    np.random.seed(cfg.RNG_SEED)

    # ----------------------------------------------------------------------------#
    # Making environment
    # ----------------------------------------------------------------------------#
    # envs init
    envs = make_train_env()
    eval_envs = make_eval_env()
    num_agents = cfg.EMAT.AGENT_NUM

    config = {
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
        "model_dir": model_dir
    }

    # ----------------------------------------------------------------------------#
    # display
    # ----------------------------------------------------------------------------#
    # Load runner
    net_option = "Shared" if cfg.NETWORK.SHARE_POLICY else "Separated"
    runner_name = cfg.ENV_NAME.split('-')[0] + net_option + 'Runner'
    try:
        Runner = globals()[runner_name]
        print("Use {} as env runner.".format(runner_name))
    except:
        print("Failed to load the customised Seperated/Shared Env Runner.")
        print("Try to use customised Env Runner...")
        print("......")
        try:
            runner_name = cfg.ENV_NAME.split('-')[0] + 'Runner'
            Runner = globals()[runner_name]
            print("Use {} as env runner.".format(runner_name))
        except:
            print("Failed to load the customised Env Runner.")
            print("Try to use basic Seperated/Shared Env Runner...")
            print("......")
            runner_name = net_option + 'Runner'
            Runner = globals()[runner_name]
            print("Use {} as env runner.".format(runner_name))

    runner = Runner(config)
    runner.run()
    runner.render()

    # post process
    envs.close()


if __name__ == "__main__":
    main(sys.argv[1:])
