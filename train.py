# !/usr/bin/envs python
import sys
import os

import setproctitle
import numpy as np
import torch
import argparse

import gymnasium as gym
from config.config import cfg
from config.config import dump_cfg

from custom.runner.multi_ant_runner import MultiAntSeparatedRunner, MultiAntSharedRunner
from custom.runner.multi_hopper_runner import MultiHopperSeparatedRunner, MultiHopperSharedRunner
from custom.runner.multi_walker2d_runner import MultiWalker2dSeparatedRunner, MultiWalker2dSharedRunner
from custom.runner.multi_half_cheetah_runner import MultiHalfCheetahSeparatedRunner, MultiHalfCheetahSharedRunner
from malib.runner.separated.env_runner import EnvRunner as SeparatedRunner
from malib.runner.shared.env_runner import EnvRunner as SharedRunner

def make_train_env():
    from envs import CUSTOM_ENVS
    def get_env_fn(rank):
        def init_env():
            assert cfg.ENV_NAME in CUSTOM_ENVS, "Alert: Undefined environment!"
            env = gym.make(cfg.ENV_NAME, agent_num=cfg.EMAT.AGENT_NUM)

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
            env = gym.make(cfg.ENV_NAME, agent_num=cfg.EMAT.AGENT_NUM)

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
        "--cfg", 
        dest="cfg_file", 
        help="Config file", 
        required=True, 
        type=str)
    args = parser.parse_args()

    # Load config options
    cfg.merge_from_file(args.cfg_file)

    # ----------------------------------------------------------------------------#
    # Define and create dirs
    # ----------------------------------------------------------------------------#
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    output_dir = cfg.OUT_DIR + '/' + cfg.ENV_NAME + '/' + str(cfg.EMAT.AGENT_NUM) + '/' + cfg.ALGO
    # run dir
    from datetime import datetime
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = str(output_dir) + '/' + time_str + '/'
    os.makedirs(run_dir, exist_ok=False)
    # Save the config file
    dump_cfg(run_dir)
    # model dir. If training, model directory is None.
    model_dir = None

    # ----------------------------------------------------------------------------#
    # Check for training
    # ----------------------------------------------------------------------------#
    if cfg.ALGO == "rmappo":
        assert (cfg.NETWORK.USE_RECURRENT_POLICY or cfg.NETWORK.USE_NAIVE_RECURRENT_POLICY), \
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

    setproctitle.setproctitle(str(cfg.ALGO) + "-" + 
                              str(cfg.ENV_NAME) + "-" + 
                              str(cfg.EMAT.AGENT_NUM) + "@" + 
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
    # Training
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

    # post process
    envs.close()
    if cfg.USE_EVAL and eval_envs is not envs:
        eval_envs.close()

    runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
    runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
