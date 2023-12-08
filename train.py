import gymnasium as gym
from config.config import Config
import argparse
import numpy as np
import torch
import gc

import logging
from logger.logger import Logger
from utils.tools import *

import time
import sys, os
sys.path.append(".")

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from runner.multi_evo_agent_runner import MultiEvoAgentRunner
from runner.multi_agent_runner import MultiAgentRunner
from runner.selfplay_agent_runner import SPAgentRunner

def main():
    # ----------------------------------------------------------------------------#
    # Load config options from terminal and predefined yaml file
    # ----------------------------------------------------------------------------#
    parser = argparse.ArgumentParser(description="User's arguments from terminal.")
    parser.add_argument("--cfg", 
                        dest="cfg_file", 
                        help="Config file", 
                        required=True, 
                        type=str)
    parser.add_argument('--use_cuda', type=str2bool, default=True)
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--ckpt_dir', type=str, default=None)
    parser.add_argument('--ckpt', type=str, default='0')
    args = parser.parse_args()
    # Load config file
    cfg = Config(args.cfg_file)

    # ----------------------------------------------------------------------------#
    # Define logger and create dirs
    # ----------------------------------------------------------------------------#
    logger = Logger(name='current', cfg=cfg)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    # set output
    logger.set_output_handler()
    logger.print_system_info()
    # only training generates log file
    logger.critical("The current environment is {}.".format(cfg.env_name))
    logger.info("Running directory: {}".format(logger.run_dir))
    logger.info('Type of current running: Training')
    logger.set_file_handler()
    # Save the config file
    cfg.save_config(logger.run_dir)

    # ----------------------------------------------------------------------------#
    # Set torch and random seed
    # ----------------------------------------------------------------------------#
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=args.gpu_index) \
        if args.use_cuda and torch.cuda.is_available() else torch.device('cpu')
    # torch.cuda.is_available() is natively False on mac m1
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    # ----------------------------------------------------------------------------#
    # Training
    # ----------------------------------------------------------------------------#
    # runner definition
    # runner = MultiEvoAgentRunner(cfg, logger, dtype, device, 
    #                              num_threads=args.num_threads, training=True)
    
    ckpt = int(args.ckpt) if args.ckpt.isnumeric() else args.ckpt
    start_epoch = ckpt if args.ckpt.isnumeric() else 0

    if cfg.runner_type == "multi-agent-runner":
        ckpt = [ckpt] * 2
        runner = MultiAgentRunner(cfg, logger, dtype, device, 
                                    num_threads=args.num_threads, training=True, ckpt_dir=args.ckpt_dir, ckpt=ckpt)
    elif cfg.runner_type == "selfplay-agent-runner":
        runner = SPAgentRunner(cfg, logger, dtype, device, 
                                    num_threads=args.num_threads, training=True, ckpt=ckpt)
    elif cfg.runner_type == "multi-evo-agent-runner":
        ckpt = [ckpt] * 2
        runner = MultiEvoAgentRunner(cfg, logger, dtype, device,
                                     num_threads=args.num_threads, training=True, ckpt_dir=args.ckpt_dir, ckpt=ckpt)
    
    # main loop
    for epoch in range(start_epoch, cfg.max_epoch_num):          
        runner.optimize(epoch)
        runner.save_checkpoint(epoch)

        """clean up gpu memory"""
        gc.collect()
        torch.cuda.empty_cache()

    runner.logger.info('training done!')

if __name__ == "__main__":
    main()
