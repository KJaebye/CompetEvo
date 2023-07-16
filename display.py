import gymnasium as gym
from config.config import Config
import argparse
import numpy as np
import torch

import logging
from logger.logger import Logger
from utils.tools import *

import time
import sys, os
sys.path.append(".")

from runner.multi_evo_agent_runner import MultiEvoAgentRunner
from runner.multi_agent_runner import MultiAgentRunner

def main():
    # ----------------------------------------------------------------------------#
    # Load config options from terminal and predefined yaml file
    # ----------------------------------------------------------------------------#
    parser = argparse.ArgumentParser(description="User's arguments from terminal.")
    parser.add_argument("--run_dir", 
                        dest="run_dir", 
                        help="run directory", 
                        required=True, 
                        type=str)
    parser.add_argument('--epoch', type=str, default='best')
    args = parser.parse_args()
    # Load config file
    cfg_file = args.run_dir + "config.yml"
    cfg = Config(cfg_file)

    # ----------------------------------------------------------------------------#
    # Define logger and create dirs
    # ----------------------------------------------------------------------------#
    # set logger
    logger = Logger(name='current', args=args, cfg=cfg)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    # set output
    logger.set_output_handler()
    logger.print_system_info()
    # only training generates log file
    logger.critical('Type of current running: Evaluation. No log file will be created')
    # redefine dir
    logger.run_dir = args.run_dir
    logger.model_dir = '%smodels' % logger.run_dir
    logger.log_dir = '%slog' % logger.run_dir
    logger.tb_dir = '%stb' % logger.run_dir

    epoch = int(args.epoch) if args.epoch.isdigit() else args.epoch

    # ----------------------------------------------------------------------------#
    # Set torch and random seed
    # ----------------------------------------------------------------------------#
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cpu')
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    # ----------------------------------------------------------------------------#
    # Evaluation
    # ----------------------------------------------------------------------------#
    # runner definition
    # runner = MultiEvoAgentRunner(cfg, logger, dtype, device, 
    #                              num_threads=args.num_threads, training=False)
    
    runner = MultiAgentRunner(cfg, logger, dtype, device, training=False, ckpt=epoch)
    runner.display(num_episode=15, mean_action=True)
    # runner.sample(min_batch_size=10000, mean_action=True, render=True, nthreads=1)

if __name__ == "__main__":
    main()

