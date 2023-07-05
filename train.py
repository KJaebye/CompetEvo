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

def main():
    # ----------------------------------------------------------------------------#
    # Load config options from terminal and predefined yaml file
    # ----------------------------------------------------------------------------#
    parser = argparse.ArgumentParser(description="User's arguments from terminal.")
    parser.add_argument("--cfg", dest="cfg_file", help="Config file", required=True, type=str)
    parser.add_argument('--use_cuda', type=str2bool, default=True)
    parser.add_argument('--gpu_index', type=int, default=0)

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
    
    


if __name__ == "__main__":
    main()
