import competevo
import gym_compete

import gymnasium as gym
from config.config import Config
import argparse

def str2bool(input_str):
    """Converts a string to a boolean value.

    Args:
        input_str (str): The string to be converted.

    Returns:
        bool: The boolean representation of the input string.
    """
    true_values = ["true", "yes", "1", "on", "y", "t"]
    false_values = ["false", "no", "0", "off", "n", "f"]

    lowercase_str = input_str.lower()
    if lowercase_str in true_values:
        return True
    elif lowercase_str in false_values:
        return False
    else:
        raise ValueError("Invalid input string. Could not convert to boolean.")

parser = argparse.ArgumentParser(description="User's arguments from terminal.")
parser.add_argument("--cfg", 
                    dest="cfg_file", 
                    help="Config file", 
                    required=True, 
                    type=str)
parser.add_argument('--use_cuda', type=str2bool, default=True)
parser.add_argument('--gpu_index', type=int, default=0)
parser.add_argument('--num_threads', type=int, default=1)
parser.add_argument('--epoch', type=str, default='0')
args = parser.parse_args()
# Load config file
cfg = Config(args.cfg_file)

env = gym.make(cfg.env_name, render_mode="human")
obs, _ = env.reset()

for _ in range(10000):
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)

   if any(terminated) or truncated:
      observation, info = env.reset()
env.close()