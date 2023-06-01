import gymnasium as gym
from config.config import cfg
import argparse
import numpy as np
import time
import sys

sys.path.append(".")

# Load config options
parser = argparse.ArgumentParser(description="User's arguments from terminal.")
parser.add_argument(
      "--cfg", 
      dest="cfg_file", 
      help="Config file", 
      required=True, 
      type=str)
args = parser.parse_args()
cfg.merge_from_file(args.cfg_file)

print("Acting Environment is {}.".format(cfg.ENV_NAME))

# register environment. Must be imported after cfg.file is merged.
import competevo

env = gym.make("sumo-evoants-v0", render_mode="human")
observation, info = env.reset()
sp = env.action_space.shape

for _ in range(1000000):
   action = env.action_space.sample() # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)

   if any(terminated) or truncated:
      observation, info = env.reset()

env.close()

