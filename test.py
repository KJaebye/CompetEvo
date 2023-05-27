import gymnasium as gym
from gym_compete import CUSTOM_ENVS
from config.config import cfg
import argparse
import numpy as np
import time


# # Load config options
# parser = argparse.ArgumentParser(description="User's arguments from terminal.")
# parser.add_argument(
#       "--cfg", 
#       dest="cfg_file", 
#       help="Config file", 
#       required=True, 
#       type=str)
# args = parser.parse_args()
# cfg.merge_from_file(args.cfg_file)

# print(cfg.TEMPLATE)

env = gym.make("sumo-ants-v0", render_mode="human")
observation, info = env.reset()
sp = env.action_space.shape

for _ in range(1000000):
   action = env.action_space.sample() # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)

   if any(terminated) or truncated:
      observation, info = env.reset()

env.close()

