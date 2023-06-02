import gymnasium as gym
from config.config import cfg
from config.config import dump_cfg
import argparse
import numpy as np
import time
import sys, os

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

# ----------------------------------------------------------------------------#
# Define and create dirs
# ----------------------------------------------------------------------------#
os.makedirs(cfg.OUT_DIR, exist_ok=True)
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg.OUT_DIR.split("/")[-1], cfg.ENV_NAME)
# run dir
from datetime import datetime
time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = str(output_dir) + '/' + time_str
os.makedirs(run_dir, exist_ok=False)
# Save the config file
dump_cfg(run_dir)
# model dir. If training, model directory is None.
model_dir = None

print("Acting Environment is {}.".format(cfg.ENV_NAME))
print(run_dir)

# register environment. Must be imported after cfg.file is merged.
import competevo

env = gym.make("sumo-evoants-v0", render_mode="human", rundir=run_dir)
observation, info = env.reset()
sp = env.action_space.shape

for _ in range(1000000):
   action = env.action_space.sample() # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)

   if any(terminated) or truncated:
      observation, info = env.reset()

env.close()

