"""Configuration file (powered by YACS)."""

import copy
import os

from config.yacs import CfgNode as CN

# Global config object
_C = CN()

# Example usage:
#   from config.config import cfg
cfg = _C

# ----------------------------------------------------------------------------#
# XML template params
# ----------------------------------------------------------------------------#
# Refer mujoco docs for what each param does

_C.XML = CN()

_C.XML.GEOM_CONDIM = 3

_C.XML.GEOM_FRICTION = [0.7, 0.1, 0.1]

_C.XML.FILTER_PARENT = "enable"

_C.XML.SHADOWCLIP = 0.5

# ----------------------------------------------------------------------------#
# Universal Env Options
# ----------------------------------------------------------------------------#
_C.ENV = CN()

_C.ENV.EPISODE_LENGTH = 1600

# environment dimension. For instance, ant of dimension is 3, hopper/cheetah is 2
_C.ENV.DIMENSION = 3

# Hopper param
_C.ENV.FORWARD_REWARD_WEIGHT = 1.0

_C.ENV.HEALTHY_REWARD = 1.0

_C.ENV.TERMINATE_WHEN_UNHEALTHY = True

_C.ENV.HEALTHY_Z_RANGE = (0.2, 1.0)

# Hopper param
_C.ENV.HEALTHY_STATE_RANGE = (-100.0, 100.0)

_C.ENV.HEALTHY_ANGLE_RANGE = (-0.2, 0.2)

_C.ENV.CTRL_COST_WEIGHT = 0.5

_C.ENV.RESET_NOISE_SCALE = 0.1

_C.ENV.USE_CONTACT_FORCES = False

_C.ENV.CONTACT_FORCE_RANGE = (-1.0, 1.0)

_C.ENV.CONTACT_COST_WEIGHT = 5.e-4

_C.ENV.USE_RELATIVE_OBS = True


# --------------------------------------------------------------------------- #
# ALGO Options
# --------------------------------------------------------------------------- #
_C.ALGO = "mappo" # "rmappo, mappo"

# Whether to use global state or concatenated obs
_C.USE_OBS_INSTEAD_OF_STATE = False

# --------------------------------------------------------------------------- #
# Network Options
# --------------------------------------------------------------------------- #
_C.NETWORK = CN()

# Whether agent share the same policy
_C.NETWORK.SHARE_POLICY = False

# Whether to use centralized V function
_C.NETWORK.USE_CENTRALIZED_V = True

# Whether to use stacked_frames
_C.NETWORK.USE_STACKED_FRAMES = False

_C.NETWORK.STACKED_FRAMES = 1

# Dimension of hidden layers for actor/critic networks
_C.NETWORK.HIDDEN_SIZE = 64

# Number of layers for actor/critic networks
_C.NETWORK.LAYER_N = 2

# Whether to use ReLU
_C.NETWORK.USE_RELU = False

# Whether to use Orthogonal initialization for weights and 0 initialization for biases
_C.NETWORK.USE_ORTHOGONAL = True

# The gain of last action layer
_C.NETWORK.GAIN = 0.01

# Use a recurrent policy
_C.NETWORK.USE_RECURRENT_POLICY = False

# Whether to use a naive recurrent policy
_C.NETWORK.USE_NAIVE_RECURRENT_POLICY = False

# The number of recurrent layers.
_C.NETWORK.RECURRENT_N = 1

# Time length of chunks used to train a recurrent_policy
_C.NETWORK.DATA_CHUNK_LENGTH = 10


# --------------------------------------------------------------------------- #
# MAPPO Options
# --------------------------------------------------------------------------- #
_C.MAPPO = CN()

# --- Optimization params ------------------------------ #
# learning rate (default: 5e-4)
_C.MAPPO.LR = 5.e-4

# use a linear schedule on the learning rate
_C.MAPPO.USE_LINEAR_LR_DECAY = False

# critic learning rate (default: 5e-4)
_C.MAPPO.CRITIC_LR = 5.e-4

# RMSprop optimizer epsilon (default: 1e-5)
_C.MAPPO.OPTI_EPS = 1.e-5

_C.MAPPO.WEIGHT_DECAY = 0

# number of ppo epochs (default: 15)
_C.MAPPO.PPO_EPOCH = 15

# by default, clip loss value
_C.MAPPO.USE_CLIPPED_VALUE_LOSS = True

# ppo clip parameter (default: 0.2)
_C.MAPPO.CLIP_PARAM = 0.2

# number of batches for ppo (default: 1)
_C.MAPPO.NUM_MINI_BATCH = 1

# entropy term coefficient (default: 0.01)
_C.MAPPO.ENTROPY_COEF = 0.01

# value loss coefficient (default: 0.5)
_C.MAPPO.VALUE_LOSS_COEF = 0.5

# by default, use max norm of gradients
_C.MAPPO.USE_MAX_GRAD_NORM = True

# max norm of gradients (default: 0.5)
_C.MAPPO.MAX_GRAD_NORM = 10.0

# Use PopArt to normalize rewards
_C.MAPPO.USE_POPART = False

# use running mean and std to normalize rewards
_C.MAPPO.USE_VALUENORM = True

# Whether to apply layernorm to the inputs
_C.MAPPO.USE_FEATURE_NORMALIZATION = True

# use generalized advantage estimation
_C.MAPPO.USE_GAE = True

# discount factor for rewards (default: 0.99)
_C.MAPPO.GAMMA = 0.99

# gae lambda parameter (default: 0.95)
_C.MAPPO.GAE_LAMBDA = 0.95

# compute returns taking into account time limits
_C.MAPPO.USE_PROPER_TIME_LIMITS = False

# by default, use huber loss
_C.MAPPO.USE_HUBER_LOSS = True

# by default True, whether to mask useless data in value loss
_C.MAPPO.USE_VALUE_ACTIVE_MASKS = True

# by default True, whether to mask useless data in policy loss
_C.MAPPO.USE_POLICY_ACTIVE_MASKS = True

# coefficience of huber loss
_C.MAPPO.HUBER_DELTA = 10

# --------------------------------------------------------------------------- #
# CUDNN options
# --------------------------------------------------------------------------- #
_C.CUDNN = CN()

_C.CUDNN.DETERMINISTIC = True

# ----------------------------------------------------------------------------#
# Misc Options
# ----------------------------------------------------------------------------#
# Name of the environment used for experience collection
_C.ENV_NAME = "MultiAnt-v0"

# Output directory
_C.OUT_DIR = "./tmp"

# Config destination (in OUT_DIR)
_C.CFG_DEST = "config.yaml"

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries. This is the only seed
# which will effect env variations.
_C.RNG_SEED = 42

# Use GPU
_C.USE_GPU = True

# Device index
_C.DEVICE = "cuda:0"

# Log period in iters.
_C.LOG_PERIOD = 1

# Checkpoint period in iters. Refer LOG_PERIOD for meaning of iter
_C.CHECKPOINT_PERIOD = 10

# by default, do not start evaluation. If set`, start evaluation alongside with training
_C.USE_EVAL = True

# Evaluate the policy after every EVAL_PERIOD iters
_C.EVAL_PERIOD = 10

# by default, do not save render video. If set, save video
_C.SAVE_VIDEO = True

# by default, do not render the envs during training. If set, start render. Note: something, 
# the environment has internal render process which is not controlled by this hyperparam
_C.USE_RENDER = False

# display episode when rendering
_C.RENDER_EPISODES = 1

# the play interval of each rendered image in saved video
_C.IFI = 0.01 

# Unimal template path relative to the basedir
_C.TEMPLATE = "./envs/assets/multi_ant.xml"

# to specify user's name for simply collecting training data.
_C.USER_NAME = "emat"


# ----------------------------------------------------------------------------#
# Multi Agent Env Options
# ----------------------------------------------------------------------------#
_C.EMAT = CN()

_C.EMAT.AGENT_NUM = 3

# Number of torch threads for training
_C.EMAT.N_TRAINING_THREADS = 1

# Number of parallel envs for training rollouts
_C.EMAT.N_ROLLOUT_THREADS = 1

# Number of parallel envs for evaluating rollouts
_C.EMAT.N_EVAL_ROLLOUT_THREADS = 1

# Number of parallel envs for rendering rollouts
_C.EMAT.N_RENDER_ROLLOUT_THREADS = 1

# Number of environment steps to train (default: 10e6)
_C.EMAT.NUM_ENV_STEPS = 10e6

# Initiate multi-agent positions as regular polygon. SPACE denotes radius of
# this regular polygon.
_C.EMAT.SPACE = 1 

# Initial team formation "line", "star"
_C.EMAT.INIT_FORMATION = "line"

# ----------------------------------------------------------------------------#
# Competition
# ----------------------------------------------------------------------------#
_C.COMP = CN()

# use competition reward inspiration
_C.COMP.USE_COMPETITION = True

_C.COMP.COMPETITION_REWARD_COEF = 0.03

# ----------------------------------------------------------------------------#
# Functions
# ----------------------------------------------------------------------------#

def dump_cfg(run_dir, cfg_name=None):
    """Dumps the config to the output directory."""
    if not cfg_name:
        cfg_name = _C.CFG_DEST
    cfg_file = os.path.join(run_dir, cfg_name)
    with open(cfg_file, "w") as f:
        _C.dump(stream=f)


def load_cfg(out_dir, cfg_dest="config.yaml"):
    """Loads config from specified output directory."""
    cfg_file = os.path.join(out_dir, cfg_dest)
    _C.merge_from_file(cfg_file)


def get_default_cfg():
    return copy.deepcopy(cfg)
