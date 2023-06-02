from gymnasium.envs.registration import register
import os
from config.config import cfg

register(
    id='sumo-evoants-v0',
    entry_point='competevo.evo_envs:SumoEvoEnv',
    disable_env_checker=True,
    kwargs={'agent_names': cfg.ENV.AGENT_NAMES,
            'world_xml_path': cfg.ENV.WORLD_XML_PATH,
            'init_pos': cfg.ENV.INIT_POS,
            'max_episode_steps': cfg.ENV.EPISODE_LENGTH,
            'min_radius': cfg.ENV.MIN_RADIUS,
            'max_radius': cfg.ENV.MAX_RADIUS,
            },
)
