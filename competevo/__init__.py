from gymnasium.envs.registration import register
import os

register(
    id='sumo-evoants-v0',
    entry_point='competevo.evo_envs:SumoEvoEnv',
    disable_env_checker=True,
    kwargs={'agent_names': ['evo_ant_fighter', 'evo_ant_fighter'],
            'world_xml_path': "./competevo/evo_envs/assets/world_body_arena.xml",
            'init_pos': [(-1, 0, 1.5), (1, 0, 1.5)],
            'max_episode_steps': 500,
            'min_radius': 4,
            'max_radius': 4,
            },
)
