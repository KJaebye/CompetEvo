from gymnasium.envs.registration import register
import os

register(
    id='sumo-evoants-v0',
    entry_point='competevo.evo_envs:SumoEvoEnv',
    disable_env_checker=True,
    kwargs={'agent_names': ['evo_ant_fighter', 'evo_ant_fighter'],
            'world_xml_path': "./competevo/evo_envs/assets/world_body_arena.xml",
            'init_pos': [(-1, 0, 1.5), (1, 0, 1.5)],
            'rgb': [(0.98, 0.54, 0.56), (0.11, 0.56, 1)],
            'max_episode_steps': 500,
            'min_radius': 4,
            'max_radius': 4,
            },
)

register(
    id='run-to-goal-evoants-v0',
    entry_point='competevo.evo_envs:MultiEvoAgentEnv',
    disable_env_checker=True,
    kwargs={'agent_names': ['evo_ant', 'evo_ant'],
            'init_pos': [(-2, 0, 0.75), (2, 0, 0.75)],
            'ini_euler': [(0, 0, 0), (0, 0, 180)],
            'rgb': [(0.98, 0.54, 0.56), (0.11, 0.56, 1)],
            'max_episode_steps': 500,
            },
)

register(
    id='run-to-goal-evoant-v0',
    entry_point='competevo.evo_envs:MultiEvoAgentEnv',
    disable_env_checker=True,
    kwargs={'agent_names': ['evo_ant'],
            'init_pos': [(2, 0, 0.75)],
            'ini_euler': [(0, 0, 0)],
            'rgb': [(0.98, 0.54, 0.56), (0.11, 0.56, 1)],
            'max_episode_steps': 500,
            },
)

register(
    id='run-to-goal-animals-v0',
    entry_point='competevo.evo_envs:MultiAnimalEnv',
    disable_env_checker=True,
    kwargs={'agent_names': ['ant', 'evo_ant'],
            'init_pos': [(-2, 0, 0.75), (2, 0, 0.75)],
            'ini_euler': [(0, 0, 0), (0, 0, 180)],
            'rgb': [(0.98, 0.54, 0.56), (0.11, 0.56, 1)],
            'max_episode_steps': 500,
            },
)

register(
    id='run-to-goal-ant-evoant-v0',
    entry_point='competevo.evo_envs:MultiAnimalEnv',
    disable_env_checker=True,
    kwargs={'agent_names': ['ant', 'evo_ant'],
            'init_pos': [(-2, 0, 0.75), (2, 0, 0.75)],
            'ini_euler': [(0, 0, 0), (0, 0, 180)],
            'rgb': [(0.98, 0.54, 0.56), (0.11, 0.56, 1)],
            'max_episode_steps': 500,
            },
)