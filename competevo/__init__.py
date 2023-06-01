from gymnasium.envs.registration import register
import os

register(
    id='sumo-evoants-v0',
    entry_point='competevo.new_envs:SumoEnv',
    disable_env_checker=True,
    kwargs={'agent_names': ['ant_fighter', 'ant_fighter'],
            'world_xml_path': os.path.join(
                os.path.dirname(__file__), "new_envs",
                "assets", 'world_body_arena.xml'
            ),
            'init_pos': [(-1, 0, 2.5), (1, 0, 2.5)],
            'max_episode_steps': 500,
            'min_radius': 4,
            'max_radius': 4
            },
)
