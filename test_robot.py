from competevo.evo_envs.robot.xml_robot import Robot
import yaml 
import numpy as np

def get_graph_fc_edges(num_nodes):
    edges = []
    for i in range(num_nodes - 1):
        for j in range(i + 1, num_nodes):
            edges.append([i, j])
            edges.append([j, i])
    edges = np.stack(edges, axis=1)
    return edges

def get_attr_fixed(cfg, robot):
    obs = []
    for i, body in enumerate(robot.bodies):
        obs_i = []
        if 'depth' in cfg.get("attr", {}):
            obs_depth = np.zeros(cfg.get('max_body_depth', 4))
            obs_depth[body.depth] = 1.0
            obs_i.append(obs_depth)
        if 'jrange' in cfg.get("attr", {}):
            obs_jrange = body.get_joint_range()
            obs_i.append(obs_jrange)
        if 'skel' in cfg.get("attr", {}):
            obs_add = allow_add_body(body)
            obs_rm = allow_remove_body(body)
            obs_i.append(np.array([float(obs_add), float(obs_rm)]))
        if len(obs_i) > 0:
            obs_i = np.concatenate(obs_i)
            obs.append(obs_i)
    
    if len(obs) == 0:
        return None
    obs = np.stack(obs)
    return obs

def get_attr_design(robot):
    obs = []
    for i, body in enumerate(robot.bodies):
        obs_i = body.get_params([], pad_zeros=True, demap_params=True)
        obs.append(obs_i)
    obs = np.stack(obs)
    return obs

def allow_add_body(cfg, body):
    add_body_condition = cfg['add_body_condition']
    max_nchild = add_body_condition.get('max_nchild', 3)
    min_nchild = add_body_condition.get('min_nchild', 0)
    return body.depth >= cfg.get('min_body_depth', 1) and body.depth < cfg.get('max_body_depth', 4) - 1 and len(body.child) < max_nchild and len(body.child) >= min_nchild

def allow_remove_body(cfg, body):
    if body.depth >= cfg.get('min_body_depth', 1) + 1 and len(body.child) == 0:
        if body.depth == 1:
            return body.parent.child.index(body) > 0
        else:
            return True
    return False


cfg_path = '/root/ws/competevo/evo_ant.yaml'
yml = yaml.safe_load(open(cfg_path, 'r'))
cfg = yml['robot']
base_ant_path = '/root/ws/competevo/competevo/evo_envs/assets/evo_ant_body_base1.xml'
xml_robot = Robot(cfg, base_ant_path, is_xml_str=False)

bodies = list(xml_robot.bodies)
skel_action = np.ones(len(bodies))

# unchanged = ["1", "2", "3", "4", "111", "112", "113", "114"]
unchanged = ["11", "12", "13", "14"]
print("pre:", bodies)
for i in range(3):
    skel_action = np.ones(len(list(xml_robot.bodies)))
    for body, a in zip(bodies, skel_action):
        if body.name in unchanged:
            continue
        if a == 1 and allow_add_body(cfg, body):
            xml_robot.add_child_to_body(body)
        if a == 2 and allow_remove_body(cfg, body):
            xml_robot.remove_body(body)
print("post:", list(xml_robot.bodies))

# write a xml
model_name = "evo_ant"
xml_robot.write_xml(f'/root/ws/competevo/{model_name}_test.xml')
asset_file = f"{model_name}_test.xml"

import time

import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path(asset_file)
d = mujoco.MjData(m)

with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  while viewer.is_running() and time.time() - start < 300:
    step_start = time.time()

    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    mujoco.mj_step(m, d)

    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)