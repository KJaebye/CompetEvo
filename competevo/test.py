from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.gymtorch import *
from isaacgym import gymutil
import time
from competevo.robot.xml_robot import Robot
import numpy as np

def print_asset_info(asset, name):
    print("======== Asset info %s: ========" % (name))
    num_bodies = gym.get_asset_rigid_body_count(asset)
    num_joints = gym.get_asset_joint_count(asset)
    num_dofs = gym.get_asset_dof_count(asset)
    print("Got %d bodies, %d joints, and %d DOFs" %
          (num_bodies, num_joints, num_dofs))

    # Iterate through bodies
    print("Bodies:")
    for i in range(num_bodies):
        name = gym.get_asset_rigid_body_name(asset, i)
        print(" %2d: '%s'" % (i, name))

    # Iterate through joints
    print("Joints:")
    for i in range(num_joints):
        name = gym.get_asset_joint_name(asset, i)
        type = gym.get_asset_joint_type(asset, i)
        type_name = gym.get_joint_type_string(type)
        print(" %2d: '%s' (%s)" % (i, name, type_name))

    # iterate through degrees of freedom (DOFs)
    print("DOFs:")
    for i in range(num_dofs):
        name = gym.get_asset_dof_name(asset, i)
        type = gym.get_asset_dof_type(asset, i)
        type_name = gym.get_dof_type_string(type)
        print(" %2d: '%s' (%s)" % (i, name, type_name))

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

import datetime
import os

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description="Asset and Environment Information")

# create simulation context
sim_params = gymapi.SimParams()

sim_params.enable_actor_creation_warning = False

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
gym.prepare_sim(sim)

if sim is None:
    print("*** Failed to create sim")
    quit()

# add ground plane
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

import yaml
cfg_path = f'/home/kjaebye/ws/competevo/competevo/robot/evo_ant.yml'
yml = yaml.safe_load(open(cfg_path, 'r'))
cfg = yml['robot']
base_ant_path = f'/home/kjaebye/ws/competevo/assets/mjcf/ant.xml'
xml_robot = Robot(cfg, base_ant_path, is_xml_str=False)

# edit skel
# skel_action = [0, 0, 0, 0, 0]
skel_action = [1, 1, 1, 1, 1]
bodies = list(xml_robot.bodies)
for body, a in zip(bodies, skel_action):
    if a == 1 and allow_add_body(cfg, body):
        xml_robot.add_child_to_body(body)
    if a == 2 and allow_remove_body(cfg, body):
        xml_robot.remove_body(body)
num_node = len(list(xml_robot.bodies))
num_dof = num_node - 1
# print("num_node:\n", num_node)

# edit attribute
params_names = xml_robot.get_params(get_name=True)
params = xml_robot.get_params()
# print(params)

new_params = params #+ .05
xml_robot.set_params(new_params)
params_new = xml_robot.get_params()

design_ref_params = get_attr_design(xml_robot)
attr_design_dim = design_ref_params.shape[-1]
# print("attr_design_shape:\n", design_ref_params.shape)
attr_fixed_dim = get_attr_fixed(cfg['obs_specs'], xml_robot).shape[-1]
print("attr_fixed_shape:\n", get_attr_fixed(cfg['obs_specs'], xml_robot).shape)
print("get_attr_fixed:\n", get_attr_fixed(cfg['obs_specs'], xml_robot))
edges = get_graph_fc_edges(num_node)
# print("fc_graph edges:\n", edges)
edges = xml_robot.get_gnn_edges()
# print("gnn edges:\n", edges)

# write a xml
os.makedirs('out', exist_ok=True)
model_name = "ant_evo"
xml_robot.write_xml(f'out/{model_name}_test.xml')
asset_file = f"{model_name}_test.xml"

assets = []
asset_root = "./out"
asset = gym.load_mjcf(sim, asset_root, asset_file)
assets.append(asset)

base_asset_file = f"ant.xml"
base_asset = gym.load_mjcf(sim, asset_root, base_asset_file)
assets.append(base_asset)


num_bodies = gym.get_asset_rigid_body_count(asset)
# print("num_bodies:\n", num_bodies)
body_names = [gym.get_asset_rigid_body_name(asset, i) for i in range(num_bodies)]
# print("body_names:\n", body_names)
asset_dof_props = gym.get_asset_dof_properties(asset)
# print("dof_props:\n", asset_dof_props)

# Setup environment spacing
spacing = 5.0
lower = gymapi.Vec3(-spacing, 0.0, -spacing)
upper = gymapi.Vec3(spacing, spacing, spacing)

# position the camera
cam_pos = gymapi.Vec3(50, 50, 0)
cam_target = gymapi.Vec3(0, 1.32, 0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

envs = []
actor_handles = []
actor_handles_op = []
for i in range(4):
    # Create one environment
    env = gym.create_env(sim, lower, upper, 2)

    # add actor
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(1.0, 1.32, 0.0)
    pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

    pose_op = gymapi.Transform()
    pose_op.p = gymapi.Vec3(-2.0, 1.32, 0.0)
    pose_op.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

    # start_pose = gymapi.Transform()
    # start_pose.p = gymapi.Vec3(-2, -2, 1.)
    # start_pose.r = gymapi.Quat(0.707107, 0.0, 0.0, -0.707107)
    # start_pose_op = gymapi.Transform()
    # start_pose_op.p = gymapi.Vec3(2, 2, 1.)
    # start_pose_op.r = gymapi.Quat(0.707107, 0.0, 0.0, -0.707107)

    actor_handle = gym.create_actor(env, asset, pose, "actor", i, -1, 0)
    actor_handle_op = gym.create_actor(env, base_asset, pose_op, "actor_op", i, -1, 0)
    actor_handles.append(actor_handle)
    actor_handles_op.append(actor_handle_op)

    envs.append(env)
    
# data aqurire
actor_root_state = gym.acquire_actor_root_state_tensor(sim)
dof_state_tensor = gym.acquire_dof_state_tensor(sim)
sensor_tensor = gym.acquire_force_sensor_tensor(sim)

gym.refresh_dof_state_tensor(sim)
gym.refresh_actor_root_state_tensor(sim)

root_states = gymtorch.wrap_tensor(actor_root_state)
print(f'root states:\n {root_states}')

# create some wrapper tensors for different slices
dof_state = gymtorch.wrap_tensor(dof_state_tensor)
print(f"dof state:\n {dof_state}")

actor_dof_prop = gym.get_actor_dof_properties(env, actor_handles_op[0])
print("dof_prop:\n", actor_dof_prop)

num_bodies = gym.get_actor_rigid_body_count(env, actor_handles_op[0])
num_dof = gym.get_actor_dof_count(env, actor_handles_op[0])
print("num_bodies:\n", num_bodies)
print("num_dof:\n", num_dof)

dof_state_tensor = gym.acquire_dof_state_tensor(sim)
dof_state = gymtorch.wrap_tensor(dof_state_tensor)

while not gym.query_viewer_has_closed(viewer):
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    gym.refresh_actor_root_state_tensor(sim)
    # print("actor_root_state:\n", root_states)


    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)


print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)