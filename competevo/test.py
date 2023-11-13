from isaacgym import gymapi
from isaacgym import gymutil
import time

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

asset_files = ["mjcf/nv_ant.xml", "mjcf/base_ant.xml"]
assets = []
t0 = time.time()
for asset_file in asset_files:
    asset_root = "../assets"
    asset = gym.load_asset(sim, asset_root, asset_file)
    assets.append(asset)
t1 = time.time()
print(t1-t0)

# Setup environment spacing
spacing = 2.0
lower = gymapi.Vec3(-spacing, 0.0, -spacing)
upper = gymapi.Vec3(spacing, spacing, spacing)

# position the camera
cam_pos = gymapi.Vec3(50, 50, 0)
cam_target = gymapi.Vec3(0, 1.32, 0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

envs = []
actor_handles = []
for asset in assets:
    # Create one environment
    env = gym.create_env(sim, lower, upper, 1)

    # add actor
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 1.32, 0.0)
    pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

    actor_handle = gym.create_actor(env, asset, pose, "actor", -1, -1, 0)

    envs.append(env)
    actor_handles.append(actor_handle)

while not gym.query_viewer_has_closed(viewer):
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)


print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)