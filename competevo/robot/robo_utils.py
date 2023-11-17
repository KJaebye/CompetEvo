from competevo.robot.xml_robot import Robot
import numpy as np

def get_graph_fc_edges(num_nodes):
    edges = []
    for i in range(num_nodes - 1):
        for j in range(i + 1, num_nodes):
            edges.append([i, j])
            edges.append([j, i])
    edges = np.stack(edges, axis=1)
    return edges

def get_attr_design(robot):
        obs = []
        for i, body in enumerate(robot.bodies):
            obs_i = body.get_params([], pad_zeros=True, demap_params=True)
            obs.append(obs_i)
        obs = np.stack(obs)
        return obs
    
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

def get_body_index(robot: Robot, index_base=5):
        index = []
        for i, body in enumerate(robot.bodies):
            ind = int(body.name, base=index_base)
            index.append(ind)
        index = np.array(index)
        return index

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

def apply_skel_action(robot: Robot, skel_action, out_dir, file_name):
    bodies = list(robot.bodies)
    for body, a in zip(bodies, skel_action):
        if a == 1 and allow_add_body(body):
            robot.add_child_to_body(body)
        if a == 2 and allow_remove_body(body):
            robot.remove_body(body)

    # robot.write_xml(f'{out_dir}/{file_name}.xml')

def set_design_params(robot: Robot, in_design_params, out_dir, file_name):
    design_params = in_design_params
    for params, body in zip(design_params, robot.bodies):
        body.set_params(params, pad_zeros=True, map_params=True)
        body.sync_node()
    
    robot.write_xml(f'{out_dir}/{file_name}.xml')
