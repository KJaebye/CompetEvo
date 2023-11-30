from lib.utils.math import *

def get_single_body_qposaddr(model, body_name):
    i = model.body(body_name).id
    start_joint_id = model.body_jntadr[i]
    end_joint_id = start_joint_id + model.body_jntnum[i]
    start_qposaddr = model.jnt_qposadr[start_joint_id]
    if end_joint_id < len(model.jnt_qposadr):
        end_qposaddr = model.jnt_qposadr[end_joint_id]
    else:
        end_qposaddr = model.nq
    return start_qposaddr, end_qposaddr

    