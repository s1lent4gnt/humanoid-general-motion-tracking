# diagnose_joint_names.py
import mujoco as mj
from general_motion_retargeting.kinematics_model import KinematicsModel

GMT_MJCF = "humanoid-general-motion-tracking/assets/robots/g1/g1.xml"
GMR_ROBOT = "unitree_g1"  # change if you used a different key

def gmt_joint_names(path):
    m = mj.MjModel.from_xml_path(path)
    names = []
    for j in range(m.njnt):
        if m.jnt_type[j] == mj.mjtJoint.mjJNT_FREE:
            continue
        nm = mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j)
        if nm: names.append(nm)
    return names

gmt = gmt_joint_names(GMT_MJCF)
print("GMT (23) joint names / order:")
for i,n in enumerate(gmt): print(f"{i:2d} {n}")

kin = KinematicsModel(robot_type=GMR_ROBOT)
gmr = list(kin.joint_names)
print("\nGMR joint names / order:", len(gmr))
for i,n in enumerate(gmr): print(f"{i:2d} {n}")

# Intersection
common = [n for n in gmr if n in gmt]
print("\nCommon names:", len(common))
print(common)
