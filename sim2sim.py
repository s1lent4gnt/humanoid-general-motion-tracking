# -----------------------------------------------------------------------------
# Copyright [2025] [Zixuan Chen, Mazeyu Ji, Xuxin Cheng, Xuanbin Peng, Xue Bin Peng, Xiaolong Wang]

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script is adapted from the open-source script:
# https://github.com/zixuan417/smooth-humanoid-locomotion/blob/main/simulation/legged_gym/legged_gym/scripts/sim2sim.py
# -----------------------------------------------------------------------------


import argparse, os, time
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
import torch

from utils.motion_lib import MotionLib

@torch.jit.script
def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c

def euler_from_quaternion(quat_angle):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        x = quat_angle[:,0]; y = quat_angle[:,1]; z = quat_angle[:,2]; w = quat_angle[:,3]
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = torch.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = torch.clip(t2, -1, 1)
        pitch_y = torch.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = torch.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians


def quatToEuler(quat):
    eulerVec = np.zeros(3)
    qw = quat[0] 
    qx = quat[1] 
    qy = quat[2]
    qz = quat[3]
    # roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    eulerVec[0] = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if np.abs(sinp) >= 1:
        eulerVec[1] = np.copysign(np.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        eulerVec[1] = np.arcsin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    eulerVec[2] = np.arctan2(siny_cosp, cosy_cosp)
    
    return eulerVec

class HumanoidEnv:
    def __init__(self, policy_path, motion_path, robot_type="g1", device="cuda", record_video=False):
        self.robot_type = robot_type
        self.device = device
        self.record_video = record_video
        self.motion_path = motion_path
        
        if robot_type == "g1":
            model_path = "assets/robots/g1/g1.xml"
            self.stiffness = np.array([
                100, 100, 100, 150, 40, 40,
                100, 100, 100, 150, 40, 40,
                150, 150, 150,
                40, 40, 40, 40,
                40, 40, 40, 40,
            ])
            self.damping = np.array([
                2, 2, 2, 4, 2, 2,
                2, 2, 2, 4, 2, 2,
                4, 4, 4,
                5, 5, 5, 5,
                5, 5, 5, 5,
            ])
            self.num_actions = 23
            self.num_dofs = 23
            self.default_dof_pos = np.array([
                -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # left leg (6)
                -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # right leg (6)
                0.0, 0.0, 0.0, # torso (1)
                0.0, 0.4, 0.0, 1.2,
                0.0, -0.4, 0.0, 1.2,
            ])
            self.torque_limits = np.array([
                88, 139, 88, 139, 50, 50,
                88, 139, 88, 139, 50, 50,
                88, 50, 50,
                25, 25, 25, 25,
                25, 25, 25, 25,
            ])
            self.dof_names = ["left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
                              "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
                              "waist_yaw", "waist_roll", "waist_pitch",
                              "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
                              "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow"]

        else:
            raise ValueError(f"Robot type {robot_type} not supported!")
        
        self.obs_indices = np.arange(self.num_dofs)
        
        self.sim_duration = 60.0
        self.sim_dt = 0.001
        self.sim_decimation = 20
        self.control_dt = self.sim_dt * self.sim_decimation
        
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_step(self.model, self.data)
        if self.record_video:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data, 'offscreen')
        else:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer.cam.distance = 5.0
        
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.action_scale = 0.5
        
        self.tar_obs_steps = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                         50, 55, 60, 65, 70, 75, 80, 85, 90, 95,]
        
        if robot_type == "g1":
            self.n_priv = 0
            self.n_proprio = 3 + 2 + 3*self.num_actions
            self.n_priv_latent = 1
            self.key_body_ids = [29, 37,  6, 14,  4, 12, 25, 33, 20]
            
        self.history_len = 20
        self.priv_latent = np.zeros(self.n_priv_latent, dtype=np.float32)
        
        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 0.05
        self.ang_vel_scale = 0.25
        
        self._motion_lib = MotionLib(self.motion_path, self.device)
        self._init_motion_buffers()
        
        self.proprio_history_buf = deque(maxlen=self.history_len)
        for _ in range(self.history_len):
            self.proprio_history_buf.append(np.zeros(self.n_proprio))
        
        print("Loading jit for policy: ", policy_path)
        self.policy_path = policy_path
        self.policy_jit = torch.jit.load(policy_path, map_location=self.device)
        
        self.last_time = time.time()
    
    def _init_motion_buffers(self):
        self.tar_obs_steps = torch.tensor(self.tar_obs_steps, device=self.device, dtype=torch.int)
        
    def _get_mimic_obs(self, curr_time_step):
        num_steps = len(self.tar_obs_steps)
        motion_times = torch.tensor([curr_time_step * self.control_dt], device=self.device).unsqueeze(-1)
        obs_motion_times = self.tar_obs_steps * self.control_dt + motion_times
        obs_motion_times = obs_motion_times.flatten()
        motion_ids = torch.zeros(num_steps, dtype=torch.int, device=self.device)
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, _ = self._motion_lib.calc_motion_frame(motion_ids, obs_motion_times)
        
        roll, pitch, yaw = euler_from_quaternion(root_rot)
        roll = roll.reshape(1, num_steps, 1)
        pitch = pitch.reshape(1, num_steps, 1)
        yaw = yaw.reshape(1, num_steps, 1)
        
        root_vel = quat_rotate_inverse(root_rot, root_vel)
        root_ang_vel = quat_rotate_inverse(root_rot, root_ang_vel)
        
        root_pos = root_pos.reshape(1, num_steps, 3)
        root_vel = root_vel.reshape(1, num_steps, 3)
        root_ang_vel = root_ang_vel.reshape(1, num_steps, 3)
        dof_pos = dof_pos.reshape(1, num_steps, -1)
        
        if self.robot_type == "g1":
            mimic_obs_buf = torch.cat((
                root_pos[..., 2:3],
                roll, pitch,
                root_vel,
                root_ang_vel[..., 2:3],
                dof_pos,
            ), dim=-1)
        
        mimic_obs_buf = mimic_obs_buf.reshape(1, -1)
        
        return mimic_obs_buf.detach().cpu().numpy().squeeze()
        
    def extract_data(self):
        dof_pos = self.data.qpos.astype(np.float32)[-self.num_dofs:]
        dof_vel = self.data.qvel.astype(np.float32)[-self.num_dofs:]
        quat = self.data.sensor('orientation').data.astype(np.float32)
        ang_vel = self.data.sensor('angular-velocity').data.astype(np.float32)
        self.dof_vel = torch.from_numpy(dof_vel).float().unsqueeze(0).to(self.device)
        return (dof_pos, dof_vel, quat, ang_vel)
        
    def run(self):
        motion_name = os.path.basename(self.motion_path).split('.')[0]
        if self.record_video:
            import imageio
            video_name = f"{self.robot_type}_{''.join(os.path.basename(self.policy_path).split('.')[:-1])}_{motion_name}.mp4"
            path = "mujoco_videos/"
            if not os.path.exists(path):
                os.makedirs(path)
            video_name = os.path.join(path, video_name)
            mp4_writer = imageio.get_writer(video_name, fps=50)
        
        for i in tqdm(range(int(self.sim_duration / self.sim_dt)), desc="Running simulation..."):
            dof_pos, dof_vel, quat, ang_vel = self.extract_data()
            
            if i % self.sim_decimation == 0:
                curr_timestep = i // self.sim_decimation
                mimic_obs = self._get_mimic_obs(curr_timestep)
                
                rpy = quatToEuler(quat)
                obs_dof_vel = dof_vel.copy()
                obs_dof_vel[[4, 5, 10, 11]] = 0.
                obs_prop = np.concatenate([
                    ang_vel * self.ang_vel_scale,
                    rpy[:2],
                    (dof_pos - self.default_dof_pos) * self.dof_pos_scale,
                    obs_dof_vel * self.dof_vel_scale,
                    self.last_action,
                ])
                
                assert obs_prop.shape[0] == self.n_proprio, f"Expected {self.n_proprio} but got {obs_prop.shape[0]}"
                obs_hist = np.array(self.proprio_history_buf).flatten()

                if self.robot_type == "g1":
                    obs_buf = np.concatenate([mimic_obs, obs_prop, obs_hist])
                
                obs_tensor = torch.from_numpy(obs_buf).float().unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    raw_action = self.policy_jit(obs_tensor).cpu().numpy().squeeze()
                
                self.last_action = raw_action.copy()
                raw_action = np.clip(raw_action, -10., 10.)
                scaled_actions = raw_action * self.action_scale
                
                step_actions = np.zeros(self.num_dofs)
                step_actions = scaled_actions
                
                pd_target = step_actions + self.default_dof_pos
                
                self.viewer.cam.lookat = self.data.qpos.astype(np.float32)[:3]
                if self.record_video:
                    img = self.viewer.read_pixels()
                    mp4_writer.append_data(img)
                else:
                    self.viewer.render()

                self.proprio_history_buf.append(obs_prop)
                
        
            torque = (pd_target - dof_pos) * self.stiffness - dof_vel * self.damping
            torque = np.clip(torque, -self.torque_limits, self.torque_limits)
            
            self.data.ctrl = torque
            
            mujoco.mj_step(self.model, self.data)
        
        self.viewer.close()
        if self.record_video:
            mp4_writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--robot', type=str, default="g1")
    parser.add_argument('--checkpoint', type=int, default=-1)
    parser.add_argument('--record_video', action='store_true')
    parser.add_argument('--motion_file', type=str, default="walk_stand.pkl")
    args = parser.parse_args()
    
    jit_policy_pth = "assets/pretrained_checkpoints/pretrained.pt"
    assert os.path.exists(jit_policy_pth), f"Policy path {jit_policy_pth} does not exist!"
    print(f"Loading model from: {jit_policy_pth}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    motion_file = os.path.join("assets/motions", args.motion_file)
    
    env = HumanoidEnv(policy_path=jit_policy_pth, motion_path=motion_file, robot_type=args.robot, device=device, record_video=args.record_video)
    
    env.run()
        
        