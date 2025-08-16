# -----------------------------------------------------------------------------
# Copyright [2025]
# Zixuan Chen, Mazeyu Ji, Xuxin Cheng, Xuanbin Peng, Xue Bin Peng, Xiaolong Wang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# -----------------------------------------------------------------------------

import argparse
import os
import time
from collections import deque

import numpy as np
import torch
import mujoco
import mujoco.viewer
# import mujoco_viewer
from tqdm import tqdm

from utils.motion_lib import MotionLib


# ----------------------------- small math utils ------------------------------

@torch.jit.script
def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Rotate vector v by the inverse (conjugate) of unit quaternion q.
    q: (..., 4) in xyzw
    v: (..., 3)
    returns: (..., 3)
    """
    qvec = q[..., 0:3]         # (… ,3)
    qw   = q[..., 3:4]         # (… ,1) keep last dim for broadcasting
    t = 2.0 * torch.cross(qvec, v, dim=-1)           # (… ,3)
    return v - qw * t + torch.cross(qvec, t, dim=-1) # rotate by q*

def euler_from_quaternion(quat_angle: torch.Tensor):
    """
    xyzw -> roll,pitch,yaw (radians), all tensors shaped [...,]
    """
    x = quat_angle[..., 0]
    y = quat_angle[..., 1]
    z = quat_angle[..., 2]
    w = quat_angle[..., 3]
    
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clamp(t2, -1.0, 1.0)
    pitch_y = torch.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)
    return roll_x, pitch_y, yaw_z

def quatToEuler(quat_np: np.ndarray) -> np.ndarray:
    """MuJoCo sensor quaternion is [w,x,y,z] — convert to RPY in numpy."""
    eulerVec = np.zeros(3, dtype=np.float32)
    qw, qx, qy, qz = quat_np[0], quat_np[1], quat_np[2], quat_np[3]

    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    eulerVec[0] = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (qw * qy - qz * qx)
    eulerVec[1] = np.pi / 2 * np.sign(sinp) if np.abs(sinp) >= 1 else np.arcsin(sinp)

    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    eulerVec[2] = np.arctan2(siny_cosp, cosy_cosp)
    return eulerVec

# --------------------------------- the env -----------------------------------

class HumanoidEnv:
    def __init__(self, policy_path, motion_path, robot_type="g1", device="cuda", record_video=False):
        self.robot_type = robot_type
        self.device = device
        self.record_video = record_video
        self.motion_path = motion_path

        # -------------------- robot config (G1) --------------------
        if robot_type == "g1":
            model_path = "assets/robots/g1/g1.xml"
            self.stiffness = np.array([
                100, 100, 100, 150,  40,  40,
                100, 100, 100, 150,  40,  40,
                150, 150, 150,
                 40,  40,  40,  40,
                 40,  40,  40,  40,
            ], dtype=np.float32)
            self.damping = np.array([
                 2,  2,  2,  4, 2, 2,
                 2,  2,  2,  4, 2, 2,
                 4,  4,  4,
                 5,  5,  5,  5,
                 5,  5,  5,  5,
            ], dtype=np.float32)
            self.num_actions = 23
            self.num_dofs = 23
            self.default_dof_pos = np.array([
                -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,   # left leg (6)
                -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,   # right leg (6)
                 0.0, 0.0, 0.0,                   # torso (3)
                 0.0, 0.4, 0.0, 1.2,              # left arm (4)
                 0.0,-0.4, 0.0, 1.2,              # right arm (4)
            ], dtype=np.float32)
            self.torque_limits = np.array([
                88, 139,  88, 139, 50, 50,
                88, 139,  88, 139, 50, 50,
                88,  50,  50,
                25,  25,  25,  25,
                25,  25,  25,  25,
            ], dtype=np.float32)
        else:
            raise ValueError(f"Robot type {robot_type} not supported!")

        # --------------------- sim + viewer setup -------------------
        self.sim_duration = 60.0
        self.sim_dt = 0.001
        self.sim_decimation = 20           # 50 Hz control
        self.control_dt = self.sim_dt * self.sim_decimation

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_step(self.model, self.data)

        # Launch passive viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.cam.distance = 5.0

        # if self.record_video:
        #     self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data, 'offscreen')
        # else:
        #     self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        # self.viewer.cam.distance = 5.0

        # --------------------- policy + obs layout -------------------
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.action_scale = 0.5

        # future steps (in control ticks) for the mimic window (~2.0 s)
        self.tar_obs_steps = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                              50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

        # proprio layout: ang_vel(3) + roll/pitch(2) + q(23) + qd(23) + last_action(23) = 74
        if robot_type == "g1":
            self.n_priv = 0
            self.n_proprio = 3 + 2 + self.num_actions + self.num_actions + self.num_actions
            self.n_priv_latent = 1

        self.history_len = 20
        self.priv_latent = np.zeros(self.n_priv_latent, dtype=np.float32)

        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 0.05
        self.ang_vel_scale = 0.25

        # motion lib
        self._motion_lib = MotionLib(self.motion_path, self.device)
        self._init_motion_buffers()

        # active motion selection + timing
        self.curr_motion_id = torch.tensor(0, dtype=torch.long, device=self.device)
        self.curr_motion_tick0 = 0
        self.global_control_tick = 0   # increments only on control ticks

        # proprio history buffer
        self.proprio_history_buf = deque(maxlen=self.history_len)
        for _ in range(self.history_len):
            self.proprio_history_buf.append(np.zeros(self.n_proprio, dtype=np.float32))

        # policy
        print("Loading jit for policy:", policy_path)
        self.policy_path = policy_path
        self.policy_jit = torch.jit.load(policy_path, map_location=self.device)

        self.last_time = time.time()

    # ----------------------------- motion mgmt ------------------------------

    def num_motions(self) -> int:
        # Works whether MotionLib loaded one or many files.
        if hasattr(self._motion_lib, "num_motions"):
            try:
                return int(self._motion_lib.num_motions())
            except Exception:
                return 1
        return 1

    def set_motion(self, motion_id: int, reset_history: bool = True):
        n = max(self.num_motions(), 1)
        motion_id = int(motion_id) % n
        self.curr_motion_id = torch.tensor(motion_id, dtype=torch.long, device=self.device)
        self.curr_motion_tick0 = self.global_control_tick
        if reset_history:
            self.proprio_history_buf.clear()
            for _ in range(self.history_len):
                self.proprio_history_buf.append(np.zeros(self.n_proprio, dtype=np.float32))
            self.last_action[:] = 0.0

    def _init_motion_buffers(self):
        self.tar_obs_steps = torch.tensor(self.tar_obs_steps, device=self.device, dtype=torch.int32)

    # ------------------------- observation builders -------------------------

    def _get_mimic_obs(self, curr_time_step: int) -> np.ndarray:
        """
        Build the 'motion targets' window from MotionLib for the currently
        selected motion, using time relative to the last switch.
        """
        num_steps = int(self.tar_obs_steps.shape[0])

        # time since switch (in seconds), as 1-D tensor [num_steps]
        rel_time_s = (curr_time_step - self.curr_motion_tick0) * self.control_dt
        rel_time_s = torch.tensor(rel_time_s, device=self.device, dtype=torch.float32)
        obs_motion_times = self.tar_obs_steps.to(torch.float32) * self.control_dt + rel_time_s

        # which motion to sample at each step
        motion_ids = torch.full((num_steps,), int(self.curr_motion_id.item()),
                                dtype=torch.long, device=self.device)

        # sample frames (all shapes: [num_steps, ...])
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, _ = \
            self._motion_lib.calc_motion_frame(motion_ids, obs_motion_times)

        # to base frame + Euler
        roll, pitch, yaw = euler_from_quaternion(root_rot)          # each [num_steps]
        roll = roll.unsqueeze(-1); pitch = pitch.unsqueeze(-1)       # -> [num_steps,1]

        root_vel = quat_rotate_inverse(root_rot, root_vel)           # [num_steps,3]
        root_ang_vel = quat_rotate_inverse(root_rot, root_ang_vel)   # [num_steps,3]

        # Assemble per-step features (height z, roll, pitch, lin vel, yaw-rate z, all DOFs)
        mimic_obs_buf = torch.cat((
            root_pos[..., 2:3],
            roll, pitch,
            root_vel,
            root_ang_vel[..., 2:3],
            dof_pos,
        ), dim=-1)  # [num_steps, C]

        # Flatten steps -> 1D
        mimic_obs_buf = mimic_obs_buf.reshape(1, -1)
        return mimic_obs_buf.detach().cpu().numpy().squeeze()

    # ------------------------------ sim utils -------------------------------

    def extract_data(self):
        dof_pos = self.data.qpos.astype(np.float32)[-self.num_dofs:]
        dof_vel = self.data.qvel.astype(np.float32)[-self.num_dofs:]
        quat = self.data.sensor('orientation').data.astype(np.float32)          # [w,x,y,z]
        ang_vel = self.data.sensor('angular-velocity').data.astype(np.float32)  # [3]
        return (dof_pos, dof_vel, quat, ang_vel)
    
    # --------------------------------- main ----------------------------------

    def run(self):
        # optional: create a writer if recording video
        motion_name = os.path.basename(self.motion_path).split('.')[0]
        if self.record_video:
            import imageio
            video_name = f"{self.robot_type}_{''.join(os.path.basename(self.policy_path).split('.')[:-1])}_{motion_name}.mp4"
            os.makedirs("mujoco_videos", exist_ok=True)
            mp4_writer = imageio.get_writer(os.path.join("mujoco_videos", video_name), fps=50)

        # Initialize PD target to a valid vector before first control tick
        pd_target = self.default_dof_pos.copy()

        # Example scripted switches (comment out or modify as you like):
        # switch_tick_10s = int(10.0 / self.control_dt)
        # switch_tick_20s = int(20.0 / self.control_dt)
        start = time.time()
        for i in tqdm(range(int(self.sim_duration / self.sim_dt)), desc="Running simulation..."):
            # Read state every sim step
            dof_pos, dof_vel, quat, ang_vel = self.extract_data()

            # Control at 50 Hz
            if i % self.sim_decimation == 0:
                curr_timestep = i // self.sim_decimation
                # rotate motions every 5 seconds, if more than one motion is loaded
                if self.num_motions() > 1:
                    period_ticks = int(5.0 / self.control_dt)          # 5 s at 50 Hz -> 250 ticks
                    if curr_timestep % period_ticks == 0:
                        next_id = (int(curr_timestep / period_ticks)) % self.num_motions()
                        if int(self.curr_motion_id.item()) != next_id: # avoid redundant resets
                            print(f"[switch] t={curr_timestep*self.control_dt:.2f}s -> motion {next_id}")
                            self.set_motion(next_id)

                # Example time-based switching (tick-accurate). Uncomment if needed.
                # if curr_timestep == switch_tick_10s and self.num_motions() > 1:
                #     self.set_motion(1)
                # elif curr_timestep == switch_tick_20s and self.num_motions() > 2:
                #     self.set_motion(2)

                # advance control tick counter ONCE per control update
                self.global_control_tick += 1

                # Build obs (mimic + proprio + history)
                mimic_obs = self._get_mimic_obs(curr_timestep)

                rpy = quatToEuler(quat)  # numpy roll/pitch/yaw from [w,x,y,z]
                obs_dof_vel = dof_vel.copy()
                # zero out ankle velocities before scaling
                obs_dof_vel[[4, 5, 10, 11]] = 0.0

                obs_prop = np.concatenate([
                    ang_vel * self.ang_vel_scale,
                    rpy[:2],  # roll, pitch
                    (dof_pos - self.default_dof_pos) * self.dof_pos_scale,
                    obs_dof_vel * self.dof_vel_scale,
                    self.last_action,
                ]).astype(np.float32)

                assert obs_prop.shape[0] == self.n_proprio, f"Expected {self.n_proprio} but got {obs_prop.shape[0]}"
                obs_hist = np.array(self.proprio_history_buf, dtype=np.float32).flatten()

                obs_buf = np.concatenate([mimic_obs, obs_prop, obs_hist]).astype(np.float32)
                obs_tensor = torch.from_numpy(obs_buf).float().unsqueeze(0).to(self.device)

                with torch.no_grad():
                    raw_action = self.policy_jit(obs_tensor).cpu().numpy().squeeze()

                self.last_action = raw_action.copy()
                raw_action = np.clip(raw_action, -10.0, 10.0)
                scaled_actions = raw_action * self.action_scale

                # target joint positions for PD
                step_actions = np.zeros(self.num_dofs, dtype=np.float32)
                step_actions = scaled_actions.astype(np.float32)
                pd_target = step_actions + self.default_dof_pos

                # # render
                # self.viewer.cam.lookat = self.data.qpos.astype(np.float32)[:3]
                # if self.record_video:
                #     img = self.viewer.read_pixels()
                #     mp4_writer.append_data(img)
                # else:
                #     # self.viewer.render()
                #     self.viewer.sync()
                #     # time.sleep(self.sim_dt)   # keep real-time pace

                # self.viewer.sync()
                # target = start + (i // self.sim_decimation + 1) * self.control_dt
                # now = time.time()
                # if now < target:
                #     time.sleep(target - now)

                # push current proprio to history
                self.proprio_history_buf.append(obs_prop)

            # low-level PD at sim rate
            torque = (pd_target - dof_pos) * self.stiffness - dof_vel * self.damping
            torque = np.clip(torque, -self.torque_limits, self.torque_limits)
            self.data.ctrl = torque

            mujoco.mj_step(self.model, self.data)

            # Render every control_dt
            if i % self.sim_decimation == 0:
                self.viewer.sync()
                target = start + (i // self.sim_decimation + 1) * self.control_dt
                now = time.time()
                if now < target:
                    time.sleep(target - now)

        self.viewer.close()
        if self.record_video:
            mp4_writer.close()


# ---------------------------------- CLI --------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', type=str, default="g1")
    parser.add_argument('--checkpoint', type=int, default=-1)
    parser.add_argument('--record_video', action='store_true')
    parser.add_argument('--motion_file', type=str, default=".",
                        help="Motion path under assets/motions. Can be a single .pkl. "
                             "If your MotionLib supports it, this can also be a comma-"
                             "separated list or a directory.")
    args = parser.parse_args()

    jit_policy_pth = "assets/pretrained_checkpoints/pretrained.pt"
    assert os.path.exists(jit_policy_pth), f"Policy path {jit_policy_pth} does not exist!"
    print(f"Loading model from: {jit_policy_pth}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    motion_file = os.path.join("assets/motions", args.motion_file)
    env = HumanoidEnv(policy_path=jit_policy_pth,
                      motion_path=motion_file,
                      robot_type=args.robot,
                      device=device,
                      record_video=args.record_video)
    env.run()
