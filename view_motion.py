import os, argparse

import torch
import mujoco, mujoco_viewer
from tqdm import tqdm

from utils.motion_lib import MotionLib


class MotionViewEnv:
    def __init__(self, motion_file, robot_type="g1", device="cuda"):
        self.robot_type = robot_type
        self.device = device
        
        self.motion_file_name = os.path.basename(motion_file)
        
        self.motion_lib = MotionLib(motion_file=motion_file, device=device)
        self.motion_ids = torch.tensor([0], dtype=torch.long, device=device)
        self.motion_len = self.motion_lib.get_motion_length(self.motion_ids)
        
        model_path_root = "assets/robots"
        
        if robot_type == "g1":
            model_path = os.path.join(model_path_root, "g1/g1.xml")
        else:
            raise NotImplementedError("Robot type not supported")
        
        self.sim_duration = 10*self.motion_len.item()
        
        self.sim_dt = 0.02
        self.sim_decimation = 1
        self.control_dt = self.sim_dt * self.sim_decimation
        
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)
        mujoco.mj_step(self.model, self.data)
        
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer.cam.distance = 5.0
        
    def run(self):
        
        for i in tqdm(range(int(self.sim_duration / self.control_dt)), desc="Running simulation..."):
            curr_time = i * self.control_dt
            motion_time = torch.tensor([curr_time], dtype=torch.float, device=self.device) % self.motion_len
            root_pos, root_rot, _, _, dof_pos, _ = self.motion_lib.calc_motion_frame(self.motion_ids, motion_time)
            self.data.qpos[:3] = root_pos[0].cpu().numpy()
            self.data.qpos[3:7] = root_rot[0].cpu().numpy()[[3, 0, 1, 2]]
            self.data.qpos[7:] = dof_pos[0].cpu().numpy()
            
            mujoco.mj_forward(self.model, self.data)
            
            self.viewer.render()
        
        self.viewer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--motion_file', type=str, default="walk_stand.pkl")
    args = parser.parse_args()
    
    motion_path = os.path.join("assets/motions", args.motion_file)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    motion_env = MotionViewEnv(motion_file=motion_path, robot_type="g1", device=device)
    motion_env.run()
    