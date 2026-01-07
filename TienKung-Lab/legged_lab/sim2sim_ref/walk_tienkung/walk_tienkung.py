from common.path_config import PROJECT_ROOT

from FSM.FSMState import FSMStateName, FSMState
from common.ctrlcomp import StateAndCmd, PolicyOutput, FSMCommand
from common.utils import scale_values
import numpy as np
import yaml
import torch
import os

MUJOCO_DOF_NAMES = [
    'left_hip_pitch_joint',
    'left_hip_roll_joint',
    'left_hip_yaw_joint',
    'left_knee_joint',
    'left_ankle_pitch_joint',
    'left_ankle_roll_joint',
    'right_hip_pitch_joint',
    'right_hip_roll_joint',
    'right_hip_yaw_joint',
    'right_knee_joint',
    'right_ankle_pitch_joint',
    'right_ankle_roll_joint',
    'waist_yaw_joint',
    'waist_roll_joint',
    'waist_pitch_joint',
    'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint',
    'left_shoulder_yaw_joint',
    'left_elbow_joint',
    'left_wrist_roll_joint',
    'left_wrist_pitch_joint',
    'left_wrist_yaw_joint',
    'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint',
    'right_elbow_joint',
    'right_wrist_roll_joint',
    'right_wrist_pitch_joint',
    'right_wrist_yaw_joint'
]

LAB_DOF_NAMES = [
'left_hip_pitch_joint',
'right_hip_pitch_joint',
'waist_yaw_joint',
'left_hip_roll_joint',
'right_hip_roll_joint',
'waist_roll_joint',
'left_hip_yaw_joint',
'right_hip_yaw_joint',
'waist_pitch_joint',
'left_knee_joint',
'right_knee_joint',
'left_shoulder_pitch_joint',
'right_shoulder_pitch_joint',
'left_ankle_pitch_joint',
'right_ankle_pitch_joint',
'left_shoulder_roll_joint',
'right_shoulder_roll_joint',
'left_ankle_roll_joint',
'right_ankle_roll_joint',
'left_shoulder_yaw_joint',
'right_shoulder_yaw_joint',
'left_elbow_joint',
'right_elbow_joint',
'left_wrist_roll_joint',
'right_wrist_roll_joint',
'left_wrist_pitch_joint',
'right_wrist_pitch_joint',
'left_wrist_yaw_joint',
'right_wrist_yaw_joint'
]

class walk_tienkung(FSMState):
    def __init__(self, state_cmd:StateAndCmd, policy_output:PolicyOutput):
        super().__init__()
        self.state_cmd = state_cmd
        self.policy_output = policy_output
        self.name = FSMStateName.walk_tienkung
        self.name_str = "walk_tienkung"
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "walk_tienkung.yaml")
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

            # 自动查找 model 文件夹下的第一个模型文件（按文件名排序）
            model_dir = os.path.join(current_dir, "model")
            model_files = [f for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f))]
            model_files.sort()
            if len(model_files) == 0:
                raise FileNotFoundError(f"No model file found in {model_dir}")
            self.policy_path = os.path.join(model_dir, model_files[0])

            self.kps = np.array(config["kps"], dtype=np.float32)
            self.kds = np.array(config["kds"], dtype=np.float32)
            self.tau_limit =  np.array(config["tau_limit"], dtype=np.float32)
            self.default_angles =  np.array(config["default_angles"], dtype=np.float32)
            self.velocity_limit =  np.array(config["velocity_limit"], dtype=np.float32)
            self.dof29_index =  np.array(config["dof29_index"], dtype=np.int32)
            
            self.num_actions = config["num_actions"]
            self.num_obs = config["num_obs"]
            self.ang_vel_scale = config["ang_vel_scale"]
            self.dof_pos_scale = config["dof_pos_scale"]
            self.dof_vel_scale = config["dof_vel_scale"]
            self.action_scale = config["action_scale"]
            self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
            self.cmd_range = config["cmd_range"]
            self.range_velx = np.array([self.cmd_range["lin_vel_x"][0], self.cmd_range["lin_vel_x"][1]], dtype=np.float32)
            self.range_vely = np.array([self.cmd_range["lin_vel_y"][0], self.cmd_range["lin_vel_y"][1]], dtype=np.float32)
            self.range_velz = np.array([self.cmd_range["ang_vel_z"][0], self.cmd_range["ang_vel_z"][1]], dtype=np.float32)
            
            self.qj_obs = np.zeros(self.num_actions, dtype=np.float32)
            self.dqj_obs = np.zeros(self.num_actions, dtype=np.float32)
            self.cmd = np.array(config["cmd_init"], dtype=np.float32)
            self.obs = np.zeros(self.num_obs)
            self.actions = np.zeros(self.num_actions)
            self.last_actions = np.zeros(self.num_actions)
            self.history_length = 10
            self.step_dt = 0.02
            self.gait_cycle = 0.85
            self.gait_air_ratio_l = 0.38
            self.gait_air_ratio_r = 0.38
            self.gait_phase_offset_l = 0.38
            self.gait_phase_offset_r = 0.88
            self.gait_phase = np.zeros((2,), dtype=np.float32)  # 左右腿
            self.phase_ratio = np.zeros((2,), dtype=np.float32)
            # load policy
            self.policy = torch.jit.load(self.policy_path)

            print(f"walk_tienkung policy initializing ...")

    def enter(self):
        self.kps_reorder = np.zeros_like(self.kps)
        self.kds_reorder = np.zeros_like(self.kds)
        self.default_angles_reorder = np.zeros_like(self.default_angles)
        self.kps_reorder = self.kps.copy()
        self.kds_reorder = self.kds.copy()
        self.default_angles_reorder = self.default_angles.copy()
        self.buffer_obs_Init()
        self.episode_length = 0

    def MJ29_to_LAB29(self, array1):
        """
        Reorder array1 (MUJOCO_DOF_NAMES order) to match the order of array2
        (LAB_DOF_NAMES). Extra elements in MUJOCO_DOF_NAMES ('waist_roll_joint',
        'waist_pitch_joint') are ignored.

        Args:
            array1 (list or np.ndarray): Input array in MUJOCO_DOF_NAMES order.
            array2 (list or np.ndarray): Output array in LAB_DOF_NAMES order.

        Returns:
            np.ndarray: Reordered array matching LAB_DOF_NAMES order.
        """
        # Create a mapping from MUJOCO_DOF_NAMES to their indices
        mujoco_indices = {
            name: idx for idx, name in enumerate(MUJOCO_DOF_NAMES)
        }

        # Reorder array1 based on LAB_DOF_NAMES
        reordered_array = [
            array1[mujoco_indices[name]] for name in LAB_DOF_NAMES
        ]

        return np.array(reordered_array)

    def LAB29_to_MJ29(self, array1):
        """
        Reorder array1 (LAB_DOF_NAMES order) to match the order of MUJOCO_DOF_NAMES.
        Extra elements in MUJOCO_DOF_NAMES ('waist_roll_joint', 'waist_pitch_joint')
        are set to 0.

        Args:
            array1 (list or np.ndarray): Input array in LAB_DOF_NAMES order.

        Returns:
            np.ndarray: Reordered array matching MUJOCO_DOF_NAMES order, with
            extra elements set to 0.
        """
        # Create a mapping from LAB_DOF_NAMES to their indices
        lab_indices = {
            name: idx for idx, name in enumerate(LAB_DOF_NAMES)
        }

        # Reorder array1 based on MUJOCO_DOF_NAMES, setting extra elements to 0
        reordered_array = [
            array1[lab_indices[name]] if name in lab_indices else 0
            for name in MUJOCO_DOF_NAMES
        ]

        return np.array(reordered_array)

    def update_buffer(self, buffer: np.ndarray, new_value: np.ndarray):
        buffer[:-1] = buffer[1:]
        buffer[-1] = new_value
        return buffer

    def clear_all_buffers(self):
        """
        一键清空所有观测缓冲区。
        """
        self.ang_vel_buffer[...] = 0
        self.projected_gravity_buffer[...] = 0
        self.command_buffer[...] = 0
        self.joint_pos_buffer[...] = 0
        self.joint_vel_buffer[...] = 0
        self.actions_buffer[...] = 0
        self.sin_buffer[...] = 0
        self.cos_buffer[...] = 0
        self.phase_ratio_buffer[...] = 0
        self.action_obs_buffer[...] = 0

    def velocity_cmd(self):
        vel_cmd = self.state_cmd.vel_cmd.copy()
        vel_cmd = scale_values(vel_cmd, [self.range_velx, self.range_vely, self.range_velz])
        # if vel_cmd[0] < -1.0:
        #     vel_cmd[0] = -1.0
        # elif -0.1 < vel_cmd[0] < 0.1:
        #     vel_cmd[0] = 0
        return vel_cmd

    def _calculate_gait_para(self) -> None:
        t = self.episode_length * self.step_dt / self.gait_cycle
        self.gait_phase[0] = (t + self.gait_phase_offset_l) % 1.0
        self.gait_phase[1] = (t + self.gait_phase_offset_r) % 1.0

    def buffer_obs_Init(self):
        self.ang_vel_buffer = np.zeros((self.history_length, 3), dtype=np.float32)
        self.projected_gravity_buffer = np.zeros((self.history_length, 3), dtype=np.float32)
        self.command_buffer = np.zeros((self.history_length, 3), dtype=np.float32)
        self.joint_pos_buffer = np.zeros((self.history_length, 29), dtype=np.float32)
        self.joint_vel_buffer = np.zeros((self.history_length, 29), dtype=np.float32)
        self.actions_buffer = np.zeros((self.history_length, 29), dtype=np.float32)
        self.sin_buffer = np.zeros((self.history_length, 2), dtype=np.float32)
        self.cos_buffer = np.zeros((self.history_length, 2), dtype=np.float32)
        self.phase_ratio_buffer = np.zeros((self.history_length, 2), dtype=np.float32)
        self.action_obs_buffer = np.zeros((self.history_length, 102), dtype=np.float32)

    def run(self):
        self.episode_length += 1
        ang_vel = self.state_cmd.ang_vel.copy()
        projected_gravity = self.state_cmd.gravity_ori.copy()
        command = self.velocity_cmd()
        joint_pos = self.MJ29_to_LAB29(self.state_cmd.q.copy()-self.default_angles.copy())
        joint_vel = self.MJ29_to_LAB29(self.state_cmd.dq.copy()-self.default_angles.copy())
        actions = self.last_actions.copy()
        sin = np.sin(2 * np.pi * self.gait_phase)
        cos = np.cos(2 * np.pi * self.gait_phase) 
        self.phase_ratio = np.array([self.gait_air_ratio_l, self.gait_air_ratio_r], dtype=np.float32)
        
        action_obs = np.concatenate([
            ang_vel,
            projected_gravity,
            command,
            joint_pos,
            joint_vel,
            actions,
            sin,
            cos,
            self.phase_ratio
        ], axis=0)

        obs = self.update_buffer(self.action_obs_buffer, action_obs)

        obs = obs.reshape(1, -1)
        obs_tensor = torch.from_numpy(obs)
        self.actions = self.policy(obs_tensor).detach().numpy().squeeze()
        self.last_actions = self.actions
        self.MJ_actions = self.LAB29_to_MJ29(self.actions)
        loco_action = self.MJ_actions * self.action_scale + self.default_angles

        self.policy_output.actions = loco_action.copy()
        self.policy_output.kps = self.kps_reorder.copy()
        self.policy_output.kds = self.kds_reorder.copy()

    def exit(self):
        self.actions = np.zeros(self.num_actions)
        self.last_actions = np.zeros(self.num_actions)
        self.observations = np.zeros(self.num_obs)
        self.episode_length = 0
        self.clear_all_buffers()
        pass
    
    def checkChange(self):
        if(self.state_cmd.skill_cmd == FSMCommand.PASSIVE):
            return FSMStateName.PASSIVE
#============================================================#
        elif(self.state_cmd.skill_cmd == FSMCommand.stand_along):
            return FSMStateName.stand_along
#============================================================#
        else:
            return FSMStateName.walk_tienkung

