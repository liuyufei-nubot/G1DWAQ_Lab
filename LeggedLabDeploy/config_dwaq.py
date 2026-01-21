# Copyright (c) 2022-2025, The unitree_rl_gym Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.

"""
DWAQ 配置类

与标准 Config 的区别:
1. 支持 dwaq_obs_history_length (DWAQ 使用 5 帧历史)
2. 支持 cenet_out_dim (VAE encoder 输出维度)
"""

import numpy as np
import yaml


class ConfigDWAQ:
    def __init__(self, file_path) -> None:
        with open(file_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

            self.control_dt = config["control_dt"]

            self.msg_type = config["msg_type"]
            self.imu_type = config["imu_type"]

            self.weak_motor = []
            if "weak_motor" in config:
                self.weak_motor = config["weak_motor"]

            self.lowcmd_topic = config["lowcmd_topic"]
            self.lowstate_topic = config["lowstate_topic"]

            self.policy_path = config["policy_path"]

            self.joint2motor_idx = config["joint2motor_idx"]
            self.kps = config["kps"]
            self.kds = config["kds"]
            self.default_joint_pos = np.array(config["default_joint_pos"], dtype=np.float32)

            if "torso_idx" in config:
                self.torso_idx = config["torso_idx"]

            self.ang_vel_scale = config["ang_vel_scale"]
            self.dof_pos_scale = config["dof_pos_scale"]
            self.dof_vel_scale = config["dof_vel_scale"]
            self.action_scale = config["action_scale"]
            self.command_scale = np.array(config["command_scale"], dtype=np.float32)

            self.num_actions = config["num_actions"]
            self.num_obs = config["num_obs"]

            # DWAQ 特有参数
            self.dwaq_obs_history_length = config.get("dwaq_obs_history_length", 5)
            self.cenet_out_dim = config.get("cenet_out_dim", 19)

            # 标准历史长度 (用于兼容性)
            self.history_length = config.get("history_length", 1)
            
            self.command_range = config["command_range"]
