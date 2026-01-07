# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

from dataclasses import dataclass
from typing import Literal

from isaaclab.sim import PinholeCameraCfg
from isaaclab.utils import configclass

from .rgb_camera import RgbCamera
from .tiled_camera_cfg import TiledCameraCfg


@dataclass
class RgbSensorNoiseCfg:
    enable: bool = False
    mode: Literal["gaussian", "dropout", "combined"] = "gaussian"
    rgb_std: float = 0.01
    rgb_std_multiplier: float = 0.0
    dropout_prob: float = 0.0
    dropout_value: float = 0.0


@configclass
class RgbCameraCfg(TiledCameraCfg):
    class_type: type = RgbCamera

    enable_rgb_camera: bool = False
    prim_body_name: str = "torso_link"
    update_period: float = 0.1
    update_interval_steps: int = 5  # Number of simulation steps between camera updates (overrides update_period)
    width: int = 128
    height: int = 128
    data_types: list[str] = ["rgb"]
    offset: TiledCameraCfg.OffsetCfg = TiledCameraCfg.OffsetCfg()
    spawn: PinholeCameraCfg = PinholeCameraCfg()
    sensor_noise: RgbSensorNoiseCfg = RgbSensorNoiseCfg()
    # CRITICAL: Must be True for body-mounted cameras to follow the robot!
    # Without this, camera pose is only read at initialization
    update_latest_camera_pose: bool = True

    debug_vis: bool = False
