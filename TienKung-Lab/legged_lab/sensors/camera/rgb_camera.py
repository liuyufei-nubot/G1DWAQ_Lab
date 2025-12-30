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

from collections.abc import Sequence
from typing import TYPE_CHECKING

import carb
import torch
from isaaclab.sensors.camera import TiledCamera as BaseTiledCamera

if TYPE_CHECKING:
    from .rgb_camera_cfg import RgbCameraCfg


class RgbCamera(BaseTiledCamera):
    """RGB camera based on tiled rendering; applies simple noise to rgb frames."""

    def __init__(self, cfg: "RgbCameraCfg"):
        self._is_headless = carb.settings.get_settings().get("/app/runLoops/main/headless")
        if self._is_headless:
            cfg.debug_vis = False

        super().__init__(cfg)
        self.cfg = cfg
        self.sensor_noise = self.cfg.sensor_noise

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        super()._update_buffers_impl(env_ids)
        self._apply_noise()

    def _apply_noise(self):
        if not self.sensor_noise.enable:
            return

        if "rgb" not in self._data.output:
            return

        images = self._data.output["rgb"]
        if self.sensor_noise.mode in ["gaussian", "combined"]:
            images = self._apply_gaussian_noise(images)
        if self.sensor_noise.mode in ["dropout", "combined"]:
            images = self._apply_dropout_noise(images)
        self._data.output["rgb"] = images

    def _apply_gaussian_noise(self, images: torch.Tensor) -> torch.Tensor:
        # Convert to float for noise operations (uint8 doesn't support randn_like)
        orig_dtype = images.dtype
        images_f = images.float()
        std_dev = self.sensor_noise.rgb_std + images_f * self.sensor_noise.rgb_std_multiplier
        noise = torch.randn_like(images_f) * std_dev
        noisy = images_f + noise
        # Clip to valid range and convert back
        if orig_dtype == torch.uint8:
            return noisy.clamp(0, 255).to(torch.uint8)
        else:
            return noisy.clamp(0, 1) if noisy.max() <= 1.0 else noisy.clamp(0, 255)

    def _apply_dropout_noise(self, images: torch.Tensor) -> torch.Tensor:
        # Use float for generating random mask, broadcast to all channels
        dropout_mask = torch.rand(images.shape[:-1], device=images.device, dtype=torch.float32) < self.sensor_noise.dropout_prob
        dropout_mask = dropout_mask.unsqueeze(-1).expand_as(images)
        noisy = images.clone()
        noisy[dropout_mask] = self.sensor_noise.dropout_value
        return noisy
