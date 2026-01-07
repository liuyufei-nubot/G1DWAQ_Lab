# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Licensed under the BSD-3-Clause license.

"""Custom event functions for domain randomization in visual RL training.

This module provides event functions for randomizing scene lighting conditions,
which is critical for RGB-based reinforcement learning and sim-to-real transfer.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import AssetBase
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# Default NVIDIA Nucleus path for HDR textures
NVIDIA_NUCLEUS_DIR = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5"


def sample_random_color(
    base: tuple[float, float, float] = (0.75, 0.75, 0.75),
    variation: float = 0.1
) -> tuple[float, float, float]:
    """Sample a random color close to the base color while preserving overall brightness.
    
    The relative balance between R, G, B components is maintained by ensuring 
    the sum of random offsets is zero.
    
    Args:
        base: The base RGB color with each component between 0 and 1.
        variation: Maximum deviation to sample for each channel before balancing.
        
    Returns:
        A new RGB color with balanced random variation.
    """
    # Generate random offsets for each channel in the range [-variation, variation]
    offsets = [random.uniform(-variation, variation) for _ in range(3)]
    # Compute the average offset
    avg_offset = sum(offsets) / 3
    # Adjust offsets so their sum is zero (maintaining brightness)
    balanced_offsets = [offset - avg_offset for offset in offsets]
    # Apply the balanced offsets to the base color and clamp each channel between 0 and 1
    new_color = tuple(
        max(0, min(1, base_component + offset))
        for base_component, offset in zip(base, balanced_offsets)
    )
    return new_color


def randomize_dome_light(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    intensity_range: tuple[float, float] = (500.0, 5000.0),
    color_variation: float = 0.3,
    textures: list[str] | None = None,
    randomize_intensity: bool = True,
    randomize_color: bool = True,
    randomize_texture: bool = False,
    default_intensity: float = 750.0,
    default_color: tuple[float, float, float] = (0.75, 0.75, 0.75),
    default_texture: str = "",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("sky_light"),
):
    """Randomize the dome light properties in the scene.
    
    This function randomizes the intensity, color, and HDR texture of a dome light,
    which is useful for domain randomization in RGB-based visual RL training.
    
    Args:
        env: The environment instance.
        env_ids: The environment IDs to randomize (not used for global lights).
        intensity_range: Range (min, max) for random intensity sampling.
        color_variation: Maximum variation for each color channel.
        textures: List of HDR texture paths to sample from. If None, texture is not changed.
        randomize_intensity: Whether to randomize light intensity.
        randomize_color: Whether to randomize light color.
        randomize_texture: Whether to randomize HDR texture.
        default_intensity: Default intensity value.
        default_color: Default RGB color (each component 0-1).
        default_texture: Default texture file path.
        asset_cfg: Configuration for the light asset in the scene.
    """
    # Get the light asset from the scene
    asset: AssetBase = env.scene[asset_cfg.name]
    light_prim = asset.prims[0]
    
    # Randomize intensity
    if randomize_intensity:
        intensity_attr = light_prim.GetAttribute("inputs:intensity")
        new_intensity = random.uniform(intensity_range[0], intensity_range[1])
        intensity_attr.Set(new_intensity)
    
    # Randomize color
    if randomize_color:
        color_attr = light_prim.GetAttribute("inputs:color")
        new_color = sample_random_color(base=default_color, variation=color_variation)
        color_attr.Set(new_color)
    
    # Randomize HDR texture (for dome light)
    if randomize_texture and textures:
        texture_file_attr = light_prim.GetAttribute("inputs:texture:file")
        new_texture = random.choice(textures)
        texture_file_attr.Set(new_texture)


def randomize_distant_light(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    intensity_range: tuple[float, float] = (1000.0, 5000.0),
    color_variation: float = 0.2,
    angle_variation: float = 30.0,  # degrees
    randomize_intensity: bool = True,
    randomize_color: bool = True,
    randomize_angle: bool = False,
    default_intensity: float = 3000.0,
    default_color: tuple[float, float, float] = (0.75, 0.75, 0.75),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("light"),
):
    """Randomize the distant light (sun-like) properties in the scene.
    
    This function randomizes the intensity, color, and angle of a distant light,
    simulating different sunlight conditions for domain randomization.
    
    Args:
        env: The environment instance.
        env_ids: The environment IDs to randomize (not used for global lights).
        intensity_range: Range (min, max) for random intensity sampling.
        color_variation: Maximum variation for each color channel.
        angle_variation: Maximum rotation angle variation in degrees.
        randomize_intensity: Whether to randomize light intensity.
        randomize_color: Whether to randomize light color.
        randomize_angle: Whether to randomize light angle/direction.
        default_intensity: Default intensity value.
        default_color: Default RGB color (each component 0-1).
        asset_cfg: Configuration for the light asset in the scene.
    """
    import math
    from pxr import Gf
    
    # Get the light asset from the scene
    asset: AssetBase = env.scene[asset_cfg.name]
    light_prim = asset.prims[0]
    
    # Randomize intensity
    if randomize_intensity:
        intensity_attr = light_prim.GetAttribute("inputs:intensity")
        new_intensity = random.uniform(intensity_range[0], intensity_range[1])
        intensity_attr.Set(new_intensity)
    
    # Randomize color
    if randomize_color:
        color_attr = light_prim.GetAttribute("inputs:color")
        new_color = sample_random_color(base=default_color, variation=color_variation)
        color_attr.Set(new_color)
    
    # Randomize direction/angle
    if randomize_angle:
        xform = light_prim.GetAttribute("xformOp:orient")
        if xform:
            # Random euler angles
            roll = random.uniform(-angle_variation, angle_variation)
            pitch = random.uniform(-angle_variation, angle_variation)
            yaw = random.uniform(-angle_variation, angle_variation)
            # Convert to quaternion (simplified)
            rot = Gf.Rotation(Gf.Vec3d(1, 0, 0), pitch) * \
                  Gf.Rotation(Gf.Vec3d(0, 1, 0), yaw) * \
                  Gf.Rotation(Gf.Vec3d(0, 0, 1), roll)
            quat = rot.GetQuat()
            xform.Set(Gf.Quatd(quat))


def randomize_scene_lighting(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    dome_light_cfg: SceneEntityCfg = SceneEntityCfg("sky_light"),
    distant_light_cfg: SceneEntityCfg = SceneEntityCfg("light"),
    dome_intensity_range: tuple[float, float] = (300.0, 2000.0),
    distant_intensity_range: tuple[float, float] = (1000.0, 6000.0),
    color_variation: float = 0.3,
    textures: list[str] | None = None,
    randomize_dome: bool = True,
    randomize_distant: bool = True,
    randomize_texture: bool = False,
):
    """Randomize both dome light and distant light in the scene.
    
    This is a convenience function that randomizes all lighting in the scene
    for comprehensive domain randomization in visual RL.
    
    Args:
        env: The environment instance.
        env_ids: The environment IDs to randomize.
        dome_light_cfg: Configuration for the dome light asset.
        distant_light_cfg: Configuration for the distant light asset.
        dome_intensity_range: Intensity range for dome light.
        distant_intensity_range: Intensity range for distant light.
        color_variation: Maximum color variation for both lights.
        textures: List of HDR textures for dome light randomization.
        randomize_dome: Whether to randomize dome light.
        randomize_distant: Whether to randomize distant light.
        randomize_texture: Whether to randomize HDR texture.
    """
    if randomize_dome:
        try:
            randomize_dome_light(
                env=env,
                env_ids=env_ids,
                intensity_range=dome_intensity_range,
                color_variation=color_variation,
                textures=textures,
                randomize_intensity=True,
                randomize_color=True,
                randomize_texture=randomize_texture,
                asset_cfg=dome_light_cfg,
            )
        except KeyError:
            pass  # Dome light not in scene
    
    if randomize_distant:
        try:
            randomize_distant_light(
                env=env,
                env_ids=env_ids,
                intensity_range=distant_intensity_range,
                color_variation=color_variation,
                randomize_intensity=True,
                randomize_color=True,
                asset_cfg=distant_light_cfg,
            )
        except KeyError:
            pass  # Distant light not in scene


# Pre-defined HDR sky textures from NVIDIA Nucleus (for convenience)
DEFAULT_SKY_TEXTURES = [
    f"{NVIDIA_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
    # Add more textures as needed - these paths should be verified
    # "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Skies/Cloudy/abandoned_parking_4k.hdr",
    # "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Skies/Cloudy/evening_road_01_4k.hdr",
]


def randomize_terrain_material(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    mdl_materials: list[dict] | None = None,
    randomize_material: bool = True,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("terrain"),
):
    """Randomize the terrain visual material by switching between MDL materials.
    
    This function changes the terrain surface material to simulate different ground types
    (concrete, asphalt, grass, etc.), which is useful for RGB-based visual RL domain randomization.
    
    Note: This is a simplified implementation. Full MDL material switching at runtime
    requires more complex USD operations. This serves as a placeholder for future implementation.
    
    Args:
        env: The environment instance.
        env_ids: The environment IDs to randomize.
        mdl_materials: List of MDL material configurations, each with 'mdl_path' and 'texture_scale'.
        randomize_material: Whether to randomize material.
        asset_cfg: Configuration for the terrain asset in the scene.
    """
    if not randomize_material or not mdl_materials:
        return
    
    # Note: Full runtime MDL material switching is complex in Isaac Sim
    # This is a placeholder - actual implementation would need to:
    # 1. Get the terrain prim
    # 2. Unbind current material
    # 3. Create/bind new MDL material
    # 4. Set texture scale
    
    # For now, material randomization should be done at environment creation time
    # by randomly selecting from MDL_TERRAIN_MATERIALS in the config
    pass
