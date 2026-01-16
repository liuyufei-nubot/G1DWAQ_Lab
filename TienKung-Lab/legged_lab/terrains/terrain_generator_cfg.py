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

"""
Configuration classes defining the different terrains available. Each configuration class must
inherit from ``isaaclab.terrains.terrains_cfg.TerrainConfig`` and define the following attributes:

- ``name``: Name of the terrain. This is used for the prim name in the USD stage.
- ``function``: Function to generate the terrain. This function must take as input the terrain difficulty
  and the configuration parameters and return a `tuple with the `trimesh`` mesh object and terrain origin.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

GRAVEL_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=False,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(-0.02, 0.04), noise_step=0.02, border_width=0.25
        )
    },
)

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # ========== 上台阶 (中心高，向外下降) - 20% ==========
        "stairs_up_28": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.0, 0.23),
            step_width=0.28,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_up_32": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.0, 0.23),
            step_width=0.32,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # ========== 下台阶 (中心低，向外上升) - 20% ==========
        "stairs_down_30": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.0, 0.23),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_down_34": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.0, 0.23),
            step_width=0.34,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # ========== 其他地形 - 60% ==========
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.1, grid_width=0.45, grid_height_range=(0.0, 0.15), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15, noise_range=(-0.02, 0.04), noise_step=0.02, border_width=0.25
        ),
        "wave": terrain_gen.HfWaveTerrainCfg(
            proportion=0.1, amplitude_range=(0.0, 0.2), num_waves=5.0
        ),
        "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.3), platform_width=2.0, inverted=False
        ),
        "high_platform": terrain_gen.MeshPitTerrainCfg(
            proportion=0.15, pit_depth_range=(0.0, 0.3), platform_width=2.0, double_pit=True
        ),
        # "gap": terrain_gen.MeshGapTerrainCfg(
        #     proportion=0.1, gap_width_range=(0.1, 0.4), platform_width=2.0
        # ),
    },
)

# ========== Play 专用: 纯台阶地形 (最大难度) ==========
STAIRS_ONLY_HARD_CFG = TerrainGeneratorCfg(
    curriculum=False,  # Play 时关闭课程学习
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=5,
    num_cols=5,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    difficulty_range=(1.0, 1.0),  # 最大难度
    sub_terrains={
        # ========== 上台阶 - 50% ==========
        "stairs_up_narrow": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.20, 0.25),  # 最大台阶高度
            step_width=0.26,  # 窄台阶，更难
            platform_width=2.5,
            border_width=1.0,
            holes=False,
        ),
        "stairs_up_wide": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.20, 0.25),
            step_width=0.32,
            platform_width=2.5,
            border_width=1.0,
            holes=False,
        ),
        # ========== 下台阶 - 50% ==========
        "stairs_down_narrow": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.20, 0.25),  # 最大台阶高度
            step_width=0.26,  # 窄台阶，更难
            platform_width=2.5,
            border_width=1.0,
            holes=False,
        ),
        "stairs_down_wide": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.20, 0.25),
            step_width=0.32,
            platform_width=2.5,
            border_width=1.0,
            holes=False,
        ),
    },
)

# ========== Play 专用: 混合台阶 + 斜坡 (高难度) ==========
STAIRS_SLOPE_HARD_CFG = TerrainGeneratorCfg(
    curriculum=False,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=5,
    num_cols=5,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    difficulty_range=(0.8, 1.0),  # 高难度
    sub_terrains={
        # ========== 上台阶 - 35% ==========
        "stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.35,
            step_height_range=(0.18, 0.25),
            step_width=0.28,
            platform_width=2.5,
            border_width=1.0,
            holes=False,
        ),
        # ========== 下台阶 - 35% ==========
        "stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.35,
            step_height_range=(0.18, 0.25),
            step_width=0.28,
            platform_width=2.5,
            border_width=1.0,
            holes=False,
        ),
        # ========== 斜坡 - 30% ==========
        "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=(0.25, 0.4), platform_width=2.0, inverted=False
        ),
        "slope_down": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=(0.25, 0.4), platform_width=2.0, inverted=True
        ),
    },
)


# ========== DWAQ 专用: 渐进式地形 (修改版：降低初期难度) ==========
# 原版 DreamWaQ 使用 70% 台阶，但对于初期学习太难
# 修改为 40% 台阶 + 60% 简单地形，便于 VAE 快速学习
DWAQ_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # ========== 上台阶 - 20% (降低难度) ==========
        "stairs_up_26": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.07,
            step_height_range=(0.0, 0.23),
            step_width=0.26,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_up_30": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.07,
            step_height_range=(0.0, 0.23),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_up_34": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.06,
            step_height_range=(0.0, 0.23),
            step_width=0.34,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # ========== 下台阶 - 20% (降低难度) ==========
        "stairs_down_26": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.07,
            step_height_range=(0.0, 0.23),
            step_width=0.26,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_down_30": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.07,
            step_height_range=(0.0, 0.23),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_down_34": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.06,
            step_height_range=(0.0, 0.23),
            step_width=0.34,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # ========== 简单地形 - 60% (增加便于学习的地形) ==========
        # 使用非常平缓的随机地形作为"平地"替代
        "flat": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.25,  # 25% 近似平地
            noise_range=(0.0, 0.02),  # 几乎无噪声
            noise_step=0.01,
            border_width=0.25,
        ),
        "smooth_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=(0.0, 0.2), platform_width=2.0, inverted=False
        ),
        "rough_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.2), platform_width=2.0, inverted=True
        ),
        "discrete": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.1, grid_width=0.45, grid_height_range=(0.0, 0.1), platform_width=2.0
        ),
    },
)


# ========== DWAQ 高难度: 窄台阶 (20cm宽度) ==========
# 用于训练后期，resume 时切换使用
DWAQ_HARD_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # ========== 上台阶 - 35% (窄台阶 20cm) ==========
        "stairs_up_20": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.12,
            step_height_range=(0.0, 0.25),  # 最高 25cm
            step_width=0.20,  # 20cm 窄台阶
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_up_24": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.12,
            step_height_range=(0.0, 0.25),
            step_width=0.24,  # 24cm
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_up_28": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.11,
            step_height_range=(0.0, 0.23),
            step_width=0.28,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # ========== 下台阶 - 35% (窄台阶 20cm) ==========
        "stairs_down_20": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.12,
            step_height_range=(0.0, 0.25),
            step_width=0.20,  # 20cm 窄台阶
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_down_24": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.12,
            step_height_range=(0.0, 0.25),
            step_width=0.24,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_down_28": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.11,
            step_height_range=(0.0, 0.23),
            step_width=0.28,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # ========== 其他高难度地形 - 30% ==========
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.1, grid_width=0.30, grid_height_range=(0.0, 0.18), platform_width=2.0
        ),
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1, noise_range=(-0.03, 0.06), noise_step=0.02, border_width=0.25
        ),
        "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.35), platform_width=2.0, inverted=False
        ),
    },
)

