"""
Legged Lab 视觉模块

主要模块：
- DepthAnythingEncoder: Depth Anything V2 视觉特征提取器
- VisionFeatureManager: 降频视觉特征管理器
"""

from .depth_anything_encoder import DepthAnythingEncoder, VisionFeatureManager

__all__ = ["DepthAnythingEncoder", "VisionFeatureManager"]
