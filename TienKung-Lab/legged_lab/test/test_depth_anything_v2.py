"""
Depth Anything V2 Metric Depth 测试脚本

测试 Depth Anything V2 带真实尺度的室内深度估计模型，
验证模型加载、深度推理以及高维特征提取功能。

用于人形机器人上台阶的视觉感知预研。
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, Tuple, Union
from torchvision import transforms

# 添加 metric_depth 模块路径
SCRIPT_DIR = Path(__file__).parent.absolute()
MODULE_DIR = SCRIPT_DIR.parent / "modules" / "Depth-Anything-V2" / "metric_depth"
sys.path.insert(0, str(MODULE_DIR))

from depth_anything_v2.dpt import DepthAnythingV2


# ImageNet 标准化参数
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class DepthAnythingV2ForRL(nn.Module):
    """
    用于强化学习的 Depth Anything V2 特征提取器
    
    特点：
    1. 支持 batch 输入（并行环境）
    2. 支持 tensor 和 numpy 输入
    3. 可选的特征投影层（384 → embedding_dim）
    4. 冻结 encoder 减少显存和计算
    5. 高效的 CLS token 特征提取
    
    使用示例：
        # 初始化
        encoder = DepthAnythingV2ForRL(
            encoder='vits',
            embedding_dim=128,
            freeze_encoder=True
        )
        
        # 在 RL 环境中使用
        # images: [B, 3, H, W] tensor, 值范围 [0, 1] 或 [0, 255]
        features = encoder(images)  # [B, embedding_dim]
    """
    
    MODEL_CONFIGS = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384], 'embed_dim': 384},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768], 'embed_dim': 768},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024], 'embed_dim': 1024},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536], 'embed_dim': 1536}
    }
    
    def __init__(
        self,
        encoder: str = 'vits',
        checkpoint_path: str = None,
        max_depth: float = 20.0,
        input_size: int = 518,
        embedding_dim: int = 128,
        freeze_encoder: bool = True,
        use_projection: bool = True,
        feature_type: str = 'cls',  # 'cls', 'avg_pool', 'concat'
        device: str = None
    ):
        """
        Args:
            encoder: 编码器类型 ('vits', 'vitb', 'vitl', 'vitg')
            checkpoint_path: 模型权重路径
            max_depth: 最大深度值 (室内20m, 室外80m)
            input_size: 输入图像尺寸
            embedding_dim: 输出特征维度
            freeze_encoder: 是否冻结 encoder 参数
            use_projection: 是否使用投影层
            feature_type: 特征类型 ('cls', 'avg_pool', 'concat')
            device: 计算设备
        """
        super().__init__()
        
        self.encoder_type = encoder
        self.max_depth = max_depth
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.freeze_encoder = freeze_encoder
        self.feature_type = feature_type
        
        # 自动选择设备
        if device is None:
            self.device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device_str = device
        
        # 获取配置
        config = self.MODEL_CONFIGS[encoder]
        self.encoder_embed_dim = config['embed_dim']
        
        # 构建 Depth Anything V2 模型
        self.depth_model = DepthAnythingV2(
            encoder=config['encoder'],
            features=config['features'],
            out_channels=config['out_channels'],
            max_depth=max_depth
        )
        
        # 加载权重
        if checkpoint_path is None:
            checkpoint_path = MODULE_DIR / "checkpoints" / f"depth_anything_v2_metric_hypersim_{encoder}.pth"
        
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        self.depth_model.load_state_dict(state_dict)
        
        # 冻结 encoder
        if freeze_encoder:
            for param in self.depth_model.parameters():
                param.requires_grad = False
        
        # 图像标准化
        self.normalize = transforms.Normalize(
            mean=IMAGENET_DEFAULT_MEAN, 
            std=IMAGENET_DEFAULT_STD
        )
        
        # 特征投影层
        if feature_type == 'concat':
            input_dim = self.encoder_embed_dim * 2
        else:
            input_dim = self.encoder_embed_dim
            
        if use_projection:
            self.projection = nn.Sequential(
                nn.Linear(input_dim, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.SiLU(),
            )
        else:
            self.projection = nn.Identity()
            self.embedding_dim = input_dim
        
        # 移动到设备
        self.to(self.device_str)
        
        print(f"[DepthAnythingV2ForRL] Initialized:")
        print(f"  - Encoder: {encoder} (embed_dim={self.encoder_embed_dim})")
        print(f"  - Feature type: {feature_type}")
        print(f"  - Output dim: {self.embedding_dim}")
        print(f"  - Freeze encoder: {freeze_encoder}")
        print(f"  - Device: {self.device_str}")
    
    def _preprocess_batch(self, images: torch.Tensor) -> torch.Tensor:
        """
        预处理 batch 图像
        
        Args:
            images: [B, 3, H, W] 或 [B, H, W, 3]，值范围 [0, 1] 或 [0, 255]
            
        Returns:
            normalized: [B, 3, H', W'] 标准化后的图像
        """
        # 确保是 [B, C, H, W] 格式
        if images.dim() == 3:
            images = images.unsqueeze(0)
        if images.shape[-1] == 3:  # [B, H, W, 3] -> [B, 3, H, W]
            images = images.permute(0, 3, 1, 2)
        
        # 移动到正确的设备
        images = images.to(self.device_str)
        
        # 转换到 [0, 1]
        if images.max() > 1.0:
            images = images.float() / 255.0
        
        # Resize 到指定尺寸
        if images.shape[-2:] != (self.input_size, self.input_size):
            images = F.interpolate(
                images, 
                size=(self.input_size, self.input_size),
                mode='bilinear',
                align_corners=False
            )
        
        # ImageNet 标准化
        images = self.normalize(images)
        
        return images
    
    def extract_cls_token(self, x: torch.Tensor) -> torch.Tensor:
        """
        高效提取 CLS token（不经过 DPT head）
        
        Args:
            x: [B, 3, H, W] 预处理后的图像
            
        Returns:
            cls_token: [B, embed_dim]
        """
        # 直接使用 DINOv2 encoder
        pretrained = self.depth_model.pretrained
        
        # 通过 patch embedding
        B, _, H, W = x.shape
        x = pretrained.patch_embed(x)
        
        # 添加 CLS token
        cls_token = pretrained.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # 添加位置编码
        x = x + pretrained.interpolate_pos_encoding(x, H, W)
        
        # 通过 transformer blocks
        for blk in pretrained.blocks:
            x = blk(x)
        
        x = pretrained.norm(x)
        
        # 返回 CLS token
        return x[:, 0]  # [B, embed_dim]
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取特征
        
        Args:
            x: [B, 3, H, W] 预处理后的图像
            
        Returns:
            features: [B, embed_dim] 或 [B, 2*embed_dim]
        """
        pretrained = self.depth_model.pretrained
        
        # 通过 patch embedding
        B, _, H, W = x.shape
        patch_tokens = pretrained.patch_embed(x)
        
        # 添加 CLS token
        cls_token = pretrained.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, patch_tokens), dim=1)
        
        # 添加位置编码
        x = x + pretrained.interpolate_pos_encoding(x, H, W)
        
        # 通过 transformer blocks
        for blk in pretrained.blocks:
            x = blk(x)
        
        x = pretrained.norm(x)
        
        cls_token = x[:, 0]  # [B, embed_dim]
        patch_tokens = x[:, 1:]  # [B, num_patches, embed_dim]
        
        if self.feature_type == 'cls':
            return cls_token
        elif self.feature_type == 'avg_pool':
            return patch_tokens.mean(dim=1)
        elif self.feature_type == 'concat':
            avg_pool = patch_tokens.mean(dim=1)
            return torch.cat([cls_token, avg_pool], dim=-1)
        else:
            raise ValueError(f"Unknown feature_type: {self.feature_type}")
    
    def forward(
        self, 
        images: torch.Tensor,
        return_depth: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播
        
        Args:
            images: [B, 3, H, W] 或 [B, H, W, 3]，值范围 [0, 1] 或 [0, 255]
            return_depth: 是否同时返回深度图
            
        Returns:
            如果 return_depth=False:
                features: [B, embedding_dim] 投影后的特征
            如果 return_depth=True:
                (features, depth): features [B, embedding_dim], depth [B, H, W]
        """
        # 预处理
        x = self._preprocess_batch(images)
        
        # 提取特征
        if self.freeze_encoder:
            with torch.no_grad():
                raw_features = self.extract_features(x)
        else:
            raw_features = self.extract_features(x)
        
        # 投影
        features = self.projection(raw_features)
        
        if return_depth:
            # 计算深度
            with torch.no_grad():
                depth = self.depth_model(x)
            return features, depth
        
        return features
    
    def get_output_dim(self) -> int:
        """获取输出特征维度"""
        return self.embedding_dim
    
    @torch.no_grad()
    def infer_depth(self, images: torch.Tensor) -> torch.Tensor:
        """
        仅推理深度
        
        Args:
            images: [B, 3, H, W]
            
        Returns:
            depth: [B, H, W] 深度图，单位米
        """
        x = self._preprocess_batch(images)
        return self.depth_model(x)


class DepthAnythingV2FeatureExtractor:
    """
    Depth Anything V2 特征提取器
    
    除了输出深度图，还可以提取 DINOv2 encoder 的高维特征，
    用于和机器人本体感知融合，驱动上台阶等任务。
    """
    
    # 模型配置
    MODEL_CONFIGS = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    # 中间层索引，用于提取多尺度特征
    INTERMEDIATE_LAYER_IDX = {
        'vits': [2, 5, 8, 11],
        'vitb': [2, 5, 8, 11], 
        'vitl': [4, 11, 17, 23], 
        'vitg': [9, 19, 29, 39]
    }
    
    def __init__(
        self, 
        encoder: str = 'vits',
        checkpoint_path: str = None,
        max_depth: float = 20.0,
        input_size: int = 518,
        device: str = None
    ):
        """
        Args:
            encoder: 编码器类型 ('vits', 'vitb', 'vitl', 'vitg')
            checkpoint_path: 模型权重路径
            max_depth: 最大深度值 (室内20m, 室外80m)
            input_size: 输入图像尺寸
            device: 计算设备
        """
        self.encoder = encoder
        self.max_depth = max_depth
        self.input_size = input_size
        
        # 自动选择设备
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"[DepthAnythingV2] Using device: {self.device}")
        print(f"[DepthAnythingV2] Encoder: {encoder}")
        print(f"[DepthAnythingV2] Max depth: {max_depth}m")
        
        # 构建模型
        config = self.MODEL_CONFIGS[encoder]
        self.model = DepthAnythingV2(
            encoder=config['encoder'],
            features=config['features'],
            out_channels=config['out_channels'],
            max_depth=max_depth
        )
        
        # 加载权重
        if checkpoint_path is None:
            checkpoint_path = MODULE_DIR / "checkpoints" / f"depth_anything_v2_metric_hypersim_{encoder}.pth"
        
        print(f"[DepthAnythingV2] Loading checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device).eval()
        
        # 获取 embedding 维度
        self.embed_dim = self.model.pretrained.embed_dim
        print(f"[DepthAnythingV2] Embed dim: {self.embed_dim}")
        
    def preprocess(self, image: np.ndarray) -> tuple:
        """
        预处理图像
        
        Args:
            image: BGR 格式的 numpy 图像
            
        Returns:
            tensor: 预处理后的 tensor
            original_size: 原始图像尺寸 (h, w)
        """
        return self.model.image2tensor(image, self.input_size)
    
    @torch.no_grad()
    def infer_depth(self, image: np.ndarray) -> np.ndarray:
        """
        推理深度图
        
        Args:
            image: BGR 格式的 numpy 图像
            
        Returns:
            depth: HxW 深度图，单位为米
        """
        return self.model.infer_image(image, self.input_size)
    
    @torch.no_grad()
    def extract_features(self, image: np.ndarray, return_depth: bool = True) -> dict:
        """
        提取高维特征和深度图
        
        Args:
            image: BGR 格式的 numpy 图像
            return_depth: 是否同时返回深度图
            
        Returns:
            dict: 包含特征和深度信息的字典
                - 'cls_token': [1, embed_dim] CLS token 特征
                - 'patch_features': [1, num_patches, embed_dim] patch 特征
                - 'intermediate_features': list of [1, num_patches, embed_dim] 中间层特征
                - 'depth': HxW 深度图 (如果 return_depth=True)
                - 'depth_tensor': [1, 1, H, W] 深度 tensor
        """
        # 预处理
        tensor, (h, w) = self.preprocess(image)
        patch_h, patch_w = tensor.shape[-2] // 14, tensor.shape[-1] // 14
        
        # 提取 DINOv2 encoder 的特征
        intermediate_idx = self.INTERMEDIATE_LAYER_IDX[self.encoder]
        features = self.model.pretrained.get_intermediate_layers(
            tensor, 
            intermediate_idx, 
            return_class_token=True
        )
        
        # 解析特征
        # features 是一个 list，每个元素是 (patch_features, cls_token)
        intermediate_features = []
        cls_tokens = []
        for feat in features:
            patch_feat, cls_token = feat
            intermediate_features.append(patch_feat)  # [1, num_patches, embed_dim]
            cls_tokens.append(cls_token)  # [1, embed_dim]
        
        # 使用最后一层的 CLS token 作为全局特征
        global_feature = cls_tokens[-1]  # [1, embed_dim]
        
        # 使用最后一层的 patch 特征
        last_patch_features = intermediate_features[-1]  # [1, num_patches, embed_dim]
        
        result = {
            'cls_token': global_feature,
            'patch_features': last_patch_features,
            'intermediate_features': intermediate_features,
            'cls_tokens': cls_tokens,
            'patch_h': patch_h,
            'patch_w': patch_w,
            'embed_dim': self.embed_dim,
        }
        
        if return_depth:
            # 通过 DPT head 获取深度
            depth = self.model.depth_head(features, patch_h, patch_w) * self.max_depth
            # depth shape: [1, H, W], 需要添加 channel 维度
            if depth.dim() == 3:
                depth = depth.unsqueeze(1)  # [1, 1, H, W]
            depth_resized = F.interpolate(
                depth, (h, w), mode="bilinear", align_corners=True
            )
            result['depth'] = depth_resized[0, 0].cpu().numpy()
            result['depth_tensor'] = depth_resized
            
        return result
    
    @torch.no_grad()
    def get_flat_feature(self, image: np.ndarray, feature_type: str = 'cls') -> torch.Tensor:
        """
        获取用于强化学习的扁平化特征向量
        
        Args:
            image: BGR 格式的 numpy 图像
            feature_type: 特征类型
                - 'cls': 使用 CLS token [1, embed_dim]
                - 'avg_pool': 对 patch 特征进行平均池化 [1, embed_dim]
                - 'concat': 连接 CLS 和平均池化特征 [1, 2*embed_dim]
                
        Returns:
            feature: 扁平化的特征向量
        """
        result = self.extract_features(image, return_depth=False)
        
        if feature_type == 'cls':
            return result['cls_token']  # [1, embed_dim]
        elif feature_type == 'avg_pool':
            # 对 patch 特征进行平均池化
            return result['patch_features'].mean(dim=1)  # [1, embed_dim]
        elif feature_type == 'concat':
            cls_feat = result['cls_token']
            avg_feat = result['patch_features'].mean(dim=1)
            return torch.cat([cls_feat, avg_feat], dim=-1)  # [1, 2*embed_dim]
        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")


def visualize_depth(depth: np.ndarray, save_path: str = None, show: bool = True):
    """可视化深度图"""
    plt.figure(figsize=(10, 8))
    
    # 使用 Spectral colormap
    plt.imshow(depth, cmap='Spectral')
    plt.colorbar(label='Depth (m)')
    plt.title(f'Depth Map\nMin: {depth.min():.2f}m, Max: {depth.max():.2f}m')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved depth visualization to {save_path}")
    
    if show:
        plt.show()
    plt.close()


def visualize_features(features: dict, save_path: str = None, show: bool = True):
    """可视化特征图"""
    patch_features = features['patch_features']  # [1, num_patches, embed_dim]
    patch_h, patch_w = features['patch_h'], features['patch_w']
    
    # Reshape to spatial
    feat_map = patch_features[0].reshape(patch_h, patch_w, -1).cpu().numpy()
    
    # PCA 降维到 3 通道用于可视化
    from sklearn.decomposition import PCA
    feat_flat = feat_map.reshape(-1, feat_map.shape[-1])
    pca = PCA(n_components=3)
    feat_pca = pca.fit_transform(feat_flat)
    feat_pca = feat_pca.reshape(patch_h, patch_w, 3)
    
    # 归一化到 [0, 1]
    feat_pca = (feat_pca - feat_pca.min()) / (feat_pca.max() - feat_pca.min())
    
    plt.figure(figsize=(10, 8))
    plt.imshow(feat_pca)
    plt.title(f'Feature Map (PCA visualization)\nShape: {patch_h}x{patch_w}x{features["embed_dim"]}')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved feature visualization to {save_path}")
    
    if show:
        plt.show()
    plt.close()


def main():
    """主测试函数"""
    print("=" * 60)
    print("Depth Anything V2 Metric Depth 测试")
    print("=" * 60)
    
    # 测试图像路径
    test_image_path = SCRIPT_DIR / "test.jpg"
    if not test_image_path.exists():
        print(f"Error: 测试图像不存在: {test_image_path}")
        return
    
    # 输出目录
    output_dir = SCRIPT_DIR / "output"
    output_dir.mkdir(exist_ok=True)
    
    # 加载测试图像
    print(f"\n加载测试图像: {test_image_path}")
    image = cv2.imread(str(test_image_path))
    print(f"图像尺寸: {image.shape}")
    
    # 创建特征提取器
    print("\n初始化 Depth Anything V2...")
    extractor = DepthAnythingV2FeatureExtractor(
        encoder='vits',  # 使用 small 模型
        max_depth=20.0,  # 室内场景
        input_size=518
    )
    
    # 测试深度推理
    print("\n" + "-" * 40)
    print("测试 1: 深度图推理")
    print("-" * 40)
    
    import time
    start = time.time()
    depth = extractor.infer_depth(image)
    inference_time = (time.time() - start) * 1000
    
    print(f"深度图尺寸: {depth.shape}")
    print(f"深度范围: {depth.min():.2f}m - {depth.max():.2f}m")
    print(f"推理时间: {inference_time:.1f}ms")
    
    # 保存深度图
    np.save(output_dir / "depth.npy", depth)
    visualize_depth(depth, save_path=output_dir / "depth_vis.png", show=False)
    
    # 测试特征提取
    print("\n" + "-" * 40)
    print("测试 2: 高维特征提取")
    print("-" * 40)
    
    start = time.time()
    features = extractor.extract_features(image, return_depth=True)
    extract_time = (time.time() - start) * 1000
    
    print(f"CLS token 尺寸: {features['cls_token'].shape}")
    print(f"Patch 特征尺寸: {features['patch_features'].shape}")
    print(f"中间层特征数量: {len(features['intermediate_features'])}")
    for i, feat in enumerate(features['intermediate_features']):
        print(f"  Layer {i}: {feat.shape}")
    print(f"提取时间: {extract_time:.1f}ms")
    
    # 可视化特征
    try:
        visualize_features(features, save_path=output_dir / "features_pca.png", show=False)
    except ImportError:
        print("Warning: sklearn 未安装，跳过 PCA 可视化")
    
    # 测试扁平化特征
    print("\n" + "-" * 40)
    print("测试 3: 强化学习特征提取")
    print("-" * 40)
    
    for feat_type in ['cls', 'avg_pool', 'concat']:
        feat = extractor.get_flat_feature(image, feature_type=feat_type)
        print(f"特征类型 '{feat_type}': shape = {feat.shape}")
    
    # 性能测试
    print("\n" + "-" * 40)
    print("测试 4: 性能基准测试 (10次推理)")
    print("-" * 40)
    
    times = []
    for _ in range(10):
        start = time.time()
        _ = extractor.infer_depth(image)
        times.append((time.time() - start) * 1000)
    
    print(f"平均推理时间: {np.mean(times):.1f}ms")
    print(f"最小推理时间: {np.min(times):.1f}ms")
    print(f"最大推理时间: {np.max(times):.1f}ms")
    print(f"推理频率: {1000 / np.mean(times):.1f} FPS")
    
    # 保存原图和深度图对比
    print("\n" + "-" * 40)
    print("保存结果")
    print("-" * 40)
    
    # 并排显示
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    im = axes[1].imshow(depth, cmap='Spectral')
    axes[1].set_title(f'Depth Map (range: {depth.min():.2f}m - {depth.max():.2f}m)')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], label='Depth (m)')
    
    plt.tight_layout()
    plt.savefig(output_dir / "comparison.png", dpi=150, bbox_inches='tight')
    print(f"结果已保存到: {output_dir}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


def test_rl_integration():
    """测试 RL 集成接口"""
    print("=" * 60)
    print("Depth Anything V2 RL 集成测试")
    print("=" * 60)
    
    # 测试图像
    test_image_path = SCRIPT_DIR / "test.jpg"
    if not test_image_path.exists():
        print(f"Error: 测试图像不存在: {test_image_path}")
        return
    
    output_dir = SCRIPT_DIR / "output"
    output_dir.mkdir(exist_ok=True)
    
    # 加载测试图像
    image = cv2.imread(str(test_image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 转换为 tensor [B, 3, H, W]
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).float()
    print(f"输入图像 tensor shape: {image_tensor.shape}")
    
    # 初始化 RL 模型
    print("\n初始化 DepthAnythingV2ForRL...")
    model = DepthAnythingV2ForRL(
        encoder='vits',
        embedding_dim=128,
        freeze_encoder=True,
        feature_type='cls'
    )
    
    # 测试单张图像
    print("\n" + "-" * 40)
    print("测试 1: 单张图像特征提取")
    print("-" * 40)
    
    import time
    start = time.time()
    features = model(image_tensor)
    elapsed = (time.time() - start) * 1000
    
    print(f"输出特征 shape: {features.shape}")
    print(f"特征范围: [{features.min().item():.4f}, {features.max().item():.4f}]")
    print(f"推理时间: {elapsed:.1f}ms")
    
    # 测试 batch 输入
    print("\n" + "-" * 40)
    print("测试 2: Batch 特征提取 (模拟并行环境)")
    print("-" * 40)
    
    batch_size = 16
    batch_images = image_tensor.repeat(batch_size, 1, 1, 1)
    print(f"Batch 输入 shape: {batch_images.shape}")
    
    start = time.time()
    batch_features = model(batch_images)
    elapsed = (time.time() - start) * 1000
    
    print(f"Batch 输出 shape: {batch_features.shape}")
    print(f"推理时间: {elapsed:.1f}ms")
    print(f"每张图像: {elapsed/batch_size:.1f}ms")
    
    # 测试同时返回深度
    print("\n" + "-" * 40)
    print("测试 3: 同时获取特征和深度")
    print("-" * 40)
    
    start = time.time()
    features, depth = model(image_tensor, return_depth=True)
    elapsed = (time.time() - start) * 1000
    
    print(f"特征 shape: {features.shape}")
    print(f"深度 shape: {depth.shape}")
    print(f"深度范围: {depth.min().item():.2f}m - {depth.max().item():.2f}m")
    print(f"推理时间: {elapsed:.1f}ms")
    
    # 测试不同 feature_type
    print("\n" + "-" * 40)
    print("测试 4: 不同特征类型")
    print("-" * 40)
    
    for feat_type in ['cls', 'avg_pool', 'concat']:
        model_test = DepthAnythingV2ForRL(
            encoder='vits',
            embedding_dim=128,
            feature_type=feat_type,
            freeze_encoder=True
        )
        features = model_test(image_tensor)
        print(f"  {feat_type}: shape = {features.shape}")
    
    # 测试梯度流（用于训练投影层）
    print("\n" + "-" * 40)
    print("测试 5: 梯度流测试（投影层可训练）")
    print("-" * 40)
    
    model.train()
    image_tensor.requires_grad_(False)
    
    features = model(image_tensor)
    loss = features.mean()
    loss.backward()
    
    # 检查梯度
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            print(f"  ✓ {name}: grad norm = {param.grad.norm().item():.6f}")
    
    if has_grad:
        print("  投影层梯度正常!")
    else:
        print("  警告: 没有检测到梯度")
    
    # 检查 encoder 是否冻结
    encoder_has_grad = False
    for name, param in model.depth_model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            encoder_has_grad = True
            break
    
    if not encoder_has_grad:
        print("  ✓ Encoder 已冻结，无梯度")
    else:
        print("  警告: Encoder 有梯度流动")
    
    # 性能基准
    print("\n" + "-" * 40)
    print("测试 6: 性能基准 (batch_size=16, 10次)")
    print("-" * 40)
    
    model.eval()
    batch_images = image_tensor.repeat(16, 1, 1, 1).cuda() if torch.cuda.is_available() else image_tensor.repeat(16, 1, 1, 1)
    
    # 预热
    for _ in range(3):
        _ = model(batch_images)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    times = []
    for _ in range(10):
        start = time.time()
        _ = model(batch_images)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.time() - start) * 1000)
    
    print(f"Batch size: 16")
    print(f"平均时间: {np.mean(times):.1f}ms")
    print(f"每张图像: {np.mean(times)/16:.2f}ms")
    print(f"吞吐量: {16000 / np.mean(times):.1f} images/s")
    
    # 模型信息
    print("\n" + "-" * 40)
    print("模型信息")
    print("-" * 40)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数: {trainable_params / 1e6:.4f}M")
    print(f"输出维度: {model.get_output_dim()}")
    
    print("\n" + "=" * 60)
    print("RL 集成测试完成!")
    print("=" * 60)


def test_video(video_path: str, save_video: bool = True):
    """测试视频深度估计和特征提取"""
    import time
    
    print("=" * 60)
    print("Depth Anything V2 视频测试")
    print("=" * 60)
    
    # 检查视频文件
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"Error: 视频文件不存在: {video_path}")
        return
    
    output_dir = SCRIPT_DIR / "output"
    output_dir.mkdir(exist_ok=True)
    
    # 打开视频
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: 无法打开视频: {video_path}")
        return
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n视频信息:")
    print(f"  - 路径: {video_path}")
    print(f"  - 分辨率: {width}x{height}")
    print(f"  - FPS: {fps:.1f}")
    print(f"  - 总帧数: {frame_count}")
    print(f"  - 时长: {frame_count/fps:.1f}s")
    
    # 初始化模型
    print("\n初始化 DepthAnythingV2ForRL...")
    model = DepthAnythingV2ForRL(
        encoder='vits',
        embedding_dim=128,
        freeze_encoder=True,
        feature_type='cls'
    )
    
    # 准备视频写入
    if save_video:
        output_path = output_dir / f"{video_path.stem}_depth.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(
            str(output_path), 
            fourcc, 
            fps, 
            (width * 2, height)  # 并排显示原图和深度图
        )
        print(f"\n输出视频: {output_path}")
    
    # 处理视频
    print("\n" + "-" * 40)
    print("开始处理视频...")
    print("-" * 40)
    
    frame_times = []
    feature_list = []
    frame_idx = 0
    
    # 创建 colormap
    cmap = plt.cm.get_cmap('Spectral')
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # BGR -> RGB -> tensor
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float()
            
            # 推理
            start = time.time()
            with torch.no_grad():
                features, depth = model(frame_tensor, return_depth=True)
            elapsed = (time.time() - start) * 1000
            frame_times.append(elapsed)
            
            # 保存特征
            feature_list.append(features.cpu())
            
            # 可视化深度图
            depth_np = depth[0].cpu().numpy()
            
            # 归一化到 [0, 1] 用于可视化
            depth_normalized = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
            
            # 应用 colormap
            depth_colored = (cmap(depth_normalized)[:, :, :3] * 255).astype(np.uint8)
            depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)
            
            # Resize 深度图到原始尺寸
            depth_colored = cv2.resize(depth_colored, (width, height))
            
            # 添加深度范围文字
            cv2.putText(
                depth_colored, 
                f"Depth: {depth_np.min():.2f}m - {depth_np.max():.2f}m", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (255, 255, 255), 
                2
            )
            
            # 添加 FPS
            cv2.putText(
                depth_colored, 
                f"FPS: {1000/elapsed:.1f}", 
                (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (255, 255, 255), 
                2
            )
            
            # 并排合并
            combined = np.hstack([frame, depth_colored])
            
            # 写入视频
            if save_video:
                out_writer.write(combined)
            
            # 进度显示
            if frame_idx % 30 == 0 or frame_idx == 1:
                avg_time = np.mean(frame_times[-30:])
                print(f"  帧 {frame_idx}/{frame_count} | "
                      f"深度范围: {depth_np.min():.2f}-{depth_np.max():.2f}m | "
                      f"耗时: {elapsed:.1f}ms | "
                      f"FPS: {1000/avg_time:.1f}")
    
    finally:
        cap.release()
        if save_video:
            out_writer.release()
    
    # 统计结果
    print("\n" + "-" * 40)
    print("处理统计")
    print("-" * 40)
    
    print(f"处理帧数: {frame_idx}")
    print(f"平均推理时间: {np.mean(frame_times):.1f}ms")
    print(f"最小推理时间: {np.min(frame_times):.1f}ms")
    print(f"最大推理时间: {np.max(frame_times):.1f}ms")
    print(f"平均 FPS: {1000/np.mean(frame_times):.1f}")
    
    # 保存特征序列
    all_features = torch.cat(feature_list, dim=0)
    feature_path = output_dir / f"{video_path.stem}_features.pt"
    torch.save(all_features, feature_path)
    print(f"\n特征序列已保存: {feature_path}")
    print(f"特征 shape: {all_features.shape}")
    
    # 分析特征变化
    print("\n" + "-" * 40)
    print("特征分析")
    print("-" * 40)
    
    # 计算帧间特征相似度
    if len(feature_list) > 1:
        similarities = []
        for i in range(len(feature_list) - 1):
            f1 = feature_list[i][0]
            f2 = feature_list[i + 1][0]
            sim = F.cosine_similarity(f1.unsqueeze(0), f2.unsqueeze(0)).item()
            similarities.append(sim)
        
        print(f"帧间余弦相似度:")
        print(f"  平均: {np.mean(similarities):.4f}")
        print(f"  最小: {np.min(similarities):.4f}")
        print(f"  最大: {np.max(similarities):.4f}")
        
        # 绘制相似度曲线
        plt.figure(figsize=(12, 4))
        plt.plot(similarities)
        plt.xlabel('Frame')
        plt.ylabel('Cosine Similarity')
        plt.title('Frame-to-Frame Feature Similarity')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"{video_path.stem}_similarity.png", dpi=150)
        plt.close()
        print(f"相似度曲线已保存: {output_dir / f'{video_path.stem}_similarity.png'}")
    
    if save_video:
        print(f"\n输出视频已保存: {output_path}")
    
    print("\n" + "=" * 60)
    print("视频测试完成!")
    print("=" * 60)


'''
# 测试默认视频
python test_depth_anything_v2.py --mode video --save-video

# 测试自定义视频
python test_depth_anything_v2.py --mode video --video /path/to/video.mp4 --save-video
'''

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='all', choices=['basic', 'rl', 'video', 'all'])
    parser.add_argument('--video', type=str, default=None, help='视频文件路径')
    parser.add_argument('--save-video', action='store_true', help='保存输出视频')
    args = parser.parse_args()
    
    if args.mode in ['basic', 'all']:
        main()
    
    if args.mode in ['rl', 'all']:
        print("\n\n")
        test_rl_integration()
    
    if args.mode == 'video':
        video_path = args.video if args.video else str(SCRIPT_DIR / "test_video.mp4")
        test_video(video_path, save_video=args.save_video)
