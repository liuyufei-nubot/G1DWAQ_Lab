"""
Depth Anything V2 视觉 Encoder

用于人形机器人强化学习的 RGB 特征提取器，
基于 Depth Anything V2 的 DINOv2 encoder。

特点：
1. 使用带真实尺度的 metric depth 预训练权重
2. 冻结 encoder，仅训练投影层
3. 支持降频特征提取（模拟真实部署）
4. 输出 CLS token 作为全局场景表示

使用方式：
    from legged_lab.modules.depth_anything_encoder import DepthAnythingEncoder, VisionFeatureManager
    
    # 初始化
    encoder = DepthAnythingEncoder(encoder='vits', embedding_dim=128)
    
    # 在环境中使用
    feature_manager = VisionFeatureManager(encoder, num_envs=4096, update_interval=5)
    
    # 每步调用
    features = feature_manager.step(rgb_images, dones)
"""

import sys
import math
import itertools
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# Depth Anything V2 模块路径
_MODULE_DIR = Path(__file__).parent / "Depth-Anything-V2" / "metric_depth"
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

from depth_anything_v2.dpt import DepthAnythingV2

# ImageNet 标准化参数
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class CenterPadding(nn.Module):
    """
    将图像 pad 到 patch_size 的倍数（仿照 gauss_gym）
    
    相比 resize 到固定尺寸，这种方式可以保持小图像的优势：
    - 85×48 图像只需 pad 到 98×56 (7×4=28 patches)
    - 而不是 resize 到 224×224 (16×16=256 patches)
    
    显存节省约 9 倍！
    """
    
    def __init__(self, multiple: int = 14):
        super().__init__()
        self.multiple = multiple  # ViT patch size
    
    def _get_pad(self, size: int) -> Tuple[int, int]:
        """计算需要 pad 的大小"""
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right
    
    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] 图像张量
        Returns:
            padded: [B, C, H', W'] pad 后的图像，H'和W'是 multiple 的倍数
        """
        # 从最后一个维度开始 pad (W, H)
        pads = list(itertools.chain.from_iterable(
            self._get_pad(m) for m in x.shape[:1:-1]
        ))
        output = F.pad(x, pads)
        return output


class DepthAnythingEncoder(nn.Module):
    """
    Depth Anything V2 视觉特征提取器
    
    基于 DINOv2 encoder，使用 metric depth 预训练权重。
    输出 CLS token 作为全局场景特征，用于和本体感知融合。
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
        checkpoint_path: Optional[str] = None,
        max_depth: float = 20.0,
        input_size: int = 224,  # 仅在 use_center_padding=False 时使用
        embedding_dim: int = 128,
        freeze_encoder: bool = True,
        use_projection: bool = False,  # 默认不投影，让 ActorCritic 处理
        feature_type: str = 'cls',
        use_center_padding: bool = True,  # 仿 gauss_gym，使用 CenterPadding 而非 resize
        patch_size: int = 14,  # ViT patch size
        device: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Args:
            encoder: 编码器类型 ('vits', 'vitb', 'vitl', 'vitg')
            checkpoint_path: 权重路径，None 则使用默认路径
            max_depth: 最大深度 (室内20m, 室外80m)
            input_size: 输入图像尺寸 (仅在 use_center_padding=False 时使用)
            embedding_dim: 投影后的维度 (use_projection=True 时)
            freeze_encoder: 是否冻结 encoder
            use_projection: 是否使用内置投影层
            feature_type: 特征类型 ('cls', 'avg_pool', 'concat')
            use_center_padding: 使用 CenterPadding (仿 gauss_gym，大幅节省显存)
            patch_size: ViT patch size，用于 CenterPadding
            device: 设备
            verbose: 是否打印信息
        """
        super().__init__()
        
        self.encoder_type = encoder
        self.max_depth = max_depth
        self.input_size = input_size
        self.freeze_encoder = freeze_encoder
        self.feature_type = feature_type
        self.use_center_padding = use_center_padding
        self.patch_size = patch_size
        
        # CenterPadding 模块 (仿 gauss_gym)
        if use_center_padding:
            self.center_padding = CenterPadding(multiple=patch_size)
        else:
            self.center_padding = None
        
        if device is None:
            self.device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device_str = device
        
        # 获取配置
        config = self.MODEL_CONFIGS[encoder]
        self.encoder_embed_dim = config['embed_dim']
        
        # 构建模型
        self.depth_model = DepthAnythingV2(
            encoder=config['encoder'],
            features=config['features'],
            out_channels=config['out_channels'],
            max_depth=max_depth
        )
        
        # 加载权重
        if checkpoint_path is None:
            checkpoint_path = _MODULE_DIR / "checkpoints" / f"depth_anything_v2_metric_hypersim_{encoder}.pth"
        
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        self.depth_model.load_state_dict(state_dict)
        
        # 冻结
        if freeze_encoder:
            for param in self.depth_model.parameters():
                param.requires_grad = False
            self.depth_model.eval()
        
        # 图像标准化
        self.register_buffer(
            'mean', 
            torch.tensor(IMAGENET_DEFAULT_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std', 
            torch.tensor(IMAGENET_DEFAULT_STD).view(1, 3, 1, 1)
        )
        
        # 可选投影层
        if feature_type == 'concat':
            raw_dim = self.encoder_embed_dim * 2
        else:
            raw_dim = self.encoder_embed_dim
        
        if use_projection:
            self.projection = nn.Sequential(
                nn.Linear(raw_dim, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.SiLU(),
            )
            self.output_dim = embedding_dim
        else:
            self.projection = None
            self.output_dim = raw_dim
        
        self.to(self.device_str)
        
        if verbose:
            print(f"[DepthAnythingEncoder] Initialized:")
            print(f"  - Encoder: {encoder} (raw_dim={self.encoder_embed_dim})")
            print(f"  - Feature type: {feature_type}")
            print(f"  - Output dim: {self.output_dim}")
            print(f"  - Freeze encoder: {freeze_encoder}")
            print(f"  - Use CenterPadding: {use_center_padding} (patch_size={patch_size})")
            if not use_center_padding:
                print(f"  - Input size: {input_size}x{input_size}")
            print(f"  - Device: {self.device_str}")
    
    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """预处理图像
        
        使用 CenterPadding (仿 gauss_gym) 或 resize 处理图像尺寸。
        CenterPadding 保持原始小尺寸，只 pad 到 patch_size 的倍数，大幅节省显存。
        
        例如 224×224 相机：
        - CenterPadding: 224×224 → 224×224 (已是14的倍数，无需pad) → 16×16=256 patches
        - 如果是 85×48: 85×48 → 98×56 (pad到14倍数) → 7×4=28 patches (节省9倍显存！)
        """
        # [B, H, W, 3] -> [B, 3, H, W]
        if images.dim() == 3:
            images = images.unsqueeze(0)
        if images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2)
        
        images = images.to(self.device_str)
        
        # [0, 255] -> [0, 1]
        if images.max() > 1.0:
            images = images.float() / 255.0
        
        # 尺寸处理：CenterPadding (仿 gauss_gym) 或 resize
        if self.use_center_padding:
            # CenterPadding: 只 pad 到 patch_size 的倍数，保持小图像优势
            images = self.center_padding(images)
        else:
            # Resize: 传统方式，resize 到固定尺寸
            if images.shape[-2:] != (self.input_size, self.input_size):
                images = F.interpolate(
                    images,
                    size=(self.input_size, self.input_size),
                    mode='bilinear',
                    align_corners=False
                )
        
        # ImageNet 标准化
        images = (images - self.mean) / self.std
        
        return images
    
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """从预处理后的图像提取特征"""
        pretrained = self.depth_model.pretrained
        
        B, _, H, W = x.shape
        patch_tokens = pretrained.patch_embed(x)
        
        cls_token = pretrained.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, patch_tokens), dim=1)
        x = x + pretrained.interpolate_pos_encoding(x, H, W)
        
        for blk in pretrained.blocks:
            x = blk(x)
        
        x = pretrained.norm(x)
        
        cls_token = x[:, 0]
        patch_tokens = x[:, 1:]
        
        if self.feature_type == 'cls':
            return cls_token
        elif self.feature_type == 'avg_pool':
            return patch_tokens.mean(dim=1)
        elif self.feature_type == 'concat':
            return torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=-1)
        else:
            return cls_token
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        提取视觉特征
        
        Args:
            images: [B, H, W, 3] 或 [B, 3, H, W]，范围 [0,255] 或 [0,1]
            
        Returns:
            features: [B, output_dim]
        """
        x = self._preprocess(images)
        
        if self.freeze_encoder:
            with torch.no_grad():
                features = self._extract_features(x)
        else:
            features = self._extract_features(x)
        
        if self.projection is not None:
            features = self.projection(features)
        
        return features
    
    @torch.no_grad()
    def infer_depth(self, images: torch.Tensor) -> torch.Tensor:
        """推理深度图"""
        x = self._preprocess(images)
        return self.depth_model(x)
    
    def get_output_dim(self) -> int:
        """获取输出维度"""
        return self.output_dim


class VisionFeatureManager:
    """
    视觉特征管理器
    
    实现降频特征提取，模拟真实部署场景：
    - encoder 每 N 步运行一次
    - 中间步骤复用缓存特征
    """
    
    def __init__(
        self,
        encoder: DepthAnythingEncoder,
        num_envs: int,
        update_interval: int = 5,
        device: str = 'cuda'
    ):
        """
        Args:
            encoder: DepthAnythingEncoder 实例
            num_envs: 环境数量
            update_interval: 更新间隔
            device: 设备
        """
        self.encoder = encoder
        self.num_envs = num_envs
        self.update_interval = update_interval
        self.device = device
        self.feature_dim = encoder.get_output_dim()
        
        # 特征缓存
        self.cached_features = torch.zeros(num_envs, self.feature_dim, device=device)
        # 步数计数
        self.step_counter = torch.zeros(num_envs, dtype=torch.long, device=device)
        
        print(f"[VisionFeatureManager] Initialized:")
        print(f"  - num_envs: {num_envs}")
        print(f"  - feature_dim: {self.feature_dim}")
        print(f"  - update_interval: {update_interval}")
    
    def step(
        self,
        images: torch.Tensor,
        dones: Optional[torch.Tensor] = None,
        force_update: bool = False
    ) -> torch.Tensor:
        """
        执行一步，返回视觉特征
        
        Args:
            images: [num_envs, H, W, 3] RGB 图像
            dones: [num_envs] 重置标志
            force_update: 强制更新所有特征
            
        Returns:
            features: [num_envs, feature_dim]
        """
        # 判断需要更新的环境
        if force_update:
            update_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            update_mask = (self.step_counter % self.update_interval) == 0
        
        # 重置的环境强制更新
        if dones is not None:
            update_mask = update_mask | dones
        
        # 更新特征
        if update_mask.any():
            update_ids = update_mask.nonzero(as_tuple=True)[0]
            # images 可能在 CPU，用 CPU indices 索引
            update_ids_cpu = update_ids.cpu()
            with torch.no_grad():
                new_features = self.encoder(images[update_ids_cpu])
            self.cached_features[update_ids] = new_features
        
        # 更新计数器
        self.step_counter += 1
        if dones is not None:
            self.step_counter[dones] = 0
        
        return self.cached_features.clone()
    
    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """重置"""
        if env_ids is None:
            self.cached_features.zero_()
            self.step_counter.zero_()
        else:
            self.cached_features[env_ids] = 0
            self.step_counter[env_ids] = 0
    
    def get_features(self) -> torch.Tensor:
        """获取当前缓存的特征"""
        return self.cached_features
