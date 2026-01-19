"""
测试 Depth Anything Encoder 与 ActorCriticVision 的集成

该脚本演示完整的视觉特征融合管线：
1. DepthAnythingEncoder 提取 CLS token (384-dim)
2. VisionFeatureManager 降频更新 (每5步)
3. ActorCriticVision 融合视觉和本体特征

使用方法：
    python test_integration.py
"""

import sys
from pathlib import Path
import time

import torch
import numpy as np

# 添加路径
sys.path.insert(0, str(Path(__file__).parents[2]))
sys.path.insert(0, str(Path(__file__).parents[2] / "rsl_rl"))

from legged_lab.modules import DepthAnythingEncoder, VisionFeatureManager
from rsl_rl.modules import ActorCriticVision, VisionFeatureBuffer
from rsl_rl.storage import RolloutStorageVision
from rsl_rl.algorithms import VisionPPO


def test_encoder_basic():
    """测试 Encoder 基本功能"""
    print("=" * 60)
    print("Test 1: DepthAnythingEncoder Basic")
    print("=" * 60)
    
    encoder = DepthAnythingEncoder(
        encoder='vits',
        embedding_dim=128,
        freeze_encoder=True,
        use_projection=False,  # 不投影，输出原始 384-dim
        verbose=True
    )
    
    # 创建假图像 [B, H, W, 3]
    batch_size = 4
    images = torch.randint(0, 256, (batch_size, 480, 640, 3), dtype=torch.uint8)
    
    # 提取特征
    with torch.no_grad():
        features = encoder(images)
    
    print(f"\nInput shape: {images.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Output dim: {encoder.get_output_dim()}")
    
    assert features.shape == (batch_size, 384)
    print("✓ Basic encoder test passed")
    
    return encoder


def test_feature_manager():
    """测试 VisionFeatureManager 降频更新"""
    print("\n" + "=" * 60)
    print("Test 2: VisionFeatureManager Frequency Control")
    print("=" * 60)
    
    encoder = DepthAnythingEncoder(
        encoder='vits',
        freeze_encoder=True,
        use_projection=False,
        verbose=False
    )
    
    num_envs = 8
    update_interval = 5
    
    manager = VisionFeatureManager(
        encoder=encoder,
        num_envs=num_envs,
        update_interval=update_interval,
        device='cuda'
    )
    
    # 模拟 20 步
    print(f"\nSimulating 20 steps (update every {update_interval} steps):")
    
    for step in range(20):
        images = torch.randint(0, 256, (num_envs, 480, 640, 3), dtype=torch.uint8)
        
        # 模拟第 10 步有些环境重置
        if step == 10:
            dones = torch.tensor([0, 1, 0, 1, 0, 0, 0, 0], dtype=torch.bool, device='cuda')
        else:
            dones = torch.zeros(num_envs, dtype=torch.bool, device='cuda')
        
        features = manager.step(images, dones)
        
        # 打印更新情况
        should_update = (manager.step_counter % update_interval) == 1  # 刚更新后
        updated = "UPDATE" if step % update_interval == 0 or step == 10 else ""
        print(f"  Step {step:2d}: features.norm()={features.norm(dim=1).mean():.3f} {updated}")
    
    print("✓ Feature manager test passed")


def test_integration_with_actor_critic():
    """测试与 ActorCriticVision 的完整集成"""
    print("\n" + "=" * 60)
    print("Test 3: Full Integration with ActorCriticVision")
    print("=" * 60)
    
    # 配置
    num_envs = 16
    num_obs = 45  # 本体观测维度
    num_critic_obs = 55  # Critic 观测维度 (可包含特权信息)
    history_len = 20
    history_dim = 24
    num_actions = 12
    
    # 1. 创建 Encoder
    print("\n[1] Creating DepthAnythingEncoder...")
    encoder = DepthAnythingEncoder(
        encoder='vits',
        freeze_encoder=True,
        use_projection=False,  # 让 ActorCritic 内部投影
        verbose=True
    )
    vision_feature_dim = encoder.get_output_dim()  # 384
    
    # 2. 创建 Feature Manager
    print("\n[2] Creating VisionFeatureManager...")
    manager = VisionFeatureManager(
        encoder=encoder,
        num_envs=num_envs,
        update_interval=5,
        device='cuda'
    )
    
    # 3. 创建 ActorCriticVision
    print("\n[3] Creating ActorCriticVision...")
    policy = ActorCriticVision(
        num_actor_obs=num_obs,
        num_critic_obs=num_critic_obs,
        num_actions=num_actions,
        # History encoder
        history_dim=history_len * history_dim,  # flattened history
        his_encoder_dims=[256, 128],
        his_latent_dim=64,
        # Vision
        vision_feature_dim=vision_feature_dim,  # 384
        vision_latent_dim=128,  # 投影到 128
        use_vision_projection=True,
        # Actor/Critic
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[256, 256],
        activation='elu'
    )
    policy = policy.to('cuda')
    
    print(f"\n  Policy vision_feature_dim: {policy.vision_feature_dim}")
    
    # 4. 模拟训练循环
    print("\n[4] Simulating training loop (100 steps)...")
    
    obs = torch.randn(num_envs, num_obs, device='cuda')
    critic_obs = torch.randn(num_envs, num_critic_obs, device='cuda')
    history = torch.randn(num_envs, history_len, history_dim, device='cuda')
    
    times_encoder = []
    times_policy = []
    
    for step in range(100):
        # 生成假图像
        images = torch.randint(0, 256, (num_envs, 480, 640, 3), dtype=torch.uint8)
        dones = torch.zeros(num_envs, dtype=torch.bool, device='cuda')
        
        # 模拟一些环境重置
        if step % 20 == 0 and step > 0:
            dones[:4] = True
        
        # VisionFeatureManager 自动处理降频
        t0 = time.time()
        vision_features = manager.step(images, dones)
        torch.cuda.synchronize()
        times_encoder.append(time.time() - t0)
        
        # Policy 前向
        t0 = time.time()
        actions = policy.act(obs, history, vision_feature=vision_features)
        torch.cuda.synchronize()
        times_policy.append(time.time() - t0)
        
        if step % 20 == 0:
            print(f"  Step {step:3d}: action.mean={actions.mean():.4f}, vision_feat.norm={vision_features.norm(dim=1).mean():.2f}")
    
    avg_encoder_ms = np.mean(times_encoder) * 1000
    avg_policy_ms = np.mean(times_policy) * 1000
    
    print(f"\n  Encoder avg time: {avg_encoder_ms:.2f} ms")
    print(f"  Policy avg time: {avg_policy_ms:.2f} ms")
    print(f"  Total avg: {avg_encoder_ms + avg_policy_ms:.2f} ms ({1000/(avg_encoder_ms + avg_policy_ms):.1f} FPS)")
    
    print("✓ Full integration test passed")


def test_end_to_end_with_images():
    """测试真实图像端到端流程"""
    print("\n" + "=" * 60)
    print("Test 4: End-to-End with Real Images")
    print("=" * 60)
    
    # 使用测试视频/图像
    test_image_path = Path(__file__).parent / "test_stair.jpg"
    
    if not test_image_path.exists():
        print(f"  [Skip] Test image not found: {test_image_path}")
        print("  Creating synthetic stair-like image instead...")
        # 创建一个合成的"台阶"图像
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        for i in range(5):
            y_start = 80 + i * 80
            image[y_start:y_start+80, :, :] = 100 + i * 30
        image = torch.from_numpy(image)
    else:
        from PIL import Image
        image = Image.open(test_image_path).convert('RGB')
        image = torch.from_numpy(np.array(image))
    
    # 创建 encoder
    encoder = DepthAnythingEncoder(
        encoder='vits',
        freeze_encoder=True,
        use_projection=False,
        verbose=False
    )
    
    # 扩展为 batch
    images = image.unsqueeze(0).repeat(4, 1, 1, 1)
    
    # 提取特征
    with torch.no_grad():
        features = encoder(images)
        depth = encoder.infer_depth(images)
    
    print(f"  Image shape: {image.shape}")
    print(f"  Features shape: {features.shape}")
    print(f"  Depth shape: {depth.shape}")
    print(f"  Depth range: [{depth.min():.2f}, {depth.max():.2f}] meters")
    
    print("✓ End-to-end test passed")


def test_different_feature_types():
    """测试不同的特征提取方式"""
    print("\n" + "=" * 60)
    print("Test 5: Different Feature Types")
    print("=" * 60)
    
    images = torch.randint(0, 256, (4, 480, 640, 3), dtype=torch.uint8)
    
    for feature_type in ['cls', 'avg_pool', 'concat']:
        encoder = DepthAnythingEncoder(
            encoder='vits',
            freeze_encoder=True,
            use_projection=False,
            feature_type=feature_type,
            verbose=False
        )
        
        with torch.no_grad():
            features = encoder(images)
        
        print(f"  {feature_type:10s}: output_dim={encoder.get_output_dim()}, shape={features.shape}")
    
    print("✓ Feature types test passed")


def benchmark_performance():
    """性能基准测试"""
    print("\n" + "=" * 60)
    print("Benchmark: Performance Analysis")
    print("=" * 60)
    
    encoder = DepthAnythingEncoder(
        encoder='vits',
        freeze_encoder=True,
        use_projection=False,
        verbose=False
    )
    
    batch_sizes = [1, 4, 16, 64]
    
    print(f"\n{'Batch Size':^12} | {'Time (ms)':^12} | {'FPS':^12}")
    print("-" * 42)
    
    for bs in batch_sizes:
        images = torch.randint(0, 256, (bs, 480, 640, 3), dtype=torch.uint8, device='cuda')
        
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = encoder(images)
        torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(20):
            t0 = time.time()
            with torch.no_grad():
                _ = encoder(images)
            torch.cuda.synchronize()
            times.append(time.time() - t0)
        
        avg_ms = np.mean(times) * 1000
        fps = bs / np.mean(times)
        
        print(f"{bs:^12} | {avg_ms:^12.2f} | {fps:^12.1f}")
    
    print("\n✓ Benchmark completed")


def test_vision_ppo_storage():
    """测试 VisionPPO 和 RolloutStorageVision 集成"""
    print("\n" + "=" * 60)
    print("Test 6: VisionPPO + RolloutStorageVision Integration")
    print("=" * 60)
    
    # 配置
    num_envs = 8
    num_steps = 10
    num_obs = 45
    num_critic_obs = 55
    history_len = 5
    history_dim = 45
    num_actions = 12
    vision_feature_dim = 384
    
    # 1. 创建 Policy
    print("\n[1] Creating ActorCriticVision...")
    policy = ActorCriticVision(
        num_actor_obs=num_obs,
        num_critic_obs=num_critic_obs,
        num_actions=num_actions,
        history_dim=history_len * history_dim,
        his_encoder_dims=[128, 64],
        his_latent_dim=32,
        vision_feature_dim=vision_feature_dim,
        vision_latent_dim=64,
        use_vision_projection=True,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[128, 128],
    ).to('cuda')
    
    # 2. 创建 VisionPPO
    print("\n[2] Creating VisionPPO...")
    alg = VisionPPO(
        policy=policy,
        num_learning_epochs=2,
        num_mini_batches=2,
        learning_rate=1e-4,
        device='cuda'
    )
    
    # 3. 初始化 Storage
    print("\n[3] Initializing RolloutStorageVision...")
    alg.init_storage(
        num_envs=num_envs,
        num_transitions_per_env=num_steps,
        actor_obs_shape=[num_obs],
        critic_obs_shape=[num_critic_obs],
        action_shape=[num_actions],
        history_len=history_len,
        history_dim=history_dim,
        vision_feature_dim=vision_feature_dim
    )
    
    # 4. 模拟 rollout
    print("\n[4] Simulating rollout collection...")
    for step in range(num_steps):
        obs = torch.randn(num_envs, num_obs, device='cuda')
        critic_obs = torch.randn(num_envs, num_critic_obs, device='cuda')
        history = torch.randn(num_envs, history_len, history_dim, device='cuda')
        vision_features = torch.randn(num_envs, vision_feature_dim, device='cuda')
        
        # Act
        actions = alg.act(obs, critic_obs, history, vision_features)
        
        # Simulate env step
        rewards = torch.randn(num_envs, device='cuda')
        dones = torch.zeros(num_envs, dtype=torch.bool, device='cuda')
        if step == 5:
            dones[:2] = True
        infos = {}
        
        # Process step
        alg.process_env_step(rewards, dones, infos)
    
    print(f"  Storage step: {alg.storage.step}")
    
    # 5. 计算 returns
    print("\n[5] Computing returns...")
    last_critic_obs = torch.randn(num_envs, num_critic_obs, device='cuda')
    last_history = torch.randn(num_envs, history_len, history_dim, device='cuda')
    alg.compute_returns(last_critic_obs, last_history)
    
    # 6. 更新 policy
    print("\n[6] Updating policy...")
    loss_dict = alg.update()
    
    print(f"  Loss dict: {loss_dict}")
    print(f"  Value loss: {loss_dict['value_function']:.4f}")
    print(f"  Surrogate loss: {loss_dict['surrogate']:.4f}")
    
    print("✓ VisionPPO + Storage test passed")


def main():
    print("=" * 60)
    print(" Depth Anything Encoder Integration Test Suite")
    print("=" * 60)
    
    test_encoder_basic()
    test_feature_manager()
    test_integration_with_actor_critic()
    test_end_to_end_with_images()
    test_different_feature_types()
    test_vision_ppo_storage()
    benchmark_performance()
    
    print("\n" + "=" * 60)
    print(" All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
