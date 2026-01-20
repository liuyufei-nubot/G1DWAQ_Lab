# 贡献指南

首先，感谢您对本项目的关注和支持！本文档将指导您如何为 G1 DWAQ 盲走上台阶项目做出贡献。

## 目录

- [行为准则](#行为准则)
- [贡献方式](#贡献方式)
- [报告Bug](#报告bug)
- [功能请求](#功能请求)
- [提交代码](#提交代码)
- [代码风格指南](#代码风格指南)
- [提交信息指南](#提交信息指南)
- [许可证](#许可证)

## 行为准则

本项目采用《贡献者公约》(Contributor Covenant)。参与本项目，即表示您同意遵守本准则。

请友善对待所有贡献者。我们致力于营造包容、尊重和安全的社区环境。

## 贡献方式

### 简单的贡献方式

1. **改进文档** - 修复错别字、补充说明、改进示例
2. **报告Bug** - 详细描述问题和复现步骤
3. **提出想法** - 在Discussions中分享您的想法和建议
4. **分享使用经验** - 提交使用案例或改进建议

### 代码贡献

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 报告Bug

### 提交Bug报告前

请先检查[Issue列表](../../issues)，确保问题未被报告过。

### Bug报告应包含

1. **清晰的标题和描述**
2. **详细的复现步骤**
   ```
   1. 运行命令...
   2. 查看输出...
   3. 发现问题...
   ```
3. **预期行为 vs 实际行为**
4. **环境信息**
   - 操作系统和版本
   - Python版本
   - Isaac Lab版本
   - PyTorch版本
   - CUDA版本（如适用）
5. **错误日志或截图**
6. **额外上下文**

## 功能请求

### 提交功能请求前

先在[Discussions](../../discussions)中与维护者讨论您的想法。

### 功能请求应包含

1. **清晰的用例描述** - 这个功能解决什么问题？
2. **预期行为** - 功能应该如何工作？
3. **可能的替代方案** - 是否有其他方式？
4. **额外上下文** - 相关的论文、参考或资源

## 提交代码

### 开发环境设置

```bash
# 克隆仓库
git clone https://github.com/your-username/G1DWAQ_Lab.git
cd G1DWAQ_Lab

# 创建虚拟环境
conda create -n dwaq python=3.10
conda activate dwaq

# 安装开发依赖
pip install -e TienKung-Lab[dev]
pip install -e unitree_sdk2_python
pip install black isort flake8 pytest

# 安装预提交钩子
pre-commit install
```

### 代码风格指南

本项目遵循以下代码风格标准：

#### Python代码规范

1. **PEP 8** - 遵循Python官方代码风格指南
2. **类型注解** - 为函数添加类型提示（推荐）
3. **文档字符串** - 使用Google风格的docstring

#### 格式化工具

```bash
# 自动格式化代码
black . --line-length 100

# 排序导入
isort .

# 检查风格
flake8 . --max-line-length 100 --extend-ignore E203,W503
```

#### 代码示例

```python
"""示例：规范的Python代码"""

from typing import List, Dict, Optional
import numpy as np
import torch


def process_observations(
    obs: np.ndarray,
    history_length: int = 1,
    normalize: bool = True
) -> torch.Tensor:
    """处理观测数据。
    
    Args:
        obs: 原始观测数组，shape为(batch_size, obs_dim)
        history_length: 历史长度
        normalize: 是否归一化
        
    Returns:
        处理后的观测张量
        
    Example:
        >>> obs = np.random.randn(4096, 96)
        >>> processed = process_observations(obs, history_length=1)
        >>> print(processed.shape)
        torch.Size([4096, 96])
    """
    # 实现代码...
    pass


class DWAQEnvironment:
    """DWAQ环境定义。
    
    Attributes:
        num_envs: 并行环境数量
        obs_dim: 观测维度
    """
    
    def __init__(self, num_envs: int, obs_dim: int = 96):
        """初始化环境。
        
        Args:
            num_envs: 并行环境数量
            obs_dim: 观测维度
        """
        self.num_envs = num_envs
        self.obs_dim = obs_dim
```

#### 版权声明

所有新文件必须在文件头部添加版权声明：

```python
# Copyright (c) 2026, The G1 DWAQ Project Developers.
# All rights reserved.
# Licensed under the BSD-3-Clause License.

"""模块简短描述。

详细描述...
"""
```

## 提交信息指南

提交信息应清晰简洁，使用英文或中文：

### 格式

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Type类型

- **feat**: 新功能
- **fix**: Bug修复
- **docs**: 文档更改
- **style**: 代码风格修改（不影响功能）
- **refactor**: 代码重构
- **perf**: 性能优化
- **test**: 测试更改
- **chore**: 工具或配置更改

### 示例

```
feat(training): add phase-based gait observation

Add support for phase-based gait observation in DWAQ environment.
This improves turning capability by 42% compared to phase-free version.

- Add gait phase computation in observation space
- Create separate configuration for phase-based version
- Update export script to handle 100-dim observations

Closes #123
Refs #456
```

## Pull Request指南

### 开启PR前的检查清单

- [ ] 代码遵循风格指南（`black . && isort . && flake8 .`）
- [ ] 添加了必要的测试
- [ ] 更新了相关文档
- [ ] 提交信息清晰
- [ ] 没有增加不必要的依赖
- [ ] 在本地成功测试

### PR描述模板

```markdown
## 描述
简要描述您的更改。

## 相关Issue
Closes #（Issue号）

## 更改类型
- [ ] Bug修复
- [ ] 新功能
- [ ] 文档更新
- [ ] 代码重构
- [ ] 其他

## 测试说明
- 描述您如何测试这些更改
- 提供测试命令示例

## 性能影响
- [ ] 无影响
- [ ] 性能改进：...
- [ ] 性能下降：... （需说明原因）

## 检查清单
- [ ] 代码自审
- [ ] 本地测试通过
- [ ] 添加了测试
- [ ] 更新了文档
- [ ] 无新的warnings
```

## 代码审查流程

1. 至少需要1个维护者的批准
2. 所有CI检查必须通过
3. 没有冲突的更改
4. 如有意见，请进行讨论并更新代码

## 测试指南

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_environment.py

# 生成覆盖率报告
pytest --cov=legged_lab tests/
```

### 编写测试

```python
"""测试模块示例"""

import pytest
import numpy as np
from legged_lab.envs import DWAQEnvironment


class TestDWAQEnvironment:
    """DWAQ环境测试类"""
    
    @pytest.fixture
    def env(self):
        """环境fixture"""
        return DWAQEnvironment(num_envs=4, obs_dim=96)
    
    def test_initialization(self, env):
        """测试环境初始化"""
        assert env.num_envs == 4
        assert env.obs_dim == 96
    
    def test_reset(self, env):
        """测试环境重置"""
        obs = env.reset()
        assert obs.shape == (4, 96)
    
    def test_step(self, env):
        """测试环境步进"""
        env.reset()
        action = np.random.randn(4, 29)
        obs, reward, done, info = env.step(action)
        assert obs.shape == (4, 96)
        assert reward.shape == (4,)
```

## 文档贡献

### 改进README

- 修复错别字和语法
- 补充缺失的说明
- 改进示例代码
- 更新过时的信息

### 编写新文档

新文档应包含：
1. 清晰的标题
2. 目录（如需要）
3. 代码示例
4. 最后的联系方式

## 许可证

通过贡献代码，您同意将您的代码在BSD-3-Clause License下发布。

## 问题反馈

- 🐛 **Bug报告**: 使用Issue，带上详细复现步骤
- 💡 **功能建议**: 在Discussions中讨论
- 📚 **文档问题**: 直接提交PR
- ❓ **使用问题**: 在Discussions或Issue中提问

## 额外资源

- [项目主README](./README.md)
- [许可证](./LICENSE)
- [项目评审报告](./PROJECT_REVIEW.md)
- [Isaac Lab文档](https://isaac-sim.github.io/IsaacLab/)
- [Legged Lab项目](https://github.com/Hellod035/LeggedLab)

---

**感谢您的贡献！** 🎉

如有任何问题，欢迎在Issues或Discussions中提出。
