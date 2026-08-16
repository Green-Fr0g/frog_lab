# frog_rl

`frog_rl` 是 Frog Lab 使用的强化学习算法包，基于 PyTorch，包含 PPO、AMP、WASABI、蒸馏、RND 和相关训练组件。

## 安装

在项目根目录执行：

```bash
python -m pip install -e source/frog_rl
```

项目中的 Isaac Lab 环境配置需要另外安装：

```bash
python -m pip install -e source/frog_lab
```

AMP/WASABI 训练入口位于 `scripts/frog_rl/train.py`。
