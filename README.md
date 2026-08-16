# Frog_lab

基于 Isaac Lab 的人形机器人强化学习训练项目，当前主要面向 Unitree G1 29 自由度与 23 自由度机器人。

项目包含常规速度跟踪任务，以及使用 `frog_rl` 实现的 AMP 和 WASABI 模仿学习训练任务。

## 环境要求

- Isaac Lab 与 Isaac Sim
- Python 3.10 或更高版本
- PyTorch
- 已安装 Isaac Lab 的 Python 环境

本项目不修改 Isaac Lab 源码，应放在 Isaac Lab 目录之外独立开发。

## 安装

在已安装 Isaac Lab 的 Python 环境中执行：

```bash
git clone git@github.com:Green-Fr0g/frog_lab.git
cd frog_lab

python -m pip install -e source/frog_lab
python -m pip install -e source/frog_rl
```

安装完成后，`frog_lab` 提供 Isaac Lab 任务与配置，`frog_rl` 提供自定义训练算法和 runner。

`frog_rl` 使用与 RSL-RL 一致的嵌套包结构：

```text
source/frog_rl/
├── pyproject.toml
├── setup.py
└── frog_rl/
    ├── algorithms/
    ├── runners/
    ├── storage/
    └── utils/
```

因此 `python -m pip install -e source/frog_rl` 会将内部 `frog_rl` 目录安装为可编辑 Python 包；修改算法源码后不需要重新安装。

## 查看任务

```bash
python scripts/list_envs.py
python scripts/list_envs.py --keyword AMP
python scripts/list_envs.py --keyword WASABI
```

## 常规 PPO 训练

常规速度跟踪任务使用原生 RSL-RL 训练入口：

```bash
python scripts/rsl_rl/train.py \
  --task FrogLab-Isaac-Velocity-Rough-Unitree-G1-29DOF-v0
```

现有 `scripts/rsl_rl` 目录保持原生 RSL-RL 行为，不用于 AMP/WASABI 算法。

## AMP 与 WASABI 训练

AMP/WASABI 使用独立的 `frog_rl` 训练入口，避免与原生 RSL-RL runner 混用。

### AMP

```bash
python scripts/frog_rl/train.py \
  --task FrogLab-Isaac-AMP-Flat-Unitree-G1-29DOF-v0
```

AMP 配置位于：

- 环境：`source/frog_lab/frog_lab/tasks/amp/config/g1_29dof/flat_env_cfg.py`
- Agent：`source/frog_lab/frog_lab/tasks/amp/config/g1_29dof/agents/amp_ppo_cfg.py`

AMP 任务使用 `AMPPPO`、运动专家数据和 `amp_state` 观察组训练判别器。

### WASABI

```bash
python scripts/frog_rl/train.py \
  --task FrogLab-Isaac-WASABI-Flat-Unitree-G1-29DOF-v0
```

WASABI 配置位于：

- 环境：`source/frog_lab/frog_lab/tasks/amp/config/g1_29dof/wasabi_flat_env_cfg.py`
- Agent：`source/frog_lab/frog_lab/tasks/amp/config/g1_29dof/agents/wasabi_ppo_cfg.py`

WASABI 使用 `WasabiPPO`，并以 `wasabi_policy` 和 `wasabi_reference` 观察组构造判别器输入。

### 常用参数

```bash
python scripts/frog_rl/train.py \
  --task FrogLab-Isaac-AMP-Flat-Unitree-G1-29DOF-v0 \
  --num_envs 1024 \
  --max_iterations 5000 \
  --seed 42
```

- `--num_envs`：并行环境数量。
- `--max_iterations`：训练迭代次数。
- `--seed`：随机种子。
- `--device`：Isaac Lab 支持的计算设备，例如 `cuda:0`。
- `--resume --load_run <运行目录> --checkpoint <模型文件>`：恢复训练。

训练日志和模型保存至：

```text
logs/frog_rl/<experiment_name>/<时间戳>/
```

## 推理与导出

使用独立的 `frog_rl` 推理脚本加载 AMP/WASABI checkpoint：

```bash
python scripts/frog_rl/play.py \
  --task FrogLab-Isaac-AMP-Flat-Unitree-G1-29DOF-v0 \
  --checkpoint /绝对路径/model_5000.pt \
  --num_envs 1
```

推理脚本会在 checkpoint 同级目录创建 `exported` 文件夹，并导出：

- `policy.pt`：TorchScript 策略
- `policy.onnx`：ONNX 策略

## 运动数据

G1 29 自由度 AMP/WASABI motion 数据位于：

```text
source/frog_lab/frog_lab/tasks/amp/config/g1_29dof/motions/
```

每个 `.npz` 文件应包含 `body_pos_w`、`body_quat_w`、`body_lin_vel_w`、`body_ang_vel_w`、`joint_pos`、`joint_vel` 和 `fps`。当前 G1 数据约定为 30 个刚体和 29 个关节。

## 开发检查

不启动 Isaac Sim 时，可以进行基础语法检查：

```bash
python -m py_compile scripts/frog_rl/train.py scripts/frog_rl/play.py
git diff --check
```

运行完整训练或推理前，需要确认 Isaac Lab、PyTorch、Isaac Sim 和 GPU 环境已正确安装。
