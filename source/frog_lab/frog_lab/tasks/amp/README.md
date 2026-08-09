# frog_lab AMP 迁移方案

## 1. 目标

把 `frog_lab` 里的 AMP 体系补齐，并把 AMP 的核心实现统一整理到 `frog_rl` 里。

先基于现有的 `motion_data/amp/g1` 数据，完成一个可训练、可扩展、可继续迭代的 G1 AMP 任务。

当前优先级：

1. 先做 `G1 29DOF`
2. 先做 locomotion AMP
3. 先保证训练链路和数据链路完整
4. 后续再考虑 23DOF、recovery、parkour 等扩展

本方案只基于本地代码阅读得出，不依赖 IsaacSim 运行验证。
本方案的重点是：**先整理 `frog_rl` 里的 AMP，再让后续所有 AMP 都基于这套统一结构继续扩展**。

## 2. 我看完两个参考项目后的结论

### `AMP_mjlab`

这个项目更适合作为主迁移蓝本，因为它已经把 AMP 作为一个完整任务链路做出来了：

- 环境里直接接 motion loader、reset 逻辑、AMP 观测、AMP 奖励
- 训练时直接挂 discriminator + replay buffer + AMP PPO
- 数据流比较清楚，和 Isaac Lab 风格也更接近
- 已经有 G1 的任务组织、配置拆分、ONNX 导出

它的关键文件大致对应：

- `src/tasks/amp_loco/amp_env_cfg.py`
- `src/tasks/amp_loco/ampmotion_loader.py`
- `src/tasks/amp_loco/mdp/*`
- `rsl_rl/algorithms/amp_ppo.py`
- `rsl_rl/runners/amp_on_policy_runner.py`

### `instinct_rl`

这个项目更适合作为算法参考，而不是直接整套迁移：

- AMP 被做成了一个更通用的算法插件
- discriminator、storage、runner 拆得比较细
- 能看出它适合扩展到更复杂的 imitation / WASABI / two-stage 流程

但它和 parkour、dataset、runner 的耦合更重，不适合直接照搬到当前仓库。

## 3. 对 `frog_rl` 的判断

当前 `frog_rl` 里已经有 AMP 相关代码，但结构还混着两代写法：

- 一部分是从 `AMP_mjlab` 思路来的
- 一部分是从 `instinct_rl` 思路来的
- 代码风格还偏旧，和现在 `rsl_rl 5.0` 的组织方式不完全一致

所以现在最重要的不是再加一套新实现，而是：

- 把现有 AMP 代码先梳理干净
- 让 AMP 的算法、runner、storage、exporter 和当前 `rsl_rl 5.0` 的结构对齐
- 后续新增 AMP 都直接基于 `frog_rl` 的统一实现

## 4. 对 `frog_lab` 当前仓库的判断

当前仓库已经有这些基础：

- `source/frog_lab/frog_lab/tasks/locomotion`
- `source/frog_lab/frog_lab/tasks/beyond_mimic/tracking`
- 任务注册机制已经通了
- `RslRlOnPolicyRunnerCfg` 风格已经在用

这意味着 AMP 最稳妥的做法不是重写一套框架，而是：

- 在 `frog_rl` 里整理 AMP 的算法层和训练层
- 在 `frog_lab` 里保留任务配置、环境配置、motion 数据入口
- 让任务侧调用 `frog_rl` 的统一 AMP 实现

## 5. 建议的迁移路线

### Phase 1: 最小可用闭环

先补一个只覆盖 locomotion 的 AMP 任务，并让它走通 `frog_rl` 里的 AMP 训练链路。

这一阶段只做：

- motion 数据加载
- actor / critic / amp 观测
- AMP discriminator 输入组织
- AMP reward
- 任务注册
- 训练配置
- `frog_rl` 内 AMP 训练闭环整理

### Phase 2: 数据转换链路

把 `motion_data/amp/g1/*.csv` 转成训练侧可直接读的 `npz`。

建议保留分层：

- 原始数据：`motion_data/amp/g1/*.csv`
- 训练资产：`source/frog_lab/frog_lab/assets/motions/g1/amp/...`

转换时要统一：

- 四元数顺序
- 关节顺序
- FPS
- 速度计算方式
- 数组 shape

### Phase 3: 扩展能力

等 v1 能训练后，再补：

- recovery 数据
- delayed reset / delayed termination
- 更复杂的 domain randomization
- 23DOF 版本

## 6. 建议的模块职责

### `mdp`

放 AMP 任务专用的观测、奖励、事件、终止。

建议先包含：

- `observations.py`
- `rewards.py`
- `events.py`
- `terminations.py`
- `utils.py`

### `config/g1`

放 G1 的任务注册和环境配置。

建议至少拆成：

- rough / flat 两个配置
- agent cfg
- `__init__.py` 做 task registration

### `utils`

放 motion loader、数据拼接、缓存、转换辅助函数。

G1 AMP v1 的 motion loader 放在：

- `source/frog_lab/frog_lab/tasks/amp/utils/motion_loader.py`

它是任务侧工具，不放进 `frog_rl.datasets`。

### `frog_rl`

放 AMP 的通用训练实现，后续新 AMP 都从这里复用：

- `algorithms/amp_ppo.py`
- `algorithms/amp_discriminator.py`
- `runners/` 里的 AMP 专用 runner
- `storage/` 里的 AMP replay / rollout 支持

目标是把这些东西整理成统一、清晰、符合当前版本风格的结构。
`frog_rl` 只通过配置解析 motion loader，不直接绑定 G1 数据格式。

## 7. 我建议的 AMP 观测与奖励设计

### 观测

建议三组：

- `actor`
- `critic`
- `amp`

其中 `amp` 只放判别器需要的状态特征，优先考虑：

- body position
- body orientation
- body linear velocity
- body angular velocity

### 奖励

先做最小闭环：

- velocity tracking
- posture / balance stabilization
- action rate penalty
- joint / torque penalty
- discriminator reward

先别把奖励做得太花，不然很难判断问题到底出在数据、判别器还是策略上。

## 8. 参考项目里的可迁移点

### 优先迁移自 `AMP_mjlab`

- motion loader 的数据读取方式
- AMP task cfg 的组织方式
- delayed reset 思路
- AMP PPO 的 discriminator 更新流程
- ONNX 导出和 metadata 附加

### 参考 `instinct_rl` 但不直接照搬

- discriminator 结构拆分
- replay buffer / amp storage 的组织
- 更通用的算法插件接口

## 9. 当前仓库里已有的落点

我阅读后认为，下面这些现有模块最适合接 AMP：

- `source/frog_lab/frog_lab/tasks/beyond_mimic/tracking`
  - 可以借鉴它的 task registration 和 runner 扩展方式
- `source/frog_lab/frog_lab/tasks/locomotion`
  - 可以借鉴它的 G1 29DOF / 23DOF 配置分层
- `motion_data/amp/g1`
  - 作为 AMP v1 唯一数据源

## 10. 风险点

- 当前只有 locomotion 数据，没有 recovery 数据
- 不能启动 IsaacSim，所以只能先做静态链路设计和 Python 侧校验
- `amp` 任务如果一开始就和 parkour 混在一起，后续会很难拆
- 关节顺序、四元数顺序、body anchor 定义一旦错了，后面 reward 和 discriminator 都会偏

## 11. 建议的实现顺序

1. 整理 `frog_rl` 里的 AMP 算法、runner、storage 结构
2. 定义 `amp` 任务目录结构
3. 写 motion 数据转换脚本
4. 写 motion loader
5. 写 AMP 观测和奖励
6. 写 G1 任务配置与注册
7. 写 AMP 训练闭环
8. 补静态校验
9. 再考虑 recovery 和 parkour

## 12. 当前已实现的 G1 29DOF locomotion AMP 闭环

当前已经按 `AMP_mjlab` 的传统 AMP 思路搭好第一版 G1 29DOF locomotion AMP：

- `frog_rl` 里保留 AMP 通用算法实现：
  - `source/frog_rl/algorithms/amp_ppo.py`
  - `source/frog_rl/algorithms/amp_discriminator.py`
  - `source/frog_rl/storage/replay_buffer.py`
- `frog_lab` 里新增 AMP 任务侧配置：
  - `source/frog_lab/frog_lab/tasks/amp/config/g1`
  - `source/frog_lab/frog_lab/tasks/amp/mdp/observations.py`
  - `source/frog_lab/frog_lab/tasks/amp/utils/motion_loader.py`
- 新增 G1 29DOF AMP 任务注册：
  - `FrogLab-Isaac-AMP-Rough-Unitree-G1-29DOF-v0`
  - `FrogLab-Isaac-AMP-Flat-Unitree-G1-29DOF-v0`
- 新增训练和回放入口：
  - `scripts/amp/frog_rl/train.py`
  - `scripts/amp/frog_rl/play.py`

AMP 判别器输入先和 `AMP_mjlab` 保持一致，只使用 body state，不加入 joint state。

当前 body state 由下面四类特征拼接：

```text
body_pos_b + body_ori_b + body_lin_vel_b + body_ang_vel_b
```

其中：

- `body_pos_b` 是目标 body 在 `torso_link` anchor frame 下的位置。
- `body_ori_b` 是目标 body 在 `torso_link` anchor frame 下的姿态，使用旋转矩阵前两列表示。
- `body_lin_vel_b` 是每个目标 body 在自身 body frame 下的线速度。
- `body_ang_vel_b` 是每个目标 body 在自身 body frame 下的角速度。

G1 29DOF AMP 的 NPZ body 轴顺序固定为：

```text
pelvis
left_hip_roll_link
left_knee_link
left_ankle_roll_link
right_hip_roll_link
right_knee_link
right_ankle_roll_link
torso_link
left_shoulder_roll_link
left_elbow_link
left_wrist_yaw_link
right_shoulder_roll_link
right_elbow_link
right_wrist_yaw_link
```

其中 `torso_link` 只作为 anchor，不送入 discriminator body subset。

## 13. CSV 到 NPZ 转换链路

原始数据位置：

```text
motion_data/amp/g1/*.csv
```

训练资产输出位置：

```text
source/frog_lab/frog_lab/assets/motions/g1/amp/locomotion
```

转换脚本：

```bash
python3 scripts/amp/convert_g1_csv_to_npz.py --overwrite
```

转换脚本做的事情：

- 按 `120 FPS` 读取原始 CSV。
- 将 CSV 的 `root_quat_xyzw` 转换为内部 FK 使用的 `xyzw`。
- 使用 `model/g1/urdf/g1_29dof_rev_1_0.urdf` 做离线 forward kinematics。
- 将 motion 重采样到 `50 FPS`。
- 导出 AMP loader 需要的 full body state NPZ。
- 输出四元数统一保存为 `wxyz`。

另有一个参考 `beyond_mimic` 方式新增的 IsaacLab replay 转换脚本：

```bash
python3 scripts/amp/convert_g1_csv_to_npz_replay.py --overwrite --headless
```

这个脚本会启动 IsaacLab，将 CSV 插值后的 root state 和 joint state 写入 G1 articulation，
调用 `sim.render()` / `scene.update()` 后从 `robot.data` 读取 body state 并保存 NPZ。

它和离线 FK 脚本的区别：

- `convert_g1_csv_to_npz.py` 不启动仿真，使用 URDF 离线 FK。
- `convert_g1_csv_to_npz_replay.py` 启动 IsaacLab，参考 `beyond_mimic` 的 replay-based 转换方式。
- replay 脚本只导出 AMP 当前使用的 14 个 body，顺序仍然是 `G1_29DOF_AMP_ALL_BODY_NAMES`。
- 当前机器配置限制下，不在本阶段启动 IsaacSim 验证该脚本。

每个 NPZ 至少包含：

```text
fps
body_names
joint_names
root_pos_w
root_quat_w
joint_pos
joint_vel
body_pos_w
body_quat_w
body_lin_vel_w
body_ang_vel_w
source_file
input_fps
```

当前已经从 `motion_data/amp/g1` 生成 17 个 locomotion NPZ 文件。

## 14. 待你确认的点

下面几个点会直接影响实现方式，建议你后续在这里继续改：

- `frog_rl` 的 AMP 算法是否保留现有类名，还是按 `rsl_rl 5.0` 风格重新命名和拆分
- motion 数据只用现有 locomotion CSV，还是同时规划 recovery 数据格式
- AMP 判别器只看 body state，还是把 joint state 也放进去
- `amp` 任务是否需要和 `beyond_mimic` 共用一些 exporter / wrapper

已确认的当前实现选择：

- 先做 `G1 29DOF`。
- 先做 locomotion AMP。
- AMP 和 WASABI 按两种不同算法处理。
- AMP 判别器先和 `AMP_mjlab` 一样，只看 body state。
- AMP 任务不和 `beyond_mimic` 共用 exporter / wrapper。
- motion loader 放在 `frog_lab.tasks.amp.utils`，不放进 `frog_rl.datasets`。

## 15. `frog_rl.datasets` 删除结论

`source/frog_rl/datasets` 已删除。G1 AMP v1 不再把 motion loader 放进 `frog_rl`，而是放在任务侧。

原因：

- 旧 `motion_loader.py` 读取的是 AMP JSON 格式，不是当前 `motion_data/amp/g1/*.csv` 或转换后的 G1 NPZ。
- 旧数据结构是四足机器人风格的固定 61 维：
  - `root_pos`
  - `root_rot`
  - `joint_pos(12)`
  - `foot_pos_local(12)`
  - `lin_vel`
  - `ang_vel`
  - `joint_vel(12)`
  - `foot_vel_local(12)`
- 旧 loader 包含 PyBullet 到 Isaac 的四足腿顺序重排逻辑，不适合 G1 29DOF。
- 旧 AMP observation 不是 `AMP_mjlab` 当前 G1 版本使用的 body-state observation。

和 `AMP_mjlab` 对齐后，G1 AMP 的 motion loader 已放到任务侧：

- `source/frog_lab/frog_lab/tasks/amp/utils/motion_loader.py`

它读取 G1 NPZ，并输出 transition-based body state：

- `body_pos_b`
- `body_ori_b`
- `body_lin_vel_b`
- `body_ang_vel_b`

其中每个 state 的维度应为：

```text
(3 + 6 + 3 + 3) * num_amp_bodies
```

如果后续需要通用四元数工具，应放到 `frog_rl/utils` 或任务侧 `utils`，不再恢复 `frog_rl.datasets` 作为数据入口。

---

这份文档的目的不是一次写死，而是先把迁移路径钉住，后面可以直接在这里改决策。
