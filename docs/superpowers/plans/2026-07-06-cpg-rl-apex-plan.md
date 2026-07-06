# APEX 上实施多步态 CPG-RL 的路线计划

日期：2026-07-06

## 1. 结论

APEX 可以作为我们后续实现“CPG 生成多种步态 + 强化学习残差控制”的主工程底座，但不建议一上来直接训练复杂结构化 CPG-RL。

更准确的判断是：

1. 可以直接开始做“数据体检 + gait feature dashboard”。
2. 可以较快开始做“CPG-only baseline”。
3. CPG+RL 需要先补一个动作中间层，不能只改奖励就算完成。

APEX 当前最适合的切入方式是：

```text
motion/imitation data
        |
        v
gait feature analysis
        |
        v
CPG base joint target q_cpg
        |
        + RL residual action delta_q
        |
        v
final target q_des = q_cpg + residual_scale * delta_q
        |
        v
existing Go2 PD / APEX control
```

也就是说，第一版不要立刻让 RL 输出 `Delta f, Delta A, Delta p_foot` 这种全结构化参数，而是先让 CPG 生成基本节律，让 RL 保持 12 维关节残差。这样改动最小，也最容易验证。

## 2. 当前代码依据

APEX 已经具备几个有利条件：

1. Go2 任务已经存在：`legged_gym/envs/__init__.py` 注册了 `go2_flat`。
2. Go2 控制链路已经存在：`legged_gym/envs/go2/go2.py` 中 `_compute_torques()` 使用 12 维动作生成关节目标或力矩。
3. 当前默认控制是 `apex_position`：`legged_gym/envs/param_config.yaml` 中 `control_type: "apex_position"`。
4. 当前 action 仍然是 12 维关节动作：基础配置 `legged_gym/envs/base/legged_robot_config.py` 中 `num_actions = 12`。
5. imitation/action prior 逻辑已经存在：`apex_position` 会叠加 imitation joint prior。
6. 已有 gait phase 相关代码：`legged_gym/envs/base/legged_robot.py` 中 `_step_contact_targets()` 维护 `gait_indices`、`foot_indices`、`clock_inputs`。
7. 已有足端接触和抬脚奖励：`legged_gym/envs/go2/go2.py` 中有 `_reward_tracking_contacts_shaped_force()`、`_reward_tracking_contacts_shaped_vel()`、`_reward_feet_clearance_cmd_linear()`。
8. 已有 Kine2Go 转换工具：`tools/kine2go/prepare_kine2go_for_apex.py`，并且显式处理了 Kine2Go 与 APEX 的腿顺序映射。
9. 已有可用参考数据：`imitation_data/kine2go/*.csv` 和 `imitation_data/animal_mocap/*.csv`。

主要缺口：

1. 目前没有真正的 CPG 动作生成器。
2. 目前 phase 更多是 reward/clock/接触目标，不是动作先验。
3. 目前还没有基于 gait family 的运动特征体检报告。
4. 目前还没有 CPG 作为动作先验接入 Go2 控制链路。
5. 需要 Isaac Gym + NVIDIA GPU 环境才能真正验证训练。

## 3. 阶段 0：先确认 APEX baseline 能跑

目标：确认不是环境安装问题、数据路径问题或 Isaac Gym 问题。

建议命令：

```powershell
cd D:\Desktop\robot\APEX
python legged_gym\tests\test_env.py --task=go2_flat --headless
```

如果测试环境可用，再跑一个短训练：

```powershell
python legged_gym\scripts\train.py --task=go2_flat --headless --num_envs=128
```

验收标准：

1. `go2_flat` 能成功创建环境。
2. `test_env.py` 可以完成若干步仿真。
3. imitation CSV 能被正确读取。
4. 没有 `isaacgym`、CUDA、URDF、CSV 路径相关报错。

风险：

1. 当前 Windows 环境通常不能直接跑 Isaac Gym 训练，需要 Linux/WSL 或服务器。
2. `param_config.yaml` 当前可能指向单一 imitation CSV，后续训练不同 gait 时需要显式管理数据源和 gait 参数。

## 4. 阶段 1：步态特征和数据体检

目标：先不引入情绪概念，只量化 walk / trot / pace / canter / run 等 gait family 的运动差异。

新增建议文件：

```text
tools/gait_analysis/compute_gait_features.py
tools/gait_analysis/plot_gait_features.py
analysis/gait_features/
```

输入数据：

```text
imitation_data/kine2go/ai4_dog_walk_00.csv
imitation_data/kine2go/ai4_dog_trot_00.csv
imitation_data/kine2go/ai4_dog_pace.csv
imitation_data/animal_mocap/go2_retarget_trot.csv
imitation_data/animal_mocap/go2_retarget_pace.csv
imitation_data/animal_mocap/go2_retarget_canter_2ms.csv
```

需要提取的指标：

1. 速度：`vx, vy, wz` 的均值、方差、分位数。
2. 步频：根据足端高度峰值或 contact pattern 估计。
3. foot clearance：每条腿足端 z 的峰值、均值、摆动期最大值。
4. body height：`height` 的均值、最小值、最大值。
5. body bounce：`height` 的标准差、主频。
6. pitch / roll：从四元数估计姿态变化。
7. contact pattern：用足端 z 和速度近似推断触地/摆动。
8. foot slip：触地期足端水平速度。
9. joint limit violation：检查 12 个关节是否接近 Go2 URDF 限位。

输出：

```text
analysis/gait_features/summary.csv
analysis/gait_features/per_motion_features.csv
analysis/gait_features/contact_patterns/*.png
analysis/gait_features/height_velocity_plots/*.png
analysis/gait_features/00_gait_feature_report.md
```

验收标准：

1. 每个 motion 都有一行统计指标。
2. 能用数据比较不同 gait family 的速度、步频、foot clearance、body bounce 和 contact pattern。
3. 产出 CPG 参数初值建议，例如：
   - walk: 较低频率、较小幅度、四拍相位。
   - trot: 对角腿同相、较高频率。
   - pace: 同侧腿同相。
   - canter/run: 更高速度和更大动态起伏。

## 5. 阶段 2：CPG-only baseline

目标：先不训练 RL，只验证 CPG 本身能生成稳定、可解释、可切换的节律。

新增建议文件：

```text
legged_gym/envs/go2/cpg.py
tools/cpg/preview_go2_cpg.py
```

第一版建议做 joint-space CPG，不立刻做完整足端 IK：

```python
q_cpg = default_dof_pos + amplitude * sin(phase + phase_offset)
```

支持的参数：

1. `frequency`
2. `phase_offsets`
3. `hip_amplitude`
4. `thigh_amplitude`
5. `calf_amplitude`
6. `duty_factor`
7. `gait_family`

支持的 gait：

1. walk
2. trot
3. pace
4. bound
5. canter

验收标准：

1. CPG 输出 shape 为 `[num_envs, 12]`。
2. 输出关节角不超过 Go2 joint limits。
3. 同一 gait 下 phase 连续，不发生跳变。
4. 不同 gait 的相位关系符合四足运动学常识，例如 trot 为对角腿同相，pace 为同侧腿同相。

## 6. 阶段 3：接入 Go2 控制链路

目标：新增一个最小侵入的控制模式，让 APEX 现有 policy 可以在 CPG 上做残差。

当前实现状态：

1. 已新增 `legged_gym/envs/go2/cpg.py`，提供 joint-space CPG 和 residual 组合函数。
2. 已在 `param_config.yaml` 中加入 `cpg_residual_position` 的配置项，但默认仍保持 `apex_position`。
3. 已在 `Go2._compute_torques()` 中加入 `cpg_residual_position` 分支。
4. 已在 `reset_idx()` 中重置对应环境的 CPG phase。
5. 尚未在 IsaacGym 中完成真实 rollout/训练验证。

建议改动：

```text
legged_gym/envs/param_config.yaml
legged_gym/envs/go2/go2_config.py
legged_gym/envs/go2/go2.py
legged_gym/envs/go2/cpg.py
```

新增控制模式：

```yaml
control_type: "cpg_residual_position"
```

核心控制逻辑：

```python
q_cpg = self.cpg.step(
    gait=self.gait_family,
    dt=self.dt,
    commands=self.commands,
)
delta_q = actions * self.cfg.control.action_scale
self.joint_pos_target = q_cpg + residual_scale * delta_q
torques = kp * (self.joint_pos_target - self.dof_pos) - kd * self.dof_vel
```

第一版 action 保持 12 维，这样不用先改 PPO 网络输出维度。

验收标准：

1. `control_type: "position"`、`"apex_position"` 旧逻辑不受影响。
2. `control_type: "cpg_residual_position"` 可以正常创建环境。
3. 零动作时机器人执行 CPG-only。
4. 非零动作时 RL residual 能调节 CPG 关节目标。

本地可验证部分：

```powershell
python -m unittest discover -s tests -p 'test_*.py'
python -m compileall .\legged_gym\envs\go2\cpg.py .\legged_gym\envs\go2\go2.py
```

## 7. 阶段 4：多步态条件输入

目标：让同一个策略能够根据 gait 条件生成不同步态，而不是根据情绪标签生成动作。

建议新增配置：

```yaml
gait_conditioned: true
gait_families: ["walk", "trot", "pace", "bound", "canter"]
```

观测中加入：

```text
gait_id 或 gait_embedding
CPG phase sin/cos
CPG base parameters
```

训练初期建议：

1. 先固定每个 episode 的 gait family。
2. 每种 gait 对应一组 CPG phase offsets、frequency、amplitude 初值。
3. reward 中加入 gait feature/contact pattern matching，而不是只 imitation joint angle。
4. 稳定后再允许策略调整 CPG 频率、幅值和小幅关节残差。

验收标准：

1. 同一速度命令下，不同 gait 的相位关系和 contact pattern 有统计差异。
2. gait 切换不导致明显相位跳变或跌倒。
3. RL residual 不破坏 CPG 的基本周期结构。

## 8. 阶段 5：从关节残差升级为结构化 CPG-RL

目标：当 joint residual 稳定后，再让 RL 输出更可解释的 CPG 参数残差。

第二版 action 可以从 12 维 joint residual 逐步变为：

```text
a_t = [
  Delta frequency,
  Delta amplitude per leg,
  Delta phase / foot placement,
  small Delta q residual
]
```

不要一次性切到全部结构化动作。建议顺序：

1. `Delta q` residual only。
2. `Delta q + Delta frequency`。
3. `Delta q + Delta frequency + Delta amplitude`。
4. 足端落点修正或 IK 版本。

验收标准：

1. 比纯 12 维 residual 更省动作、更平滑。
2. gait 切换更连续。
3. foot slip 和 action rate 不恶化。

## 9. 阶段 6：评估和论文实验

建议比较四组：

1. APEX imitation/action prior baseline。
2. CPG-only。
3. CPG + 12 维 RL residual。
4. CPG + structured RL residual。

核心指标：

1. 速度跟踪误差。
2. fall rate。
3. foot slip。
4. torque/energy。
5. action rate。
6. contact pattern match。
7. body height/bounce。
8. foot clearance。
9. gait family 可分性。
10. gait 切换平滑性。

## 10. 第一周可执行任务

第一周不要改 PPO，不要直接长训。

建议顺序：

1. 跑通或确认无法本地跑通 APEX baseline。
2. 写 `tools/gait_analysis/compute_gait_features.py`。
3. 对 Kine2Go 和 animal_mocap 数据生成 `summary.csv`。
4. 写 `analysis/gait_features/00_gait_feature_report.md`。
5. 根据报告定出 walk/trot/pace/canter 的 CPG 初始参数。
6. 写 `legged_gym/envs/go2/cpg.py` 的纯张量单元测试。
7. 再决定是否接入 `go2.py` 的控制链路。

## 11. 需要用户协助的部分

1. 确认是否有可用的 Linux + NVIDIA GPU + Isaac Gym 环境。
2. 确认研究对象主要是 Unitree Go2，还是未来要迁移到 A1/Go1。
3. 确认第一版 gait 集合：建议先做 walk/trot/pace，再扩展 canter/bound。
4. 如果要做真实机器狗，需要提供 Unitree SDK/部署机器的环境信息。
5. 如果要做用户感知实验，需要后续设计问卷和伦理/数据记录方式。

## 12. 不建议现在做的事

1. 不建议马上把 action dim 改成复杂的 `Delta f, Delta A, Delta p_foot, Delta q`。
2. 不建议只靠奖励函数强行学出 gait，因为相位结构会不稳定、解释性也弱。
3. 不建议一开始就做足端 IK，全 foot-space CPG 容易把问题扩大。
4. 不建议直接上真实机器狗，先在仿真里证明 CPG-only 和 CPG-RL 的差异。

## 13. 最小可行里程碑

最小论文原型可以定义为：

1. APEX/Go2 仿真环境能跑。
2. motion 数据体检证明不同 gait family 的运动特征不同。
3. CPG-only 能生成可解释的 walk/trot/pace/canter 节律。
4. CPG+RL residual 比 CPG-only 更稳，比纯 imitation/action prior 更容易控制步态。
5. 视频或指标能清楚区分不同 gait，并证明切换足够平滑。

达到这个程度，就可以开始写项目立项和初版实验设计。
