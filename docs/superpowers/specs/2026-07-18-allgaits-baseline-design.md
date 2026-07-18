# Go2 AllGaits 基线设计

日期：2026-07-18

## 1. 目标

在 APEX 中新增独立的 `go2_allgaits` 研究任务，严格复现 AllGaits 的核心控制范式：

```text
前向速度命令 + 本体感知 + 足底接触 + CPG 状态
                         |
                         v
                  PPO 策略（8 维动作）
                         |
                         v
          每条腿的目标振幅 mu_i 与频率 omega_i
                         |
                         v
       强耦合 CPG -> 足端轨迹 -> 解析 IK -> 关节 PD
```

第一阶段只研究前向运动和用户指定的步态，不加入侧向速度、转向、足端残差、关节残差、自动步态选择、模仿奖励或情绪标签。

## 2. 设计原则

1. 新任务与现有 `go2_flat` 隔离，保留原 APEX imitation 和 multi-critic 行为。
2. 策略动作严格为 8 维，不保留无效的 12 维兼容动作。
3. 策略不观察 gait coupling matrix，也不观察 pattern formation 参数，与论文保持一致。
4. 足底接触布尔量和完整 CPG 状态进入 actor observation。
5. CPG 只生成足端目标；在线控制必须经过唯一的 `foot trajectory -> IK -> joint target` 通路。
6. CPG、环境适配和 PPO 配置分别维护，便于独立单元测试和消融实验。

## 3. 任务边界

### 3.1 第一阶段包含

- Go2 机器人和 Isaac Gym 仿真。
- 前向速度命令 `v_x*`。
- 9 种 AllGaits 步态：walk、amble、trot、pace、bound、pronk、canter、transverse gallop、rotary gallop。
- 用户指定步态；训练时每 3 秒随机切换 gait coupling matrix。
- 每 5 秒重新采样前向速度命令。
- 每个环境 reset 时随机化 body height、ground clearance、ground penetration 和 foot offset。
- 纯 PPO 训练，不使用 APEX imitation prior、AMP discriminator 或 multi-critic。
- CPG-only 和 CPG-RL rollout、训练及指标验证。

### 3.2 第一阶段不包含

- `v_y`、yaw rate 或摇杆转向。
- 自动选择步态的高层网络。
- RL 足端位置残差或关节位置残差。
- 视觉、地形高度图和复杂地形课程。
- Kine2Go 或动物 mocap imitation reward。
- 真机摇杆和 Unitree SDK 接入。

## 4. CPG 数学模型

采用 AllGaits 的 amplitude-controlled phase oscillator，不继续使用当前 `hopf_cpg.py` 中的一阶 Hopf normal form：

```text
ddot(r_i) = a * (a / 4 * (mu_i - r_i) - dot(r_i))

dot(theta_i) = omega_i
             + sum_j r_j * w_ij
               * sin(theta_j - theta_i - phi_ij)
```

其中：

- `r_i` 是第 i 条腿的当前振幅。
- `mu_i` 是策略给出的目标振幅。
- `theta_i` 是当前相位。
- `omega_i` 是策略给出的固有频率。
- `phi_ij` 是目标步态的相位偏置。
- `w_ij` 默认使用论文的强耦合值 `10.0`。
- `a` 是振幅收敛系数，作为配置参数保存。

每个环境维护 4 条腿的 `r`、`r_dot`、`theta` 和 `theta_dot`。reset 必须按照当前 gait 初始化相位，不能固定初始化为 trot。

9 种步态的相位偏置以 AllGaits Figure 3 的腿序 `FR, FL, HR, HL` 为唯一来源。代码中使用显式、不可变的相位表，并用接触时序单元测试防止腿序映射错误。

## 5. 动作空间

PPO actor 输出标准化动作 `a in [-1, 1]^8`：

```text
a = [a_mu_FR, a_mu_FL, a_mu_HR, a_mu_HL,
     a_f_FR,  a_f_FL,  a_f_HR,  a_f_HL]
```

映射为论文范围：

```text
mu_i        = 1.5 + 0.5 * a_mu_i       # [1, 2]
frequency_i = 4.0 + 4.0 * a_f_i        # [0, 8] Hz
omega_i     = 2 * pi * frequency_i     # CPG 内部使用 rad/s
```

动作在 policy rate 更新，并在相邻 policy step 之间保持。CPG 在每个 physics step 积分。

## 6. Pattern Formation 与 IK

对每条腿生成 sagittal 足端轨迹：

```text
x_foot = x_off - d_step * (r_i - 1) * cos(theta_i)

z_foot = -h + gc * sin(theta_i),  sin(theta_i) > 0
z_foot = -h + gp * sin(theta_i),  otherwise
```

`y_foot` 使用各腿相对 hip frame 的固定横向名义位置。第一阶段不学习横向足端偏移。

论文中的 pattern formation 随机化范围为：

| 参数 | 范围 |
|---|---:|
| body height `h` | `[0.18, 0.35] m` |
| foot offset `x_off` | `[-0.08, 0.03] m` |
| swing clearance `gc` | `[0.02, 0.12] m` |
| stance penetration `gp` | `[0.0, 0.015] m` |

这些参数在每个环境 reset 时采样，但不加入 actor observation。`d_step` 保留为显式配置参数。

解析 IK 必须：

- 同时计算 hip ab/ad、thigh 和 calf 三个关节。
- 使用与 Go2 URDF 一致的腿长、hip offset、关节符号和腿顺序。
- 对不可达目标先投影到可达工作空间，再计算关节角。
- 对三角函数输入做数值保护，并记录 IK projection 计数。
- 在输出 PD 前执行 Go2 joint limit clamp。

控制链中不允许存在另一套直接从振荡器状态生成关节偏移的旁路。

## 7. 观测空间

Actor observation 固定为 62 维，顺序如下：

| 顺序 | 内容 | 维数 |
|---:|---|---:|
| 1 | 前向速度命令 `v_x*` | 1 |
| 2 | base linear velocity | 3 |
| 3 | base angular velocity | 3 |
| 4 | projected gravity | 3 |
| 5 | joint position relative to default | 12 |
| 6 | joint velocity | 12 |
| 7 | foot contact booleans | 4 |
| 8 | previous policy action | 8 |
| 9 | CPG state `[r, r_dot, theta, theta_dot]` | 16 |
| | 合计 | 62 |

足底接触从仿真接触力计算：

```python
foot_contacts = (
    contact_forces[:, feet_indices, 2] > 1.0
).to(dtype=torch.float)
```

计算 observation 时不得修改 `last_contacts`，避免影响 airtime reward 的状态。足顺序必须与 CPG 的 `FR, FL, HR, HL` 顺序显式对齐；不能假设 Isaac Gym asset 的 body handle 顺序天然一致。

第一阶段 actor 和 critic 使用相同的 62 维信息。后续引入 asymmetric critic 时另行设计，不在本次范围内。

## 8. 命令与步态切换

- 训练速度范围为 `[0.2, 3.0] m/s`。
- 每 5 秒为各环境重新采样 `v_x*`。
- 每 3 秒为各环境重新采样 9 种 coupling matrix 之一。
- gait 切换只替换目标相位偏置，不重置 `theta`、`r` 或机器人状态，以验证真实连续过渡。
- episode reset 时按采样到的 gait 初始化相位。
- 部署接口允许用户直接设置 `v_x*` 和 `gait_id`。

策略不直接接收 `gait_id`。强耦合项改变 CPG 动力学，策略通过 CPG state、接触和本体反馈适应当前步态。

## 9. Reward 与终止条件

使用 AllGaits 的简化 reward，不添加 gait style 或 imitation reward：

```text
forward velocity tracking:
    exp(-||v_x* - v_x||^2 / 0.25)              weight 3.0

lateral and vertical velocity penalty:
    -||[v_y, v_z]||^2                           weight 2.0

base angular velocity penalty:
    -||omega_base_xyz||^2                       weight 0.1

mechanical power penalty:
    -abs(sum(tau * q_dot))                      weight 0.001
```

各权重继续由环境按 policy `dt` 缩放。episode 长度为 20 秒；base 或 thigh 接触地面时终止。

不得加入 contact-pattern reward。步态由 coupling matrix 强制形成，接触感知用于闭环调节而不是用 reward 硬编码步态。

## 10. 训练与随机化

新增专用 PPO 配置：

- actor/critic MLP：`[512, 256, 128]`，ELU。
- PPO runner：`ActorCritic + PPO`。
- 默认 4096 个并行环境；smoke test 可覆盖为 64 或 256。
- policy rate：100 Hz。
- physics、CPG 和 PD rate：1000 Hz。
- episode：20 秒。

随机化采用论文范围：

- joint `Kp`：`[30, 100]`。
- joint `Kd`：`[0.5, 2.0]`。
- 每个 body link mass：标称值的 `[70%, 130%]`。
- added base mass：`[0, 5] kg`。
- friction：`[0.3, 1.0]`。
- 每 15 秒施加最大 `0.5 m/s` 的随机方向 base push。

第一阶段只在 flat terrain 训练。

## 11. 代码边界

计划新增以下组件：

```text
legged_gym/envs/go2/allgaits_cpg.py
    纯 PyTorch CPG、9 种 coupling matrix、pattern formation 和 IK

legged_gym/envs/go2/allgaits_env.py
    8 维动作解释、62 维 observation、命令/步态采样和控制链

legged_gym/envs/go2/allgaits_config.py
    独立环境、reward、PPO 和随机化配置

tests/test_go2_allgaits_cpg.py
    动力学、相位关系、轨迹、IK 和 reset 单元测试

tests/test_go2_allgaits_contract.py
    动作/观测维数、接触顺序和配置契约测试
```

现有文件只做必要接入：

- `legged_gym/envs/__init__.py` 注册 `go2_allgaits`。
- `legged_gym/envs/base/legged_robot.py` 将 `num_actions` 与 `num_dof` 解耦；action buffers 使用 8 维，torque、PD gain 和 joint buffers 使用 12 维。
- 原 `go2_flat`、`cpg_residual_position` 和 `cpg_hopf_position` 保留，默认行为不改变。

## 12. 验证门槛

### 12.1 无 Isaac Gym 的单元测试

1. 8 维标准化动作正确映射到论文的 `mu` 和 frequency 范围。
2. `r_i` 收敛到 `mu_i`，而不是 `sqrt(mu_i)`。
3. 9 种 gait 的目标相位关系与 Figure 3 一致。
4. gait 切换过程中相位状态连续，无 reset 跳变。
5. 足端轨迹、IK 和关节目标均为有限值且 shape 正确。
6. 足端目标在工作空间边界外时产生 projection，不产生 NaN。
7. observation shape 为 `[N, 62]`，其中接触值只为 `0` 或 `1`。
8. mechanical power reward 严格计算为 `-abs(sum(tau * q_dot))`。
9. 现有测试全部继续通过。

### 12.2 Isaac Gym smoke test

1. `go2_allgaits` 能以 64 或 256 environments 创建。
2. CPG-only 固定 trot rollout 至少运行 20 秒，无 NaN、无 shape error。
3. 9 种 gait 分别运行，并验证接触相位的定性关系。
4. episode 内 gait 切换不会导致 CPG 状态被重置。
5. PPO 能运行 5、50、400 iteration 三档测试。

### 12.3 训练验收

1. 前向速度 tracking error 随训练下降。
2. 单一策略能覆盖 9 种 gait 和 `[0.2, 3.0] m/s`。
3. gait 切换不产生高 fall rate 或明显关节跳变。
4. 记录 COT、速度误差、base angular velocity、joint acceleration、foot slip、IK projection rate 和 fall rate。
5. 固定随机种子下可以复现实验趋势。

## 13. 风险与处理

### 策略动作数与关节数耦合

APEX 当前多处默认 `num_actions == num_dof == 12`。实施时先完成最小解耦，并用原 `go2_flat` 环境契约测试防止回归。

### 腿顺序不一致

论文使用 `FR, FL, HR, HL`，APEX 和 URDF 的关节顺序不完全相同。所有 CPG、足端接触、IK 和关节输出均通过命名映射转换，不使用隐式 reshape 猜测顺序。

### 训练频率增加计算成本

严格配置使用 1 kHz physics 和 100 Hz policy。先用 64/256 environments 验证，再扩大到 4096；显存或吞吐不足时只降低环境数，不改变时间尺度。

### Go2 与论文 Go1 形态不同

控制架构、动力学、动作/观测和训练流程严格复现，几何参数使用 Go2 URDF。实验报告明确标注机器人形态差异，不把结果表述为数值复现 Go1。

## 14. 完成定义

第一阶段完成必须同时满足：

1. `go2_allgaits` 使用纯 PPO 和真实 8 维动作。
2. observation 含 4 维足底接触和 16 维 CPG 状态，总维数 62。
3. 在线执行链唯一且完整：`CPG -> foot trajectory -> IK -> 12 joint targets -> PD`。
4. 支持 9 种 gait，并能在 episode 内连续切换。
5. 单元测试、现有回归测试和 Isaac Gym smoke test 通过。
6. 形成可用于后续足端残差、高层 gait selector 和论文消融实验的 AllGaits 基线。
