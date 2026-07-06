# APEX CPG-RL 进度交接文档

日期：2026-07-06

## 1. 仓库信息

当前工作仓库：

```text
D:\Desktop\robot\APEX
```

远程仓库：

```text
origin   git@github.com:Starwander-Scott/APEX.git
upstream https://github.com/marmotlab/APEX.git
```

当前开发分支：

```text
main
```

当前交接时最新提交：

```text
24dbf1d Add CPG residual Go2 control mode
```

在另一台电脑继续时，建议先执行：

```bash
git clone git@github.com:Starwander-Scott/APEX.git
cd APEX
git status --short --branch
git log --oneline --decorate --max-count=5
```

如果已经 clone 过：

```bash
cd APEX
git fetch origin
git checkout main
git pull --ff-only origin main
```

## 2. 当前研究方向

当前方向已经从“情绪步态”调整为更基础、更稳的：

```text
CPG 生成不同四足步态
        +
RL 学习关节残差或少量 CPG 参数修正
```

第一阶段不引入 calm / active / emotion / style 标签。

当前优先目标是：

1. 用已有 motion CSV 量化不同 gait family 的运动特征。
2. 用 CPG 生成 walk / trot / pace / bound / canter / run 的基础周期关节目标。
3. 在 Go2 控制链路里加入 `cpg_residual_position`。
4. 后续在 IsaacGym 中验证 CPG-only 和 CPG+RL residual 是否能稳定 rollout。

## 3. 已完成内容

### 3.1 步态特征分析工具

新增文件：

```text
tools/gait_analysis/compute_gait_features.py
tests/test_gait_analysis.py
analysis/gait_features/summary.csv
analysis/gait_features/per_motion_features.csv
analysis/gait_features/00_gait_feature_report.md
```

功能：

1. 读取 `imitation_data/kine2go` 和 `imitation_data/animal_mocap` 下的 motion CSV。
2. 计算速度、步频、foot clearance、body height、body bounce、pitch / roll、contact duty、foot slip、joint limit violation。
3. 根据文件名关键词生成 `gait_family`，例如 `walk`、`trot`、`pace`、`canter`、`run`。
4. 生成多步态特征体检报告。

运行方式：

```bash
python tools/gait_analysis/compute_gait_features.py
```

输出位置：

```text
analysis/gait_features/
```

### 3.2 Go2 joint-space CPG

新增文件：

```text
legged_gym/envs/go2/cpg.py
tests/test_go2_cpg.py
```

当前支持 gait：

```text
walk
trot
pace
bound
canter
run
```

核心类和函数：

```python
Go2JointCPG
CPGAmplitudes
phase_offsets()
apply_residual_action()
```

当前 CPG 是 joint-space 版本，不是 foot-space IK 版本。它生成 `[num_envs, 12]` 的 Go2 关节目标。

### 3.3 Go2 控制模式接入

修改文件：

```text
legged_gym/envs/go2/go2.py
legged_gym/envs/param_config.yaml
```

新增控制模式：

```yaml
control_type: "cpg_residual_position"
```

默认仍保持：

```yaml
control_type: "apex_position"
```

所以当前代码不会默认改变原 APEX 训练流程。

`cpg_residual_position` 的控制逻辑：

```text
q_cpg = CPG(gait, frequency, amplitudes)
delta_q = scaled RL action
q_des = q_cpg + cpg_residual_scale * delta_q
torque = PD(q_des)
```

零动作时，如果 `cpg_residual_scale = 0.0` 或 action 为 0，就是 CPG-only。

## 4. 当前本地可验证命令

在没有 IsaacGym 的 Windows 环境里，可以验证纯 Python / torch 层：

```bash
python -m unittest discover -s tests -p 'test_*.py'
python -m compileall legged_gym/envs/go2/cpg.py legged_gym/envs/go2/go2.py tools/gait_analysis/compute_gait_features.py
python tools/gait_analysis/compute_gait_features.py
```

最近一次验证结果：

```text
Ran 6 tests
OK

Analyzed 46 motions
```

## 5. IsaacGym 环境测试方式

这一步还没有在当前 Windows 电脑完成。需要 Linux + NVIDIA GPU + IsaacGym。

### 5.1 CPG-only smoke test

修改 `legged_gym/envs/param_config.yaml`：

```yaml
# control_type: "apex_position"
control_type: "cpg_residual_position"

cpg_gait: "trot"
cpg_frequency_hz: 1.5
cpg_residual_scale: 0.0
cpg_hip_amplitude: 0.08
cpg_thigh_amplitude: 0.25
cpg_calf_amplitude: 0.35
```

运行：

```bash
python legged_gym/tests/test_env.py --task=go2_flat --headless
```

预期：

1. 环境能创建。
2. step 不报错。
3. 数值不出现 NaN。
4. 不一定能稳定行走，因为这是 joint-space CPG 初版。

### 5.2 可视化不同步态

把 `headless` 去掉，逐个修改：

```yaml
cpg_gait: "walk"
cpg_gait: "trot"
cpg_gait: "pace"
cpg_gait: "bound"
cpg_gait: "canter"
```

运行：

```bash
python legged_gym/tests/test_env.py --task=go2_flat
```

观察重点：

1. `walk`：四条腿错相。
2. `trot`：对角腿同相。
3. `pace`：同侧腿同相。
4. `bound`：前腿同相、后腿同相。
5. `canter`：非对称相位。

### 5.3 RL residual 训练前检查

CPG-only 能 step 后，再设置：

```yaml
cpg_residual_scale: 1.0
```

然后才考虑训练：

```bash
python legged_gym/scripts/train.py --task=go2_flat --headless
```

第一轮建议不要长训。先用较小 `num_envs` 做 smoke training，确认 loss、reward、reset rate 没有明显异常。

## 6. 已知限制

1. 当前 CPG 是 joint-space CPG，不是 foot-space CPG + IK。
2. 当前没有显式 duty factor / stance-swing 分段轨迹。
3. 当前 `cpg_gait` 是全局配置，不是每个 episode 随机采样。
4. 当前没有把 gait id 加入 observation。
5. 当前没有 gait switching 逻辑。
6. 当前没有在 IsaacGym 里完成 rollout 验证。
7. 当前没有真实 Go2 部署验证。

## 7. 下一步建议

建议按下面顺序继续：

1. 在 IsaacGym 环境中跑 `cpg_residual_position` 的 CPG-only smoke test。
2. 逐个可视化 `walk / trot / pace / bound / canter`，确认相位方向正确。
3. 如果 CPG-only 数值稳定，加入一个小脚本记录 `joint_pos_target`、base velocity、foot contact。
4. 再开始短训练 RL residual，不要直接长训。
5. 如果 joint-space CPG 不稳定，再考虑 foot-space CPG + 简化 IK。
6. 稳定后再做 gait-conditioned policy，让策略根据 gait id 或 command 选择步态。

## 8. 关键文件速查

```text
legged_gym/envs/go2/cpg.py
legged_gym/envs/go2/go2.py
legged_gym/envs/param_config.yaml
tools/gait_analysis/compute_gait_features.py
analysis/gait_features/00_gait_feature_report.md
docs/superpowers/plans/2026-07-06-cpg-rl-apex-plan.md
tests/test_go2_cpg.py
tests/test_gait_analysis.py
```

## 9. 注意事项

1. 不要把当前阶段描述成“情绪步态生成已经完成”。
2. 当前只完成了“多步态 CPG + residual 控制模式的代码接入”。
3. 论文贡献应围绕 CPG-RL 多步态控制是否更稳定、更可解释，而不是直接宣称情绪陪伴。
4. 在正式实验前，所有 IsaacGym rollout 结果都需要重新记录并保存。
