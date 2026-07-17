# CPG-RL 进度记录 — IsaacGym 验证

日期：2026-07-10

承接文档：`docs/handoff/2026-07-06-cpg-rl-progress-handoff.md`

## 1. 环境信息

| 项目 | 值 |
|------|-----|
| 机器 | Linux (sjtu) |
| GPU | NVIDIA RTX 4090 (24 GB) |
| CUDA | 12.8 |
| Driver | 570.211.01 |
| PyTorch | 2.2.2+cu121 |
| IsaacGym | Preview 4 (gym_38) |
| Conda env | `apex` (Python 3.8.20) |

> **注意**：IsaacGym 需要 `LD_LIBRARY_PATH` 指向 conda env 的 lib 目录，否则会报 `libpython3.8.so.1.0: cannot open shared object file`。
>
> ```bash
> export LD_LIBRARY_PATH=/home/sjtu/miniconda3/envs/apex/lib:$LD_LIBRARY_PATH
> ```

## 2. 本次完成内容

按照交接文档 Section 5（IsaacGym 环境测试方式）完成了以下三步：

### 2.1 CPG-only smoke test ✅

- 将 `control_type` 切换为 `cpg_residual_position`，`cpg_residual_scale: 0.0`
- 运行 `test_env.py --task=go2_flat --headless`
- 结果：环境创建成功，step 无报错，数值无 NaN

### 2.2 多步态相位验证 ✅

通过 CPG 单元测试验证了全部 6 种步态的相位关系：

| 步态 | 相位 (FL, FR, RL, RR) | 关系 | 状态 |
|------|----------------------|------|------|
| walk | (0.00, 0.50, 0.75, 0.25) | 4-beat | ✅ |
| trot | (0.00, 0.50, 0.50, 0.00) | 对角同相 | ✅ |
| pace | (0.00, 0.50, 0.00, 0.50) | 同侧同相 | ✅ |
| bound | (0.00, 0.00, 0.50, 0.50) | 前/后同相 | ✅ |
| canter | (0.00, 0.15, 0.55, 0.70) | 非对称 | ✅ |
| run | (0.00, 0.50, 0.50, 0.00) | 同 trot | ✅ |

### 2.3 RL residual 训练 + demo ✅

- 控制模式：`cpg_residual_position`，步态 `trot`，`cpg_residual_scale: 1.0`
- 训练参数：256 envs，400 iterations
- 训练时间：约 10 分钟
- Wandb 已禁用（`WANDB_MODE=disabled`），TensorBoard 日志正常写入

**训练产物**：

```text
logs/apex_IROS_pronk/Jul10_13-45-19_/
├── model_0.pt                  # 初始权重
├── model_200.pt                # 中期 checkpoint（save_interval=200）
├── model_400.pt                # 最终模型
├── events.out.tfevents.*       # TensorBoard 日志
└── exported/policies/
    └── policy_1.pt             # TorchScript JIT 导出
```

**Demo 评估（model_400.pt，1 episode）**：

| 指标 | 值 | 说明 |
|------|-----|------|
| `imitate_quat` | +0.488 | 身体姿态跟踪良好 |
| `tracking_ang_vel` | +0.234 | 角速度跟踪较好 |
| `tracking_lin_vel` | +0.030 | 线速度跟踪尚弱（训练初期正常） |
| `action_rate` | -0.001 | 动作平滑 |
| `feet_slip` | -0.003 | 足端滑移低 |
| `collision` | 0.000 | 无碰撞 |
| `imitation_height` | -0.259 | 高度跟踪有待改善 |

策略已在 IsaacGym 中成功 rollout，无报错、无 NaN。

## 3. 已知问题和注意事项

1. **GLFW headless 渲染**：`play.py` 在纯 headless 环境下会因 GLFW 窗口创建失败而 segfault。需要加 `--headless` 参数跳过 viewer 创建。如需录制画面，可尝试 Xvfb + Mesa 软件渲染（当前环境暂未配置成功）。
2. **wandb**：训练默认依赖 wandb 登录。使用 `export WANDB_MODE=disabled` 绕过。
3. **isaacgym 导入顺序**：必须 `import isaacgym` 在 `import torch` 之前，否则报错。
4. **测试依赖**：`test_gait_analysis.py` 需要 `pandas`，已在 apex 环境中安装。
5. **rsl_rl + legged_gym**：需要以 editable mode 安装（`pip install -e rsl_rl/ && pip install -e .`）。

## 4. 下一步建议

按照交接文档 Section 7 的路线，后续建议：

1. **更长训练**：以 1200-2000 iterations 完整训练 CPG+RL residual，观察 reward curve 是否收敛。
2. **添加 gait-conditioned policy**：将 gait_id 加入 observation，让策略根据 gait 条件选择步态。
3. **记录 rollout 数据**：写脚本记录 `joint_pos_target`、base velocity、foot contact，用于定量分析。
4. **多步态切换**：实现 episode 间或 episode 内 gait switching。
5. **对比实验**：按 plan 文档的阶段 6 设计四组对比（APEX baseline / CPG-only / CPG+12D residual / CPG+structured residual）。

## 5. 关键文件速查

```text
docs/handoff/2026-07-06-cpg-rl-progress-handoff.md  # 原始交接文档
docs/superpowers/plans/2026-07-06-cpg-rl-apex-plan.md  # 路线计划
logs/apex_IROS_pronk/Jul10_13-45-19_/                 # 本次训练产物
legged_gym/envs/go2/cpg.py                             # CPG 实现
legged_gym/envs/go2/go2.py                             # 控制链路（cpg_residual_position 分支）
legged_gym/envs/param_config.yaml                      # 配置文件
legged_gym/scripts/train.py                            # 训练入口
legged_gym/scripts/play.py                             # 推理/demo 入口
```

## 6. 常用命令

```bash
# 激活环境
conda activate apex
export LD_LIBRARY_PATH=/home/sjtu/miniconda3/envs/apex/lib:$LD_LIBRARY_PATH

# 运行单元测试
python -m unittest discover -s tests -p 'test_*.py'

# CPG-only 测试
# 先修改 param_config.yaml: control_type → cpg_residual_position, cpg_residual_scale → 0.0
python legged_gym/tests/test_env.py --task=go2_flat --headless

# 训练 CPG+RL
export WANDB_MODE=disabled
python legged_gym/scripts/train.py --task=go2_flat --headless --max_iterations=1200

# 运行 demo（加 --headless 避免 GLFW 报错）
python legged_gym/scripts/play.py --task=go2_flat \
  --load_run Jul10_13-45-19_ --checkpoint 400 --headless
```
