---
license: other
license_name: multi-license-see-card
license_link: LICENSE
pretty_name: "Kine2Go: Kinematic dataset for the Unitree Go2 robot with diverse gaits and motions"
size_categories:
  - n<1K
task_categories:
  - robotics
  - reinforcement-learning
tags:
  - kine2go
  - quadruped
  - unitree-go2
  - motion-retargeting
  - imitation-learning
  - kinematic-dataset
  - mocap
  - genesis-simulator
  - ppo
  - gait
---

# Kine2Go: Kinematic dataset for the Unitree Go2 robot with diverse gaits and motions

A kinematic motion dataset for the Unitree Go2 quadruped robot. Forty reference clips (dog, horse, and synthetic robot motions) are retargeted to the Go2 morphology and paired with a per-clip imitation-learning policy (PPO) and 20 perturbed rollouts with rendered video. Designed to support training and regularization of behavioral foundation models for legged locomotion (Meta Motivo style), with secondary applicability to single-clip behavioral cloning and motion-retargeting evaluation.

[**Paper**](TBD) · [**Pipeline code (GitHub)**](TBD)

## At a glance

| | |
|---|---|
| Robot | Unitree Go2 (12 actuated joints) |
| Simulator | Genesis |
| Clips | 40 |
| Rollouts per clip | 20 |
| Total rollouts | 800 |
| Reference state dim | 61 (float32) |
| Control / sim FPS | 60 Hz |

## Quick start

```python
import sys; sys.path.insert(0, "scripts")
from load_clip import Clip, list_clips

for name in list_clips("."):
    print(name)

clip   = Clip(".", "ai4_dog_canter")
motion = clip.motion          # np.ndarray (T, 61) float32
config = clip.config          # env / obs / reward / command / policy configs
meta   = clip.metadata        # source, license, fps, n_frames, ...
frames = clip.rollout(0)      # list of per-step dicts (requires torch)
policy = clip.policy_path     # path to the trained PPO checkpoint
```

Validate a downloaded copy:

```bash
python scripts/verify_schema.py
```

## Per-clip layout

```
<clip_name>/
├── motion.npy           # (T, 61) float32 reference trajectory
├── config.json          # env / obs / reward / command / policy configs
├── cfgs.pkl             # same configs, pickle (for tooling)
├── clip.json            # per-clip metadata (source, license, n_frames, ...)
├── traj_NNNN/           # 20 perturbed PPO rollouts (NNNN = 0000..0019)
│   ├── init_state.pkl
│   ├── traj.pkl         # list of per-step frame dicts
│   └── traj.mp4         # rendered video
└── logs/
    └── model.pt         # final PPO checkpoint
```

## Subsets and licensing

Multi-license dataset. Each `clip.json` carries the authoritative per-clip license. Full license texts are in `LICENSES/`.

| Subset | Clip prefix | Clips | Rollouts | License |
|---|---|---|---|---|
| AI4Animation - natural dog mocap | `ai4_dog_*` (excl. `synth`) | 15 | 300 | CC BY-NC 4.0 |
| AI4Animation - synthetic via SIGGRAPH 2018 controller | `ai4_dog_synth_*` | 7 | 140 | CC BY-NC 4.0 |
| Vienna Horse Data Collection (Horse 1, two sessions) | `vhdc_horse1_*` | 12 | 240 | CC BY-SA 4.0 |
| cassi - Solo8 robot motions | `solo8_*` | 6 | 120 | BSD-3-Clause |

## Schema

### `motion.npy` (reference trajectory)

Shape `(T, 61)` float32. Per-frame layout:

| Slice | Field | Description |
|---|---|---|
| `[0:18]`  | `dofs_position` | 6 floating-base DOFs + 12 joint DOFs |
| `[18:36]` | `dofs_velocity` | matching DOF order |
| `[36:48]` | `feet_pos` | 4 feet × xyz, world frame |
| `[48:51]` | `base_pos` | world-frame xyz |
| `[51:55]` | `base_quat` | scalar-first quaternion (w, x, y, z) |
| `[55:58]` | `base_lin_vel` | linear velocity |
| `[58:61]` | `base_ang_vel` | angular velocity |

### `traj_NNNN/traj.pkl` (rollouts)

A `list` (one entry per simulator step). Each entry is a `dict[str, torch.Tensor]` with the following keys:

| Key | Shape | dtype | Description |
|---|---|---|---|
| `dof_pos` | (1, 12) | float32 | joint angles |
| `dof_vel` | (1, 12) | float32 | joint velocities |
| `base_quat` | (1, 4) | float32 | base orientation, scalar-first quaternion |
| `base_ang_vel` | (1, 3) | float32 | base angular velocity |
| `actions` | (1, 12) | float32 | PPO policy output |
| `frame` | (1,) | int32 | episode-relative simulator step counter |
| `links_pos` | (17, 3) | float32 | world-frame xyz per link |
| `links_rot` | (17, 6) | float32 | 6-d continuous rotation per link |

### Conventions

- Z-up coordinates (X forward, Y left, Z up).
- Linear units: meters; angular units: radians; quaternions are scalar-first.
- Joint order: `FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf, RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf`.
- Feet order (in `motion.npy[36:48]`): `FL_foot, RL_foot, FR_foot, RR_foot`.

The full machine-readable schema is in `metadata.json`.

## Collection and preprocessing

Source motions were retargeted to the Go2 12-DOF morphology and used to train per-clip imitation policies in the Genesis simulator:

- **AI4Animation - natural** (Zhang et al., SIGGRAPH 2018): dog mocap data.
- **AI4Animation - synthetic**: a custom trajectory collector was injected into the AI4Animation SIGGRAPH 2018 controller to drive the trained model along authored paths (circle, ellipse, figure-eight, square, strafes, and a half-flipping jump).
- **Vienna Horse Data Collection**: biomechanical horse kinematics across two measurement sessions × {walk, trot} × 3 repetitions.
- **cassi - Solo8** (Li et al., ICRA 2023): six trajectories from cassi's Solo8 motion bank (slow/fast crawl, scoot, walk, two jump variants).

Each retargeted reference was used to train a PPO policy (`rsl_rl_lib`) in Genesis with hyperparameters captured in `config.json`. After training, 20 perturbed rollouts were recorded per clip.

## Citations

Please cite the Kine2Go paper when referencing this dataset. Final citation metadata will be added after review. Also cite the upstream sources for whichever subsets you use.

```bibtex
% AI4Animation - applies to ai4_dog_* and ai4_dog_synth_*
@article{zhang2018modeadaptive,
  title   = {Mode-Adaptive Neural Networks for Quadruped Motion Control},
  author  = {Zhang, He and Starke, Sebastian and Komura, Taku and Saito, Jun},
  journal = {ACM Transactions on Graphics},
  year    = {2018}
}

% Vienna Horse Data Collection - applies to vhdc_horse1_*
@misc{vhdc,
  title = {Vienna Horse Data Collection (VHDC)},
  url   = {https://horse.cs.uni-bonn.de/vhdc-home.html}
}

% cassi - applies to solo8_*
@inproceedings{li2023versatile,
  title     = {Versatile Skill Control via Self-supervised Adversarial Imitation of Unlabeled Mixed Motions},
  author    = {Li, Chenhao and Blaes, Sebastian and Kolev, Pavel and Vlastelica, Marin and Frey, Jonas and Martius, Georg},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2023}
}
```
