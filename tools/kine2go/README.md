# Kine2Go to APEX

This folder prepares Kine2Go reference motions for the APEX Go2 training code.
It downloads only lightweight reference files, not rollout videos or trajectory
pickle files.

## Quick Start

From `D:\Desktop\robot\APEX`:

```powershell
python tools\kine2go\prepare_kine2go_for_apex.py --clips ai4_dog_pace ai4_dog_walk_00 ai4_dog_trot_00
```

Outputs:

- Dataset reference files: `datasets\kine2go_refs`
- APEX CSV files: `imitation_data\kine2go`
- Conversion summary: `imitation_data\kine2go\conversion_summary.csv`

To convert only an existing local copy:

```powershell
python tools\kine2go\prepare_kine2go_for_apex.py --convert-only --clips ai4_dog_pace
```

To download and convert all clips:

```powershell
python tools\kine2go\prepare_kine2go_for_apex.py
```

## Use In APEX

Edit `legged_gym\envs\param_config.yaml`:

```yaml
control_type: "apex_position"
train_multi_skills: False
path_to_imitation_data: "imitation_data/kine2go/ai4_dog_pace.csv"
use_imitation_commands: True
number_observations: 45
number_privileged_observations: 77
```

Then train:

```powershell
python legged_gym\scripts\train.py --task=go2_flat --headless
```

## Notes

- Kine2Go reference motions are 60 Hz. The converter writes 50 Hz CSV by
  default because this APEX config runs at `sim.dt * decimation = 0.02s`.
- The current Kine2Go `motion.npy` files are numerically consistent with joint
  values grouped as `hip[FR,FL,RR,RL], thigh[FR,FL,RR,RL], calf[FR,FL,RR,RL]`;
  the converter detects this and writes APEX order `FL,FR,RL,RR`.
- Kine2Go feet are documented as `FL,RL,FR,RR`; the converter writes
  `FL,FR,RL,RR`.
- The APEX command columns `com_vx/com_vy/com_wz` are written as per-clip mean
  velocity commands by default. Use `--command-mode instantaneous` if you want
  frame-wise commands instead.
- Dog clips are CC BY-NC 4.0. Do not redistribute the data in a commercial
  project.
