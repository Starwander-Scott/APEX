"""Joint-space CPG utilities for Go2 gait generation.

The first version is intentionally independent from IsaacGym. It gives the RL
environment a deterministic gait prior that can later be combined with policy
residuals in Go2._compute_torques.
"""

from __future__ import annotations

from typing import NamedTuple

import torch


LEG_ORDER = ("FL", "FR", "RL", "RR")
JOINT_NAMES = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
)

GAIT_PHASE_OFFSETS = {
    # LEG_ORDER = FL, FR, RL, RR
    "walk": (0.0, 0.5, 0.75, 0.25),
    "trot": (0.0, 0.5, 0.5, 0.0),
    "pace": (0.0, 0.5, 0.0, 0.5),
    "bound": (0.0, 0.0, 0.5, 0.5),
    "canter": (0.0, 0.15, 0.55, 0.7),
    "run": (0.0, 0.5, 0.5, 0.0),
}

HIP_SIDE_SIGN = torch.tensor([1.0, -1.0, 1.0, -1.0])


class CPGAmplitudes(NamedTuple):
    hip: float = 0.08
    thigh: float = 0.25
    calf: float = 0.35


def phase_offsets(gait: str, device: torch.device | str | None = None) -> torch.Tensor:
    try:
        offsets = GAIT_PHASE_OFFSETS[gait]
    except KeyError as exc:
        known = ", ".join(sorted(GAIT_PHASE_OFFSETS))
        raise ValueError(f"Unknown gait '{gait}'. Known gaits: {known}") from exc
    return torch.tensor(offsets, dtype=torch.float32, device=device)


class Go2JointCPG:
    """Simple sinusoidal joint-space CPG for Unitree Go2.

    This module does not try to solve foot-space IK yet. It is the smallest
    useful prior for CPG+RL: CPG proposes a periodic joint target, and a policy
    can later add residual joint corrections.
    """

    def __init__(
        self,
        num_envs: int,
        default_dof_pos: torch.Tensor,
        device: torch.device | str,
        initial_phase: float = 0.0,
    ) -> None:
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        default_dof_pos = torch.as_tensor(default_dof_pos, dtype=torch.float32, device=self.device)
        if default_dof_pos.numel() != 12:
            raise ValueError("default_dof_pos must contain 12 Go2 joint values")
        self.default_dof_pos = default_dof_pos.reshape(1, 12)
        self.phase = torch.full((self.num_envs,), float(initial_phase), device=self.device)

    def reset(self, env_ids: torch.Tensor | None = None, phase: float = 0.0) -> None:
        if env_ids is None:
            self.phase.fill_(float(phase))
            return
        self.phase[env_ids] = float(phase)

    def leg_phases(self, gait: str) -> torch.Tensor:
        offsets = phase_offsets(gait, device=self.device)
        return torch.remainder(self.phase.unsqueeze(1) + offsets.unsqueeze(0), 1.0)

    def step(
        self,
        gait: str,
        frequency_hz: torch.Tensor | float,
        dt: float,
        amplitudes: CPGAmplitudes = CPGAmplitudes(),
    ) -> torch.Tensor:
        frequency = torch.as_tensor(frequency_hz, dtype=torch.float32, device=self.device)
        if frequency.ndim == 0:
            frequency = frequency.repeat(self.num_envs)
        if frequency.shape != (self.num_envs,):
            raise ValueError(f"frequency_hz must be scalar or shape ({self.num_envs},)")

        self.phase = torch.remainder(self.phase + frequency * float(dt), 1.0)
        leg_phase = self.leg_phases(gait)
        oscillation = torch.sin(2.0 * torch.pi * leg_phase)
        swing = torch.clamp(oscillation, min=0.0)

        targets = self.default_dof_pos.repeat(self.num_envs, 1)
        hip_sign = HIP_SIDE_SIGN.to(device=self.device)
        for leg_idx in range(4):
            start = leg_idx * 3
            leg_osc = oscillation[:, leg_idx]
            leg_swing = swing[:, leg_idx]
            targets[:, start] += hip_sign[leg_idx] * amplitudes.hip * leg_osc
            targets[:, start + 1] += amplitudes.thigh * leg_osc
            targets[:, start + 2] += -amplitudes.calf * leg_swing

        return targets
