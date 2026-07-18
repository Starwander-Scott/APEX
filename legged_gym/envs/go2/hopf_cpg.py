"""Hopf-oscillator CPG with foot-space trajectory generation and IK for Go2.

Based on the Hopf CPG approach from:
  "CPG-RL: Learning Central Pattern Generators for Quadruped Locomotion" (IEEE RA-L 2022)
  Guyueju ROS tutorial: Hopf oscillator CPG for quadruped robots

Key improvements over the joint-space sin-wave CPG:
- Hopf oscillators with coupled phase dynamics (stable limit cycles)
- Explicit stance/swing phase separation with different frequencies
- Foot-end trajectory generation in Cartesian space
- Analytical IK converts foot positions to joint angles
- RL can output residual corrections in foot space (more intuitive than joint space)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import NamedTuple


LEG_ORDER = ("FL", "FR", "RL", "RR")
JOINT_NAMES = (
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
)


# ── Phase coupling matrices ──────────────────────────────────────────────
# Each entry PHI[i,j] is the desired phase difference θ_j - θ_i.
# Values in radians; trot = diagonal legs 180° out of phase.

GAIT_COUPLING = {
    # LEG_ORDER = FL, FR, RL, RR
    "trot": torch.tensor([
        [ 0.0,  3.1416,  3.1416,  0.0],
        [-3.1416,  0.0,  0.0, -3.1416],
        [-3.1416,  0.0,  0.0, -3.1416],
        [ 0.0,  3.1416,  3.1416,  0.0],
    ]),
    "walk": torch.tensor([
        [ 0.0,  1.5708,  2.3562, -0.7854],
        [-1.5708,  0.0,  0.7854, -2.3562],
        [-2.3562, -0.7854,  0.0, -1.5708],
        [ 0.7854,  2.3562,  1.5708,  0.0],
    ]),
    "pace": torch.tensor([
        [ 0.0,  3.1416,  0.0,  3.1416],
        [-3.1416,  0.0, -3.1416,  0.0],
        [ 0.0,  3.1416,  0.0,  3.1416],
        [-3.1416,  0.0, -3.1416,  0.0],
    ]),
    "bound": torch.tensor([
        [ 0.0,  0.0,  3.1416,  3.1416],
        [ 0.0,  0.0,  3.1416,  3.1416],
        [-3.1416, -3.1416,  0.0,  0.0],
        [-3.1416, -3.1416,  0.0,  0.0],
    ]),
    "canter": torch.tensor([
        [ 0.0,  0.4712,  1.7279,  2.1991],
        [-0.4712,  0.0,  1.2566,  1.7279],
        [-1.7279, -1.2566,  0.0,  0.4712],
        [-2.1991, -1.7279, -0.4712,  0.0],
    ]),
    "run": torch.tensor([  # same as trot
        [ 0.0,  3.1416,  3.1416,  0.0],
        [-3.1416,  0.0,  0.0, -3.1416],
        [-3.1416,  0.0,  0.0, -3.1416],
        [ 0.0,  3.1416,  3.1416,  0.0],
    ]),
}


class HopfCPGParams(NamedTuple):
    """Configurable Hopf CPG parameters."""
    mu: float = 1.0                # target amplitude of limit cycle
    alpha: float = 50.0            # convergence speed to limit cycle
    omega_swing: float = 10.0      # angular frequency during swing (rad/s)
    omega_stance: float = 5.0      # angular frequency during stance (rad/s)
    ground_clearance: float = 0.06  # foot lift height during swing (m)
    ground_penetration: float = 0.005  # foot penetration during stance (m)
    des_step_len: float = 0.08     # desired step length per stride (m)
    robot_height: float = 0.30     # nominal body height above ground (m)
    coupling_strength: float = 1.0 # coupling strength between oscillators
    # IK: Go2 leg link lengths
    thigh_length: float = 0.20     # thigh link length (m)
    calf_length: float = 0.20      # calf link length (m)
    # Hip lateral offset per leg (body frame y)
    hip_lateral_offset: float = 0.08  # half hip width (m)


class Go2HopfCPG:
    """Hopf-oscillator CPG with foot-space trajectory generation and IK.

    State per leg: amplitude r and phase θ.
    - r ∈ [0, ∞): radius of the Hopf limit cycle
    - θ ∈ [0, 2π): phase angle on the limit cycle

    The oscillator dynamics:
        r_dot  = α (μ - r²) r
        θ_dot  = ω + Σ c_ij r_j sin(θ_j - θ_i - Φ_ij)

    Foot trajectory (hip frame, sagittal plane):
        x_i  = -step_len * r_i * cos(θ_i)        # forward/back
        z_i  = -h + clearance * sin(θ_i)          # up/down (swing > 0)

    RL interface: policy can output residual corrections to foot positions.
    """

    def __init__(
        self,
        num_envs: int,
        default_dof_pos: torch.Tensor,
        device: torch.device | str,
        params: HopfCPGParams | None = None,
        hip_sign: torch.Tensor | None = None,
    ):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.p = params or HopfCPGParams()
        self.default_dof_pos = torch.as_tensor(
            default_dof_pos, dtype=torch.float32, device=self.device
        ).reshape(1, 12)

        # ── Oscillator state: [num_envs, 4] each ──
        self.r = torch.full((self.num_envs, 4), 0.1, device=self.device)
        # Initialize phases from the trot coupling matrix row 0
        # so oscillators start with proper phase differences
        init_phases = torch.tensor([0.0, 3.1416, 3.1416, 0.0], device=self.device)
        # Add small random perturbation to break perfect symmetry
        init_phases = init_phases + torch.rand(self.num_envs, 4, device=self.device) * 0.1
        self.theta = init_phases % (2.0 * torch.pi)
        self.r_dot_prev = torch.zeros(self.num_envs, 4, device=self.device)
        self.theta_dot_prev = torch.zeros(self.num_envs, 4, device=self.device)

        # Hip side sign: FL=+1, FR=-1, RL=+1, RR=-1 (lateral mirror)
        self.hip_sign = hip_sign if hip_sign is not None else torch.tensor(
            [1.0, -1.0, 1.0, -1.0], device=self.device
        )

        # ── Foot neutral positions in body frame [num_envs, 4, 3] ──
        # These are the default foot positions when r → 0
        self._init_neutral_feet()

        # ── RL interface ──
        self.use_rl = False
        self.omega_rl = torch.zeros(self.num_envs, 4, device=self.device)
        self.mu_rl = torch.zeros(self.num_envs, 4, device=self.device)

    def _init_neutral_feet(self):
        """Compute neutral foot positions from default standing pose."""
        d = self.default_dof_pos[0]  # [12]
        L1, L2 = self.p.thigh_length, self.p.calf_length
        y_off = self.p.hip_lateral_offset

        feet = torch.zeros(4, 3, device=self.device)
        for leg_idx in range(4):
            start = leg_idx * 3
            hip = d[start]
            thigh = d[start + 1]
            calf = d[start + 2]

            # Sagittal plane: foot position relative to hip joint
            s1, c1 = torch.sin(thigh), torch.cos(thigh)
            s12, c12 = torch.sin(thigh + calf), torch.cos(thigh + calf)

            x = L1 * s1 + L2 * s12
            z = -(L1 * c1 + L2 * c12)

            # Lateral
            y_sign = 1.0 if leg_idx in (0, 2) else -1.0  # FL,RL left; FR,RR right
            y = y_sign * y_off

            feet[leg_idx] = torch.tensor([x, y, z])

        self.neutral_feet = feet.unsqueeze(0)  # [1, 4, 3]

    def reset(self, env_ids: torch.Tensor | None = None, gait: str = "trot"):
        """Reset oscillator state for given environments.

        Phases are initialized from the gait coupling matrix (row 0)
        to preserve proper inter-leg phase relationships.
        """
        init_phases = torch.tensor([0.0, 3.1416, 3.1416, 0.0], device=self.device)
        init_phases = init_phases + torch.rand(4, device=self.device) * 0.1

        if env_ids is None:
            self.r.fill_(0.1)
            self.theta = init_phases.unsqueeze(0).repeat(self.num_envs, 1)
            self.r_dot_prev.zero_()
            self.theta_dot_prev.zero_()
            return
        self.r[env_ids] = 0.1
        self.theta[env_ids] = init_phases.unsqueeze(0)
        self.r_dot_prev[env_ids] = 0.0
        self.theta_dot_prev[env_ids] = 0.0

    def get_phase_sin_cos(self) -> torch.Tensor:
        """Return [sin(θ), cos(θ)] per leg → [num_envs, 8] for observation."""
        return torch.cat([
            torch.sin(self.theta),  # [N, 4]
            torch.cos(self.theta),  # [N, 4]
        ], dim=-1)

    # ══════════════════════════════════════════════════════════════════════
    # Core: Hopf oscillator integration
    # ══════════════════════════════════════════════════════════════════════

    def step(
        self,
        gait: str,
        dt: float,
        coupling_matrix: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance oscillator state and return joint targets.

        Uses joint-space offsets from the default standing pose for thigh/calf,
        calibrated by the oscillator state. Hip stays at default angle.

        Returns:
            feet_target: [num_envs, 4, 3] foot positions (for RL residual interface)
            joint_targets: [num_envs, 12] joint angles
        """
        if coupling_matrix is None:
            coupling_matrix = GAIT_COUPLING.get(gait, GAIT_COUPLING["trot"])
            coupling_matrix = coupling_matrix.to(device=self.device, dtype=torch.float32)

        self._integrate(dt, coupling_matrix)
        feet_target = self._foot_trajectory()  # [N, 4, 3]
        joint_targets = self._joint_targets()
        return feet_target, joint_targets

    def _integrate(self, dt: float, PHI: torch.Tensor):
        """Semi-implicit Euler integration of the Hopf oscillator network."""
        r, theta = self.r.clone(), self.theta.clone()
        N = self.num_envs

        # Determine omega per leg based on phase
        # sin(θ) > 0 → swing phase (faster), sin(θ) ≤ 0 → stance (slower)
        swing_mask = (torch.sin(theta) > 0).float()
        omega = swing_mask * self.p.omega_swing + (1 - swing_mask) * self.p.omega_stance

        if self.use_rl:
            omega = omega + self.omega_rl
            mu = self.mu_rl
        else:
            mu = self.p.mu

        # Amplitude dynamics: r_dot = α (μ - r²) r
        r_dot = self.p.alpha * (mu - r * r) * r

        # Phase dynamics with coupling
        theta_dot = omega.clone()
        for i in range(4):
            coupling = torch.zeros(N, device=self.device)
            for j in range(4):
                if i == j:
                    continue
                coupling += r[:, j] * torch.sin(theta[:, j] - theta[:, i] - PHI[i, j])
            theta_dot[:, i] += self.p.coupling_strength * coupling

        # Semi-implicit Euler (velocity average)
        dt_f = float(dt)
        r_new = r + (self.r_dot_prev + r_dot) * dt_f / 2.0
        theta_new = theta + (self.theta_dot_prev + theta_dot) * dt_f / 2.0

        self.r = torch.clamp(r_new, min=0.0)
        self.theta = torch.remainder(theta_new, 2.0 * torch.pi)
        self.r_dot_prev = r_dot
        self.theta_dot_prev = theta_dot

    # ══════════════════════════════════════════════════════════════════════
    # Foot trajectory generation
    # ══════════════════════════════════════════════════════════════════════

    def _foot_trajectory(self) -> torch.Tensor:
        """Generate foot position OFFSETS from neutral standing position.

        Returns [num_envs, 4, 3] — displacements from neutral foot positions.
        neutral_foot + offset = target foot position in body frame.

        Sagittal offsets:
          dx_i = -step_len * r_i * cos(θ_i)       # forward/back
          dz_i = clearance(θ_i) * |sin(θ_i)|      # up (swing lift) / down (stance)

        The neutral foot positions come from the standing pose IK.
        """
        N = self.num_envs
        sin_theta = torch.sin(self.theta)  # [N, 4]
        cos_theta = torch.cos(self.theta)  # [N, 4]

        swing = (sin_theta > 0).float()  # [N, 4]

        # Forward/backward offset (sagittal plane)
        dx = -self.p.des_step_len * self.r * cos_theta  # [N, 4]

        # Vertical offset: lift during swing, push slightly during stance
        dz = torch.where(
            sin_theta > 0,
            self.p.ground_clearance * sin_theta,      # swing: lift foot
            self.p.ground_penetration * sin_theta,    # stance: slight push
        )

        # Lateral offset (small hip ab/ad)
        dy = torch.zeros(N, 4, device=self.device)

        # Stack → [N, 4, 3]
        offsets = torch.stack([dx, dy, dz], dim=-1)
        return offsets

    # ══════════════════════════════════════════════════════════════════════
    # Inverse Kinematics: foot position → joint angles
    # ══════════════════════════════════════════════════════════════════════

    def _inverse_kinematics(self, feet_body: torch.Tensor) -> torch.Tensor:
        """Convert foot positions to 12 joint angles via analytical IK.

        Each Go2 leg is a 3-DOF serial chain:
          - hip:  lateral rotation (roll in body frame)
          - thigh: sagittal pitch (primary forward/back)
          - calf:  sagittal pitch (knee bend)

        The IK treats the leg as:
          1. Hip base position in body frame → static offset
          2. Foot position relative to hip → (x_hip, y_hip, z_hip)
          3. 2-link planar IK for thigh + calf in the sagittal plane
          4. Hip angle from lateral foot offset

        Args:
            feet_body: [N, 4, 3] target foot positions in body frame

        Returns:
            joint_angles: [N, 12] in order FL(hip,thigh,calf), FR(...), ...
        """
        N = self.num_envs
        L1 = self.p.thigh_length
        L2 = self.p.calf_length
        y_off = self.p.hip_lateral_offset

        joint_angles = torch.zeros(N, 12, device=self.device)

        for leg_idx in range(4):
            start = leg_idx * 3
            foot = feet_body[:, leg_idx, :]  # [N, 3]

            # ── Hip position in body frame ──
            y_sign = 1.0 if leg_idx in (0, 2) else -1.0
            hip_body_y = y_sign * y_off
            hip_body_z = 0.0  # hip near body CoM vertically
            hip_body_x = 0.0

            # ── Foot relative to hip ──
            fx = foot[:, 0] - hip_body_x  # [N]
            fy = foot[:, 1] - hip_body_y  # [N]
            fz = foot[:, 2] - hip_body_z  # [N]

            # ── 2-link planar IK in sagittal plane ──
            # Go2 leg: thigh + calf form a 2-link mechanism.
            # Convention: x=forward, z=down (negative in body frame).
            #
            # Distance from hip to foot:
            d = torch.sqrt(fx * fx + fz * fz + 1e-8)

            # Angle of the foot line from vertical (straight down):
            #   alpha = atan2(fx, -fz)  since z is negative (below hip)
            alpha = torch.atan2(fx, -fz + 1e-8)  # [N]

            # Knee angle via cosine law:
            #   d² = L1² + L2² - 2*L1*L2*cos(π - knee)
            #   → cos(knee) = (L1² + L2² - d²) / (2*L1*L2)
            cos_knee = (L1 * L1 + L2 * L2 - d * d) / (2.0 * L1 * L2 + 1e-8)
            cos_knee = torch.clamp(cos_knee, -0.999, 0.999)
            knee_angle = torch.acos(cos_knee)  # [N], positive (leg bent)

            # calf = -knee (calf is negative when knee is bent)
            calf_angle = -knee_angle

            # Thigh angle from vertical:
            #   beta = asin(L2 * sin(knee) / d)
            beta = torch.asin(
                torch.clamp(L2 * torch.sin(knee_angle) / (d + 1e-8), -0.999, 0.999)
            )
            thigh_angle = alpha + beta

            # ── Hip angle ──
            # Go2 hip is ab/ad (lateral roll). Use the default standing hip
            # angle for neutral stance; thigh+calf handle sagittal motion.
            default_hip = self.default_dof_pos[0, start]

            # ── Assign joint angles ──
            joint_angles[:, start] = default_hip
            joint_angles[:, start + 1] = thigh_angle
            joint_angles[:, start + 2] = calf_angle

        return joint_angles

    # ══════════════════════════════════════════════════════════════════════
    # Joint-space target generation (uses default pose as baseline)
    # ══════════════════════════════════════════════════════════════════════

    def _joint_targets(self) -> torch.Tensor:
        """Generate joint targets from oscillator state.

        Uses the default standing pose as baseline and adds oscillator-driven
        offsets for thigh (forward/back swing) and calf (knee bend/extend).
        Hip stays at default.

        Mapping (per leg):
          thigh_offset = -amp * r * cos(θ)  → forward in swing, backward in stance
          calf_offset  = -amp * r * sin_clipped(θ)  → retract in swing, extend in stance
        """
        N = self.num_envs
        targets = self.default_dof_pos.repeat(N, 1)  # [N, 12]

        sin_theta = torch.sin(self.theta)  # [N, 4]
        cos_theta = torch.cos(self.theta)  # [N, 4]

        # Swing = sin>0, Stance = sin<0
        # During swing: thigh swings forward, calf retracts (bends, lifts foot)
        # During stance: thigh pushes backward, calf extends (pushes against ground)

        thigh_osc = -self.p.des_step_len * self.r * cos_theta  # [N,4]
        calf_osc = -self.p.ground_clearance * self.r * torch.clamp(sin_theta, min=0.0)

        # Scale to reasonable joint angle ranges
        # thigh: map ~0.08m foot motion → ~0.25 rad joint motion
        # calf: map ~0.06m clearance → ~0.30 rad joint motion
        thigh_angle_scale = 0.25 / max(self.p.des_step_len, 0.01)
        calf_angle_scale = 0.30 / max(self.p.ground_clearance, 0.01)

        for leg_idx in range(4):
            start = leg_idx * 3
            targets[:, start + 1] += thigh_angle_scale * thigh_osc[:, leg_idx]
            targets[:, start + 2] += calf_angle_scale * calf_osc[:, leg_idx]

        return targets

    # ══════════════════════════════════════════════════════════════════════
    # RL: apply residual corrections to foot positions
    # ══════════════════════════════════════════════════════════════════════

    def apply_residual(
        self,
        feet_target: torch.Tensor,
        residual_actions: torch.Tensor,
        residual_scale: float,
    ) -> torch.Tensor:
        """Apply RL residual corrections to foot positions.

        Args:
            feet_target: [N, 4, 3] CPG-generated foot target positions
            residual_actions: [N, 12] RL policy output (dx,dy,dz per leg × 4)
            residual_scale: scaling factor

        Returns:
            corrected_feet: [N, 4, 3]
        """
        residual = residual_actions.view(-1, 4, 3) * residual_scale
        return feet_target + residual

    # ══════════════════════════════════════════════════════════════════════
    # RL: directly modulate CPG parameters
    # ══════════════════════════════════════════════════════════════════════

    def set_rl_params(self, omega_rl: torch.Tensor, mu_rl: torch.Tensor):
        """Enable RL modulation of frequency and amplitude per leg.

        Args:
            omega_rl: [N, 4] frequency adjustment per leg
            mu_rl: [N, 4] target amplitude adjustment per leg
        """
        self.use_rl = True
        self.omega_rl = omega_rl
        self.mu_rl = torch.clamp(mu_rl, min=0.01)
