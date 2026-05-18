"""Download Kine2Go reference clips and convert them to APEX Go2 CSV files.

This script intentionally downloads only the small reference files needed by
APEX imitation training: motion.npy, clip.json, config.json, cfgs.pkl, and
license/metadata files. It does not download rollout videos or traj.pkl files.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ID = "MIMUW-Robotics/kine2go"
REPO_TYPE = "dataset"
INPUT_FPS = 60.0
DEFAULT_OUTPUT_FPS = 50.0

ROOT_FILES = {
    ".gitattributes",
    "LICENSE",
    "README.md",
    "metadata.json",
}

CLIP_FILES = {
    "motion.npy",
    "clip.json",
    "config.json",
    "cfgs.pkl",
}

KINE_LEG_ORDER = ("FR", "FL", "RR", "RL")
APEX_LEG_ORDER = ("FL", "FR", "RL", "RR")

APEX_COLUMNS = [
    "vx",
    "vy",
    "vz",
    "wx",
    "wy",
    "wz",
    "base1",
    "shoulder1",
    "elbow1",
    "base2",
    "shoulder2",
    "elbow2",
    "base3",
    "shoulder3",
    "elbow3",
    "base4",
    "shoulder4",
    "elbow4",
    "com_vx",
    "com_vy",
    "com_wz",
    "height",
    "e1x",
    "e1y",
    "e1z",
    "e2x",
    "e2y",
    "e2z",
    "e3x",
    "e3y",
    "e3z",
    "e4x",
    "e4y",
    "e4z",
    "com_x",
    "com_y",
    "quat_x",
    "quat_y",
    "quat_z",
    "quat_w",
    "e1x_w",
    "e1y_w",
    "e1z_w",
    "e2x_w",
    "e2y_w",
    "e2z_w",
    "e3x_w",
    "e3y_w",
    "e3z_w",
    "e4x_w",
    "e4y_w",
    "e4z_w",
    "jv_base1",
    "jv_shoulder1",
    "jv_elbow1",
    "jv_base2",
    "jv_shoulder2",
    "jv_elbow2",
    "jv_base3",
    "jv_shoulder3",
    "jv_elbow3",
    "jv_base4",
    "jv_shoulder4",
    "jv_elbow4",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_clip_patterns(values: list[str] | None) -> list[str]:
    if not values:
        return ["*"]
    patterns: list[str] = []
    for value in values:
        patterns.extend(part.strip() for part in value.split(",") if part.strip())
    return patterns or ["*"]


def matches_any(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def list_remote_clips() -> tuple[list[str], list[str]]:
    from huggingface_hub import list_repo_files

    files = list_repo_files(REPO_ID, repo_type=REPO_TYPE)
    clips = sorted(
        {path.split("/")[1] for path in files if path.startswith("data/") and path.endswith("/motion.npy")}
    )
    return clips, files


def list_local_clips(dataset_dir: Path) -> list[str]:
    data_dir = dataset_dir / "data"
    if not data_dir.is_dir():
        return []
    return sorted(path.name for path in data_dir.iterdir() if (path / "motion.npy").exists())


def download_reference(dataset_dir: Path, patterns: list[str], force: bool = False) -> list[str]:
    from huggingface_hub import hf_hub_download

    clips, files = list_remote_clips()
    selected_clips = [clip for clip in clips if matches_any(clip, patterns)]
    if not selected_clips:
        raise ValueError(f"No Kine2Go clips matched patterns: {patterns}")

    wanted: list[str] = []
    wanted.extend(path for path in files if path in ROOT_FILES or path.startswith("LICENSES/"))
    selected = set(selected_clips)
    for path in files:
        parts = path.split("/")
        if len(parts) == 3 and parts[0] == "data" and parts[1] in selected and parts[2] in CLIP_FILES:
            wanted.append(path)

    dataset_dir.mkdir(parents=True, exist_ok=True)
    for idx, filename in enumerate(sorted(set(wanted)), start=1):
        local_file = dataset_dir / filename
        if local_file.exists() and not force:
            continue
        print(f"[download {idx:03d}/{len(wanted):03d}] {filename}")
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            filename=filename,
            local_dir=dataset_dir,
        )

    return selected_clips


def normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return q / norms


def quat_conjugate_wxyz(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def quat_rotate_inverse_wxyz(q: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    q = normalize_quat_wxyz(q)
    zeros = np.zeros((*vectors.shape[:-1], 1), dtype=np.float64)
    pure = np.concatenate([zeros, vectors], axis=-1)
    q_inv = quat_conjugate_wxyz(q)
    return quat_mul_wxyz(quat_mul_wxyz(q_inv, pure), q)[..., 1:]


def yaw_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = normalize_quat_wxyz(q)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def rotate_yaw_inverse(vectors: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    c = np.cos(yaw)
    s = np.sin(yaw)
    out = vectors.copy()
    out[..., 0] = c * vectors[..., 0] + s * vectors[..., 1]
    out[..., 1] = -s * vectors[..., 0] + c * vectors[..., 1]
    return out


def slerp_quat_wxyz(q0: np.ndarray, q1: np.ndarray, blend: np.ndarray) -> np.ndarray:
    q0 = normalize_quat_wxyz(q0)
    q1 = normalize_quat_wxyz(q1)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.abs(dot)

    close = dot > 0.9995
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * blend[..., None]
    sin_theta = np.sin(theta)
    s0 = np.cos(theta) - dot * sin_theta / np.maximum(sin_theta_0, 1e-12)
    s1 = sin_theta / np.maximum(sin_theta_0, 1e-12)
    out = s0 * q0 + s1 * q1

    lerp = q0 + blend[..., None] * (q1 - q0)
    out = np.where(close, lerp, out)
    return normalize_quat_wxyz(out)


def resample_linear(values: np.ndarray, src_times: np.ndarray, dst_times: np.ndarray) -> np.ndarray:
    flat = values.reshape(values.shape[0], -1)
    cols = [np.interp(dst_times, src_times, flat[:, i]) for i in range(flat.shape[1])]
    return np.stack(cols, axis=-1).reshape((len(dst_times),) + values.shape[1:])


def resample_quat(values: np.ndarray, src_times: np.ndarray, dst_times: np.ndarray) -> np.ndarray:
    upper = len(src_times) - 1
    idx0 = np.searchsorted(src_times, dst_times, side="right") - 1
    idx0 = np.clip(idx0, 0, upper)
    idx1 = np.clip(idx0 + 1, 0, upper)
    denom = np.maximum(src_times[idx1] - src_times[idx0], 1e-12)
    blend = (dst_times - src_times[idx0]) / denom
    return slerp_quat_wxyz(values[idx0], values[idx1], blend)


def resample_motion(motion: np.ndarray, in_fps: float, out_fps: float) -> np.ndarray:
    if abs(in_fps - out_fps) < 1e-9:
        return motion.astype(np.float64, copy=True)

    src_times = np.arange(motion.shape[0], dtype=np.float64) / in_fps
    duration = src_times[-1]
    n_out = int(round(duration * out_fps)) + 1
    dst_times = np.arange(n_out, dtype=np.float64) / out_fps
    dst_times[-1] = min(dst_times[-1], duration)

    out = np.zeros((n_out, motion.shape[1]), dtype=np.float64)
    out[:, 0:51] = resample_linear(motion[:, 0:51], src_times, dst_times)
    out[:, 51:55] = resample_quat(motion[:, 51:55], src_times, dst_times)
    out[:, 55:61] = resample_linear(motion[:, 55:61], src_times, dst_times)
    return out


def detect_joint_layout(joint_pos: np.ndarray) -> str:
    """Detect whether 12 joints are triplets or grouped by joint type.

    Kine2Go's public metadata says triplets, but current reference files are
    numerically consistent with grouped-by-joint-type ordering:
    hip[FR,FL,RR,RL], thigh[FR,FL,RR,RL], calf[FR,FL,RR,RL].
    """

    def score(layout: str) -> float:
        if layout == "triplet":
            hip = joint_pos[:, [0, 3, 6, 9]]
            thigh = joint_pos[:, [1, 4, 7, 10]]
            calf = joint_pos[:, [2, 5, 8, 11]]
        elif layout == "joint_type":
            hip = joint_pos[:, 0:4]
            thigh = joint_pos[:, 4:8]
            calf = joint_pos[:, 8:12]
        else:
            raise ValueError(layout)
        hip_penalty = np.mean(np.maximum(np.abs(hip) - 0.7, 0.0))
        thigh_penalty = np.mean(np.maximum(0.15 - thigh, 0.0))
        calf_penalty = np.mean(np.maximum(calf + 0.4, 0.0))
        return float(hip_penalty + thigh_penalty + calf_penalty)

    triplet_score = score("triplet")
    joint_type_score = score("joint_type")
    return "joint_type" if joint_type_score < triplet_score else "triplet"


def reorder_joints_to_apex(values: np.ndarray, layout: str) -> np.ndarray:
    if layout == "auto":
        layout = detect_joint_layout(values)

    if layout == "triplet":
        kine = values.reshape(values.shape[0], 4, 3)
        leg_to_idx = {leg: idx for idx, leg in enumerate(KINE_LEG_ORDER)}
        apex = np.stack([kine[:, leg_to_idx[leg], :] for leg in APEX_LEG_ORDER], axis=1)
        return apex.reshape(values.shape[0], 12)

    if layout == "joint_type":
        hip = values[:, 0:4]
        thigh = values[:, 4:8]
        calf = values[:, 8:12]
        leg_to_idx = {leg: idx for idx, leg in enumerate(KINE_LEG_ORDER)}
        legs = []
        for leg in APEX_LEG_ORDER:
            idx = leg_to_idx[leg]
            legs.append(np.stack([hip[:, idx], thigh[:, idx], calf[:, idx]], axis=-1))
        return np.stack(legs, axis=1).reshape(values.shape[0], 12)

    raise ValueError(f"Unknown joint layout: {layout}")


def reorder_feet_to_apex(values: np.ndarray) -> np.ndarray:
    feet = values.reshape(values.shape[0], 4, 3)
    # Kine2Go feet: FL, RL, FR, RR. APEX/Isaac Go2 feet: FL, FR, RL, RR.
    return feet[:, [0, 2, 1, 3], :].reshape(values.shape[0], 12)


def convert_motion_to_apex_df(
    motion: np.ndarray,
    input_fps: float,
    output_fps: float,
    joint_layout: str = "auto",
    center_xy: bool = True,
    command_mode: str = "mean",
) -> tuple[pd.DataFrame, dict]:
    if motion.ndim != 2 or motion.shape[1] != 61:
        raise ValueError(f"Expected motion shape (T, 61), got {motion.shape}")

    motion = resample_motion(motion, input_fps, output_fps)

    raw_joint_pos = motion[:, 6:18]
    raw_joint_vel = motion[:, 24:36]
    detected_layout = detect_joint_layout(raw_joint_pos) if joint_layout == "auto" else joint_layout

    joint_pos = reorder_joints_to_apex(raw_joint_pos, detected_layout)
    joint_vel = reorder_joints_to_apex(raw_joint_vel, detected_layout)

    base_pos = motion[:, 48:51].copy()
    base_quat_wxyz = normalize_quat_wxyz(motion[:, 51:55])
    base_lin_vel_body = quat_rotate_inverse_wxyz(base_quat_wxyz, motion[:, 55:58])
    base_ang_vel_body = quat_rotate_inverse_wxyz(base_quat_wxyz, motion[:, 58:61])

    feet_world = reorder_feet_to_apex(motion[:, 36:48]).reshape(-1, 4, 3)
    if center_xy:
        xy0 = base_pos[0, 0:2].copy()
        base_pos[:, 0:2] -= xy0
        feet_world[:, :, 0:2] -= xy0.reshape(1, 1, 2)

    feet_delta = feet_world - base_pos[:, None, :]
    feet_body_yaw = rotate_yaw_inverse(feet_delta, yaw_from_quat_wxyz(base_quat_wxyz)[:, None])

    base_quat_xyzw = base_quat_wxyz[:, [1, 2, 3, 0]]

    if command_mode == "mean":
        command = np.repeat(
            np.mean(
                np.concatenate([base_lin_vel_body[:, 0:2], base_ang_vel_body[:, 2:3]], axis=1),
                axis=0,
                keepdims=True,
            ),
            repeats=motion.shape[0],
            axis=0,
        )
    elif command_mode == "instantaneous":
        command = np.concatenate([base_lin_vel_body[:, 0:2], base_ang_vel_body[:, 2:3]], axis=1)
    else:
        raise ValueError(f"Unknown command mode: {command_mode}")

    data = np.concatenate(
        [
            base_lin_vel_body,
            base_ang_vel_body,
            joint_pos,
            command,
            base_pos[:, 2:3],
            feet_body_yaw.reshape(-1, 12),
            base_pos[:, 0:2],
            base_quat_xyzw,
            feet_world.reshape(-1, 12),
            joint_vel,
        ],
        axis=1,
    )
    df = pd.DataFrame(data, columns=APEX_COLUMNS)
    summary = {
        "frames": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "input_fps": float(input_fps),
        "output_fps": float(output_fps),
        "duration_s": float((df.shape[0] - 1) / output_fps),
        "joint_layout": detected_layout,
        "height_min": float(df["height"].min()),
        "height_mean": float(df["height"].mean()),
        "height_max": float(df["height"].max()),
        "mean_abs_vx": float(np.mean(np.abs(df["vx"].to_numpy()))),
        "mean_abs_wz": float(np.mean(np.abs(df["wz"].to_numpy()))),
        "command_mode": command_mode,
        "command_vx": float(df["com_vx"].iloc[0]),
        "command_vy": float(df["com_vy"].iloc[0]),
        "command_wz": float(df["com_wz"].iloc[0]),
    }
    return df, summary


def read_clip_fps(clip_dir: Path) -> float:
    clip_json = clip_dir / "clip.json"
    if not clip_json.exists():
        return INPUT_FPS
    try:
        return float(json.loads(clip_json.read_text(encoding="utf-8")).get("fps", INPUT_FPS))
    except Exception:
        return INPUT_FPS


def convert_clips(
    dataset_dir: Path,
    output_dir: Path,
    patterns: list[str],
    output_fps: float,
    joint_layout: str,
    center_xy: bool,
    command_mode: str,
) -> pd.DataFrame:
    clips = [clip for clip in list_local_clips(dataset_dir) if matches_any(clip, patterns)]
    if not clips:
        raise ValueError(f"No local Kine2Go clips found in {dataset_dir} for patterns: {patterns}")

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for clip in clips:
        clip_dir = dataset_dir / "data" / clip
        motion_path = clip_dir / "motion.npy"
        motion = np.load(motion_path)
        input_fps = read_clip_fps(clip_dir)
        df, summary = convert_motion_to_apex_df(
            motion,
            input_fps=input_fps,
            output_fps=output_fps,
            joint_layout=joint_layout,
            center_xy=center_xy,
            command_mode=command_mode,
        )
        out_path = output_dir / f"{clip}.csv"
        df.to_csv(out_path, index=False, float_format="%.8f")

        clip_meta = {}
        clip_json = clip_dir / "clip.json"
        if clip_json.exists():
            clip_meta = json.loads(clip_json.read_text(encoding="utf-8"))
        summary.update(
            {
                "clip": clip,
                "source_frames": int(motion.shape[0]),
                "csv": str(out_path),
                "license": clip_meta.get("license", ""),
                "tags": ",".join(clip_meta.get("tags", [])),
            }
        )
        summaries.append(summary)
        print(
            f"[convert] {clip}: {motion.shape[0]} -> {df.shape[0]} frames, "
            f"layout={summary['joint_layout']}, csv={out_path}"
        )

    summary_df = pd.DataFrame(summaries)
    summary_df = summary_df[
        [
            "clip",
            "source_frames",
            "frames",
            "input_fps",
            "output_fps",
            "duration_s",
            "joint_layout",
            "height_min",
            "height_mean",
            "height_max",
            "mean_abs_vx",
            "mean_abs_wz",
            "command_mode",
            "command_vx",
            "command_vy",
            "command_wz",
            "license",
            "tags",
            "csv",
        ]
    ]
    summary_path = output_dir / "conversion_summary.csv"
    summary_df.to_csv(summary_path, index=False, float_format="%.6f")
    print(f"[summary] {summary_path}")
    return summary_df


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=root / "datasets" / "kine2go_refs")
    parser.add_argument("--output-dir", type=Path, default=root / "imitation_data" / "kine2go")
    parser.add_argument(
        "--clips",
        nargs="*",
        help="Clip names or glob patterns. Default: all clips. Example: ai4_dog_pace ai4_dog_walk_*",
    )
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--convert-only", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--output-fps", type=float, default=DEFAULT_OUTPUT_FPS)
    parser.add_argument(
        "--joint-layout",
        choices=("auto", "joint_type", "triplet"),
        default="auto",
        help="Use auto for current Kine2Go reference files.",
    )
    parser.add_argument(
        "--no-center-xy",
        action="store_true",
        help="Keep world x/y instead of shifting each clip to start near x=y=0.",
    )
    parser.add_argument(
        "--command-mode",
        choices=("mean", "instantaneous"),
        default="mean",
        help=(
            "How to fill APEX columns com_vx/com_vy/com_wz. "
            "Use mean for APEX-style per-clip command defaults."
        ),
    )
    args = parser.parse_args()

    patterns = parse_clip_patterns(args.clips)
    do_download = not args.convert_only
    do_convert = not args.download_only

    if do_download:
        selected = download_reference(args.dataset_dir, patterns, force=args.force_download)
        print(f"[download] prepared {len(selected)} reference clips in {args.dataset_dir}")

    if do_convert:
        convert_clips(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            patterns=patterns,
            output_fps=args.output_fps,
            joint_layout=args.joint_layout,
            center_xy=not args.no_center_xy,
            command_mode=args.command_mode,
        )


if __name__ == "__main__":
    main()
