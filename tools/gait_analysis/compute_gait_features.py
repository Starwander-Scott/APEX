"""Compute gait feature summaries for APEX imitation motion CSV files."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FOOT_IDS = (1, 2, 3, 4)
JOINT_COLUMNS = (
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
)

# Broad CSV-space sanity bounds. These are intentionally conservative because
# several files are animal/Kine2Go retargets rather than hardware commands.
DEFAULT_JOINT_LIMITS = {name: (-4.0, 4.0) for name in JOINT_COLUMNS}


def finite_or_nan(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan
    return float(values.mean())


def percentile_or_nan(values: np.ndarray, percentile: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan
    return float(np.percentile(values, percentile))


def column_array(df: pd.DataFrame, name: str) -> np.ndarray | None:
    if name not in df.columns:
        return None
    return df[name].to_numpy(dtype=float)


def safe_mean(df: pd.DataFrame, name: str) -> float:
    values = column_array(df, name)
    return math.nan if values is None else finite_or_nan(values)


def safe_std(df: pd.DataFrame, name: str) -> float:
    values = column_array(df, name)
    if values is None:
        return math.nan
    values = values[np.isfinite(values)]
    return math.nan if values.size == 0 else float(values.std(ddof=0))


def safe_min(df: pd.DataFrame, name: str) -> float:
    values = column_array(df, name)
    if values is None:
        return math.nan
    values = values[np.isfinite(values)]
    return math.nan if values.size == 0 else float(values.min())


def safe_max(df: pd.DataFrame, name: str) -> float:
    values = column_array(df, name)
    if values is None:
        return math.nan
    values = values[np.isfinite(values)]
    return math.nan if values.size == 0 else float(values.max())


def quaternion_roll_pitch(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    required = ("quat_x", "quat_y", "quat_z", "quat_w")
    if not all(name in df.columns for name in required):
        empty = np.array([], dtype=float)
        return empty, empty

    x = df["quat_x"].to_numpy(dtype=float)
    y = df["quat_y"].to_numpy(dtype=float)
    z = df["quat_z"].to_numpy(dtype=float)
    w = df["quat_w"].to_numpy(dtype=float)

    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sin_pitch, -1.0, 1.0))
    return roll, pitch


def count_peaks(values: np.ndarray, fps: float, min_separation_s: float = 0.25) -> int:
    values = np.asarray(values, dtype=float)
    if values.size < 3:
        return 0
    finite_mask = np.isfinite(values)
    if finite_mask.sum() < 3:
        return 0

    filled = values.copy()
    median = float(np.nanmedian(filled))
    filled[~finite_mask] = median
    threshold = float(np.percentile(filled, 60.0))
    min_separation = max(1, int(round(min_separation_s * fps)))
    peaks: list[int] = []

    for idx in range(1, len(filled) - 1):
        if filled[idx] <= threshold:
            continue
        if filled[idx] >= filled[idx - 1] and filled[idx] > filled[idx + 1]:
            if peaks and idx - peaks[-1] < min_separation:
                if filled[idx] > filled[peaks[-1]]:
                    peaks[-1] = idx
            else:
                peaks.append(idx)
    return len(peaks)


def compute_foot_features(df: pd.DataFrame, fps: float) -> dict[str, float]:
    clearances: list[float] = []
    duty_factors: list[float] = []
    step_frequencies: list[float] = []
    slip_speeds: list[float] = []
    duration_s = len(df) / fps if fps > 0 else math.nan

    for foot_id in FOOT_IDS:
        z_name = f"e{foot_id}z"
        z = column_array(df, z_name)
        if z is None:
            continue

        z_low = percentile_or_nan(z, 5.0)
        z_high = percentile_or_nan(z, 95.0)
        if math.isfinite(z_low) and math.isfinite(z_high):
            clearances.append(max(0.0, z_high - z_low))

        contact_threshold = percentile_or_nan(z, 30.0)
        if math.isfinite(contact_threshold):
            contact = z <= contact_threshold
            duty_factors.append(float(np.mean(contact)))
        else:
            contact = np.zeros_like(z, dtype=bool)

        peaks = count_peaks(z, fps=fps)
        if math.isfinite(duration_s) and duration_s > 0:
            step_frequencies.append(peaks / duration_s)

        x_w = column_array(df, f"e{foot_id}x_w")
        y_w = column_array(df, f"e{foot_id}y_w")
        if x_w is not None and y_w is not None and len(x_w) > 1:
            vx = np.gradient(x_w) * fps
            vy = np.gradient(y_w) * fps
            speed = np.sqrt(vx * vx + vy * vy)
            if contact.any():
                slip_speeds.append(finite_or_nan(speed[contact]))

    return {
        "foot_clearance_mean": finite_or_nan(np.array(clearances)),
        "foot_clearance_max": percentile_or_nan(np.array(clearances), 100.0),
        "contact_duty_factor_mean": finite_or_nan(np.array(duty_factors)),
        "step_frequency_hz": finite_or_nan(np.array(step_frequencies)),
        "foot_slip_speed_mean": finite_or_nan(np.array(slip_speeds)),
    }


def joint_limit_violation_count(df: pd.DataFrame) -> int:
    count = 0
    for name, (lower, upper) in DEFAULT_JOINT_LIMITS.items():
        values = column_array(df, name)
        if values is None:
            continue
        count += int(np.sum((values < lower) | (values > upper)))
    return count


def infer_gait_family(motion_name: str) -> str:
    name = motion_name.lower()
    keyword_map = (
        ("canter", "canter"),
        ("trot", "trot"),
        ("pace", "pace"),
        ("bound", "bound"),
        ("run", "run"),
        ("walk", "walk"),
        ("crawl", "crawl"),
        ("jump", "jump"),
        ("hopturn", "turn"),
        ("turn", "turn"),
        ("sidestep", "lateral"),
        ("strafe", "lateral"),
    )
    for keyword, family in keyword_map:
        if keyword in name:
            return family
    return "unknown"


def compute_motion_features(csv_path: Path | str, fps: float = 50.0) -> dict[str, float | str | int]:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    duration_s = len(df) / fps if fps > 0 else math.nan

    vx = column_array(df, "vx")
    vy = column_array(df, "vy")
    if vx is not None and vy is not None:
        speed_abs = np.sqrt(vx * vx + vy * vy)
    elif vx is not None:
        speed_abs = np.abs(vx)
    else:
        speed_abs = np.array([], dtype=float)

    wz = column_array(df, "wz")
    roll, pitch = quaternion_roll_pitch(df)
    foot_features = compute_foot_features(df, fps=fps)

    features: dict[str, float | str | int] = {
        "motion": csv_path.stem,
        "source_dir": csv_path.parent.name,
        "gait_family": infer_gait_family(csv_path.stem),
        "path": str(csv_path),
        "frames": int(len(df)),
        "duration_s": float(duration_s),
        "speed_abs_mean": finite_or_nan(speed_abs),
        "speed_abs_p90": percentile_or_nan(speed_abs, 90.0),
        "forward_speed_mean": safe_mean(df, "vx"),
        "lateral_speed_mean": safe_mean(df, "vy"),
        "yaw_rate_abs_mean": math.nan if wz is None else finite_or_nan(np.abs(wz)),
        "body_height_mean": safe_mean(df, "height"),
        "body_height_min": safe_min(df, "height"),
        "body_height_max": safe_max(df, "height"),
        "body_bounce_std": safe_std(df, "height"),
        "roll_abs_mean": finite_or_nan(np.abs(roll)),
        "roll_std": math.nan if roll.size == 0 else float(np.std(roll, ddof=0)),
        "pitch_abs_mean": finite_or_nan(np.abs(pitch)),
        "pitch_std": math.nan if pitch.size == 0 else float(np.std(pitch, ddof=0)),
        "joint_limit_violation_count": joint_limit_violation_count(df),
    }
    features.update(foot_features)
    return features


def discover_motion_csvs(paths: Iterable[Path]) -> list[Path]:
    csvs: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix.lower() == ".csv":
            csvs.append(path)
        elif path.is_dir():
            csvs.extend(path.rglob("*.csv"))

    return sorted(
        path for path in csvs
        if path.name.lower() not in {"conversion_summary.csv", "summary.csv", "per_motion_features.csv"}
    )


def write_report(summary: pd.DataFrame, output_dir: Path) -> Path:
    report_path = output_dir / "00_gait_feature_report.md"
    family_counts = summary["gait_family"].value_counts().to_dict() if "gait_family" in summary else {}
    columns = [
        "motion",
        "source_dir",
        "gait_family",
        "speed_abs_mean",
        "step_frequency_hz",
        "foot_clearance_mean",
        "body_height_mean",
        "body_bounce_std",
        "contact_duty_factor_mean",
        "foot_slip_speed_mean",
        "joint_limit_violation_count",
    ]
    available_columns = [column for column in columns if column in summary.columns]
    ranked = summary.sort_values(
        by=["gait_family", "speed_abs_mean", "foot_clearance_mean"],
        ascending=[True, False, False],
    )[available_columns]
    family_summary = summary.groupby("gait_family", dropna=False).agg(
        motions=("motion", "count"),
        speed_abs_mean=("speed_abs_mean", "mean"),
        step_frequency_hz=("step_frequency_hz", "mean"),
        foot_clearance_mean=("foot_clearance_mean", "mean"),
        body_bounce_std=("body_bounce_std", "mean"),
    ).reset_index().sort_values("gait_family")

    lines = [
        "# 多步态特征体检报告",
        "",
        "本报告由 APEX imitation motion CSV 自动生成，用于在正式训练前判断不同 gait family 的运动特征差异。",
        "",
        "## 数据集概况",
        "",
        f"- 分析 motion 数量：{len(summary)}",
        f"- gait family 计数：{family_counts}",
        "",
        "## 第一版多步态建模建议",
        "",
        "- 先不要引入情绪标签，优先把 walk / trot / pace / canter / bound 等 gait 的周期结构做对。",
        "- CPG 负责提供相位、频率、基础关节轨迹；RL 后续只学习 residual 或少量 CPG 参数修正。",
        "- `gait_family` 来自 motion 文件名关键词，只用于整理数据，不代表已完成接触序列识别。",
        "",
        "## gait family 均值",
        "",
        family_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 全量 motion 指标",
        "",
        ranked.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 注意事项",
        "",
        "- contact pattern 目前由足端高度低位区间近似得到，因为 CSV 没有显式触地标签。",
        "- foot slip 只在存在 world foot position 列时计算；缺失时记为 `nan`。",
        "- joint limit violation 目前使用保守的 CSV 空间宽限位，只作为第一轮数据健康检查。",
        "- 后续做真实 Go2 安全边界时，应切换到 URDF 精确关节限位。",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def write_outputs(features: list[dict[str, float | str | int]], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(features)
    summary_path = output_dir / "summary.csv"
    per_motion_path = output_dir / "per_motion_features.csv"
    summary.to_csv(summary_path, index=False)
    summary.to_csv(per_motion_path, index=False)
    report_path = write_report(summary, output_dir)
    return summary_path, report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        nargs="*",
        type=Path,
        default=[Path("imitation_data/kine2go"), Path("imitation_data/animal_mocap")],
        help="CSV files or directories to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/gait_features"),
        help="Directory for generated CSV and Markdown reports.",
    )
    parser.add_argument("--fps", type=float, default=50.0, help="Motion sample rate in Hz.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_paths = discover_motion_csvs(args.input)
    if not csv_paths:
        raise SystemExit("No motion CSV files found.")

    features = [compute_motion_features(path, fps=args.fps) for path in csv_paths]
    summary_path, report_path = write_outputs(features, args.output_dir)
    print(f"Analyzed {len(features)} motions")
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
