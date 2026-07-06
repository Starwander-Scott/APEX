import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "tools" / "gait_analysis" / "compute_gait_features.py"


def load_module():
    spec = importlib.util.spec_from_file_location("compute_gait_features", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GaitAnalysisTests(unittest.TestCase):
    def write_motion_csv(self, directory: Path, name: str, fast: bool = False) -> Path:
        rows = []
        fps = 50.0
        frequency_hz = 2.0 if fast else 1.0
        speed = 1.2 if fast else 0.35
        clearance = 0.12 if fast else 0.04
        for i in range(100):
            phase = 2.0 * math.pi * frequency_hz * (i / fps)
            row = {
                "vx": speed,
                "vy": 0.0,
                "vz": 0.0,
                "wx": 0.0,
                "wy": 0.0,
                "wz": 0.05 if fast else 0.01,
                "com_vx": speed,
                "com_vy": 0.0,
                "com_wz": 0.0,
                "height": 0.32 + (0.015 if fast else 0.004) * math.sin(phase),
                "quat_x": 0.0,
                "quat_y": 0.0,
                "quat_z": 0.0,
                "quat_w": 1.0,
                "com_x": speed * i / fps,
                "com_y": 0.0,
            }
            for joint in (
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
            ):
                row[joint] = 0.0
            for leg in range(1, 5):
                leg_phase = phase + (math.pi if leg in (2, 3) else 0.0)
                z = -0.22 + max(0.0, math.sin(leg_phase)) * clearance
                row[f"e{leg}x"] = 0.2 if leg <= 2 else -0.2
                row[f"e{leg}y"] = 0.12 if leg % 2 else -0.12
                row[f"e{leg}z"] = z
                row[f"e{leg}x_w"] = row[f"e{leg}x"] + speed * i / fps
                row[f"e{leg}y_w"] = row[f"e{leg}y"]
                row[f"e{leg}z_w"] = z
            rows.append(row)

        csv_path = directory / name
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        return csv_path

    def test_compute_motion_features_extracts_core_gait_metrics(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = self.write_motion_csv(Path(temp_dir), "ai4_dog_walk_00.csv", fast=False)

            features = module.compute_motion_features(csv_path, fps=50.0)

        self.assertEqual(features["motion"], "ai4_dog_walk_00")
        self.assertEqual(features["gait_family"], "walk")
        self.assertNotIn("style_guess", features)
        self.assertAlmostEqual(features["speed_abs_mean"], 0.35, places=2)
        self.assertAlmostEqual(features["body_height_mean"], 0.32, places=2)
        self.assertGreater(features["body_bounce_std"], 0.0)
        self.assertGreater(features["foot_clearance_mean"], 0.02)
        self.assertGreater(features["step_frequency_hz"], 0.5)
        self.assertGreater(features["contact_duty_factor_mean"], 0.0)
        self.assertLess(features["contact_duty_factor_mean"], 1.0)
        self.assertEqual(features["joint_limit_violation_count"], 0)

    def test_gait_family_comes_from_motion_identity_not_emotion_style(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            walk_csv = self.write_motion_csv(Path(temp_dir), "ai4_dog_walk_00.csv", fast=False)
            trot_csv = self.write_motion_csv(Path(temp_dir), "go2_retarget_trot.csv", fast=True)

            walk = module.compute_motion_features(walk_csv, fps=50.0)
            trot = module.compute_motion_features(trot_csv, fps=50.0)

        self.assertEqual(walk["gait_family"], "walk")
        self.assertEqual(trot["gait_family"], "trot")
        self.assertGreater(trot["speed_abs_mean"], walk["speed_abs_mean"])
        self.assertGreater(trot["foot_clearance_mean"], walk["foot_clearance_mean"])
        self.assertGreater(trot["body_bounce_std"], walk["body_bounce_std"])


if __name__ == "__main__":
    unittest.main()
