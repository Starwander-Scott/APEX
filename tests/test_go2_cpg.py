import importlib.util
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "legged_gym" / "envs" / "go2" / "cpg.py"


def load_module():
    spec = importlib.util.spec_from_file_location("go2_cpg", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Go2CPGTests(unittest.TestCase):
    def test_phase_offsets_encode_common_quadruped_gaits(self):
        module = load_module()

        self.assertTrue(torch.allclose(module.phase_offsets("trot"), torch.tensor([0.0, 0.5, 0.5, 0.0])))
        self.assertTrue(torch.allclose(module.phase_offsets("pace"), torch.tensor([0.0, 0.5, 0.0, 0.5])))
        self.assertTrue(torch.allclose(module.phase_offsets("bound"), torch.tensor([0.0, 0.0, 0.5, 0.5])))
        self.assertEqual(torch.unique(module.phase_offsets("walk")).numel(), 4)

    def test_zero_amplitude_cpg_returns_default_pose(self):
        module = load_module()
        default_pose = torch.linspace(-0.2, 0.2, 12)
        cpg = module.Go2JointCPG(num_envs=3, default_dof_pos=default_pose, device="cpu")

        targets = cpg.step(
            gait="trot",
            frequency_hz=torch.ones(3),
            dt=0.02,
            amplitudes=module.CPGAmplitudes(hip=0.0, thigh=0.0, calf=0.0),
        )

        self.assertEqual(targets.shape, (3, 12))
        self.assertTrue(torch.allclose(targets, default_pose.repeat(3, 1)))

    def test_cpg_generates_finite_joint_targets_for_multiple_gaits(self):
        module = load_module()
        default_pose = torch.zeros(12)
        cpg = module.Go2JointCPG(num_envs=2, default_dof_pos=default_pose, device="cpu")

        for gait in ("walk", "trot", "pace", "bound", "canter"):
            targets = cpg.step(gait=gait, frequency_hz=torch.tensor([1.0, 1.5]), dt=0.05)
            self.assertEqual(targets.shape, (2, 12))
            self.assertTrue(torch.isfinite(targets).all())
            self.assertLessEqual(float(targets.abs().max()), 0.8)


if __name__ == "__main__":
    unittest.main()
