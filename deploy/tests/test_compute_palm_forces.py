import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from compute_palm_forces import FORCE_COLUMNS, JTS_FORCE_COLUMNS, PalmForceEstimator, process_csv
from data_collector.pink_ik import DCP_ACTIVE_JOINT_NAMES
from helper.eir_pink_ik_visualizer import DEFAULT_CONFIG_PATH, _load_eir_settings


class PalmForceCSVTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "joint_states.csv"
        input_patch = patch("builtins.input", return_value="n")
        self.keyboard = input_patch.start()
        self.addCleanup(input_patch.stop)

    def write_rows(self, rows):
        # Reverse the order to catch accidental reliance on numeric column offsets.
        with self.path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(reversed(rows[0])))
            writer.writeheader()
            writer.writerows(rows)

    def raw_row(self):
        return {
            "elapsed_time_s": "0.0200", "command_example_deg": "12.3400",
            **{f"q_{name}_deg": str(i) for i, name in enumerate(DCP_ACTIVE_JOINT_NAMES)},
            **{f"tau_ext_{name}": str(i / 10) for i, name in enumerate(DCP_ACTIVE_JOINT_NAMES)},
        }

    def test_append_preserve_and_repeat_without_recomputing(self):
        original = self.raw_row()
        self.write_rows([original, {**original, "elapsed_time_s": "0.0400"}])
        with patch("compute_palm_forces._load_eir_settings", return_value={"kinematics_urdf_path": "unused"}), \
                patch("compute_palm_forces.PalmForceEstimator") as factory, patch("builtins.print"):
            factory.return_value.estimate.side_effect = [np.array([5., 10.]), np.array([6., np.nan])]
            plot = process_csv(self.path, torque_source="tau_ext")
        np.testing.assert_equal(factory.return_value.estimate.call_args.args[0], np.arange(18))
        np.testing.assert_allclose(factory.return_value.estimate.call_args.args[1], np.arange(18) / 10)
        with self.path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        for name, value in original.items():
            self.assertEqual(rows[0][name], value)
        self.assertEqual([rows[0][name] for name in FORCE_COLUMNS], ["5.0", "10.0", "7.5"])
        self.assertEqual([rows[1][name] for name in FORCE_COLUMNS], ["6.0", "", ""])
        self.assertEqual(plot.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        saved = self.path.read_bytes()
        with patch("compute_palm_forces.PalmForceEstimator") as factory, patch("builtins.print"):
            process_csv(self.path, torque_source="tau_ext")
        factory.assert_not_called()
        self.assertEqual(self.path.read_bytes(), saved)
        self.keyboard.assert_called_once()

    def test_yes_replaces_all_force_columns_and_preserves_raw_data(self):
        original = self.raw_row()
        self.write_rows([{**original, **dict.fromkeys(FORCE_COLUMNS, "999")}])
        self.keyboard.return_value = " Y "
        with patch("compute_palm_forces._load_eir_settings", return_value={"kinematics_urdf_path": "unused"}), \
                patch("compute_palm_forces.PalmForceEstimator") as factory, patch("builtins.print"):
            factory.return_value.estimate.return_value = np.array([5., 10.])
            plot = process_csv(self.path, torque_source="tau_ext")
        factory.return_value.estimate.assert_called_once()
        with self.path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            saved = next(reader)
            for name in FORCE_COLUMNS:
                self.assertEqual(reader.fieldnames.count(name), 1)
        self.assertEqual(saved, {**original, **dict(zip(FORCE_COLUMNS, ["5.0", "10.0", "7.5"]))})
        self.assertTrue(plot.is_file())

    def test_failed_recomputation_keeps_existing_csv(self):
        self.write_rows([{"elapsed_time_s": "0", **dict.fromkeys(FORCE_COLUMNS, "999")}])
        saved = self.path.read_bytes()
        self.keyboard.return_value = "y"
        with self.assertRaisesRegex(ValueError, "Record a new run"):
            process_csv(self.path, torque_source="tau_ext")
        self.assertEqual(self.path.read_bytes(), saved)

    def test_add_only_missing_mean_without_torque_data(self):
        row = {"elapsed_time_s": "1.0", FORCE_COLUMNS[0]: "2.000", FORCE_COLUMNS[1]: "4.000"}
        self.write_rows([row])
        with patch("compute_palm_forces.PalmForceEstimator") as factory, patch("builtins.print"):
            process_csv(self.path, torque_source="tau_ext")
        factory.assert_not_called()
        with self.path.open(newline="") as stream:
            saved = next(csv.DictReader(stream))
        self.assertEqual(saved, {**row, FORCE_COLUMNS[2]: "3.0"})

    def test_old_log_is_rejected_without_changes(self):
        self.write_rows([{"elapsed_time_s": "0", "error_norm_deg": "2.0"}])
        saved = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "Record a new run"):
            process_csv(self.path, torque_source="tau_ext")
        self.assertEqual(self.path.read_bytes(), saved)

    def test_unavailable_torques_are_not_saved_as_zero_force(self):
        self.write_rows([self.raw_row()])
        saved = self.path.read_bytes()
        with patch("compute_palm_forces._load_eir_settings", return_value={"kinematics_urdf_path": "unused"}), \
                patch("compute_palm_forces.PalmForceEstimator") as factory:
            factory.return_value.estimate.return_value = np.array([np.nan, np.nan])
            with self.assertRaisesRegex(ValueError, "no samples with usable tau_ext"):
                process_csv(self.path, torque_source="tau_ext")
        self.assertEqual(self.path.read_bytes(), saved)

    def test_jts_gravity_bias_and_source_columns_are_independent(self):
        # Negative sensor convention: raw=-[gravity+bias+contact].
        row = {"elapsed_time_s": "0.02",
               **{f"q_{name}_deg": "0" for name in DCP_ACTIVE_JOINT_NAMES},
               **{f"tau_jts_{name}": "-16" for name in DCP_ACTIVE_JOINT_NAMES},
               **dict.fromkeys(FORCE_COLUMNS, "99")}
        self.write_rows([row])
        bias_path = self.path.with_name("no_contact.csv")
        with bias_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow({name: "-12" if name.startswith("tau_jts_") else value
                             for name, value in row.items()})
        with patch("compute_palm_forces._load_eir_settings", return_value={"kinematics_urdf_path": "unused"}), \
                patch("compute_palm_forces.PalmForceEstimator") as factory, patch("builtins.print"):
            factory.return_value.gravity.return_value = np.full(18, 10.)
            factory.return_value.estimate.return_value = np.array([5., 10.])
            plot = process_csv(self.path, bias_csv=bias_path, sensor_sign=-1)
        np.testing.assert_array_equal(factory.return_value.estimate.call_args.args[1][4:18], np.full(14, 4.))
        self.keyboard.assert_not_called()  # Existing tau_ext columns are a different method.
        with self.path.open(newline="") as stream:
            saved = next(csv.DictReader(stream))
        self.assertEqual(saved, {**row, **dict(zip(JTS_FORCE_COLUMNS, ["5.0", "10.0", "7.5"]))})
        self.assertTrue(plot.name.endswith("_tau_jts_gravity_force_norm.png"))

        # Recompute only the JTS columns, preserving the old tau_ext result.
        self.keyboard.return_value = "y"
        with patch("compute_palm_forces._load_eir_settings", return_value={"kinematics_urdf_path": "unused"}), \
                patch("compute_palm_forces.PalmForceEstimator") as factory, patch("builtins.print"):
            factory.return_value.gravity.return_value = np.full(18, 10.)
            factory.return_value.estimate.return_value = np.array([2., 4.])
            process_csv(self.path)
        self.keyboard.assert_called_once()
        with self.path.open(newline="") as stream:
            saved = next(csv.DictReader(stream))
        self.assertEqual(saved, {**row, **dict(zip(JTS_FORCE_COLUMNS, ["2.0", "4.0", "3.0"]))})


class PalmForceKinematicsTests(unittest.TestCase):
    def test_known_tcp_wrench_recovery_and_missing_arm(self):
        if not DEFAULT_CONFIG_PATH.is_file():
            self.skipTest("EIR model/configuration is not installed")
        try:
            import pinocchio as pin
        except ImportError:
            self.skipTest("Pinocchio is not installed")
        estimator = PalmForceEstimator(_load_eir_settings(DEFAULT_CONFIG_PATH)["kinematics_urdf_path"])
        q_deg = np.array([10., -5., 7., 2., 30., -20., 15., -65., 25., 12., -8.,
                          -30., 20., -15., -65., -25., -12., 8.])
        q = estimator.neutral.copy()
        q[estimator.q_indices] = np.deg2rad(q_deg)
        data = estimator.model.createData()
        pin.computeJointJacobians(estimator.model, data, q)
        pin.updateFramePlacements(estimator.model, data)
        tau = np.full(18, np.nan)
        for arm, (frame, indices, velocity_indices) in enumerate(estimator.arms):
            jacobian = pin.getFrameJacobian(
                estimator.model, data, frame, pin.LOCAL_WORLD_ALIGNED)[:, velocity_indices]
            self.assertEqual(np.linalg.matrix_rank(jacobian), 6)
            wrench = np.array([3., 4., 0., .1, -.2, .3]) * (arm + 1)
            tau[indices] = jacobian.T @ wrench
        np.testing.assert_allclose(estimator.estimate(q_deg, tau), [5., 10.], atol=1e-9)

        # Gravity must use velocity indices in DCP order, not the URDF's order.
        expected_gravity = pin.rnea(estimator.model, data, q,
                                    np.zeros(estimator.model.nv), np.zeros(estimator.model.nv))
        np.testing.assert_allclose(estimator.gravity(q_deg), expected_gravity[estimator.v_indices])
        gravity = estimator.gravity(q_deg)
        self.assertGreater(np.linalg.norm(gravity[4:18]), 1.)
        measured = gravity + tau
        np.testing.assert_allclose(estimator.estimate(q_deg, measured - estimator.gravity(q_deg)),
                                   [5., 10.], atol=1e-9)
        tau[4] = np.nan
        force = estimator.estimate(q_deg, tau)
        self.assertTrue(np.isnan(force[0]))
        self.assertAlmostEqual(force[1], 10.)
        np.testing.assert_array_equal(estimator.estimate(q_deg, np.zeros(18)), [0., 0.])


if __name__ == "__main__":
    unittest.main()
