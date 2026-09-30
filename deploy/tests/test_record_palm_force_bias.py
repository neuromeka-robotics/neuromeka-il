import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from compute_palm_forces import _sensor_bias
from record_palm_force_bias import record_bias


class BiasRecorderTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "bias.csv"
        self.now = 0.
        self.ready_at = None
        self.client = MagicMock()
        self.client.get_robot_data.return_value = {"op_state": 5}
        def robot_data():
            mode = 17 if self.client.start_teleop.called else 5
            if mode == 17 and self.ready_at is None:
                self.ready_at = self.now
            return {"op_state": mode}
        self.client.get_robot_data.side_effect = robot_data
        self.client.get_compliance_mode.return_value = {"enable": True}
        self.client.get_control_state.side_effect = lambda: self.state()
        for name in ("start_teleop", "stop_teleop", "movetelej_abs"):
            getattr(self.client, name).return_value = {"code": "0", "msg": ""}

    def state(self):
        return {"q": [10. + .01 * self.recording_time] * 18 + [0.] * 4,
                "qdot": [0.] * 22, "tau_jts": [3.] * 18}

    @property
    def recording_time(self):
        return self.now - (self.ready_at or 0.)

    def sleep(self, dt):
        self.now += dt

    def run_recording(self):
        with patch("record_palm_force_bias.time.monotonic", side_effect=lambda: self.now), \
                patch("record_palm_force_bias.time.sleep", side_effect=self.sleep), \
                patch("builtins.print"):
            return record_bias(self.client, self.path, duration=.3, settle=.2, rate=10.)

    def read(self):
        with self.path.open(newline="") as stream:
            return list(csv.DictReader(stream))

    def test_fixed_target_settling_and_bias_csv_compatibility(self):
        self.run_recording()
        rows = self.read()
        self.assertGreaterEqual(len(rows), 3)
        self.assertTrue(all(0. <= float(row["elapsed_time_s"]) < .3 for row in rows))
        for call in self.client.movetelej_abs.call_args_list:
            self.assertEqual(call.kwargs["jpos"], [10.] * 18 + [0.] * 4)
        self.assertEqual(float(rows[0]["elapsed_time_s"]), 0.)
        self.assertAlmostEqual(float(rows[0]["q_Joint_L2_L_deg"]), 10.002)
        self.assertEqual(rows[0]["tau_jts_Dummy_0"], "")
        self.assertEqual(rows[0]["tau_jts_count"], "18")
        self.assertEqual(self.client.stop_teleop.call_count, 2)
        self.client.set_compliance_mode.assert_not_called()
        estimator = MagicMock()
        estimator.gravity.return_value = np.ones(18)
        np.testing.assert_array_equal(_sensor_bias(estimator, self.path, 1)[4:18], np.full(14, 2.))

    def test_moving_samples_excluded(self):
        def state():
            result = self.state()
            if .29 < self.recording_time < .31:
                result["qdot"][4] = 1.
            return result
        self.client.get_control_state.side_effect = state
        self.run_recording()
        rows = self.read()
        self.assertGreaterEqual(len(rows), 2)
        self.assertTrue(all(float(row["qdot_Joint_L2_L_deg_s"]) == 0. for row in rows))
        self.assertTrue(all(abs(float(row["elapsed_time_s"]) - .1) > 1e-8 for row in rows))
        self.assertEqual(self.client.stop_teleop.call_count, 2)

    def test_drift_stops_and_preserves_previous_samples(self):
        def state():
            result = self.state()
            if self.recording_time > .29:
                result["q"][4] += 1.
            return result
        self.client.get_control_state.side_effect = state
        with self.assertRaisesRegex(RuntimeError, "drift exceeded"):
            self.run_recording()
        self.assertEqual(len(self.read()), 1)
        self.assertEqual(self.client.stop_teleop.call_count, 2)

    def test_interruption_stops_and_preserves_samples(self):
        def state():
            if self.recording_time > .29:
                raise KeyboardInterrupt
            return self.state()
        self.client.get_control_state.side_effect = state
        with self.assertRaises(KeyboardInterrupt):
            self.run_recording()
        self.assertEqual(len(self.read()), 1)
        self.assertEqual(self.client.stop_teleop.call_count, 2)

    def test_rejected_send_is_not_recorded(self):
        self.client.movetelej_abs.return_value = {"code": "1", "msg": "rejected"}
        with self.assertRaisesRegex(RuntimeError, "movetelej_abs failed"):
            self.run_recording()
        self.assertFalse(self.path.exists())
        self.assertEqual(self.client.stop_teleop.call_count, 2)

    def test_no_takeover_of_existing_teleoperation(self):
        self.client.get_robot_data.side_effect = None
        self.client.get_robot_data.return_value = {"op_state": 17}
        with self.assertRaisesRegex(ValueError, "already be idle"):
            self.run_recording()
        self.client.start_teleop.assert_not_called()
        self.client.stop_teleop.assert_not_called()

    def test_retries_start_until_mode_changes_before_sending(self):
        def robot_data():
            mode = 17 if self.client.start_teleop.call_count >= 3 else 5
            if mode == 17 and self.ready_at is None:
                self.ready_at = self.now
            return {"op_state": mode}
        self.client.get_robot_data.side_effect = robot_data
        def move(**kwargs):
            self.assertEqual(self.client.start_teleop.call_count, 3)
            self.assertGreaterEqual(self.now, .6)
            return {"code": "0"}
        self.client.movetelej_abs.side_effect = move
        self.run_recording()
        self.assertGreater(len(self.read()), 0)
        self.assertEqual(self.client.start_teleop.call_count, 3)
        self.assertEqual(self.client.stop_teleop.call_count, 4)

    def test_mode_transition_timeout_stops_without_sending(self):
        self.client.get_robot_data.side_effect = None
        self.client.get_robot_data.return_value = {"op_state": 5}
        with self.assertRaisesRegex(RuntimeError, "did not enter teleoperation"):
            self.run_recording()
        self.client.movetelej_abs.assert_not_called()
        self.assertGreater(self.client.start_teleop.call_count, 1)
        self.assertEqual(self.client.stop_teleop.call_count, self.client.start_teleop.call_count + 1)
        self.assertFalse(self.path.exists())

    def test_invalid_torque_and_existing_file_prevent_start(self):
        self.client.get_control_state.side_effect = lambda: {**self.state(), "tau_jts": []}
        with self.assertRaisesRegex(ValueError, "requires finite tau_jts"):
            self.run_recording()
        self.client.start_teleop.assert_not_called()
        self.client.get_control_state.side_effect = lambda: self.state()
        self.path.write_text("existing data")
        with self.assertRaises(FileExistsError):
            self.run_recording()
        self.assertEqual(self.path.read_text(), "existing data")
        self.client.start_teleop.assert_not_called()


if __name__ == "__main__":
    unittest.main()
