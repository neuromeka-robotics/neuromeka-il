import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from data_collector.device.vive import TrackpadButtons
from data_collector.pink_ik import DCP_ACTIVE_JOINT_NAMES
from data_collector.rl_constraint import CompliantPlaneObservation, RLConstraintTeleop
from helper.config_utils import TELEOP_CONFIG
import test_pink_collection_fallback as fallback


MODEL = Path(os.environ.get(
    "RL_CONSTRAINT_TEST_MODEL",
    Path(__file__).resolve().parents[1] / "data_collector/models/dual_arm_plane_800.onnx",
))
TARGETS = {1: [400., 250., 250., 0., 0., 0.],
           2: [400., -250., 250., 0., 0., 0.]}


class CompliantPlaneObservationTest(unittest.TestCase):
    def setUp(self):
        self.contract = {
            "policy_interface": "compliant_plane_history_v3",
            "pose_frame": "robot_base",
            "orientation_representation": "rotation_6d_columns",
            "history_order": "term_major_offset_order",
            "action_semantics": "encoder_joint_delta_rad",
            "observe_constraint_normal": False,
            "observe_palm_plane_distance": False,
            "joint_names": list(DCP_ACTIVE_JOINT_NAMES[4:18]),
            "joint_pos_history_offsets_steps": [2, 0],
            "observe_joint_target_error_history": True,
            "include_previous_actions": False,
            "include_last_last_action": False,
            "num_obs": 132,
            "control_dt": .02,
            "action_scale": .15,
            "palm_reference_local_m": [[0., -.02, 0.], [0., .02, 0.]],
        }
        self.builder = CompliantPlaneObservation(self.contract, .02)

    def test_history_reset_and_last_sent_target_error(self):
        q0 = np.arange(22, dtype=float)
        q1 = q0 + 10.
        self.builder.update(TARGETS, q0, measured_targets=TARGETS,
                            compliance_mode=[0., 1.])
        self.builder.record_command(q1 + 2., q0)
        self.builder.update(TARGETS, q1, measured_targets=TARGETS,
                            compliance_mode=[1., 0.])
        obs = self.builder.build()
        indices = self.builder.indices
        np.testing.assert_allclose(obs[:14], np.deg2rad(q0[indices]), atol=1e-7)
        np.testing.assert_allclose(obs[14:28], np.deg2rad(q1[indices]), atol=1e-7)
        np.testing.assert_allclose(obs[28:42], 0., atol=1e-7)
        np.testing.assert_allclose(obs[42:56], np.deg2rad(2.), atol=1e-7)
        np.testing.assert_array_equal(obs[-4:], [0., 1., 1., 0.])

    def test_reset_uses_measured_pose_then_admits_requested_pose(self):
        q = np.zeros(22)
        measured = {arm: [0.] * 6 for arm in (1, 2)}
        self.builder.update(TARGETS, q, measured_targets=measured, compliance_mode=[1., 1.])
        obs = self.builder.build()
        np.testing.assert_array_equal(obs[56:92], obs[92:128])
        self.builder.record_command(q, q)
        self.builder.update(TARGETS, q, measured_targets=measured, compliance_mode=[1., 1.])
        self.assertFalse(np.array_equal(self.builder.build()[74:92], obs[74:92]))
        self.builder.reset()
        self.assertIsNone(self.builder.sent_command)

    def test_action_is_unbiased_encoder_relative_command_without_clipping(self):
        q = np.zeros(22)
        reference = np.arange(22, dtype=float)
        command = np.asarray(self.builder.joint_command(np.ones(14), q, reference))
        np.testing.assert_allclose(
            command[self.builder.indices], np.rad2deg(.15), atol=1e-6)
        np.testing.assert_array_equal(command[:4], reference[:4])
        np.testing.assert_array_equal(command[18:], 0.)

    def test_current_command_only_layout_with_and_without_explicit_metadata(self):
        contract = dict(self.contract, joint_pos_history_offsets_steps=[0],
                        observe_joint_target_error_history=False, num_obs=34)
        for explicit_flag in (False, True):
            if explicit_flag:
                contract["observe_measured_palm_pose_history"] = False
            builder = CompliantPlaneObservation(contract, .02)
            for tick in range(3):
                q = np.arange(22, dtype=float) + tick
                builder.update(TARGETS, q, compliance_mode=[1., 0.])
                obs = builder.build()
                self.assertEqual(obs.shape, (34,))
                np.testing.assert_allclose(obs[:14], builder.joint_positions(q))
                np.testing.assert_allclose(
                    obs[14:32], builder.pose_observation(builder.task_poses(TARGETS)))
                np.testing.assert_array_equal(obs[-2:], [1., 0.])
            self.assertFalse(builder.observe_measured)

    def test_explicit_pose_flag_must_match_graph_dimensions(self):
        contract = dict(self.contract, observe_measured_palm_pose_history=False)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            CompliantPlaneObservation(contract, .02)


class PlaneObservationTest(unittest.TestCase):
    def setUp(self):
        self.policy = RLConstraintTeleop(str(MODEL), .05)
        self.builder = self.policy.observation

    def test_history_and_initial_reference_survive_repeated_switches(self):
        for tick in range(4):
            self.policy.update(TARGETS, np.full(22, tick))
        history = self.builder.joint_history.copy()
        self.assertEqual(self.policy.active_mode, "pink")
        self.policy.toggle(np.full(22, 10.), TARGETS)
        np.testing.assert_array_equal(history, self.builder.joint_history)
        np.testing.assert_allclose(self.builder.initial_joints, np.deg2rad(10.), atol=1e-7)
        self.policy.toggle(np.full(22, 11.), TARGETS)
        self.policy.update(TARGETS, np.full(22, 12.))
        self.policy.toggle(np.full(22, 13.), TARGETS)
        np.testing.assert_allclose(self.builder.initial_joints, np.deg2rad(13.), atol=1e-7)
        np.testing.assert_allclose(self.builder.joint_history[-2], np.deg2rad(3.), atol=1e-7)
        np.testing.assert_allclose(self.builder.joint_history[-1], np.deg2rad(12.), atol=1e-7)
        self.policy.reset()
        self.assertEqual(self.policy.active_mode, "pink")
        self.assertIsNone(self.builder.initial_joints)
        self.assertIsNone(self.builder.joint_history)

    def test_observation_order_units_and_previous_command(self):
        self.policy.update(TARGETS, np.arange(22))
        self.policy.toggle(np.full(22, 5.), TARGETS)
        moved = {arm: np.array(target) + [10., 0., 0., 0., 0., 0.]
                 for arm, target in TARGETS.items()}
        self.policy.update(moved, np.arange(22) + 1)
        obs = self.builder.build()
        self.assertEqual(obs.shape, (182,))
        self.assertEqual(obs.dtype, np.float32)
        np.testing.assert_allclose(obs[14:28], np.deg2rad(np.arange(22)[self.builder.indices]), atol=1e-7)
        np.testing.assert_allclose(obs[140:154], np.deg2rad((np.arange(22)+1)[self.builder.indices]), atol=1e-7)
        np.testing.assert_allclose(obs[154:168], np.deg2rad(5.), atol=1e-7)
        current, previous = obs[:14].reshape(2, 7), obs[-14:].reshape(2, 7)
        np.testing.assert_allclose(current[:, 0]-previous[:, 0], .01, atol=1e-7)
        np.testing.assert_allclose(previous[:, :3],
                                   np.array([TARGETS[1][:3], TARGETS[2][:3]]) / 1000.
                                   + [0., 0., .88] + self.builder.palm_reference, atol=1e-7)
        np.testing.assert_allclose(np.linalg.norm(current[:, 3:], axis=1), 1.)

    def test_action_mapping_limits_and_locked_commands(self):
        q = np.linspace(-10., 10., 22)
        reference = np.arange(22, dtype=float)
        action = np.linspace(-200., 200., 14)
        command = np.asarray(self.builder.joint_command(action, q, reference))
        contract = self.builder.contract
        expected = np.clip(np.deg2rad(q[self.builder.indices]) + .02 * action,
                           contract["joint_lower_rad"], contract["joint_upper_rad"])
        np.testing.assert_allclose(np.deg2rad(command[self.builder.indices]), expected, atol=1e-6)
        np.testing.assert_array_equal(command[:4], reference[:4])
        np.testing.assert_array_equal(command[18:], 0.)
        with self.assertRaises(ValueError):
            self.builder.joint_command(np.full(14, np.nan), q, reference)

    def test_real_onnx_inference_and_frequency_validation(self):
        self.policy.update(TARGETS, np.zeros(22))
        self.policy.toggle(np.zeros(22), TARGETS)
        command = self.policy.command(np.zeros(22), np.zeros(22))
        self.assertEqual(len(command), 22)
        self.assertTrue(np.isfinite(command).all())
        with self.assertRaisesRegex(ValueError, "control_dt"):
            RLConstraintTeleop(str(MODEL), .01)


class TrackpadTest(unittest.TestCase):
    def test_split_latches_until_release(self):
        buttons = TrackpadButtons(split=True)
        self.assertEqual(buttons.read(True, -.8), (False, True))
        self.assertEqual(buttons.read(True, .8), (False, True))
        self.assertEqual(buttons.read(False, .8), (False, False))
        self.assertEqual(buttons.read(True, .8), (True, False))
        self.assertEqual(buttons.read(True, -.8), (True, False))
        buttons.read(False, 0.)
        self.assertEqual(buttons.read(True, 0.), (False, False))
        self.assertEqual(buttons.read(True, -.8), (False, False))

    def test_original_pink_keeps_whole_pad_for_recording(self):
        self.assertEqual(TrackpadButtons().read(True, -.8), (True, False))


class RLCollectionTest(unittest.TestCase):
    def test_repeated_switches_keep_recording_and_save_exact_sent_commands(self):
        solve = MagicMock(return_value={"success": True, "jpos": [0.] * 18})
        scheduler, robot = fallback.CollectionFallbackTest._scheduler(solve)
        scheduler.ik_type = "rl_constraint"
        scheduler.teleop_config.rl_constraint_dry_run = False
        scheduler.rl_constraint = RLConstraintTeleop(str(MODEL), .05)
        policy = scheduler.rl_constraint
        # Simulate the state left by a preceding recording that ended in RL.
        policy.update(TARGETS, np.full(22, 99.))
        policy.toggle(np.full(22, 99.), TARGETS)
        policy.session = MagicMock()
        policy.session.run.return_value = [np.ones((1, 14), dtype=np.float32)]
        policy.toggle = MagicMock(wraps=policy.toggle)
        policy.update = MagicMock(wraps=policy.update)
        measurements = []

        def state():
            q = [float(len(measurements))] * 18 + [0.] * 4
            measurements.append(q)
            return {"q": q, "p": [0.] * 6 + TARGETS[1] + TARGETS[2]}

        robot.get_state.side_effect = state

        def inputs(record=False, switch=False):
            result = fallback.CollectionFallbackTest._valid_device_data(button=record)
            result["final_switch_button"] = switch
            # VIVE's old absolute anchors request a pose far from the measured
            # robot pose after RL. Pink must not use these on the return tick.
            result[0]["control"] = [TARGETS[1][0] + 150.] + TARGETS[1][1:]
            result[1]["control"] = [TARGETS[2][0] + 150.] + TARGETS[2][1:]
            return result

        scheduler.data_collector.get_device_input.side_effect = [
            inputs(True, True), inputs(True, True),  # start/reset: always starts Pink
            inputs(), inputs(switch=True), inputs(switch=True), inputs(),
            inputs(switch=True), inputs(), inputs(switch=True), inputs(), inputs(True),
        ]
        modes = []
        robot.tele_move.side_effect = lambda **kwargs: modes.append(policy.active_mode)
        with patch("builtins.print"):
            scheduler._collection_fn()
        self.assertIsNone(scheduler._collection_error)
        self.assertEqual(modes, ["pink", "pink", "rl_constraint", "rl_constraint",
                                 "rl_constraint", "pink", "pink", "rl_constraint", "rl_constraint"])
        self.assertEqual(policy.toggle.call_count, 3)
        self.assertEqual(solve.call_count, 4)
        self.assertEqual(policy.session.run.call_count, 5)
        # Each re-entry uses that tick's measured q, not home/recording-start q.
        last_entry = policy.toggle.call_args_list[-1].args[0]
        np.testing.assert_allclose(policy.observation.initial_joints,
                                   np.deg2rad(np.array(last_entry)[policy.observation.indices]), atol=1e-7)
        np.testing.assert_allclose(policy.observation.joint_history[-1],
                                   np.deg2rad(np.array(measurements[-2])[policy.observation.indices]), atol=1e-7)
        np.testing.assert_allclose(policy.observation.joint_history[0],
                                   np.deg2rad(np.array(measurements[1])[policy.observation.indices]), atol=1e-7)
        saved = scheduler.data_collector.update_data_buffer.call_args_list
        self.assertEqual(len(saved), len(modes))
        for send, save in zip(robot.tele_move.call_args_list, saved):
            self.assertEqual(send.kwargs["action"], save.kwargs["joint_abs_control_0"])
            self.assertEqual(send.kwargs["mode"], "joint_abs")
            self.assertEqual(send.kwargs["action"][:4], measurements[0][:4])
        # Latest measured joints are passed to Pink on return from RL.
        self.assertEqual(solve.call_args_list[2].kwargs["init_jpos"], measurements[6])
        self.assertNotEqual(solve.call_args_list[0].kwargs["targets"], TARGETS)
        self.assertEqual(solve.call_args_list[2].kwargs["targets"], TARGETS)
        # Both devices re-anchor at recording start and on RL -> Pink only.
        for device in scheduler.data_collector.device.values():
            self.assertEqual(device.reset.call_count, 2)
        # Advance history exactly once per tick, using the corrected target.
        self.assertEqual(policy.update.call_count, len(modes))
        self.assertEqual(policy.update.call_args_list[5].args[0], TARGETS)
        scheduler.exec_finish_movement.assert_called_once_with()

    def test_dry_run_prints_rl_but_sends_saves_and_stops_with_last_pink_command(self):
        solve = MagicMock(side_effect=[
            {"success": True, "jpos": [float(i)] * 18} for i in (1, 2, 3)
        ])
        scheduler, robot = fallback.CollectionFallbackTest._scheduler(solve)
        scheduler.ik_type = "rl_constraint"
        scheduler.teleop_config.rl_constraint_dry_run = True
        scheduler.rl_constraint = RLConstraintTeleop(str(MODEL), .05)
        projection = [90.] * 18 + [0.] * 4
        scheduler.rl_constraint.command = MagicMock(return_value=projection)

        def inputs(record=False, switch=False):
            result = fallback.CollectionFallbackTest._valid_device_data(button=record)
            result["final_switch_button"] = switch
            return result

        scheduler.data_collector.get_device_input.side_effect = [
            inputs(True), inputs(True), inputs(switch=True), inputs(),
            inputs(switch=True), inputs(), inputs(switch=True), inputs(True),
        ]
        with patch("builtins.print") as output:
            scheduler._collection_fn()
        self.assertIsNone(scheduler._collection_error)
        self.assertEqual(scheduler.rl_constraint.command.call_count, 3)
        printed = [call.args[0] for call in output.call_args_list
                   if str(call.args[0]).startswith("RL dry run")]
        self.assertEqual(len(printed), 3)
        self.assertTrue(all("90.0" in line for line in printed))
        expected = [[0.] * 4 + [float(i)] * 14 + [0.] * 4 for i in (1, 1, 1, 2, 3, 3)]
        sent = [call.kwargs["action"] for call in robot.tele_move.call_args_list]
        saved = [call.kwargs["joint_abs_control_0"]
                 for call in scheduler.data_collector.update_data_buffer.call_args_list]
        self.assertEqual(sent, expected)
        self.assertEqual(saved, expected)
        self.assertEqual(scheduler.exec_soft_stop.call_args.kwargs["last_action"], {0: expected[-1]})
        scheduler.exec_enable_compliance.assert_called_once_with()

    def test_config_requires_both_arms_and_model(self):
        TELEOP_CONFIG(ik_type="rl_constraint", arm_index=[1, 2],
                      pink_config_path="eir.yaml", rl_constraint_model_path=str(MODEL))
        with self.assertRaises(ValueError):
            TELEOP_CONFIG(ik_type="rl_constraint", arm_index=1,
                          pink_config_path="eir.yaml", rl_constraint_model_path=str(MODEL))
        with self.assertRaises(ValueError):
            TELEOP_CONFIG(ik_type="rl_constraint", arm_index=[1, 2], pink_config_path="eir.yaml")


if __name__ == "__main__":
    unittest.main()
