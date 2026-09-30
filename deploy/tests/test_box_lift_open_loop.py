import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from middle_level_controller.box_lift_open_loop.controller import NN_controller
from middle_level_controller.box_lift_open_loop import controller as controller_module


class BoxLiftOpenLoopTest(unittest.TestCase):
    @staticmethod
    def _controller(trajectory):
        controller = NN_controller.__new__(NN_controller)
        controller.trajectory = np.asarray(trajectory, dtype=float)
        controller.task_targets = [
            {1: [0.] * 6, 2: [0.] * 6} for _ in controller.trajectory]
        controller._control_triggered = True
        controller._control_thread = None
        controller._control_error = None

        state = {"q": [0.] * 22, "qdot": [0.] * 22, "qdes": [0.] * 22,
                 "p": [0.] * 18, "op_state": 9}
        robot = MagicMock()
        robot.get_state.return_value = state
        robot.robot_client.get_control_state.return_value = dict(
            state, tau=[1.] * 22, tau_act=[2.] * 22, tau_ext=[3.] * 18, tau_jts=[])
        robot.get_task_pose.return_value = [0.] * 6
        robot.validate_joint_command.side_effect = lambda q: list(q)
        controller.pink_solver = MagicMock()
        controller.pink_solver.forward_multi.return_value = {1: [1.] * 6, 2: [2.] * 6}
        controller.robot = {0: robot}
        controller.robot_cluster = MagicMock()
        controller.exec_enable_compliance = MagicMock()
        controller.exec_start_movement = MagicMock()
        controller.exec_soft_stop = MagicMock()
        controller.exec_finish_movement = MagicMock()
        controller._save_joint_states = MagicMock(return_value=Path("states.csv"))
        return controller

    def test_20_hz_commands_are_held_at_50_hz_without_interpolation(self):
        source = np.array([[0.], [10.], [20.]])
        held = NN_controller._hold_trajectory(source, .05, .02)
        np.testing.assert_array_equal(held[:, 0], [0., 0., 0., 10., 10., 20.])

    def test_saved_errors_norm_mean_and_plot(self):
        records = []
        for index, (first, second) in enumerate(((3., 4.), (-6., -8.))):
            current = [10.] * 22
            # Large torso/head/dummy errors must not affect the arm norm.
            command = [999.] * 4 + [10. + first, 10. + second] + [10.] * 12 + [999.] * 4
            records.append(
                [index, index * .02, 1000. + index * .02, 9]
                + current + [0.] * 22 + ["pink"] + command
                + NN_controller._torque_record({"tau_ext": [2.] * 18}))

        with TemporaryDirectory() as directory, \
                patch.object(controller_module, "RESULT_DIR", Path(directory)), \
                patch.object(controller_module, "IK_TYPE", "pink"), \
                patch.object(controller_module, "TRAJECTORY_PATH", Path("traj/example.csv")), \
                patch("builtins.print") as output:
            path = NN_controller._save_joint_states(records)
            self.assertEqual(path.parent, Path(directory) / "pink_example")
            with path.open(newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))
            names = controller_module.DCP_ACTIVE_JOINT_NAMES
            self.assertEqual([float(row["error_norm_deg"]) for row in rows], [5., 10.])
            self.assertEqual(float(rows[0][f"error_{names[4]}_deg"]), 3.)
            self.assertEqual(float(rows[1][f"error_{names[5]}_deg"]), -8.)
            error_columns = [name for name in rows[0]
                             if name.startswith("error_") and name != "error_norm_deg"]
            self.assertEqual(len(error_columns), 14)
            # Preserve current and sent positions for every joint in the log.
            joint_names = names + tuple(f"Dummy_{i}" for i in range(4))
            for row, record in zip(rows, records):
                self.assertEqual([float(row[f"q_{name}_deg"]) for name in joint_names],
                                 record[4:26])
                self.assertEqual([float(row[f"command_{name}_deg"]) for name in joint_names],
                                 record[49:71])
            self.assertEqual(float(rows[1]["elapsed_time_s"]), .02)
            self.assertEqual(rows[0]["tau_ext_count"], "18")
            self.assertEqual(float(rows[0][f"tau_ext_{names[4]}"]), 2.)
            self.assertEqual(rows[0]["tau_ext_Dummy_0"], "")
            self.assertEqual(rows[0]["tau_jts_count"], "0")
            self.assertEqual(rows[0][f"tau_jts_{names[4]}"], "")
            output.assert_any_call(
                "Mean joint error norm across 2 time steps: "
                "7.500000 deg (14 arm joints, command - current)")
            plot = path.with_name(path.name.replace("joint_states_", "joint_errors_")).with_suffix(".png")
            self.assertTrue(plot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            # Repeated executions preserve earlier results.
            self.assertNotEqual(NN_controller._save_joint_states(records), path)

    def test_partial_run_saves_only_successfully_sent_validated_commands(self):
        controller = self._controller(np.zeros((3, 22)))
        controller.rl_constraint = None
        controller.robot[0].validate_joint_command.side_effect = (
            lambda q: list(q[:18]) + [0.] * 4)
        controller.trajectory[0, -4:] = 99.
        controller.robot_cluster.tele_move.side_effect = [None, RuntimeError("send failed")]
        with patch.object(controller_module.time, "sleep"), patch("builtins.print"):
            # Dummy targets should not interfere with the start-position check.
            with patch.object(controller, "_check_start_position"):
                controller._open_loop_control_fn(1200.)

        self.assertIsInstance(controller._control_error, RuntimeError)
        records = controller._save_joint_states.call_args.args[0]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0][49:71], [0.] * 22)
        controller.exec_finish_movement.assert_called_once_with()

    def test_pink_mode_replays_exact_recorded_commands(self):
        trajectory = np.stack([np.full(22, value) for value in (0., 10., 20.)])
        controller = self._controller(trajectory)
        controller.rl_constraint = None

        with patch("middle_level_controller.box_lift_open_loop.controller.IK_TYPE", "pink"), \
                patch("middle_level_controller.box_lift_open_loop.controller.time.sleep"), \
                patch("builtins.print"):
            controller._open_loop_control_fn(1200.)

        sent = [call.kwargs["action"][0]
                for call in controller.robot_cluster.tele_move.call_args_list]
        self.assertEqual(sent, trajectory.tolist())
        controller.exec_enable_compliance.assert_called_once_with()
        controller.exec_finish_movement.assert_called_once_with()

    def test_rl_observes_free_motion_while_robot_compliance_stays_enabled(self):
        controller = self._controller(np.zeros((4, 22)))
        policy = MagicMock()
        policy.uses_compliant_history = True
        policy.observation = SimpleNamespace(mode_dim=2)
        policy.command.return_value = [1.] * 18 + [0.] * 4
        controller.rl_constraint = policy

        with patch("middle_level_controller.box_lift_open_loop.controller.IK_TYPE",
                   "rl_constraint"), \
                patch.object(controller_module, "RL_START_DELAY_S", 0.), \
                patch("middle_level_controller.box_lift_open_loop.controller.RL_COMPLIANCE_INTERACTIVE", False), \
                patch("middle_level_controller.box_lift_open_loop.controller.RL_COMPLIANCE_COMMAND", False), \
                patch("middle_level_controller.box_lift_open_loop.controller.time.sleep"), \
                patch("builtins.print"):
            controller._open_loop_control_fn(1200.)

        self.assertIsNone(controller._control_error)
        controller.exec_enable_compliance.assert_called_once_with()
        controller.exec_finish_movement.assert_called_once_with()
        policy.toggle.assert_called_once()
        self.assertEqual(policy.command.call_count, 4)
        self.assertEqual(policy.update.call_count, 4)
        for call in policy.update.call_args_list:
            self.assertEqual(call.kwargs["compliance_mode"], [0., 0.])
        controller.robot[0].get_control_state.assert_not_called()
        controller.robot[0].get_task_pose.assert_not_called()
        for call in policy.update.call_args_list:
            self.assertEqual(call.kwargs["measured_targets"], {1: [1.] * 6, 2: [2.] * 6})
            self.assertNotIn("applied_command", call.kwargs)
        modes = [row[48] for row in controller._save_joint_states.call_args.args[0]]
        self.assertEqual(modes, ["rl_constraint"] * 4)

    def test_interactive_command_toggles_at_ticks_and_restarts_from_config(self):
        controller = self._controller(np.zeros((4, 22)))
        policy = MagicMock()
        policy.uses_compliant_history = True
        policy.observation = SimpleNamespace(mode_dim=2)
        policy.command.return_value = [1.] * 18 + [0.] * 4
        controller.rl_constraint = policy

        def switch_after_send(*args):
            if policy.record_command.call_count in (1, 2, 4):
                controller.toggle_rl_compliance_command()

        policy.record_command.side_effect = switch_after_send
        with patch("middle_level_controller.box_lift_open_loop.controller.IK_TYPE", "rl_constraint"), \
                patch.object(controller_module, "RL_START_DELAY_S", 0.), \
                patch("middle_level_controller.box_lift_open_loop.controller.RL_COMPLIANCE_INTERACTIVE", True), \
                patch("middle_level_controller.box_lift_open_loop.controller.RL_COMPLIANCE_COMMAND", False), \
                patch("middle_level_controller.box_lift_open_loop.controller.time.sleep"), \
                patch("builtins.print"):
            controller._open_loop_control_fn(1200.)
            self.assertIsNone(controller._control_error)
            modes = [call.kwargs["compliance_mode"] for call in policy.update.call_args_list]
            self.assertEqual(modes, [[0., 0.], [1., 1.], [0., 0.], [0., 0.]])
            controller._control_triggered = True
            controller._open_loop_control_fn(1200.)
        self.assertIsNone(controller._control_error)
        self.assertEqual(policy.update.call_args_list[4].kwargs["compliance_mode"], [0., 0.])
        self.assertEqual(policy.toggle.call_count, 2)  # Backend is selected once per run.
        self.assertEqual(controller.exec_enable_compliance.call_count, 2)

    def test_rl_start_delay_replays_then_initializes_at_current_sample(self):
        for delay, first_rl_step in ((0., 0), (1., 2), (1.1, 3), (5., 4)):
            with self.subTest(delay=delay):
                trajectory = np.stack([np.full(22, i) for i in range(4)])
                controller = self._controller(trajectory)
                controller.robot_config = SimpleNamespace(
                    control_dt=.5, robot_params=controller.robot_config.robot_params)
                policy = MagicMock()
                policy.uses_compliant_history = False
                policy.command.return_value = [9.] * 18 + [0.] * 4
                controller.rl_constraint = policy
                now = 100.

                def sleep(seconds):
                    nonlocal now
                    now += seconds

                def state():
                    return {"q": [now - 100.] * 22, "qdot": [0.] * 22, "op_state": 9}

                controller.robot[0].get_state.side_effect = state
                controller.robot[0].robot_client.get_control_state.side_effect = state
                with patch.object(controller_module, "IK_TYPE", "rl_constraint"), \
                        patch.object(controller_module, "RL_START_DELAY_S", delay), \
                        patch.object(controller_module, "RL_CONSTRAINT_DRY_RUN", False), \
                        patch.object(controller_module.time, "monotonic", side_effect=lambda: now), \
                        patch.object(controller_module.time, "sleep", side_effect=sleep), \
                        patch("builtins.print"):
                    # Every new execution starts with the full configured delay.
                    for _ in range(2):
                        now = 100.
                        policy.reset_mock()
                        controller.robot_cluster.tele_move.reset_mock()
                        controller._control_triggered = True
                        controller._open_loop_control_fn(1200.)
                        self.assertIsNone(controller._control_error)
                        records = controller._save_joint_states.call_args.args[0]
                        self.assertEqual([row[48] for row in records],
                                         ["pink"] * first_rl_step
                                         + ["rl_constraint"] * (4 - first_rl_step))
                        sent = [call.kwargs["action"][0] for call in
                                controller.robot_cluster.tele_move.call_args_list]
                        self.assertEqual(sent, trajectory[:first_rl_step].tolist()
                                         + [policy.command.return_value] * (4 - first_rl_step))
                        self.assertEqual(policy.record_command.call_count, 4 - first_rl_step)
                        self.assertEqual(policy.update.call_count, 4 - first_rl_step)
                        if first_rl_step < 4:
                            policy.toggle.assert_called_once_with(
                                [first_rl_step * .5] * 22,
                                controller.pink_solver.forward_multi.return_value)
                            self.assertEqual(policy.command.call_args_list[0].args[1],
                                             trajectory[first_rl_step].tolist())
                        else:
                            policy.toggle.assert_not_called()

    def test_invalid_rl_delay_is_rejected_before_motion(self):
        for delay in (-1., float("nan"), float("inf")):
            with self.subTest(delay=delay):
                controller = self._controller(np.zeros((2, 22)))
                controller.rl_constraint = MagicMock()
                with patch.object(controller_module, "RL_START_DELAY_S", delay), \
                        patch("builtins.print"):
                    controller._open_loop_control_fn(1200.)
                self.assertIsInstance(controller._control_error, ValueError)
                controller.exec_start_movement.assert_not_called()
                controller.robot_cluster.tele_move.assert_not_called()

    def test_logged_position_and_torque_use_the_same_control_response(self):
        controller = self._controller(np.zeros((1, 22)))
        controller.rl_constraint = None
        controller.robot[0].robot_client.get_control_state.return_value = {
            "q": [4.] * 22, "qdot": [5.] * 22,
            "tau": [1.] * 22, "tau_act": [2.] * 22, "tau_ext": [3.] * 18}
        with patch("builtins.print"):
            controller._open_loop_control_fn(1200.)
        self.assertIsNone(controller._control_error)
        controller.robot[0].robot_client.get_control_state.assert_called_once_with()
        row = controller._save_joint_states.call_args.args[0][0]
        self.assertEqual(row[4:26], [4.] * 22)
        self.assertEqual(row[26:48], [5.] * 22)
        self.assertEqual(row[71:], NN_controller._torque_record(
            controller.robot[0].robot_client.get_control_state.return_value))


if __name__ == "__main__":
    unittest.main()
