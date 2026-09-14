import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from middle_level_controller.box_lift_open_loop.controller import NN_controller


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

    def test_interactive_command_toggles_at_ticks_and_restarts_enabled(self):
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
                patch("middle_level_controller.box_lift_open_loop.controller.RL_COMPLIANCE_INTERACTIVE", True), \
                patch("middle_level_controller.box_lift_open_loop.controller.RL_COMPLIANCE_COMMAND", False), \
                patch("middle_level_controller.box_lift_open_loop.controller.time.sleep"), \
                patch("builtins.print"):
            controller._open_loop_control_fn(1200.)
            self.assertIsNone(controller._control_error)
            modes = [call.kwargs["compliance_mode"] for call in policy.update.call_args_list]
            self.assertEqual(modes, [[1., 1.], [0., 0.], [1., 1.], [1., 1.]])
            controller._control_triggered = True
            controller._open_loop_control_fn(1200.)
        self.assertIsNone(controller._control_error)
        self.assertEqual(policy.update.call_args_list[4].kwargs["compliance_mode"], [1., 1.])
        self.assertEqual(policy.toggle.call_count, 2)  # Backend is selected once per run.
        self.assertEqual(controller.exec_enable_compliance.call_count, 2)


if __name__ == "__main__":
    unittest.main()
