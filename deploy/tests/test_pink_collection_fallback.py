import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from data_collector.collector import DataCollectionScheduler
from data_collector.pink_ik import PinkTeleopIK


class PinkFailureResultTest(unittest.TestCase):
    def test_non_convergence_returns_final_iterate(self):
        config_path = (
            Path(__file__).resolve().parents[3]
            / "robot_interface/robot_interface/config/eir.yaml"
        )
        solver = PinkTeleopIK(config_path)
        solver._settings = dict(solver._settings)
        solver._settings["max_iterations"] = 0

        result = solver.solve_multi(
            targets={1: [0.] * 6, 2: [0.] * 6},
            init_jpos=[0.] * 22,
            lock_non_selected_joints=True,
        )

        self.assertFalse(result["success"])
        self.assertEqual(len(result["jpos"]), 18)
        self.assertEqual(result["jpos"], [0.] * 18)
        self.assertEqual(set(result["failed_arms"]), {1, 2})


class CollectionFallbackTest(unittest.TestCase):
    @staticmethod
    def _scheduler(solve_multi):
        scheduler = DataCollectionScheduler.__new__(DataCollectionScheduler)
        scheduler.robot_ids = [0]
        scheduler.arm_index = [1, 2]
        scheduler.is_multi_arm = True
        scheduler.device_output_mode = "task_abs"
        scheduler.control_mode = "joint_abs"
        scheduler.ik_type = "pink"
        scheduler.teleop_config = SimpleNamespace(
            lock_non_selected_joints=True)
        scheduler._collection_triggered = True
        scheduler._collection_error = None

        robot = MagicMock()
        state = {"q": [0.] * 22, "p": [0.] * 18}
        robot.get_state.return_value = state
        robot.get_task_pose.side_effect = (
            lambda current_state, arm_index: current_state["p"][
                arm_index * 6:(arm_index + 1) * 6])
        robot.validate_task_command.side_effect = (
            lambda command, arm_index: list(command))
        robot.validate_joint_command.side_effect = lambda command: list(command)
        robot.DUMMY_JOINT_DOF = 4
        scheduler.robot = {0: robot}

        scheduler.robot_config = SimpleNamespace(
            control_dt=0.,
            robot_ids=[0],
            robot_params={
                0: {"control": {"vel_scale": 1., "acc_scale": 1.}},
            },
        )
        scheduler.task_config = SimpleNamespace(
            extra_config=SimpleNamespace(
                control_post_process_fn=lambda command: command),
        )
        scheduler.config = SimpleNamespace(
            data_to_collect={"control": ["joint_abs_control"]})
        scheduler.pink_solvers = {
            0: SimpleNamespace(solve_multi=solve_multi),
        }

        devices = {0: MagicMock(), 1: MagicMock()}
        scheduler.data_collector = SimpleNamespace(
            device_ids=[0, 1],
            device=devices,
            get_device_input=MagicMock(),
            update_data_buffer=MagicMock(),
        )
        scheduler.collect_buffer = MagicMock(return_value={})
        scheduler.exec_enable_compliance = MagicMock()
        scheduler.exec_start_movement = MagicMock()
        scheduler.exec_soft_stop = MagicMock()
        scheduler.exec_finish_movement = MagicMock()
        return scheduler, robot

    @staticmethod
    def _valid_device_data(button=False):
        return {
            "final_valid": True,
            "final_button": button,
            0: {"control": [0.] * 6, "trigger": 0.},
            1: {"control": [0.] * 6, "trigger": 0.},
        }

    def test_repeated_failures_execute_until_user_stops(self):
        solve_count = 0

        def solve_multi(**kwargs):
            nonlocal solve_count
            solve_count += 1
            success = solve_count == 3
            return {
                "success": success,
                "jpos": [float(solve_count)] * 18,
                "error": None if success else "did not converge",
                "failed_arms": [] if success else [2],
            }

        scheduler, robot = self._scheduler(solve_multi)
        input_count = 0

        def get_device_input(**kwargs):
            nonlocal input_count
            input_count += 1
            # First press starts recording. Five solves run, then the second
            # press stops it. IK failures must never end the loop themselves.
            return self._valid_device_data(button=input_count in (1, 7))

        scheduler.data_collector.get_device_input.side_effect = get_device_input

        with patch("builtins.print"):
            scheduler._collection_fn()

        expected_solves = 5
        expected_commands = expected_solves
        self.assertEqual(solve_count, expected_solves)
        self.assertEqual(robot.tele_move.call_count, expected_commands)
        self.assertEqual(
            scheduler.data_collector.update_data_buffer.call_count,
            expected_commands,
        )
        self.assertEqual(
            robot.tele_move.call_args.kwargs["action"],
            [float(expected_commands)] * 18 + [0.] * 4,
        )
        # Both devices reset once when recording begins. Only device 1 maps to
        # failed arm 2, so only it is re-anchored after each of four failures.
        self.assertEqual(
            scheduler.data_collector.device[0].reset.call_count, 1)
        self.assertEqual(
            scheduler.data_collector.device[1].reset.call_count,
            5,
        )
        self.assertIsNone(scheduler._collection_error)
        scheduler.exec_finish_movement.assert_called_once_with()

    def test_device_loss_ends_without_collection_error(self):
        solve_multi = MagicMock()
        scheduler, robot = self._scheduler(solve_multi)
        scheduler.data_collector.get_device_input.return_value = {
            "final_valid": False,
            "final_button": False,
        }

        with patch("builtins.print"):
            scheduler._collection_fn()

        solve_multi.assert_not_called()
        robot.tele_move.assert_not_called()
        self.assertIsNone(scheduler._collection_error)
        scheduler.exec_finish_movement.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
