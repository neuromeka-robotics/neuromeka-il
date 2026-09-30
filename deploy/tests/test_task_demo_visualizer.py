import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

import task_demo
import test_box_lift_open_loop as open_loop_tests
from middle_level_controller.box_lift_rl import controller as rl_module


class TaskDemoVisualizerTest(unittest.TestCase):
    def test_config_flag_and_cli_override(self):
        args = SimpleNamespace(demo_task="box_lift_open_loop", viser=None,
                               viser_host="127.0.0.1", viser_port=8081, viser_hz=10.)
        config = SimpleNamespace(
            VISER_ENABLED=False,
            CUSTOM_TASK_CONFIG=SimpleNamespace(control_config=SimpleNamespace(
                teleop_config=SimpleNamespace(pink_config_path="eir.yaml"))))
        with patch("data_collector.visualizer.CollectionVisualizer") as factory:
            self.assertIsNone(task_demo._start_visualizer(args, config))
            factory.assert_not_called()
            config.VISER_ENABLED = True
            self.assertIs(task_demo._start_visualizer(args, config), factory.return_value)
            factory.assert_called_once_with("eir.yaml", host="127.0.0.1", port=8081, hz=10.)
            args.viser = False
            self.assertIsNone(task_demo._start_visualizer(args, config))
            self.assertEqual(factory.call_count, 1)
            config.VISER_ENABLED = False
            args.viser = True
            task_demo._start_visualizer(args, config)
            self.assertEqual(factory.call_count, 2)

    def test_viewer_closes_on_startup_failure(self):
        visualizer = MagicMock()
        controller_type = MagicMock(side_effect=RuntimeError("startup failed"))
        with patch("sys.argv", ["task_demo.py", "box_lift_open_loop"]), \
                patch("helper.extra_utils.load_NN_controller", return_value=controller_type), \
                patch.object(task_demo, "_start_visualizer", return_value=visualizer):
            with self.assertRaisesRegex(RuntimeError, "startup failed"):
                task_demo.main()
        visualizer.close.assert_called_once_with()

    def test_quit_stops_controller_and_closes_viewer(self):
        controller = MagicMock()
        controller.task_config.data_config = None
        visualizer = MagicMock()
        with patch("sys.argv", ["task_demo.py", "box_lift_open_loop"]), \
                patch("helper.extra_utils.load_NN_controller", return_value=MagicMock(return_value=controller)), \
                patch.object(task_demo, "_start_visualizer", return_value=visualizer), \
                patch.object(task_demo, "Process") as process, \
                patch.object(task_demo, "Queue") as queue:
            queue.return_value.get.return_value = "q"
            process.return_value.is_alive.return_value = False
            task_demo.main()
        controller.exec_nn_control_stop.assert_called_once_with()
        visualizer.close.assert_called_once_with()
        queue.return_value.close.assert_called_once_with()

    def test_open_loop_shadow_matches_sent_rl_command_and_dry_run(self):
        for dry_run in (False, True):
            with self.subTest(dry_run=dry_run):
                trajectory = np.stack([np.zeros(22), np.full(22, 3.)])
                controller = open_loop_tests.BoxLiftOpenLoopTest._controller(trajectory)
                controller.visualizer = MagicMock()
                controller.rl_constraint = MagicMock()
                controller.rl_constraint.uses_compliant_history = False
                controller.rl_constraint.command.return_value = [7.] * 18 + [0.] * 4
                prefix = "middle_level_controller.box_lift_open_loop.controller."
                with patch(prefix + "IK_TYPE", "rl_constraint"), \
                        patch(prefix + "RL_START_DELAY_S", 0.), \
                        patch(prefix + "COMMAND_OFFSET_S", controller.robot_config.control_dt), \
                        patch(prefix + "RL_CONSTRAINT_DRY_RUN", dry_run), \
                        patch(prefix + "time.sleep"), patch("builtins.print"):
                    controller._open_loop_control_fn(1200.)
                self.assertIsNone(controller._control_error)
                sent = [c.kwargs["action"][0]
                        for c in controller.robot_cluster.tele_move.call_args_list]
                shown = [c.args[1] for c in controller.visualizer.publish.call_args_list
                         if len(c.args) == 2]
                self.assertEqual(shown, sent)
                self.assertEqual(shown[0], [3.] * 22 if dry_run else [7.] * 18 + [0.] * 4)
                raw = [c.kwargs["raw_command"] for c in controller.visualizer.publish.call_args_list
                       if "raw_command" in c.kwargs]
                self.assertEqual(raw, [[3.] * 22, [3.] * 22])
                self.assertTrue(controller.visualizer.publish.call_args_list[0].kwargs["reset"])
                self.assertEqual(controller.robot[0].get_state.call_count, 3)

    def test_failed_send_does_not_publish_unsent_shadow(self):
        controller = open_loop_tests.BoxLiftOpenLoopTest._controller(np.zeros((1, 22)))
        controller.rl_constraint = None
        controller.visualizer = MagicMock()
        controller.robot_cluster.tele_move.side_effect = RuntimeError("send failed")
        with patch("builtins.print"):
            controller._open_loop_control_fn(1200.)
        self.assertIsInstance(controller._control_error, RuntimeError)
        self.assertTrue(all(len(c.args) == 1 for c in controller.visualizer.publish.call_args_list))

    def test_rl_policy_shadow_keeps_measured_updates_during_missing_camera_pose(self):
        controller = rl_module.NN_controller.__new__(rl_module.NN_controller)
        controller.visualizer = MagicMock()
        controller._control_triggered = True
        controller._control_error = None
        controller._box_pose_visualizer = None
        controller._joint_velocity_plotter = None
        controller.robot_cluster = MagicMock()
        robot = MagicMock()
        controller.robot = {0: robot}
        measurements = []

        def get_state():
            q = [float(len(measurements))] * 22
            measurements.append(q)
            if len(measurements) == 3:
                controller._control_triggered = False
            return {"q": q, "op_state": 9}

        robot.get_state.side_effect = get_state
        # Make the validated command different from the raw policy output.
        robot.validate_joint_command.side_effect = lambda q: list(q[:18]) + [0.] * 4
        camera = MagicMock()
        camera.get_all.side_effect = [{"rgb": None}, None]
        controller.camera = {rl_module.CAMERA_NAME: camera}
        controller._check_home_position = MagicMock()
        controller._box_pose_in_robot_base = MagicMock(return_value=np.eye(4))
        controller.nn_policy = MagicMock(return_value={
            "control_state": rl_module.NN_CONTROL_STATE.TASK_IN_PROGRESS,
            "action": [1.] * 14, "robot_action_0": [5.] * 22})
        controller.exec_enable_compliance = MagicMock()
        controller.exec_start_movement = MagicMock()
        controller.exec_finish_movement = MagicMock()
        controller.exec_soft_stop = MagicMock()
        with patch.object(rl_module, "RECORD_POLICY_DEPLOYMENT", False), \
                patch.object(rl_module.time, "sleep"), patch("builtins.print"):
            controller._nn_control_fn(1200.)
        self.assertIsNone(controller._control_error)
        calls = controller.visualizer.publish.call_args_list
        self.assertTrue(calls[0].kwargs["reset"])
        shown = [c.args[1] for c in calls if len(c.args) == 2]
        self.assertEqual(shown, [[5.] * 18 + [0.] * 4])
        self.assertEqual(calls[-1].args, (measurements[-1],))
        controller.nn_policy.assert_called_once()
        controller.nn_policy.record_joint_position.assert_called_once_with(measurements[-1])


if __name__ == "__main__":
    unittest.main()
