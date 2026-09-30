import threading
import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import numpy as np

from data_collector.visualizer import CollectionVisualizer, joint_values, _run_viewer
import test_pink_collection_fallback as fallback


class CollectionVisualizerTest(unittest.TestCase):
    def setUp(self):
        # Exercise the real publisher without a server or hardware connection.
        self.viewer = CollectionVisualizer.__new__(CollectionVisualizer)
        self.viewer._sample = [0.] * 90
        self.viewer._lock = threading.Lock()
        self.viewer._process = MagicMock()
        self.viewer._process.is_alive.return_value = True
        self.viewer._disabled = False
        self.viewer._command = None
        self.viewer._raw_command = None
        self.viewer._task_commands = {}

    def test_maps_dcp_arm_order_and_degrees_ignoring_dummy_joints(self):
        q = np.arange(22) * 10.
        joints = joint_values(q)
        self.assertEqual(len(joints), 18)
        self.assertAlmostEqual(joints["Joint_L2_L"], np.deg2rad(40.))
        self.assertAlmostEqual(joints["Joint_L2_R"], np.deg2rad(110.))
        self.assertAlmostEqual(joints["Joint_L8_R"], np.deg2rad(170.))
        for bad in ([0.] * 18, [np.nan] * 22):
            with self.assertRaises(ValueError):
                joint_values(bad)

    def test_copies_last_sent_command_and_clears_it_for_new_collection(self):
        measured, command = [1.] * 22, [2.] * 22
        self.viewer.publish(measured, command)
        command[0] = 100.
        self.viewer.publish([3.] * 22)
        self.assertEqual(self.viewer._sample[1], 1.)
        self.assertEqual(self.viewer._sample[2:24], [3.] * 22)
        self.assertEqual(self.viewer._sample[24:46], [2.] * 22)
        self.viewer.publish(measured, reset=True)
        self.assertEqual(self.viewer._sample[1], 0.)

    def test_renderer_lock_never_blocks_publisher_and_latest_command_survives(self):
        self.viewer._lock.acquire()
        try:
            self.viewer.publish([0.] * 22, [4.] * 22)
            self.assertEqual(self.viewer._sample, [0.] * 90)
        finally:
            self.viewer._lock.release()
        self.viewer.publish([1.] * 22)
        self.assertEqual(self.viewer._sample[24:46], [4.] * 22)

    def test_task_commands_keep_both_arms_and_clear_between_episodes(self):
        left, right = [100.] * 6, [200.] * 6
        self.viewer.publish([0.] * 22, task_commands={1: left})
        self.viewer.publish([0.] * 22, task_commands={2: right})
        left[0] = 999.
        self.assertEqual(self.viewer._sample[1], 2.)
        self.assertEqual(self.viewer._sample[46:49], [0., 1., 1.])
        self.assertEqual(self.viewer._sample[55:61], [100.] * 6)
        self.assertEqual(self.viewer._sample[61:67], right)
        self.viewer.publish([0.] * 22, reset=True)
        self.assertEqual(self.viewer._sample[46:49], [0.] * 3)

    def test_raw_csv_command_is_copied_retained_and_reset_independently(self):
        raw = [3.] * 22
        self.viewer.publish([1.] * 22, [2.] * 22, raw_command=raw)
        raw[0] = 999.
        self.viewer.publish([4.] * 22)
        self.assertFalse(self.viewer._disabled)
        self.assertEqual(self.viewer._sample[24:46], [2.] * 22)
        self.assertEqual(self.viewer._sample[67], 1.)
        self.assertEqual(self.viewer._sample[68:90], [3.] * 22)
        self.viewer.publish([4.] * 22, reset=True)
        self.assertEqual(self.viewer._sample[1], 0.)
        self.assertEqual(self.viewer._sample[67], 0.)

    def test_renderer_updates_three_robots_and_hides_raw_on_checkbox_or_reset(self):
        server = MagicMock()
        robots = [MagicMock() for _ in range(3)]
        actual_root, target_root, raw_root = [SimpleNamespace(visible=False) for _ in range(3)]
        show_actual, show_target, show_raw = [SimpleNamespace(value=True) for _ in range(3)]
        scene = (server, robots, actual_root, target_root, show_actual, show_target,
                 SimpleNamespace(content=""), SimpleNamespace(content=""), raw_root, show_raw)
        stop = MagicMock()
        raw_visibility = []

        def next_tick(_):
            step = stop.wait.call_count
            if step > 1:
                raw_visibility.append(raw_root.visible)
            if step == 1:
                self.viewer.publish([1.] * 22, [2.] * 22, raw_command=[3.] * 22)
            elif step == 2:
                show_raw.value = False
            elif step == 3:
                show_raw.value = True
                self.viewer.publish([4.] * 22, reset=True)
            return step == 4

        stop.wait.side_effect = next_tick
        with patch("data_collector.visualizer._make_scene", return_value=scene):
            _run_viewer("unused.yaml", "127.0.0.1", 8080, 15.,
                        self.viewer._sample, self.viewer._lock, stop)
        self.assertEqual(raw_visibility, [True, False, False])
        robots[0].update_cfg.assert_any_call(joint_values([1.] * 22))
        robots[1].update_cfg.assert_any_call(joint_values([2.] * 22))
        robots[2].update_cfg.assert_called_once_with(joint_values([3.] * 22))
        self.assertFalse(target_root.visible)
        server.stop.assert_called_once()

    def test_renderer_exit_or_bad_telemetry_does_not_raise_into_control(self):
        self.viewer._process.is_alive.return_value = False
        with patch("builtins.print") as output:
            self.viewer.publish([0.] * 22)
            self.viewer.publish([0.] * 22)
        self.assertTrue(self.viewer._disabled)
        output.assert_called_once()
        self.viewer._disabled = False
        self.viewer._process.is_alive.return_value = True
        with patch("builtins.print"):
            self.viewer.publish([np.nan] * 22)
        self.assertTrue(self.viewer._disabled)


class ViewerCollectionIntegrationTest(unittest.TestCase):
    def test_dry_run_shadow_is_hold_command_not_rl_projection(self):
        scheduler, robot = fallback.CollectionFallbackTest._scheduler(
            MagicMock(return_value={"success": True, "jpos": [1.] * 18}))
        scheduler.visualizer = MagicMock()
        scheduler.ik_type = "rl_constraint"
        scheduler.teleop_config.rl_constraint_dry_run = True
        policy = SimpleNamespace(
            active_mode="pink", reset=MagicMock(), uses_compliant_history=False,
            update=MagicMock(), record_command=MagicMock(),
            command=MagicMock(return_value=[90.] * 22))

        def toggle(*args):
            policy.active_mode = "rl_constraint"
            return policy.active_mode

        policy.toggle = toggle
        scheduler.rl_constraint = policy
        inputs = [fallback.CollectionFallbackTest._valid_device_data(button=button)
                  for button in (True, True, False, False, True)]
        inputs[2]["final_switch_button"] = True
        scheduler.data_collector.get_device_input.side_effect = inputs
        with patch("builtins.print"):
            scheduler._collection_fn()
        self.assertIsNone(scheduler._collection_error)
        policy.command.assert_called()
        sent = [call.kwargs["action"] for call in robot.tele_move.call_args_list]
        self.assertEqual(len(sent), 3)
        self.assertTrue(all(q == [0.] * 4 + [1.] * 14 + [0.] * 4 for q in sent))
        shown = [call.args[1] for call in scheduler.visualizer.publish.call_args_list
                 if call.args[1] is not None]
        self.assertEqual(shown, sent)

    def test_cartesian_viewer_receives_each_arm_target_after_send(self):
        scheduler, robot = fallback.CollectionFallbackTest._scheduler(MagicMock())
        scheduler.visualizer = MagicMock()
        scheduler.control_mode = "task_abs"
        scheduler.config.data_to_collect["control"] = ["task_abs_control"]
        inputs = [fallback.CollectionFallbackTest._valid_device_data(button=True)
                  for _ in range(2)]
        inputs.append({"final_valid": False})
        inputs[1][0]["control"] = [100.] * 6
        inputs[1][1]["control"] = [200.] * 6
        scheduler.data_collector.get_device_input.side_effect = inputs
        sent_count_at_publish = []
        scheduler.visualizer.publish.side_effect = lambda *args, **kwargs: (
            sent_count_at_publish.append(robot.tele_move.call_count)
            if kwargs["task_commands"] else None)
        with patch("builtins.print"):
            scheduler._collection_fn()
        self.assertIsNone(scheduler._collection_error)
        shown = [call.kwargs["task_commands"]
                 for call in scheduler.visualizer.publish.call_args_list
                 if call.kwargs["task_commands"]]
        self.assertEqual(shown, [{1: [100.] * 6}, {2: [200.] * 6}])
        self.assertEqual(sent_count_at_publish, [1, 2])


if __name__ == "__main__":
    unittest.main()
