"""Hardware-free regression checks for perception, labeling and Nami visual servoing."""
import contextlib
import io
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

import cv2
import h5py
import numpy as np
import torch
import zmq
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'deploy'), str(ROOT / 'train'), str(ROOT / 'train/nrmk_il/src')]
from perception.zero_shot_detection import DetectionClient, DinoV2MaskLabeler
from preprocess_visual_servo import goal_corrections, process_episode, select_task_poses
from middle_level_controller.visual_servo.model import NN_policy, ACTPolicyWrapper, angle_delta, constrain_target
from middle_level_controller.visual_servo.config import SERVO_CONFIG
from helper.extra_utils import NN_CONTROL_STATE as State


class FakePredictor:
    def __init__(self):
        self.calls = []
        self.mask = torch.full((1, 1, 32, 40), -1.)
        self.mask[:, :, 8:20, 10:25] = 1

    def load_first_frame(self, image):
        self.calls.append('load')

    def add_new_prompt(self, **kwargs):
        self.calls.append('prompt')
        return 0, [1], self.mask

    def track(self, image):
        self.calls.append('track')
        return [1], self.mask

    def reset_state(self):
        self.calls.append('reset')


class PerceptionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        reference = Path(self.temp.name) / 'ref.png'
        cv2.imwrite(str(reference), np.full((32, 40, 3), [5, 10, 200], np.uint8))
        self.target = {'name': 'object', 'image_path': str(reference)}
        self.rgb = np.zeros((32, 40, 3), np.uint8)
        self.client, self.predictor = Mock(), FakePredictor()
        self.points = np.array([[y, x] for y in range(8, 20) for x in range(10, 25)])
        self.client.detect.return_value = {'object_coords': [self.points]}
        self.tracker = DinoV2MaskLabeler(predictor=self.predictor, client=self.client)

    def tearDown(self):
        self.temp.cleanup()

    def test_register_rgb_and_track_without_detecting_again(self):
        self.assertTrue(self.tracker(self.rgb, self.target)['valid'])
        np.testing.assert_array_equal(self.client.register.call_args.args[1][0, 0], [200, 10, 5])
        self.assertTrue(self.tracker(self.rgb, self.target)['valid'])
        self.assertEqual(self.client.detect.call_count, 1)
        self.assertEqual(self.predictor.calls, ['load', 'prompt', 'track'])

    def test_empty_detection_retries_without_initializing_sam(self):
        self.client.detect.side_effect = [{'object_coords': []}, {'object_coords': [self.points]}]
        self.assertFalse(self.tracker(self.rgb, self.target)['valid'])
        self.assertFalse(self.tracker.is_init)
        self.assertEqual(self.predictor.calls, [])
        self.assertTrue(self.tracker(self.rgb, self.target)['valid'])

    def test_lost_mask_resets_then_redetects(self):
        self.tracker(self.rgb, self.target)
        self.predictor.mask.fill_(-1)
        self.assertFalse(self.tracker(self.rgb, self.target)['valid'])
        self.assertFalse(self.tracker.is_init)
        self.predictor.mask[:, :, 8:20, 10:25] = 1
        self.assertTrue(self.tracker(self.rgb, self.target)['valid'])
        self.assertEqual(self.client.detect.call_count, 2)

    def test_roi_filters_yx_as_xy(self):
        result = self.tracker(self.rgb, self.target, obj_center_limit={'min': [30, 0], 'max': [39, 31]})
        self.assertFalse(result['valid'])
        self.assertFalse(self.predictor.calls)

    def test_server_restart_forces_reregistration(self):
        self.client.detect.side_effect = [RuntimeError('unknown target'), {'object_coords': [self.points]}]
        with self.assertRaises(RuntimeError):
            self.tracker(self.rgb, self.target)
        self.tracker(self.rgb, self.target)
        self.assertEqual(self.client.register.call_count, 2)

    def test_timeout_recreates_req_socket(self):
        with patch('perception.zero_shot_detection.zmq.Context') as factory:
            first, second = Mock(), Mock()
            factory.return_value.socket.side_effect = [first, second]
            first.recv_pyobj.side_effect = zmq.Again()
            second.recv_pyobj.return_value = {'status': 'success', 'object_coords': []}
            client = DetectionClient(timeout=10)
            with self.assertRaises(ConnectionError):
                client.detect('a', self.rgb)
            self.assertEqual(client.detect('a', self.rgb)['object_coords'], [])
            first.close.assert_called_once_with(linger=0)
            client.close()


class LabelTests(unittest.TestCase):
    def test_goal_corrections_reconstruct_first_pose(self):
        poses = np.array([[100, 200, 300, 170, 5, 179], [110, 180, 330, 170, 5, -179]], float)
        dp, dr = goal_corrections(poses)
        rot = Rotation.from_euler('xyz', poses[:, 3:], degrees=True).as_matrix()
        np.testing.assert_allclose(poses[:, :3] / 1000 + np.einsum('tij,tj->ti', rot, dp),
                                   np.repeat(poses[:1, :3] / 1000, 2, axis=0), atol=1e-7)
        np.testing.assert_allclose(rot @ dr, np.repeat(rot[:1], 2, axis=0), atol=1e-7)
        np.testing.assert_allclose(dp[0], 0, atol=1e-7)

    def test_preprocessing_uses_left_arm_and_keeps_goal_when_dropping_frames(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            poses = np.zeros((3, 18), np.float32)
            poses[:, 6] = [100, 150, 130]
            with h5py.File(root/'raw.h5', 'w') as f:
                f['p_0'] = poses
                f['images.rgb.wrist'] = np.full((3, 32, 40, 3), 255, np.uint8)
                f.attrs['arm_indices'] = [1]
            masks = np.zeros((3, 32, 40, 1), np.uint8)
            masks[[0, 2], 8:20, 10:25] = 1
            with h5py.File(root/'mask.h5', 'w') as f:
                f['target'] = masks
            self.assertEqual(process_episode(root/'raw.h5', root/'mask.h5', root/'0.h5'), 2)
            with h5py.File(root/'0.h5') as f:
                np.testing.assert_allclose(f['action.visual_servoing.end_pos'][:, 0], [0, -.03])
                np.testing.assert_array_equal(f.attrs['source_frame_indices'], [0, 2])
                self.assertEqual(f['observation.images.rgb.wrist'][1, 0, 0].sum(), 0)
                self.assertEqual(f['observation.images.rgb.wrist'][1, 10, 12].sum(), 765)
            with self.assertRaises(FileExistsError):
                process_episode(root/'raw.h5', root/'mask.h5', root/'0.h5')
            np.testing.assert_array_equal(select_task_poses(poses, 1)[:, 0], [100, 150, 130])

    def test_mask_save_tracks_renamed_object_and_alignment_metadata(self):
        from generate_mask import MaskSession
        with tempfile.TemporaryDirectory() as temp:
            session = MaskSession(Mock(), temp)
            session.source = Path('/data/7.h5')
            session.output_name = '7_wrist'
            session.frames = [np.zeros((32, 40, 3), np.uint8)] * 2
            session.add_object('target')
            session.masks['target'] = {i: np.ones((32, 40), bool) for i in range(2)}
            session.tracked = True
            session.rename_object('target', 'syringe')
            session.save()
            with h5py.File(Path(temp)/'7_wrist.h5') as f:
                self.assertEqual(f['syringe'].shape, (2, 32, 40, 1))
                self.assertEqual(f.attrs['source_episode'], '7.h5')
            with self.assertRaises(FileExistsError):
                session.save()

    def test_negative_box_survives_retracking(self):
        from generate_mask import MaskSession
        session = MaskSession(Mock(), '/tmp/unused_mask_output')
        session.frames = [np.zeros((32, 40, 3), np.uint8)]
        session.state = {}
        session.add_object('target')
        session.masks['target'] = {0: np.ones((32, 40), bool)}
        session.prompt(0, 'target', 'Negative box', (5, 5))
        session.prompt(0, 'target', 'Negative box', (15, 15))
        session._store(0, [1], torch.ones((1, 1, 32, 40)))
        self.assertFalse(session.masks['target'][0][5:15, 5:15].any())
        self.assertTrue(session.masks['target'][0][0, 0])


class MotionTests(unittest.TestCase):
    def test_nami_joint_locks_always_send_individual_commands_without_reading_state(self):
        from communication.robot import NamiRobot
        robot = NamiRobot.__new__(NamiRobot)
        robot.robot_client = Mock()
        robot.robot_client.set_locked_joint.return_value = {'code': '0', 'msg': ''}
        for _ in range(2):
            robot.set_locked_joints([0, 1, 2, 3])
        self.assertEqual(robot.robot_client.mock_calls,
                         [call.set_locked_joint(i) for i in (1, 2, 3, 4) * 2])
        robot.robot_client.reset_mock()
        for _ in range(2):
            robot.set_locked_joints([])
        self.assertEqual(robot.robot_client.mock_calls, [call.set_locked_joint(0)] * 2)

    def test_nami_invalid_locks_and_rejected_command_do_not_continue(self):
        from communication.robot import NamiRobot
        robot = NamiRobot.__new__(NamiRobot)
        robot.robot_client = Mock()
        for indices in ([0, 0], [0, -1], [0, 18], [True], [1.5]):
            with self.assertRaises(ValueError):
                robot.set_locked_joints(indices)
        self.assertFalse(robot.robot_client.mock_calls)
        robot.robot_client.set_locked_joint.side_effect = [
            {'code': '0', 'msg': ''}, {'code': '7', 'msg': 'rejected'}]
        with self.assertRaisesRegex(RuntimeError, r'set_locked_joint\(2\) failed'):
            robot.set_locked_joints([0, 1, 2, 3])
        self.assertEqual(robot.robot_client.mock_calls,
                         [call.set_locked_joint(1), call.set_locked_joint(2)])

    def test_nami_collection_and_rollout_apply_locks_before_teleop(self):
        from data_collector.config import get_config
        from middle_level_controller.visual_servo.config import CUSTOM_TASK_CONFIG
        from communication.robot import NamiRobot
        config = get_config('nami_visual_servo')
        robot = NamiRobot.__new__(NamiRobot)
        robot.robot_client = Mock()
        robot.robot_client.set_locked_joint.return_value = {'code': '0', 'msg': ''}
        controller = Mock(robot_ids=[0], robot={0: robot}, robot_config=config.robot_config,
                          control_mode='task_abs')
        events = Mock()
        events.attach_mock(robot.robot_client.set_locked_joint, 'lock')
        events.attach_mock(controller.set_teleop, 'teleop')
        for task_config in (config.task_config, CUSTOM_TASK_CONFIG):
            events.reset_mock()
            task_config.extra_config.start_movement_fn(controller)
            self.assertEqual(events.mock_calls, [call.lock(1), call.lock(2), call.lock(3),
                                                call.lock(4), call.teleop(0, mode='task_abs')])

    def test_visual_servo_spacemouse_preserves_legacy_sensitivity(self):
        from data_collector.config import get_config
        from data_collector.device.spacemouse import SpaceMouse
        from helper.math_utils import TaskControlTransformation
        from third_party.spacemouse.spacemouse import Spacemouse

        config = get_config('nami_visual_servo')
        params = config.task_config.data_config.device_params
        driver = Spacemouse.__new__(Spacemouse)
        driver.max_value = params['init_params']['max_value']
        driver.deadzone = np.full(6, params['init_params']['deadzone'])
        driver.dtype = np.float32
        driver.tx_zup_spnav = np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]])
        driver.ring_buffer = Mock()
        driver.ring_buffer.get.return_value = {
            'motion_event': np.full(7, 39), 'button_state': [False, False]}
        np.testing.assert_array_equal(driver.get_motion_state_transformed(), np.zeros(6))
        driver.ring_buffer.get.return_value['motion_event'][:] = 100

        device = SpaceMouse.__new__(SpaceMouse)
        device.control_transform = TaskControlTransformation([0., 0., 0.])
        device.device_params = params
        device.control_dt = config.robot_config.control_dt
        device.device = driver
        device.reset()
        initial = np.array([100., 200., 300., -160., 20., 40.])
        # The first input anchors the target at the current TCP pose.
        np.testing.assert_allclose(device.get_input(robot_pose=initial)['control'], initial, atol=1e-5)
        for _ in range(round(1. / device.control_dt)):
            result = np.array(device.get_input(robot_pose=initial)['control'])
        # Raw input 100 = one mouse unit: 2 mm and 0.1 rad over one second.
        np.testing.assert_allclose(result[:3] - initial[:3], [2., -2., 2.], atol=1e-5)
        np.testing.assert_allclose(angle_delta(result[3:5], initial[3:5]), 0., atol=1e-5)
        self.assertAlmostEqual(angle_delta(result[5], initial[5]), np.rad2deg(.1), places=5)

    def test_spacemouse_locks_roll_pitch_and_translation_axis(self):
        from data_collector.device.spacemouse import SpaceMouse
        from helper.math_utils import TaskControlTransformation
        device = SpaceMouse.__new__(SpaceMouse)
        device.control_transform = TaskControlTransformation([0., 0., 0.])
        device.device_params = {'max_pos_speed': .01, 'max_rot_speed': .1,
                                'fix': {'roll': True, 'pitch': True, 'x': True}}
        device.control_dt = .05
        device.device = Mock()
        device.device.is_button_pressed.return_value = False
        device.device.get_motion_state_transformed.return_value = np.ones(6)
        device.reset()
        initial = np.array([100, 200, 300, -160, 20, 40.])
        for _ in range(5):
            result = np.array(device.get_input(robot_pose=initial)['control'])
        np.testing.assert_allclose(angle_delta(result[3:5], initial[3:5]), 0, atol=1e-5)
        self.assertAlmostEqual(result[0], initial[0])
        self.assertGreater(result[1], initial[1])
        self.assertGreater(angle_delta(result[5], initial[5]), 0)

    def test_nami_adapter_routes_only_left_task_chain(self):
        from data_collector.config import get_config
        from communication.robot import NamiRobot
        config = get_config('nami_visual_servo')
        adapter = config.teleop_adapter_factory(config)
        robot = Mock(gripper_arm_indices=())
        robot.validate_task_command.side_effect = lambda pose, arm: list(pose)
        robot.validate_joint_command.side_effect = list
        robot.get_task_pose.side_effect = lambda state, arm: state['p'][6*arm:6*(arm+1)]
        adapter.bind(Mock(robots={0: robot}), {1: Mock()}, 'task_abs')
        buffer = {'p_0': list(range(18)), 'q_0': [0.] * 22}
        self.assertEqual(adapter.device_poses(buffer), {'p_1': list(range(6, 12))})
        adapter.reset(buffer)
        action, _ = adapter.convert({1: {'control': [1., 2., 3., 4., 5., 6.], 'trigger': 0}}, buffer)
        adapter.send(action, 'task_abs', {0: .5}, {0: 1.})
        self.assertEqual(robot.tele_move.call_args.kwargs['arm_index'], 1)
        self.assertEqual(adapter.recorded_actions(action)[0], [1., 2., 3., 4., 5., 6.])
        self.assertEqual(adapter.recording_metadata['arm_indices'], (1,))

    def test_correction_clips_norm_wraps_yaw_and_fixes_roll_pitch(self):
        current = np.array([0., 0., 0., 170, 10, 179])
        target = constrain_target(current, [300, 400, 0, 0, 0, -179], [170, 10], SERVO_CONFIG)
        self.assertAlmostEqual(np.linalg.norm(target[:3]), 50)
        np.testing.assert_array_equal(target[3:5], [170, 10])
        self.assertAlmostEqual(target[5], 181.)

    def test_policy_waits_for_target_and_stops_after_convergence(self):
        policy = NN_policy.__new__(NN_policy)
        policy.policy = Mock()
        policy.use_gripper = False
        policy.tracker = Mock(return_value={'valid': True, 'mask': np.ones((32, 40, 1))})
        policy.settings = SERVO_CONFIG.copy()
        policy.reset()
        image = np.ones((32, 40, 3), np.uint8)
        predicted = np.array([20., 0, 0, 0, 0, 5])
        result = {'robot_action_0': predicted, 'control_state': State.TASK_IN_PROGRESS}
        with patch.object(ACTPolicyWrapper, '__call__', return_value=result) as forward:
            policy(end_pose=np.zeros(6), **{'images.rgb.wrist': image})
            policy(end_pose=np.array([1., 0, 0, 0, 0, 1]), **{'images.rgb.wrist': image})
            self.assertEqual(forward.call_count, 1)
            done = policy(end_pose=predicted.copy(), **{'images.rgb.wrist': image})
            self.assertEqual(forward.call_count, 2)
            self.assertEqual(done['control_state'], State.TASK_FINISH)

    def test_controller_stops_on_exception_and_sends_no_wrong_arm_command(self):
        from middle_level_controller.visual_servo.controller import NN_controller
        controller = NN_controller.__new__(NN_controller)
        robot = Mock()
        robot.get_state.return_value = {'op_state': 17, 'q': [0.]*22, 'qdot': [0.]*22, 'p': [0.]*18, 'pdot': [0.]*18}
        robot.get_task_pose.return_value = [0.]*6
        controller.robot = {0: robot}
        camera = Mock()
        camera.get_all.return_value = {'rgb': np.zeros((32, 40, 3), np.uint8), 'timestamp': time.monotonic()}
        controller.camera = {'wrist': camera}
        controller.nn_policy = Mock(side_effect=ConnectionError('server down'))
        controller.exec_start_movement = Mock(return_value=State.TASK_IN_PROGRESS)
        controller.exec_finish_movement = Mock(return_value=State.TASK_FAIL)
        controller._control_triggered = True
        with contextlib.redirect_stderr(io.StringIO()):
            controller._nn_control_fn(1.)
        robot.stop_teleop.assert_called_once()
        robot.tele_move.assert_not_called()
        self.assertFalse(controller._control_triggered)
        self.assertEqual(controller.control_state, State.TASK_FAIL)


if __name__ == '__main__':
    unittest.main()
