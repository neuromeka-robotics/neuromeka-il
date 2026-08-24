import unittest

import numpy as np

from data_collector.config import CONFIGS
from helper.math_utils import MathFunc, TaskControlTransformation


class TaskControlTransformationTest(unittest.TestCase):
    def test_absolute_mode_remains_anchored_to_initial_robot_pose(self):
        transform = TaskControlTransformation(
            fixed_robot_to_fixed_device_euler=[0., 0., 0.],
            tracking_mode="absolute",
        )
        transform.current_device.pos = np.asarray([100., 50., -20.])
        transform.current_device.rot = np.eye(3)
        transform.initialize([10., 20., 30., 0., 0., 20.])

        transform.current_device.pos = np.asarray([105., 48., -17.])
        transform.current_device.rot = MathFunc.euler_to_rotMat(
            0., 0., np.deg2rad(10.))
        output = transform.apply(
            # Absolute mode intentionally ignores this newer robot pose.
            robot_pose=[1000., 1000., 1000., 0., 0., 90.],
        )

        np.testing.assert_allclose(output[:3], [15., 18., 33.], atol=1e-6)
        np.testing.assert_allclose(output[3:], [0., 0., 30.], atol=1e-5)

    def test_relative_mode_applies_latest_delta_to_measured_robot_pose(self):
        transform = TaskControlTransformation(
            fixed_robot_to_fixed_device_euler=[0., 0., 0.],
            tracking_mode="relative",
        )
        transform.current_device.pos = np.zeros(3)
        transform.current_device.rot = np.eye(3)
        transform.initialize([100., 200., 300., 0., 0., 20.])

        initial_output = transform.apply(
            robot_pose=[100., 200., 300., 0., 0., 20.])
        np.testing.assert_allclose(
            initial_output, [100., 200., 300., 0., 0., 20.], atol=1e-5)

        transform.current_device.pos = np.asarray([10., 0., 0.])
        transform.current_device.rot = MathFunc.euler_to_rotMat(
            0., 0., np.deg2rad(10.))
        moved_output = transform.apply(
            robot_pose=[95., 205., 300., 0., 0., 20.])

        np.testing.assert_allclose(
            moved_output[:3], [105., 205., 300.], atol=1e-6)
        np.testing.assert_allclose(
            moved_output[3:], [0., 0., 30.], atol=1e-5)

    def test_relative_mode_accepts_compliance_motion_when_device_is_still(self):
        transform = TaskControlTransformation(
            fixed_robot_to_fixed_device_euler=[0., 0., 0.],
            tracking_mode="relative",
        )
        transform.current_device.pos = np.zeros(3)
        transform.current_device.rot = np.eye(3)
        transform.initialize([100., 200., 300., 0., 0., 0.])
        transform.apply(robot_pose=[100., 200., 300., 0., 0., 0.])

        # Move the device once so this becomes the new previous device pose.
        transform.current_device.pos = np.asarray([5., 0., 0.])
        transform.apply(robot_pose=[100., 200., 300., 0., 0., 0.])

        # With no new device motion, the target follows the displaced robot
        # exactly rather than returning to either earlier absolute target.
        compliant_pose = [92., 207., 296., 2., -3., 4.]
        output = transform.apply(robot_pose=compliant_pose)

        np.testing.assert_allclose(output, compliant_pose, atol=1e-5)

    def test_relative_mode_requires_current_robot_pose(self):
        transform = TaskControlTransformation(
            fixed_robot_to_fixed_device_euler=[0., 0., 0.],
            tracking_mode="relative",
        )
        with self.assertRaisesRegex(ValueError, "current robot task pose"):
            transform.apply()

    def test_invalid_tracking_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unavailable tracking mode"):
            TaskControlTransformation(
                fixed_robot_to_fixed_device_euler=[0., 0., 0.],
                tracking_mode="invalid",
            )

    def test_lift_box_vive_uses_accumulated_absolute_tracking(self):
        params = CONFIGS["lift_box"].task_config.data_config.device_params
        self.assertEqual(params["tracking_mode"], "absolute")


if __name__ == "__main__":
    unittest.main()
