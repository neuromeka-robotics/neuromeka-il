import unittest

import numpy as np

from middle_level_controller.box_lift_rl.config import (
    POLICY_ACTION_SCALES_RAD,
    POLICY_ROBOT_JOINT_INDICES,
)
from middle_level_controller.box_lift_rl.model import (
    MoveBoxObservationBuilder,
    action_to_robot_command,
    transform_from_position_rpy,
)


class MoveBoxObservationBuilderTest(unittest.TestCase):
    def test_observation_matches_51_dim_exported_model_layout(self):
        builder = MoveBoxObservationBuilder()
        qpos_deg = np.arange(22, dtype=np.float64)
        transform_base_box = transform_from_position_rpy(
            [0.5, -0.1, 1.2], [0.0, 0.0, np.pi / 2.0]
        )

        observation = builder.build(transform_base_box, qpos_deg)

        self.assertEqual(observation.shape, (51,))
        np.testing.assert_allclose(observation[:3], [0.5, -0.1, 1.2])
        np.testing.assert_allclose(
            observation[3:9], [0.0, 1.0, 0.0, 1.0, 1.0, 0.0], atol=1e-7
        )
        np.testing.assert_allclose(
            observation[9:23],
            np.deg2rad(qpos_deg[list(POLICY_ROBOT_JOINT_INDICES)]),
        )
        np.testing.assert_array_equal(observation[23:], 0.0)

    def test_action_history_matches_training_observation_layout(self):
        builder = MoveBoxObservationBuilder()
        first_action = np.arange(14, dtype=np.float32)
        second_action = first_action + 20.0

        builder.advance_action(first_action)
        first_observation = builder.build(np.eye(4), np.zeros(22))
        builder.advance_action(second_action)
        second_observation = builder.build(np.eye(4), np.zeros(22))

        np.testing.assert_array_equal(first_observation[23:37], first_action)
        np.testing.assert_array_equal(first_observation[37:51], 0.0)
        np.testing.assert_array_equal(second_observation[23:37], second_action)
        np.testing.assert_array_equal(second_observation[37:51], first_action)


class MoveBoxActionConversionTest(unittest.TestCase):
    def test_action_scales_match_saved_training_configuration(self):
        np.testing.assert_array_equal(
            POLICY_ACTION_SCALES_RAD,
            np.full(14, 0.02),
        )

    def test_relative_policy_action_maps_to_robot_q22_degrees(self):
        current_qpos_deg = np.arange(22, dtype=np.float64)
        home_qpos_deg = np.full(22, -100.0)
        policy_action = np.ones(14)

        command = action_to_robot_command(
            policy_action, current_qpos_deg, home_qpos_deg
        )

        indices = np.asarray(POLICY_ROBOT_JOINT_INDICES)
        expected_policy_joints = (
            current_qpos_deg[indices]
            + np.rad2deg(np.asarray(POLICY_ACTION_SCALES_RAD))
        )
        np.testing.assert_allclose(command[indices], expected_policy_joints)
        locked_indices = sorted(set(range(22)) - set(indices.tolist()))
        np.testing.assert_array_equal(command[locked_indices], -100.0)

    # Temporarily disabled: action clamping is active instead of raising.
    # def test_rejects_unsafe_policy_target_step_without_clipping(self):
    #     with self.assertRaisesRegex(RuntimeError, "unsafe one-cycle"):
    #         action_to_robot_command(
    #             np.full(14, 10.0),
    #             np.zeros(22),
    #             np.zeros(22),
    #         )


if __name__ == "__main__":
    unittest.main()
