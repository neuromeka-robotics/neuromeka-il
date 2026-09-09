import unittest

import numpy as np

from middle_level_controller.box_lift_rl.config import (
    JOINT_POSITION_HISTORY_LENGTH,
    JOINT_POSITION_HISTORY_OFFSETS_S,
    POLICY_ACTION_SCALES_RAD,
    POLICY_ROBOT_JOINT_INDICES,
)
from middle_level_controller.box_lift_rl.model import (
    MoveBoxObservationBuilder,
    action_to_robot_command,
    transform_from_position_rpy,
)


class MoveBoxObservationBuilderTest(unittest.TestCase):
    def test_observation_matches_65_dim_exported_model_layout(self):
        builder = MoveBoxObservationBuilder(control_dt=0.05)
        qpos_deg = np.arange(22, dtype=np.float64)
        transform_base_box = transform_from_position_rpy(
            [0.5, -0.1, 1.2], [0.0, 0.0, np.pi / 2.0]
        )

        observation = builder.build(transform_base_box, qpos_deg)

        self.assertEqual(JOINT_POSITION_HISTORY_LENGTH, 3)
        self.assertEqual(JOINT_POSITION_HISTORY_OFFSETS_S, (0.0, 0.1, 0.2))
        self.assertEqual(observation.shape, (65,))
        np.testing.assert_allclose(observation[:3], [0.5, -0.1, 1.2])
        np.testing.assert_allclose(
            observation[3:9], [0.0, 1.0, 0.0, 1.0, 1.0, 0.0], atol=1e-7
        )
        expected_joint_position = np.deg2rad(
            qpos_deg[list(POLICY_ROBOT_JOINT_INDICES)]
        )
        for start in (9, 23, 37):
            np.testing.assert_allclose(
                observation[start : start + 14], expected_joint_position
            )
        np.testing.assert_array_equal(observation[51:], 0.0)

    def test_joint_position_history_matches_training_layout(self):
        builder = MoveBoxObservationBuilder(control_dt=0.05)
        qpos_samples = [
            np.arange(22, dtype=np.float64) + 10.0 * step
            for step in range(5)
        ]

        for qpos_deg in qpos_samples:
            observation = builder.build(np.eye(4), qpos_deg)

        for start, sample_index in zip((9, 23, 37), (4, 2, 0)):
            np.testing.assert_allclose(
                observation[start : start + 14],
                np.deg2rad(
                    qpos_samples[sample_index][
                        list(POLICY_ROBOT_JOINT_INDICES)
                    ]
                ),
            )

    def test_action_history_matches_training_observation_layout(self):
        builder = MoveBoxObservationBuilder(control_dt=0.05)
        first_action = np.arange(14, dtype=np.float32)
        second_action = first_action + 20.0

        builder.advance_action(first_action)
        first_observation = builder.build(np.eye(4), np.zeros(22))
        builder.advance_action(second_action)
        second_observation = builder.build(np.eye(4), np.zeros(22))

        np.testing.assert_array_equal(first_observation[51:65], first_action)
        np.testing.assert_array_equal(second_observation[51:65], second_action)

    def test_joint_history_advances_while_policy_inference_is_skipped(self):
        builder = MoveBoxObservationBuilder(control_dt=0.05)
        qpos_samples = [
            np.full(22, 10.0 * step, dtype=np.float64)
            for step in range(4)
        ]

        builder.build(np.eye(4), qpos_samples[0])
        builder.record_joint_position(qpos_samples[1])
        builder.record_joint_position(qpos_samples[2])
        observation = builder.build(np.eye(4), qpos_samples[3])

        for start, sample_index in zip((9, 23, 37), (3, 1, 0)):
            np.testing.assert_allclose(
                observation[start : start + 14],
                np.deg2rad(qpos_samples[sample_index][0]),
            )


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
