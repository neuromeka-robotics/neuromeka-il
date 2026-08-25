import unittest

from helper.config_utils import (
    COMPLIANCE_CONFIG,
    CONTROL_CONFIG,
    TASK_CONFIG,
    TELEOP_CONFIG,
)


class ControlConfigTest(unittest.TestCase):
    def test_robot_control_and_compliance_are_not_teleop_fields(self):
        task_config = TASK_CONFIG(
            control_config=CONTROL_CONFIG(
                robot_control_mode="joint_abs",
                compliance=COMPLIANCE_CONFIG(
                    enable=True,
                    stiffness=[100] * 22,
                ),
            )
        )

        self.assertEqual(
            task_config.control_config.robot_control_mode, "joint_abs"
        )
        self.assertTrue(task_config.control_config.compliance.enable)
        self.assertIsNone(task_config.control_config.teleop_config)

    def test_teleop_specific_settings_are_nested_under_control(self):
        teleop_config = TELEOP_CONFIG(
            arm_index=[1, 2],
            ik_type="pink",
            pink_config_path="eir.yaml",
            lock_non_selected_joints=True,
        )
        control_config = CONTROL_CONFIG(teleop_config=teleop_config)

        self.assertIs(control_config.teleop_config, teleop_config)
        self.assertEqual(control_config.teleop_config.arm_index, [1, 2])


if __name__ == "__main__":
    unittest.main()
