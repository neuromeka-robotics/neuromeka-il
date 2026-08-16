import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from calibrate_vive import execute_and_collect
from communication.humanoid import HumanoidRobot
from data_collector.control_adapter import (
    ControlConversionError,
    convert_device_control,
)


class HumanoidControlConversionTest(unittest.TestCase):
    def setUp(self):
        self.robot = HumanoidRobot.__new__(HumanoidRobot)
        self.robot.robot_client = MagicMock()
        self.state = {
            "q": [float(index) for index in range(22)],
            "p": [float(index) for index in range(18)],
        }

    def test_joint_passthrough_preserves_all_22_values(self):
        command = [float(index) for index in range(22)]
        converted = convert_device_control(
            self.robot, command, "joint_abs", "joint_abs",
            self.state, arm_index=1)
        self.assertEqual(converted.command, command)

    def test_joint_passthrough_rejects_partial_arm_vector(self):
        with self.assertRaisesRegex(ValueError, "exactly 22"):
            convert_device_control(
                self.robot, [0.] * 18, "joint_abs", "joint_abs",
                self.state, arm_index=1)

    def test_task_passthrough_is_six_values(self):
        command = [1., 2., 3., 4., 5., 6.]
        converted = convert_device_control(
            self.robot, command, "task_abs", "task_abs",
            self.state, arm_index=1)
        self.assertEqual(converted.command, command)

    def test_pink_task_to_joint_uses_locked_solver_result(self):
        active_joints = [float(index + 1) for index in range(18)]
        locked_result = self.state["q"][:18]
        locked_result[4:11] = active_joints[4:11]
        initial_reference = [200. + index for index in range(22)]
        expected = initial_reference.copy()
        expected[4:11] = locked_result[4:11]
        expected[18:] = [0.] * 4
        pink_solver = MagicMock()
        pink_solver.solve.return_value = {
            "jpos": locked_result,
            "success": True,
        }
        converted = convert_device_control(
            self.robot, [0.] * 6, "task_abs", "joint_abs",
            self.state, arm_index=1,
            ik_type="pink", lock_non_selected_joints=True,
            locked_joint_reference=initial_reference,
            pink_solver=pink_solver)

        self.assertEqual(converted.command, expected)
        pink_solver.solve.assert_called_once_with(
            tpos=[0.] * 6,
            init_jpos=self.state["q"],
            arm_index=1,
            lock_non_selected_joints=True,
        )
        self.robot.robot_client.inverse_kin.assert_not_called()

    def test_step_ik_rejects_joint_locking(self):
        with self.assertRaisesRegex(ValueError, "unavailable for STEP IK"):
            convert_device_control(
                self.robot, [0.] * 6, "task_abs", "joint_abs",
                self.state, arm_index=1,
                ik_type="step", lock_non_selected_joints=True)

    def test_locked_command_uses_initial_non_selected_joints(self):
        ik_active = [100. + index for index in range(18)]
        initial_reference = [200. + index for index in range(22)]
        expected = initial_reference.copy()
        expected[4:11] = ik_active[4:11]
        expected[18:] = [0.] * 4

        command = self.robot.make_joint_command_from_ik(
            ik_active,
            joint_reference=initial_reference,
            arm_index=1,
            lock_non_selected_joints=True,
        )

        self.assertEqual(command, expected)

    def test_unlocked_ik_applies_complete_active_joint_result(self):
        ik_active = [100. + index for index in range(18)]
        command = self.robot.make_joint_command_from_ik(
            ik_active,
            joint_reference=self.state["q"],
            arm_index=1,
            lock_non_selected_joints=False,
        )
        self.assertEqual(command, ik_active + [0.] * 4)

    def test_failed_ik_ends_conversion_without_fallback_command(self):
        self.robot.robot_client.inverse_kin.return_value = {
            "response": {"code": "7", "msg": "no solution"},
        }
        with self.assertRaisesRegex(ControlConversionError, "no solution"):
            convert_device_control(
                self.robot, [0.] * 6, "task_abs", "joint_abs",
                self.state, arm_index=2)

    def test_joint_to_task_uses_arm(self):
        expected = [10., 20., 30., 40., 50., 60.]
        self.robot.robot_client.forward_kin.return_value = {
            "tpos": expected,
            "response": {"code": "0"},
        }
        converted = convert_device_control(
            self.robot, self.state["q"], "joint_abs", "task_abs",
            self.state, arm_index=2)

        self.assertEqual(converted.command, expected)
        self.robot.robot_client.forward_kin.assert_called_once_with(
            jpos=self.state["q"], arm_index=2)


class IntegratedDhGripperTest(unittest.TestCase):
    def test_activation_and_position_commands_use_four_pvt_values(self):
        client = MagicMock()
        with patch("communication.robot.RobotClient", return_value=client):
            robot = HumanoidRobot(
                robot_ip="unused",
                gripper_config={
                    "enable": True,
                    "backend": "integrated_dh",
                    "tool_index": 0,
                },
            )

        first_call = client.set_gripper_command.call_args_list[0]
        self.assertEqual(first_call.kwargs["command"], 0)
        self.assertEqual(first_call.kwargs["gripper_type"], 2)
        self.assertEqual(first_call.kwargs["pvt_data"], [0, 0, 0, 0])

        robot.move_gripper(mode="thread", value=1.)
        move_call = client.set_gripper_command.call_args_list[-1]
        self.assertEqual(move_call.kwargs["command"], 2)
        self.assertEqual(move_call.kwargs["pvt_data"], [1000, 0, 0, 0])

        call_count = client.set_gripper_command.call_count
        robot.move_gripper(mode="thread", value=1.)
        self.assertEqual(client.set_gripper_command.call_count, call_count)


class CalibrationComplianceLifecycleTest(unittest.TestCase):
    def test_compliance_wraps_task_teleoperation(self):
        events = []

        class FakeRobot:
            state_count = 0

            def get_state(self):
                self.state_count += 1
                return {
                    "op_state": 5 if self.state_count == 1 else 17,
                    "p": [0.] * 18,
                }

            def get_task_pose(self, state, arm_index):
                start = arm_index * 6
                return state["p"][start:start + 6]

            def set_compliance_mode(self, enable, stiffness=None):
                events.append("compliance_on" if enable else "compliance_off")

            def start_teleop(self, mode):
                events.append("start_teleop")

            def tele_move(self, **kwargs):
                events.append("move")

            def stop_teleop(self):
                events.append("stop_teleop")

        pose = SimpleNamespace(m=[
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
        ])
        vive = SimpleNamespace(get_pose_matrix=lambda: pose)
        compliance = SimpleNamespace(enable=True, stiffness=[100] * 22)

        with patch("calibrate_vive.time.sleep"):
            execute_and_collect(
                robot=FakeRobot(),
                robot_params={"control": {"vel_scale": 1., "acc_scale": 1.}},
                vive_device=vive,
                trajectory=np.asarray([[0.] * 6]),
                control_dt=0.,
                arm_index=1,
                compliance=compliance,
            )

        self.assertEqual(events, [
            "compliance_on",
            "start_teleop",
            "move",
            "stop_teleop",
            "compliance_off",
        ])


if __name__ == "__main__":
    unittest.main()
