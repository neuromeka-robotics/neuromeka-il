"""Stream a recorded humanoid joint trajectory at 20 Hz."""

from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path
from threading import Thread, current_thread
from typing import Dict

import numpy as np

from communication.humanoid import HumanoidRobot
from communication.robot import Robot
from data_collector.pink_ik import DCP_ACTIVE_JOINT_NAMES
from helper.controller_utils import Controller

from .config import (
    CUSTOM_ROBOT_CONFIG,
    CUSTOM_TASK_CONFIG,
    JOINT_STATE_LOG_DIR,
    START_POSITION_TOLERANCE_DEG,
    TRAJECTORY_PATH,
)


class NN_controller(Controller):
    """`task_demo.py`-compatible controller for one open-loop trajectory."""

    robot_config = CUSTOM_ROBOT_CONFIG
    robot_ids = CUSTOM_ROBOT_CONFIG.robot_ids
    task_config = CUSTOM_TASK_CONFIG
    task_name = CUSTOM_TASK_CONFIG.name

    def __init__(self, robot: Dict[int, Robot] | None = None, **kwargs):
        if len(self.robot_ids) != 1:
            raise ValueError("The box-lift trajectory requires exactly one robot")

        super().__init__(robot=robot, **kwargs)
        self.camera = kwargs.get("camera", {})
        self.trajectory: np.ndarray | None = None
        self._control_triggered = False
        self._control_thread: Thread | None = None
        self._control_error: Exception | None = None

    @staticmethod
    def _load_trajectory(path: str | Path) -> np.ndarray:
        path = Path(path)
        try:
            with path.open(newline="") as csv_file:
                header = next(csv.reader(csv_file))
        except FileNotFoundError:
            raise FileNotFoundError(f"Trajectory file does not exist: {path}") from None
        except StopIteration:
            raise ValueError(f"Trajectory file is empty: {path}") from None

        if len(header) != len(set(header)):
            raise ValueError("Trajectory CSV contains duplicate column names")

        expected_columns = tuple(
            f"{joint_name}_deg" for joint_name in DCP_ACTIVE_JOINT_NAMES)
        missing_columns = [
            name for name in expected_columns if name not in header]
        unexpected_columns = [
            name for name in header if name not in expected_columns]
        if missing_columns or unexpected_columns:
            raise ValueError(
                "Trajectory CSV joint columns do not match the humanoid; "
                f"missing={missing_columns}, unexpected={unexpected_columns}")

        try:
            source_trajectory = np.loadtxt(
                path, delimiter=",", skiprows=1, ndmin=2)
        except ValueError as exc:
            raise ValueError(
                f"Trajectory CSV contains invalid numeric data: {exc}") from exc

        if source_trajectory.shape[0] == 0:
            raise ValueError("Trajectory CSV does not contain any commands")
        if source_trajectory.shape[1] != len(header):
            raise ValueError(
                "Trajectory row width does not match its header: "
                f"expected {len(header)}, got {source_trajectory.shape[1]}")
        if not np.all(np.isfinite(source_trajectory)):
            raise ValueError("Trajectory CSV contains NaN or infinite values")

        # The file interleaves left/right joints. Reorder by header into the
        # DCP q18 convention, then add the controller's four dummy joints.
        source_indices = [header.index(name) for name in expected_columns]
        q18_trajectory = source_trajectory[:, source_indices]
        dummy_joints = np.zeros(
            (q18_trajectory.shape[0], HumanoidRobot.DUMMY_JOINT_DOF),
            dtype=q18_trajectory.dtype,
        )
        q22_trajectory = np.concatenate(
            (q18_trajectory, dummy_joints), axis=1)

        if q22_trajectory.shape[1] != HumanoidRobot.JOINT_DOF:
            raise ValueError(
                "Converted trajectory has an invalid joint count: "
                f"expected {HumanoidRobot.JOINT_DOF}, "
                f"got {q22_trajectory.shape[1]}")
        return q22_trajectory

    def load_policy(self):
        """Load and validate the CSV in place of a neural-network policy."""
        self.trajectory = self._load_trajectory(TRAJECTORY_PATH)
        duration = (len(self.trajectory) - 1) * self.robot_config.control_dt
        print(
            f"Loaded {len(self.trajectory)} joint commands from "
            f"{TRAJECTORY_PATH} ({duration:.2f} s at "
            f"{1. / self.robot_config.control_dt:.1f} Hz)")

    def exec_nn_control(self, duration: float):
        if self._control_thread is not None and self._control_thread.is_alive():
            print("Control loop is already triggered")
            return
        if duration <= 0.:
            raise ValueError("Control duration must be positive")
        if self.trajectory is None:
            self.load_policy()

        self._control_error = None
        self._control_triggered = True
        self._control_thread = Thread(
            target=self._open_loop_control_fn,
            args=(duration,),
            daemon=True,
        )
        self._control_thread.start()

    def exec_nn_control_stop(self):
        self._control_triggered = False
        control_thread = self._control_thread
        if control_thread is not None and control_thread is not current_thread():
            control_thread.join()
        if self._control_thread is control_thread:
            self._control_thread = None

    def _check_start_position(self) -> None:
        robot_id = self.robot_ids[0]
        state = self.robot[robot_id].get_state()
        current_joint_pos = np.asarray(state["q"], dtype=np.float64)
        first_command = self.trajectory[0]
        if current_joint_pos.shape != first_command.shape:
            raise ValueError(
                "Robot state has an invalid joint count: "
                f"expected {first_command.size}, got {current_joint_pos.size}")

        error = float(np.linalg.norm(first_command - current_joint_pos))
        if error > START_POSITION_TOLERANCE_DEG:
            raise RuntimeError(
                "Move the robot to the task home position before executing "
                f"the trajectory (joint error norm: {error:.3f} deg)")

    def _send_joint_command(self, command: np.ndarray) -> Dict[int, list[float]]:
        robot_id = self.robot_ids[0]
        action = {
            robot_id: self.robot[robot_id].validate_joint_command(command)}
        control = self.robot_config.robot_params[robot_id]["control"]
        self.robot_cluster.tele_move(
            action=action,
            mode="joint_abs",
            vel_scale={robot_id: control["vel_scale"]},
            acc_scale={robot_id: control["acc_scale"]},
        )
        return action

    def _read_joint_state(
            self, sample_index: int, start_time: float) -> list[float]:
        robot_id = self.robot_ids[0]
        state = self.robot[robot_id].get_state()
        joint_position = self.robot[robot_id].validate_joint_command(
            state["q"])
        joint_velocity = np.asarray(state["qdot"], dtype=np.float64)
        if (joint_velocity.ndim != 1
                or joint_velocity.size != HumanoidRobot.JOINT_DOF):
            raise ValueError(
                "Robot joint velocity has an invalid shape: expected "
                f"({HumanoidRobot.JOINT_DOF},), got {joint_velocity.shape}")
        if not np.all(np.isfinite(joint_velocity)):
            raise ValueError(
                "Robot joint velocity contains NaN or infinite values")

        return [
            sample_index,
            time.monotonic() - start_time,
            time.time(),
            int(state["op_state"]),
            *joint_position,
            *joint_velocity.tolist(),
        ]

    @staticmethod
    def _save_joint_states(records: list[list[float]]) -> Path:
        JOINT_STATE_LOG_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = JOINT_STATE_LOG_DIR / f"joint_states_{timestamp}.csv"

        dummy_joint_names = tuple(
            f"Dummy_{index}"
            for index in range(HumanoidRobot.DUMMY_JOINT_DOF))
        joint_names = DCP_ACTIVE_JOINT_NAMES + dummy_joint_names
        header = [
            "sample_index",
            "elapsed_time_s",
            "wall_time_s",
            "op_state",
            *(f"q_{name}_deg" for name in joint_names),
            *(f"qdot_{name}_deg_s" for name in joint_names),
        ]

        with output_path.open("x", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(header)
            writer.writerows(records)
        return output_path

    def _open_loop_control_fn(self, duration: float):
        compliance_attempted = False
        teleop_attempted = False
        last_action = None
        commands_sent = 0
        joint_state_records = []

        try:
            self._check_start_position()

            compliance_attempted = (
                self.task_config.control_config.compliance.enable)
            if compliance_attempted:
                self.exec_enable_compliance()

            teleop_attempted = True
            self.exec_start_movement(control_mode="joint_abs")

            start_time = time.monotonic()
            next_tick = start_time
            for index, command in enumerate(self.trajectory):
                if not self._control_triggered:
                    break
                if time.monotonic() - start_time >= duration:
                    print("Open-loop execution reached its duration limit")
                    break

                joint_state = self._read_joint_state(index, start_time)
                last_action = self._send_joint_command(command)
                joint_state_records.append(joint_state)
                commands_sent += 1

                if index + 1 < len(self.trajectory):
                    next_tick += self.robot_config.control_dt
                    wait_time = next_tick - time.monotonic()
                    if wait_time > 0.:
                        time.sleep(wait_time)

            if commands_sent == len(self.trajectory):
                print("Open-loop trajectory completed")
            elif not self._control_triggered:
                print("Open-loop trajectory stopped")
        except Exception as exc:
            self._control_error = exc
            print(f"Open-loop trajectory failed: {exc}")
        finally:
            if teleop_attempted:
                if last_action is not None:
                    try:
                        self.exec_soft_stop(
                            last_action=last_action,
                            control_period=self.robot_config.control_dt,
                            mode="joint_abs",
                        )
                    except Exception as exc:
                        if self._control_error is None:
                            self._control_error = exc
                        print(f"Open-loop soft stop failed: {exc}")
                try:
                    # Stop teleoperation before disabling compliance.
                    self.exec_finish_movement()
                except Exception as exc:
                    if self._control_error is None:
                        self._control_error = exc
                    print(f"Failed to leave teleoperation mode: {exc}")

            if compliance_attempted:
                try:
                    # self.exec_disable_compliance()
                    pass
                except Exception as exc:
                    if self._control_error is None:
                        self._control_error = exc
                    print(f"Failed to disable compliance: {exc}")

            if joint_state_records:
                try:
                    output_path = self._save_joint_states(joint_state_records)
                    print(
                        f"Saved {len(joint_state_records)} joint states to "
                        f"{output_path}")
                except Exception as exc:
                    if self._control_error is None:
                        self._control_error = exc
                    print(f"Failed to save joint states: {exc}")

            self._control_triggered = False
            self._control_thread = None
