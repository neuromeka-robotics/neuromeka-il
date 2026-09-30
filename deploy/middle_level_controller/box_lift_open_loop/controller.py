"""Follow a fixed CSV trajectory through Pink or the compliant-plane projector."""

from __future__ import annotations

import csv
import time
from queue import Empty, SimpleQueue
from datetime import datetime
from pathlib import Path
from threading import Thread, current_thread
from typing import Dict

import numpy as np

from communication.humanoid import HumanoidRobot
from communication.robot import Robot
from data_collector.pink_ik import DCP_ACTIVE_JOINT_NAMES, PinkTeleopIK
from data_collector.rl_constraint import RLConstraintTeleop
from helper.controller_utils import Controller

from .config import (
    CUSTOM_ROBOT_CONFIG,
    CUSTOM_TASK_CONFIG,
    RESULT_DIR,
    START_POSITION_TOLERANCE_DEG,
    TRAJECTORY_PATH,
    TRAJECTORY_DT,
    COMMAND_OFFSET_S,
    HOLD_FIRST_TARGET,
    IK_TYPE,
    RL_START_DELAY_S,
    RL_CONSTRAINT_MODEL_PATH,
    RL_CONSTRAINT_DRY_RUN,
    RL_COMPLIANCE_COMMAND,
    RL_COMPLIANCE_INTERACTIVE,
)

TORQUE_FIELDS = ("tau", "tau_act", "tau_ext", "tau_jts")


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
        self.visualizer = kwargs.get("visualizer")
        self.camera = kwargs.get("camera", {})
        self.trajectory: np.ndarray | None = None
        self._control_triggered = False
        self._control_thread: Thread | None = None
        self._control_error: Exception | None = None
        self.pink_solver = None
        self.rl_constraint = None
        self.task_targets = None
        self._compliance_switches = SimpleQueue()

    @property
    def interactive_compliance_enabled(self):
        return (RL_COMPLIANCE_INTERACTIVE and IK_TYPE == "rl_constraint"
                and self.rl_constraint is not None
                and self.rl_constraint.uses_compliant_history)

    def toggle_rl_compliance_command(self):
        if self.interactive_compliance_enabled and self._control_triggered:
            self._compliance_switches.put(True)
        else:
            print("Policy compliance switching is available during interactive RL execution")

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
        """Load exact recorded commands and any FK targets needed by RL."""
        if IK_TYPE not in ("pink", "rl_constraint"):
            raise ValueError("IK_TYPE must be pink or rl_constraint")
        if IK_TYPE == "rl_constraint":
            if not RL_CONSTRAINT_MODEL_PATH:
                raise ValueError("Set RL_CONSTRAINT_MODEL_PATH in box_lift_open_loop/config.py")
            self.rl_constraint = RLConstraintTeleop(RL_CONSTRAINT_MODEL_PATH, self.robot_config.control_dt)
        source = self._load_trajectory(TRAJECTORY_PATH)
        self.trajectory = self._hold_trajectory(
            source, TRAJECTORY_DT, self.robot_config.control_dt)
        if HOLD_FIRST_TARGET:
            self.trajectory[:] = self.trajectory[0]
            print("Holding the first recorded target for the complete run")
        if IK_TYPE == "rl_constraint":
            self.pink_solver = PinkTeleopIK(
                self.task_config.control_config.teleop_config.pink_config_path)
            if HOLD_FIRST_TARGET:
                first_target = self.pink_solver.forward_multi(self.trajectory[0])
                self.task_targets = [first_target] * len(self.trajectory)
            else:
                self.task_targets = [
                    self.pink_solver.forward_multi(q) for q in self.trajectory]
        else:
            self.task_targets = [None] * len(self.trajectory)
        duration = (len(self.trajectory) - 1) * self.robot_config.control_dt
        print(
            f"Loaded {len(self.trajectory)} joint commands from "
            f"{TRAJECTORY_PATH} ({duration:.2f} s at "
            f"{1. / self.robot_config.control_dt:.1f} Hz)")

    @staticmethod
    def _hold_trajectory(source, source_dt, control_dt):
        if min(source_dt, control_dt) <= 0:
            raise ValueError("Trajectory and controller sample periods must be positive")
        times = np.arange(len(source)) * source_dt
        ticks = np.arange(int(np.ceil(times[-1] / control_dt)) + 1) * control_dt
        source_indices = np.floor(
            (ticks + source_dt * 1e-9) / source_dt).astype(int)
        return source[np.minimum(source_indices, len(source) - 1)].copy()

    def exec_home_movement(self, wait=False):
        self.exec_enable_compliance()
        return super().exec_home_movement(wait=wait)

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
            self, sample_index: int, start_time: float, state=None) -> list[float]:
        robot_id = self.robot_ids[0]
        if state is None:
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
    def _torque_record(state):
        """Preserve DCP torque values; unreported joints remain blank, not zero."""
        record = []
        for field in TORQUE_FIELDS:
            values = list(state.get(field, []))
            if len(values) > HumanoidRobot.JOINT_DOF:
                raise ValueError(f"Unexpected {field} joint count: {len(values)}")
            record.extend([len(values), *values,
                           *([None] * (HumanoidRobot.JOINT_DOF - len(values)))])
        return record

    @staticmethod
    def _save_joint_states(records: list[list[float]]) -> Path:
        output_dir = RESULT_DIR / f"{IK_TYPE}_{Path(TRAJECTORY_PATH).stem}"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = output_dir / f"joint_states_{timestamp}.csv"

        # Compare the state sampled at this tick with the validated command
        # sent at the same tick. The policy controls the 14 arm joints at
        # DCP indices 4:18; exclude the torso, head, and dummy joints.
        arm_slice = slice(4, 18)
        error_joint_names = DCP_ACTIVE_JOINT_NAMES[arm_slice]
        command_offset = 4 + 2 * HumanoidRobot.JOINT_DOF + 1
        current = np.asarray([
            row[4:4 + HumanoidRobot.JOINT_DOF] for row in records])[:, arm_slice]
        commanded = np.asarray([
            row[command_offset:command_offset + HumanoidRobot.JOINT_DOF]
            for row in records])[:, arm_slice]
        errors = commanded - current
        error_norms = np.linalg.norm(errors, axis=1)
        mean_error_norm = float(np.mean(error_norms))
        print(
            f"Mean joint error norm across {len(records)} time steps: "
            f"{mean_error_norm:.6f} deg (14 arm joints, command - current)")

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
            "ik_mode",
            *(f"command_{name}_deg" for name in joint_names),
            *(column for field in TORQUE_FIELDS for column in (
                f"{field}_count", *(f"{field}_{name}" for name in joint_names))),
            *(f"error_{name}_deg" for name in error_joint_names),
            "error_norm_deg",
        ]

        with output_path.open("x", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(header)
            writer.writerows(
                [*row, *error.tolist(), float(norm)]
                for row, error, norm in zip(records, errors, error_norms))
        print(f"Saved {len(records)} joint states and errors to {output_path}")

        # Render after the robot has stopped, using a headless canvas that is
        # safe to create in the controller's worker thread.
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        elapsed = np.asarray([row[1] for row in records])
        figure = Figure(figsize=(12, 8), constrained_layout=True)
        FigureCanvasAgg(figure)
        norm_ax, joints_ax = figure.subplots(2, 1, sharex=True)
        norm_ax.plot(elapsed, error_norms, label="14-arm-joint error L2 norm")
        norm_ax.axhline(
            mean_error_norm, color="tab:red", linestyle="--",
            label=f"Mean: {mean_error_norm:.6f} deg")
        norm_ax.set_title(f"{IK_TYPE}: {Path(TRAJECTORY_PATH).stem}")
        norm_ax.set_ylabel("Error norm (deg)")
        norm_ax.legend()
        norm_ax.grid(True, alpha=0.3)
        for index, name in enumerate(error_joint_names):
            joints_ax.plot(elapsed, errors[:, index], label=name)
        joints_ax.set_xlabel("Elapsed time (s)")
        joints_ax.set_ylabel("Command - current (deg)")
        joints_ax.legend(ncol=3, fontsize="small")
        joints_ax.grid(True, alpha=0.3)
        plot_path = output_dir / f"joint_errors_{timestamp}.png"
        figure.savefig(plot_path, dpi=150)
        print(f"Saved joint error plot to {plot_path}")
        return output_path

    def _open_loop_control_fn(self, duration: float):
        visualizer = getattr(self, "visualizer", None)
        teleop_attempted = False
        last_action = None
        commands_sent = 0
        joint_state_records = []
        self._compliance_switches = SimpleQueue()
        compliance_command = RL_COMPLIANCE_COMMAND
        rl_started = False

        try:
            if not np.isfinite(COMMAND_OFFSET_S) or COMMAND_OFFSET_S < 0.:
                raise ValueError("COMMAND_OFFSET_S must be finite and non-negative")
            control_dt = self.robot_config.control_dt
            offset_steps = int(np.floor(
                min(COMMAND_OFFSET_S, (len(self.trajectory) - 1) * control_dt)
                / control_dt + 1e-9))
            if self.rl_constraint is not None and (
                    not np.isfinite(RL_START_DELAY_S) or RL_START_DELAY_S < 0.):
                raise ValueError("RL_START_DELAY_S must be finite and non-negative")
            self._check_start_position()

            self.exec_enable_compliance()

            teleop_attempted = True
            self.exec_start_movement(control_mode="joint_abs")

            robot_id = self.robot_ids[0]
            robot = self.robot[robot_id]
            if self.rl_constraint is not None:
                self.rl_constraint.reset()
                print(f"RL compliance command: {compliance_command}")
                print(f"RL starts after {RL_START_DELAY_S:g} s of Pink replay")
            print(
                f"Configured IK mode: {IK_TYPE}. "
                "Robot compliance remains enabled for both arms.")
            print(
                f"Command offset: {COMMAND_OFFSET_S:g} s "
                f"({offset_steps} control steps ahead, clamped to the last sample)")

            start_time = time.monotonic()
            next_tick = start_time
            for index in range(len(self.trajectory)):
                if not self._control_triggered:
                    break
                elapsed = time.monotonic() - start_time
                if elapsed >= duration:
                    print("Open-loop execution reached its duration limit")
                    break

                state = robot.get_state()
                control_state = robot.robot_client.get_control_state()
                # q/qdot and raw torques come from the same DCP response.
                # Keep op_state from get_robot_data(), which provides it.
                state = {**state, **control_state,
                         "q": control_state["q"], "qdot": control_state["qdot"]}
                torque_record = self._torque_record(control_state)
                if visualizer is not None:
                    visualizer.publish(state["q"], reset=index == 0)
                use_rl = self.rl_constraint is not None and (
                    rl_started or elapsed >= RL_START_DELAY_S)
                mode = "rl_constraint" if use_rl else "pink"
                command_index = min(index + offset_steps, len(self.trajectory) - 1)
                targets = self.task_targets[command_index]
                recorded_command = self.trajectory[command_index].tolist()
                if use_rl:
                    measured = self.pink_solver.forward_multi(state["q"])
                    if not rl_started:
                        self.rl_constraint.toggle(state["q"], measured)
                        rl_started = True
                        print(f"RL activated at {elapsed:.3f} s (step {index})")
                    if self.rl_constraint.uses_compliant_history:
                        while self.interactive_compliance_enabled:
                            try:
                                self._compliance_switches.get_nowait()
                            except Empty:
                                break
                            compliance_command = not compliance_command
                            print(f"RL compliance command: {compliance_command}")
                        modes = [float(compliance_command)] * self.rl_constraint.observation.mode_dim
                        self.rl_constraint.update(targets, state["q"], measured_targets=measured,
                                                  compliance_mode=modes)
                    else:
                        self.rl_constraint.update(targets, state["q"])
                    command = self.rl_constraint.command(
                        state["q"], recorded_command)
                    if RL_CONSTRAINT_DRY_RUN:
                        print(f"RL dry run — projected q22 (deg): {command}")
                        command = recorded_command
                else:
                    # These are the exact joint commands originally produced by
                    # Pink during recording; replay them without another IK solve.
                    command = recorded_command
                joint_state = self._read_joint_state(index, start_time, state)
                last_action = self._send_joint_command(command)
                joint_state_records.append(
                    joint_state + [mode] + last_action[robot_id] + torque_record)
                commands_sent += 1
                if visualizer is not None:
                    visualizer.publish(
                        state["q"], last_action[robot_id], raw_command=recorded_command)
                if use_rl:
                    self.rl_constraint.record_command(last_action[robot_id], state["q"])

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
                    # Finish teleoperation; leave compliance enabled.
                    self.exec_finish_movement()
                except Exception as exc:
                    if self._control_error is None:
                        self._control_error = exc
                    print(f"Failed to leave teleoperation mode: {exc}")

            if joint_state_records:
                try:
                    self._save_joint_states(joint_state_records)
                except Exception as exc:
                    if self._control_error is None:
                        self._control_error = exc
                    print(f"Failed to save joint states or error plot: {exc}")

            self._control_triggered = False
            self._control_thread = None
