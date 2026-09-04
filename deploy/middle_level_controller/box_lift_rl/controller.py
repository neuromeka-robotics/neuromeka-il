"""Real EIR controller for the Genesis move-box ONNX policy."""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
import threading
from threading import Thread, current_thread
from typing import Dict

import numpy as np

from communication.robot import Robot
from helper.controller_utils import Base_NN_controller
from helper.extra_utils import NN_CONTROL_STATE, ROBOT_STATE
from perception.aruco_box_pose import (
    ArucoBoxPoseEstimator,
    PsfCameraCalibration,
    _create_box_geometries,
    _GeometryTracker,
    robot_pose_command_to_transform,
    tissue_box_aruco_config,
)
from perception.realsense import RealsenseCamHandler

from .config import (
    BOX_FRAME_CORRECTION,
    CAMERA_NAME,
    CUSTOM_ROBOT_CONFIG,
    CUSTOM_TASK_CONFIG,
    MAX_CONSECUTIVE_BOX_POSE_MISSES,
    PLOT_AVERAGE_ABS_JOINT_VELOCITY,
    POLICY_DEPLOYMENT_RECORD_DIR,
    POLICY_ROBOT_JOINT_INDICES,
    PSF_CAMERA_CALIBRATION_PATH,
    RECORD_POLICY_DEPLOYMENT,
    START_POSITION_TOLERANCE_DEG,
    VISUALIZE,
)
from .model import NN_policy, rotation_matrix_to_rpy
from .recording import DeploymentRecorder


def _box_pose_visualizer_process(
    pose_queue: "mp.Queue[np.ndarray | None]",
    box_config,
) -> None:
    """Open3D visualizer worker that runs in a separate process."""

    import open3d as o3d

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(
        "Box Pose - Robot Base Frame", width=1280, height=720
    )
    visualizer.add_geometry(
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    )
    box_tracker = _GeometryTracker(
        visualizer, lambda: _create_box_geometries(o3d, box_config)
    )

    while True:
        try:
            pose = pose_queue.get(timeout=0.05)
        except queue.Empty:
            pass
        else:
            if pose is None:
                break
            box_tracker.update(pose)
        if not visualizer.poll_events():
            break
        visualizer.update_renderer()

    visualizer.destroy_window()


class _BoxPoseVisualizer:
    """Open3D box-pose visualizer backed by a persistent child process."""

    def __init__(self, box_config) -> None:
        self._queue: mp.Queue = mp.Queue(maxsize=1)
        self._process: mp.Process | None = None
        self._box_config = box_config

    def start(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        self._process = mp.Process(
            target=_box_pose_visualizer_process,
            args=(self._queue, self._box_config),
            daemon=True,
        )
        self._process.start()

    def update(self, pose: np.ndarray) -> None:
        try:
            self._queue.put_nowait(pose.copy())
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(pose.copy())

    def stop(self) -> None:
        if self._process is not None and self._process.is_alive():
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
            self._process.join(timeout=2.0)
        self._process = None


def _joint_velocity_plotter_process(
    sample_queue: "mp.Queue[tuple[float, float] | None]",
) -> None:
    """Plot the mean absolute joint velocity in a separate process."""

    import matplotlib.pyplot as plt

    plt.ion()
    figure, axes = plt.subplots()
    (line,) = axes.plot([], [], linewidth=1.5)
    axes.set_title("Average Absolute Policy-Joint Velocity")
    axes.set_xlabel("Elapsed time (s)")
    axes.set_ylabel("Mean |joint velocity| (rad/s)")
    axes.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.show()

    elapsed_times: list[float] = []
    average_velocities: list[float] = []
    running = True

    try:
        while running and plt.fignum_exists(figure.number):
            samples: list[tuple[float, float]] = []
            try:
                sample = sample_queue.get(timeout=0.05)
            except queue.Empty:
                sample = None
            else:
                if sample is None:
                    break
                samples.append(sample)

            while True:
                try:
                    sample = sample_queue.get_nowait()
                except queue.Empty:
                    break
                if sample is None:
                    running = False
                    break
                samples.append(sample)

            if samples:
                elapsed_times.extend(sample[0] for sample in samples)
                average_velocities.extend(sample[1] for sample in samples)
                line.set_data(elapsed_times, average_velocities)
                axes.relim()
                axes.autoscale_view(scalex=True, scaley=False)
                maximum_velocity = max(average_velocities)
                y_margin = max(0.1 * maximum_velocity, 0.01)
                axes.set_ylim(0.0, maximum_velocity + y_margin)

            figure.canvas.draw_idle()
            figure.canvas.flush_events()
    finally:
        plt.close(figure)


class _JointVelocityPlotter:
    """Non-blocking producer for the live joint-velocity plot."""

    def __init__(self) -> None:
        self._queue: mp.Queue | None = None
        self._process: mp.Process | None = None

    def start(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        # Start each control run with an empty timeline, including after the
        # user closes a previous plot window before the controller stops.
        sample_queue: mp.Queue = mp.Queue(maxsize=256)
        self._queue = sample_queue
        self._process = mp.Process(
            target=_joint_velocity_plotter_process,
            args=(sample_queue,),
            daemon=True,
        )
        self._process.start()

    def update(self, elapsed_time: float, joint_velocity) -> None:
        sample_queue = self._queue
        if sample_queue is None:
            return

        velocity = np.asarray(joint_velocity, dtype=np.float64)
        if velocity.ndim != 1 or velocity.size == 0:
            return
        if not np.all(np.isfinite(velocity)):
            return

        sample = (
            float(elapsed_time),
            float(np.mean(np.abs(np.deg2rad(velocity)))),
        )
        try:
            sample_queue.put_nowait(sample)
        except queue.Full:
            try:
                sample_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                sample_queue.put_nowait(sample)
            except queue.Full:
                pass

    def stop(self) -> None:
        process = self._process
        sample_queue = self._queue
        if (
            process is not None
            and process.is_alive()
            and sample_queue is not None
        ):
            while True:
                try:
                    sample_queue.put_nowait(None)
                    break
                except queue.Full:
                    try:
                        sample_queue.get_nowait()
                    except queue.Empty:
                        break
            process.join(timeout=2.0)
        if sample_queue is not None:
            sample_queue.close()
        self._process = None
        self._queue = None


class NN_controller(Base_NN_controller):
    """`task_demo.py`-compatible compliant joint controller."""

    robot_config = CUSTOM_ROBOT_CONFIG
    robot_ids = CUSTOM_ROBOT_CONFIG.robot_ids
    task_config = CUSTOM_TASK_CONFIG
    task_name = CUSTOM_TASK_CONFIG.name

    def __init__(self, robot: Dict[int, Robot] | None = None, **kwargs):
        if len(self.robot_ids) != 1:
            raise ValueError("The move-box policy requires exactly one robot")
        super().__init__(robot=robot, **kwargs)

        self.camera: Dict[str, RealsenseCamHandler] = kwargs.get("camera", {})
        if CAMERA_NAME not in self.task_config.camera_config.cam_params:
            raise ValueError(f"Camera '{CAMERA_NAME}' is missing from task config")
        if CAMERA_NAME in self.camera:
            if not getattr(self.camera[CAMERA_NAME], "_thread_running", False):
                self.camera[CAMERA_NAME].start()
        else:
            camera_config = self.task_config.camera_config.cam_params[CAMERA_NAME]
            self.camera[CAMERA_NAME] = RealsenseCamHandler(
                serial_number=camera_config["serial"],
                align=True,
                clipping_distance_m=1.0,
                exposure=camera_config.get("exposure"),
            )
            self.camera[CAMERA_NAME].start()

        self.box_pose_estimator = ArucoBoxPoseEstimator(
            tissue_box_aruco_config()
        )
        self.camera_calibration = PsfCameraCalibration.load(
            PSF_CAMERA_CALIBRATION_PATH
        )
        if self.camera_calibration.calibration_type not in {"Tbc", "Thc"}:
            raise ValueError(
                "Box deployment supports fixed Tbc or head-mounted Thc "
                f"calibration, got {self.camera_calibration.calibration_type}"
            )
        configured_serial = self.task_config.camera_config.cam_params[
            CAMERA_NAME
        ]["serial"]
        calibrated_serial = self.camera_calibration.camera_serial
        if calibrated_serial and configured_serial != calibrated_serial:
            raise ValueError(
                f"Camera serial {configured_serial} does not match calibration "
                f"serial {calibrated_serial}"
            )

        self._control_triggered = False
        self._control_thread: Thread | None = None
        self._control_error: Exception | None = None
        self._box_pose_visualizer = (
            _BoxPoseVisualizer(tissue_box_aruco_config())
            if VISUALIZE
            else None
        )
        self._joint_velocity_plotter = (
            _JointVelocityPlotter()
            if PLOT_AVERAGE_ABS_JOINT_VELOCITY
            else None
        )

    def load_policy(self):
        if not isinstance(self.nn_policy, NN_policy):
            self.nn_policy = NN_policy(
                robot_config=self.robot_config,
                task_config=self.task_config,
            )
            print(f"Loaded move-box ONNX policy: {self.nn_policy.model_path}")

    def exec_nn_control(self, duration: float):
        if self._control_thread is not None and self._control_thread.is_alive():
            print("Control loop is already triggered")
            return
        if duration <= 0.0:
            raise ValueError("Control duration must be positive")
        if not isinstance(self.nn_policy, NN_policy):
            self.load_policy()

        self._reset_control()
        self.box_pose_estimator.reset()
        self._control_error = None
        self._control_triggered = True
        if self._box_pose_visualizer is not None:
            self._box_pose_visualizer.start()
        if self._joint_velocity_plotter is not None:
            self._joint_velocity_plotter.start()
        self._control_thread = Thread(
            target=self._nn_control_fn,
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
        if self._box_pose_visualizer is not None:
            self._box_pose_visualizer.stop()
        if self._joint_velocity_plotter is not None:
            self._joint_velocity_plotter.stop()

    def _check_home_position(self, state: dict) -> None:
        robot_id = self.robot_ids[0]
        current_qpos = np.asarray(state["q"], dtype=np.float64)
        home_qpos = np.asarray(
            self.robot_config.robot_params[robot_id]["home_pos"],
            dtype=np.float64,
        )
        if current_qpos.shape != home_qpos.shape:
            raise ValueError(
                f"Robot q shape {current_qpos.shape} does not match home "
                f"shape {home_qpos.shape}"
            )
        error = float(np.linalg.norm(current_qpos - home_qpos))
        if error > START_POSITION_TOLERANCE_DEG:
            raise RuntimeError(
                "Move the robot to the task home position before executing "
                f"the policy (joint error norm: {error:.3f} deg)"
            )

    def _box_pose_in_robot_base(
        self, camera_output: dict, robot_state: dict
    ) -> np.ndarray | None:
        transform_camera_box = self.box_pose_estimator.estimate(
            camera_output["rgb"],
            camera_output["intrinsics"],
            camera_output.get("dist_coeffs"),
        )
        if transform_camera_box is None:
            return None

        transform_base_mount = None
        if self.camera_calibration.calibration_type == "Thc":
            transform_base_mount = robot_pose_command_to_transform(
                self.robot[self.robot_ids[0]].get_task_pose(
                    robot_state, arm_index=0
                )
            )
        transform_base_camera = self.camera_calibration.base_to_camera(
            transform_base_mount
        )
        return transform_base_camera @ transform_camera_box

    def _send_joint_command(self, command: np.ndarray) -> Dict[int, list[float]]:
        robot_id = self.robot_ids[0]
        action = {
            robot_id: self.robot[robot_id].validate_joint_command(command)
        }
        control = self.robot_config.robot_params[robot_id]["control"]
        self.robot_cluster.tele_move(
            action=action,
            mode="joint_abs",
            vel_scale={robot_id: control["vel_scale"]},
            acc_scale={robot_id: control["acc_scale"]},
        )
        return action

    def _nn_control_fn(self, duration: float):
        robot_id = self.robot_ids[0]
        compliance_attempted = False
        teleop_started = False
        last_robot_action = None
        consecutive_pose_misses = 0
        recorder = (
            DeploymentRecorder(
                output_dir=POLICY_DEPLOYMENT_RECORD_DIR,
                model_path=self.nn_policy.model_path,
                control_dt=self.robot_config.control_dt,
            )
            if RECORD_POLICY_DEPLOYMENT
            else None
        )

        try:
            initial_state = self.robot[robot_id].get_state()
            self._check_home_position(initial_state)

            compliance_attempted = (
                self.task_config.control_config.compliance.enable
            )
            if compliance_attempted:
                self.exec_enable_compliance()

            self.exec_start_movement(control_mode="joint_abs")
            teleop_started = True

            start_time = time.monotonic()
            next_tick = start_time
            while (
                self._control_triggered
                and time.monotonic() - start_time < duration
            ):
                robot_state = self.robot[robot_id].get_state()
                elapsed_s = time.monotonic() - start_time
                if self._joint_velocity_plotter is not None:
                    joint_velocity = np.asarray(
                        robot_state.get("qdot", ()), dtype=np.float64
                    )
                    if joint_velocity.shape == (22,):
                        joint_velocity = joint_velocity[
                            list(POLICY_ROBOT_JOINT_INDICES)
                        ]
                    else:
                        joint_velocity = np.asarray((), dtype=np.float64)
                    self._joint_velocity_plotter.update(
                        elapsed_s,
                        joint_velocity,
                    )
                if ROBOT_STATE.in_failure_state(robot_state["op_state"]):
                    self.control_state = NN_CONTROL_STATE.ROBOT_FAIL
                    raise RuntimeError(
                        f"Robot entered failure state {robot_state['op_state']}"
                    )

                camera_output = self.camera[CAMERA_NAME].get_all()
                transform_base_box = (
                    None
                    if camera_output is None
                    else self._box_pose_in_robot_base(
                        camera_output, robot_state
                    )
                )
                policy_action = None
                command_q_deg = None
                if transform_base_box is None:
                    consecutive_pose_misses += 1
                    if (
                        consecutive_pose_misses
                        == MAX_CONSECUTIVE_BOX_POSE_MISSES + 1
                    ):
                        print(
                            "ArUco box pose was unavailable for "
                            f"{consecutive_pose_misses} consecutive cycles; "
                            "pausing policy actions until the pose is "
                            "available again"
                        )
                else:
                    if (
                        consecutive_pose_misses
                        > MAX_CONSECUTIVE_BOX_POSE_MISSES
                    ):
                        print(
                            "ArUco box pose is available again; resuming "
                            "policy actions"
                        )
                    consecutive_pose_misses = 0
                    if self._box_pose_visualizer is not None:
                        self._box_pose_visualizer.update(transform_base_box)
                    # Reorient the box frame from real-world ArUco convention
                    # (x=long, y=short) to Genesis sim convention (x=short, y=long).
                    transform_base_box = transform_base_box @ BOX_FRAME_CORRECTION
                    policy_output = self.nn_policy(
                        transform_base_box=transform_base_box,
                        qpos_deg=robot_state["q"],
                    )
                    self.control_state = policy_output["control_state"]
                    policy_action = policy_output["action"]
                    command_q_deg = policy_output["robot_action_0"]
                    last_robot_action = self._send_joint_command(
                        command_q_deg
                    )

                if recorder is not None:
                    recorder.append(
                        elapsed_s=elapsed_s,
                        wall_time_s=time.time(),
                        robot_state=robot_state,
                        policy_action=policy_action,
                        command_q_deg=command_q_deg,
                        box_transform_base=transform_base_box,
                    )

                next_tick += self.robot_config.control_dt
                wait_time = next_tick - time.monotonic()
                if wait_time > 0.0:
                    time.sleep(wait_time)

            if self._control_triggered:
                print("Move-box policy reached its duration limit")
            else:
                print("Move-box policy stopped")
            self.control_state = NN_CONTROL_STATE.TASK_FINISH
        except Exception as exc:
            self._control_error = exc
            if self.control_state != NN_CONTROL_STATE.ROBOT_FAIL:
                self.control_state = NN_CONTROL_STATE.TASK_FAIL
            print(f"Move-box policy failed: {exc}")
        finally:
            if teleop_started:
                if last_robot_action is not None:
                    try:
                        self.exec_soft_stop(
                            last_action=last_robot_action,
                            control_period=self.robot_config.control_dt,
                            mode="joint_abs",
                        )
                    except Exception as exc:
                        print(f"Move-box soft stop failed: {exc}")
                try:
                    self.exec_finish_movement()
                except Exception as exc:
                    print(f"Failed to leave teleoperation mode: {exc}")

            if compliance_attempted:
                try:
                    # self.exec_disable_compliance()
                    pass
                except Exception as exc:
                    print(f"Failed to disable compliance: {exc}")

            self._control_triggered = False
            self._control_thread = None
            if recorder is not None:
                try:
                    recording_path = recorder.save()
                    if recording_path is not None:
                        print(f"Saved policy deployment: {recording_path}")
                except Exception as exc:
                    print(f"Failed to save policy deployment: {exc}")
            if self._joint_velocity_plotter is not None:
                self._joint_velocity_plotter.stop()
