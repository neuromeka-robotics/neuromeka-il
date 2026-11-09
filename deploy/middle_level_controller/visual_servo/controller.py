"""Nami left-arm visual servoing entry point for task_demo.py and FSM callers."""
from threading import Thread
import time
import traceback

import numpy as np
import torch

from helper.controller_utils import Base_NN_controller
from helper.extra_utils import NN_CONTROL_STATE, ROBOT_STATE
from perception.realsense import RealsenseCamHandler
from .config import CUSTOM_ROBOT_CONFIG, CUSTOM_TASK_CONFIG, ROBOT_ID, ARM_INDEX, SERVO_CONFIG
from .model import NN_policy


class NN_controller(Base_NN_controller):
    robot_config = CUSTOM_ROBOT_CONFIG
    robot_ids = CUSTOM_ROBOT_CONFIG.robot_ids
    task_config = CUSTOM_TASK_CONFIG
    task_name = CUSTOM_TASK_CONFIG.name

    def __init__(self, robot=None, **kwargs):
        super().__init__(robot=robot, **kwargs)
        self.camera = kwargs.get("camera", {})
        self._owns_camera = "wrist" not in self.camera
        if self._owns_camera:
            self.camera["wrist"] = RealsenseCamHandler(
                serial_number=self.task_config.camera_config.cam_params["wrist"]["serial"])
            self.camera["wrist"].start()
        self._control_triggered = False
        self._control_thread = None

    def load_policy(self):
        if not isinstance(self.nn_policy, NN_policy):
            self.nn_policy = NN_policy(robot_config=self.robot_config, task_config=self.task_config)

    def exec_nn_control(self, duration=120.):
        if self._control_thread is not None and self._control_thread.is_alive():
            print("Visual servo loop is already running")
            return
        self._reset_control()
        self._control_triggered = True
        self._control_thread = Thread(target=self._nn_control_fn,
                                      args=(min(duration, SERVO_CONFIG["max_runtime_s"]),), daemon=True)
        self._control_thread.start()

    def exec_nn_control_stop(self):
        self._control_triggered = False
        if self._control_thread is not None:
            self._control_thread.join()
            self._control_thread = None

    def _nn_control_fn(self, duration):
        robot = self.robot[ROBOT_ID]
        control = self.robot_config.robot_params[ROBOT_ID]["control"]
        try:
            self.control_state = self.exec_start_movement()
            deadline = time.monotonic() + duration
            while self._control_triggered and time.monotonic() < deadline:
                start = time.monotonic()
                state = robot.get_state()
                if ROBOT_STATE.in_failure_state(state["op_state"]):
                    self.control_state = NN_CONTROL_STATE.ROBOT_FAIL
                    break
                frame = self.camera["wrist"].get_all()
                if frame is None or time.monotonic() - frame["timestamp"] > 1.:
                    raise RuntimeError("Wrist camera has no fresh frame")
                with torch.inference_mode():
                    output = self.nn_policy(
                        qpos=np.asarray(state["q"]), qvel=np.asarray(state["qdot"]),
                        end_pose=np.asarray(robot.get_task_pose(state, ARM_INDEX)),
                        end_vel=np.asarray(state["pdot"])[6 * ARM_INDEX:6 * (ARM_INDEX + 1)],
                        **{"images.rgb.wrist": frame["rgb"]})
                self.control_state = output["control_state"]
                # A stop requested during inference must not send another target.
                if not self._control_triggered or NN_CONTROL_STATE.get_out_control_loop(self.control_state):
                    break
                if time.monotonic() - frame["timestamp"] > 1.:
                    raise RuntimeError("Visual inference exceeded the frame age limit; stopping")
                robot.tele_move(output["robot_action_0"].tolist(), mode="task_abs", arm_index=ARM_INDEX,
                                vel_scale=control["vel_scale"], acc_scale=control["acc_scale"])
                time.sleep(max(0., self.robot_config.control_dt - (time.monotonic() - start)))
        except Exception:
            self.control_state = NN_CONTROL_STATE.TASK_FAIL
            traceback.print_exc()
        finally:
            # Runs on stop, timeout, missing masks, camera/server errors and exceptions.
            try:
                robot.stop_teleop()
                self.control_state = self.exec_finish_movement()
            finally:
                self._control_triggered = False

    def close(self):
        self.exec_nn_control_stop()
        if hasattr(self.nn_policy, "close"):
            self.nn_policy.close()
        if self._owns_camera:
            self.camera["wrist"].stop()
