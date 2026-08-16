# base
import os
import h5py
import json
import cv2
import select
import sys
import termios
import time
from threading import Thread
from queue import Empty, Queue
from typing import Dict
import tty
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# helper functions
from helper.extra_utils import ROBOT_STATE
from helper.controller_utils import Controller
from perception.realsense import RealsenseCamHandler
from data_collector.device.base import BaseDevice
from data_collector.control_adapter import convert_device_control
import getch

# communication
from communication.robot import Robot

# config
from helper.config_utils import ROBOT_CONFIG, TASK_CONFIG
from data_collector.config import DataCollectorConfig, CONFIGS as DATA_COLLECTOR_CONFIGS


class DataCollectionScheduler(Controller):
    
    def __init__(self, robot: Dict[int, Robot] | None = None, **kwargs):
        # set robot
        self.config: DataCollectorConfig = DATA_COLLECTOR_CONFIGS[kwargs["config_name"]]
        self.robot_config = self.config.robot_config
        self.robot_ids = self.robot_config.robot_ids
        self.task_config = self.config.task_config
        self.task_name = self.task_config.name
        super(DataCollectionScheduler, self).__init__(robot=robot, **kwargs)
        
        # set camera
        self.camera: Dict[str, RealsenseCamHandler] = kwargs.get("camera", dict())
        for cam_name in self.task_config.camera_config.cam_names:
            if cam_name in self.camera.keys():
                if not self.camera[cam_name]._thread_running:
                    self.camera[cam_name].start()
            else:
                cam_config = self.task_config.camera_config.cam_params[cam_name]
                
                self.camera[cam_name] = RealsenseCamHandler(
                    serial_number=cam_config["serial"], 
                    align=True, 
                    clipping_distance_m=1., 
                    exposure=cam_config.get("exposure", None)
                )
                self.camera[cam_name].start()
        
        # set data collector
        self.data_collector = TeleopDataCollector(
            robot_config=self.robot_config,
            task_config=self.task_config,
            data_collector_config=self.config,
            dagger_mode=kwargs.get("dagger_mode", False),
            device=kwargs.get("device", None),
        )
        self.device_output_mode = self.data_collector.get_control_mode()
        self.control_mode = self.task_config.teleop_config.robot_control_mode
        self.arm_index = self.task_config.teleop_config.arm_index
        self.ik_type = self.task_config.teleop_config.ik_type
        self.pink_solvers = {}
        if (self.device_output_mode == "task_abs"
                and self.control_mode == "joint_abs"
                and self.ik_type == "pink"):
            from data_collector.pink_ik import PinkTeleopIK

            self.pink_solvers = {
                robot_id: PinkTeleopIK(
                    self.task_config.teleop_config.pink_config_path)
                for robot_id in self.robot_ids
            }
        
        # set empty variable
        self._collection_triggered = False
        self._collection_thread = None
        self._final_button_queue = Queue()
        self._collection_error = None

    def exec_collection(self, mode: str):
        if not self._collection_triggered:
            assert mode in ["start", "current"]
            if mode == "start":
                # stop nn control + move to start joint position
                self.exec_home_movement(wait=True)

            # Do not reuse an extra key press from an earlier collection.
            while not self._final_button_queue.empty():
                self._final_button_queue.get_nowait()

            self._collection_triggered = True
            self._collection_error = None
            
            self._collection_thread = Thread(target=self._collection_fn, daemon=True)
            self._collection_thread.start()
            print("Press 't' to start/stop recording")

            # Temporarily use keyboard input while the Vive buttons are unavailable.
            stdin_fd = sys.stdin.fileno()
            terminal_settings = termios.tcgetattr(stdin_fd)
            try:
                tty.setcbreak(stdin_fd)
                while self._collection_thread.is_alive():
                    readable, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if readable and getch.getch() == "t":
                        self._final_button_queue.put(True)
            finally:
                termios.tcsetattr(
                    stdin_fd, termios.TCSADRAIN, terminal_settings)
                self._collection_thread.join()

            if self._collection_error is not None:
                print(
                    "Collection failed; the partial episode will not be saved: "
                    f"{self._collection_error}")
                self.data_collector.init_data_buffer()
                self._collection_triggered = False
                return

            if self.data_collector.traj_len == 0:
                print("No samples were recorded; nothing to save.")
                self.data_collector.init_data_buffer()
                self._collection_triggered = False
                return
            
            # save trajectory if it's OK
            print("Click 's' to save / Click 'e' to not save")
            while True:
                #command = command_listener.get_key_states()
                char = getch.getch()
                if char == "s":
                    print(f"Trajectory {self.data_collector.traj_num} saved.")
                    self.data_collector.save_data_buffer()
                    self.data_collector.visualize_last_data()
                    break
                elif char == "e":
                    print("Not saved")
                    break
                        
            self.data_collector.init_data_buffer()
                
            self._collection_triggered = False

    def collect_buffer(self, robot_states=None):
        buffer_data = dict()
        robot_states = robot_states or {
            robot_id: self.robot[robot_id].get_state()
            for robot_id in self.robot_ids
        }
            
        # get proprioception
        for robot_id in self.robot_ids:
            if "proprio" in self.config.data_to_collect:
                control_dat = robot_states[robot_id]
                for k in self.config.data_to_collect["proprio"]:
                    buffer_data[f"{k}_{robot_id}"] = control_dat[k]
            if "gripper" in self.config.data_to_collect:
                gripper_state = self.robot[robot_id].get_gripper_state()
                if "gripper_position" in self.config.data_to_collect["gripper"]:
                    buffer_data[f"gripper_position_{robot_id}"] = gripper_state["gripper_pos"]
                if "grasp_state" in self.config.data_to_collect["gripper"]:
                    buffer_data[f"grasp_state_{robot_id}"] = gripper_state["grasp_state"]
            if "ft" in self.config.data_to_collect:
                ft_data = self.robot[robot_id].get_transformed_ft_sensor_data()
                for k in self.config.data_to_collect["ft"]:
                    if k not in ft_data:
                        continue
                    buffer_data[f"{k}_{robot_id}"] = ft_data[k]
            if "force_gain" in self.config.data_to_collect:
                force_gain = self.robot[robot_id].get_force_control_gain()
                for k in self.config.data_to_collect["force_gain"]:
                    fg_k = k.removeprefix("fg_")
                    if fg_k not in force_gain:
                        continue
                    buffer_data[f"{k}_{robot_id}"] = force_gain[fg_k]
            if "force_mode" in self.config.data_to_collect:
                if self.robot_config.robot_params[robot_id].get("init_kwargs", {}).get("force_mode", None) is not None:
                    force_mode_dict = self.robot_config.robot_params[robot_id]["init_kwargs"]["force_mode"]
                else:
                    enabled = False
                    des_force = np.zeros(6)
                    enabled_force = [False] * 6
                    force_mode_dict = {"enable": enabled, "des_force": des_force, "enabled_force": enabled_force}
                for k in self.config.data_to_collect["force_mode"]:
                    fm_k = k.removeprefix("fm_")
                    if fm_k not in force_mode_dict:
                        continue
                    buffer_data[f"{k}_{robot_id}"] = force_mode_dict[fm_k]
        
        # get exteroception
        if "camera" in self.config.data_to_collect:
            for cam_name in self.config.data_to_collect["camera"]:
                cam_output = self.camera[cam_name].get_all()
                for k in self.config.data_to_collect["camera"][cam_name]:
                    buffer_data[f"images.{k}.{cam_name}"] = cam_output[k]
        return buffer_data

    def _collection_fn(self):
        movement_started = False
        value = None

        try:
            # Compliance must be active before the teleoperation protocol starts.
            self.exec_enable_compliance()
            self.exec_start_movement(control_mode=self.control_mode)
            movement_started = True

            prev_button = False
            is_recording = False
            initial_states = {
                robot_id: self.robot[robot_id].get_state()
                for robot_id in self.robot_ids
            }
            if self.control_mode == "joint_abs":
                value = {
                    robot_id: self.robot[robot_id].validate_joint_command(
                        initial_states[robot_id]["q"])
                    for robot_id in self.robot_ids
                }
            else:
                value = {
                    robot_id: self.robot[robot_id].get_task_pose(
                        initial_states[robot_id], arm_index=self.arm_index)
                    for robot_id in self.robot_ids
                }

            while self._collection_triggered:
                control_start = time.time()
                robot_states = {
                    robot_id: self.robot[robot_id].get_state()
                    for robot_id in self.robot_ids
                }
                buffer_data = self.collect_buffer(robot_states=robot_states)
                references = {
                    robot_id: (
                        self.robot[robot_id].get_task_pose(
                            robot_states[robot_id], arm_index=self.arm_index)
                        if self.device_output_mode == "task_abs"
                        else self.robot[robot_id].validate_joint_command(
                            robot_states[robot_id]["q"])
                    )
                    for robot_id in self.robot_ids
                }

                device_data = self.data_collector.get_device_input(
                    robot_references=references)

                # Temporary keyboard replacement for the Vive final button.
                try:
                    device_data["final_button"] = (
                        self._final_button_queue.get_nowait())
                except Empty:
                    device_data["final_button"] = False

                if not device_data["final_valid"]:
                    raise RuntimeError(
                        "Device input is invalid or the connection was lost")

                if (is_recording and not prev_button
                        and device_data["final_button"]):
                    print("Recording stopped")
                    break

                if not prev_button and device_data["final_button"]:
                    print("Recording started")
                    is_recording = True
                    for robot_id in self.robot_ids:
                        self.data_collector.device[robot_id].reset()
                    device_data = self.data_collector.get_device_input(
                        robot_references=references)
                    if not device_data["final_valid"]:
                        raise RuntimeError(
                            "Device input became invalid after reset")

                prev_button = device_data["final_button"]

                if is_recording:
                    converted = {}
                    for robot_id in self.robot_ids:
                        device_command = (
                            self.task_config.extra_config
                            .control_post_process_fn(
                                device_data[robot_id]["control"]))
                        converted[robot_id] = convert_device_control(
                            robot=self.robot[robot_id],
                            device_command=device_command,
                            device_mode=self.device_output_mode,
                            robot_mode=self.control_mode,
                            current_state=robot_states[robot_id],
                            arm_index=self.arm_index,
                            ik_type=self.ik_type,
                            lock_non_selected_joints=(
                                self.task_config.teleop_config
                                .lock_non_selected_joints),
                            locked_joint_reference=(
                                initial_states[robot_id]["q"]),
                            pink_solver=self.pink_solvers.get(robot_id),
                        )
                    value = {
                        robot_id: converted[robot_id].command
                        for robot_id in self.robot_ids
                    }
                    gripper_command = {
                        robot_id: float(
                            1. - np.round(device_data[robot_id]["trigger"]))
                        for robot_id in self.robot_ids
                    }

                    # Send only commands that passed full dimensional validation.
                    self.robot_cluster.tele_move(
                        action=value,
                        mode=self.control_mode,
                        vel_scale={robot_id: self.robot_config.robot_params[robot_id]["control"]["vel_scale"] for robot_id in self.robot_config.robot_ids},
                        acc_scale={robot_id: self.robot_config.robot_params[robot_id]["control"]["acc_scale"] for robot_id in self.robot_config.robot_ids},
                        arm_index=self.arm_index,
                    )
                    self.robot_cluster.move_gripper(
                        mode="thread", value=gripper_command)

                    # Store the converted command actually sent to the robot.
                    control_key = f"{self.control_mode}_control"
                    configured_controls = self.config.data_to_collect["control"]
                    for robot_id in self.robot_ids:
                        if control_key in configured_controls:
                            buffer_data[f"{control_key}_{robot_id}"] = value[robot_id]
                        if "gripper_command" in configured_controls:
                            buffer_data[f"gripper_command_{robot_id}"] = (
                                gripper_command[robot_id])
                    self.data_collector.update_data_buffer(**buffer_data)

                    wait_time = self.robot_config.control_dt - (
                        time.time() - control_start)
                    if wait_time > 0.:
                        time.sleep(wait_time)

        except Exception as exc:
            self._collection_error = exc
        finally:
            if movement_started and value is not None:
                try:
                    self.exec_soft_stop(
                        last_action=value,
                        control_period=self.robot_config.control_dt,
                        mode=self.control_mode,
                        arm_index=self.arm_index,
                    )
                except Exception as exc:
                    if self._collection_error is None:
                        self._collection_error = exc
            try:
                self.exec_finish_movement()
            except Exception as exc:
                if self._collection_error is None:
                    self._collection_error = exc
            finally:
                try:
                    # Teleop is stopped before compliance is deactivated.
                    # self.exec_disable_compliance()
                    pass
                except Exception as exc:
                    if self._collection_error is None:
                        self._collection_error = exc
                self._collection_triggered = False
        

class TeleopDataCollector:
    def __init__(self, 
                 robot_config: ROBOT_CONFIG, 
                 task_config: TASK_CONFIG, 
                 data_collector_config: DataCollectorConfig,
                 dagger_mode=False, 
                 device=None):
        self.robot_config = robot_config
        self.robot_ids = robot_config.robot_ids
        self.task_config = task_config
        self.task_name = task_config.name
        self.data_collector_config = data_collector_config
        
        self.device: Dict[int, BaseDevice] = dict()
        if device is None:
            for robot_id in self.robot_ids:
                device_class = self.task_config.data_config.device_class
                if device_class is not None:
                    if not issubclass(device_class, BaseDevice):
                        raise TypeError("device_class must inherit BaseDevice")
                    self.device[robot_id] = device_class(
                        device_params=self.task_config.data_config.device_params,
                        control_dt=self.robot_config.control_dt,
                    )
                elif self.task_config.data_config.device_type == "vive":
                    from data_collector.device.vive import Vive
                    self.device[robot_id] = Vive(
                        device_params=self.task_config.data_config.device_params,
                    )
                elif self.task_config.data_config.device_type == "spacemouse":
                    from data_collector.device.spacemouse import SpaceMouse
                    self.device[robot_id] = SpaceMouse(
                        device_params=self.task_config.data_config.device_params,
                        control_dt=self.robot_config.control_dt
                    )
                else:
                    raise ValueError
        else:
            self.device = device
        
        # check data colleciton progress
        self.DATA_DIR = os.path.join(self.task_config.data_config.data_dir, self.task_name)
        self.DATA_VIZ_DIR = os.path.join(self.task_config.data_config.data_viz_dir, self.task_name)
        
        if dagger_mode:
            assert os.path.isdir(self.DATA_DIR), "Data directory does not exist"
        
        if not os.path.isdir(self.DATA_DIR):
            os.makedirs(self.DATA_DIR)

        if len(os.listdir(self.DATA_DIR)) == 0:
            self.traj_num = 0
        else:
            traj_nums = [int(f.split(".")[0]) for f in os.listdir(self.DATA_DIR) if f.endswith('.h5')]
            self.traj_num = max(traj_nums) + 1
        print("==========================")
        print(f"[Task: {self.task_name}] Continue data saving from {self.traj_num}")
        print("==========================")
        
        # save data collection progress (DAGGER)
        if dagger_mode:
            collect_progress_file = os.path.join(self.task_config.data_config.data_dir, f"{self.task_name}_progress.json")
            if os.path.exists(collect_progress_file):
                with open(collect_progress_file, "r") as f:
                    collect_progress = json.load(f)
                
                collect_iters = []
                for k in collect_progress.keys():
                    collect_iters.append(int(k.split("_")[-1]))
                    
                if collect_progress[f"iter_{max(collect_iters)}"] != self.traj_num:
                    # New iteration
                    current_collect_iter = max(collect_iters) + 1
                else:
                    current_collect_iter = max(collect_iters)
            else:
                collect_progress = dict()
                current_collect_iter = 1
                
            collect_progress[f"iter_{current_collect_iter}"] = self.traj_num
            
            with open(collect_progress_file, "w") as f:
                json.dump(collect_progress, f)
            
        # initialize data buffer
        self.init_data_buffer()
        
    def __del__(self):
        for robot_id in self.robot_ids:
            self.device[robot_id].exit()
            
    def get_device_input(self, robot_references):
        output = dict()
        output["final_button"] = False
        output["final_valid"] = True
        
        for robot_id in self.robot_ids:
            reference = robot_references[robot_id]
            output[robot_id] = self.device[robot_id].get_input(
                robot_pose=reference,
                robot_joint=reference,
            )
            # Temporary: ignore the Vive button as the final button.
            # output["final_button"] = output["final_button"] or output[robot_id]["button"]
            output["final_valid"] = output["final_valid"] and output[robot_id]["valid"]
        return output
    
    def get_control_mode(self):
        from helper.extra_utils import ROBOT_CONTROL_MODE

        device_modes = {
            self.device[robot_id].CONTROL_MODE
            for robot_id in self.robot_ids
        }
        if len(device_modes) != 1:
            raise ValueError(
                "All teleoperation devices must use the same output mode")
        control_mode = device_modes.pop()
        if control_mode == ROBOT_CONTROL_MODE.TELE_JOINT_ABSOLUTE:
            return "joint_abs"
        if control_mode == ROBOT_CONTROL_MODE.TELE_TASK_ABSOLUTE:
            return "task_abs"
        raise ValueError(f"Unsupported device control mode: {control_mode}")

    def init_data_buffer(self):
        """
        "q", "qdot", "p", "pdot",   # robot state (T, D)
        "gripper_position", "grasp_state",   # gripper state  (T, 1)
        "images.rgb.{cam_name}", "images.depth.{cam_name}", "images.intrinsics.{cam_name}",  # rgb (T, H, W, C) [rgb], depth (T, H, W) [float32], intrinsics (3, 3) [float32]
        "task_abs_control" or "joint_abs_control"  # sent robot control (T, D)
        "gripper_command",   # gripper control (T, 1)
        """
        # Reset data
        self.data_types = []
        data_to_collect = []
        for k in self.data_collector_config.data_to_collect.keys():
            if k == "camera":
                continue
            else:
                data_to_collect.extend(
                    self.data_collector_config.data_to_collect[k])
        for type in data_to_collect:
            for robot_id in self.robot_ids:
                self.data_types.append(f"{type}_{robot_id}")
                
        for cam_name in self.data_collector_config.data_to_collect["camera"]:
            for data in self.data_collector_config.data_to_collect["camera"][cam_name]:
                self.data_types.append(f"images.{data}.{cam_name}")
        
        self.traj = dict()
        for data_type in self.data_types:
            self.traj[data_type] = None
        
        self.traj_len = 0
        
        # Reset device
        for robot_id in self.robot_ids:
            self.device[robot_id].reset()
        
    def update_data_buffer(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.data_types:
                if self.traj[k] is None:
                    self.traj[k] = [v]
                    if "intrinsics" in k:
                        self.traj[k] = v # To match legacy code
                elif k.rpartition("_")[0] not in self.data_collector_config.data_to_collect_once:
                    if "intrinsics" in k:
                        continue
                    self.traj[k].append(v)
            else:
                print(f"Ignoring key: {k}")

        self.traj_len += 1
        
    def save_data_buffer(self):
        for k in self.traj.keys():
            if self.traj[k] is not None:
                if ("rgb" in k) or ("depth" in k):
                    self.traj[k] = np.asarray(self.traj[k])
                else:
                    self.traj[k] = np.asarray(self.traj[k]).astype(np.float32)
                
                if len(self.traj[k].shape) == 1:
                    self.traj[k] = self.traj[k][..., np.newaxis]  # change shape from (T,) to (T, 1)
                    
                print(f"{k}: {self.traj[k].shape} ({self.traj[k].dtype})")

        with h5py.File(f"{self.DATA_DIR}/{self.traj_num}.h5", "w") as hf:
            for k, v in self.traj.items():
                if v is not None:
                    hf.create_dataset(k, data=v)
        
        print(f"Data saved to {self.DATA_DIR}/{self.traj_num}.h5")
        self.traj_num += 1
        self.init_data_buffer()
    
    def visualize_last_data(self):
        # SET DATA ID TO VISUALIZE
        VISUALIZE_DATA_ID = self.traj_num - 1
        FREQUENCY = int(1 / self.robot_config.control_dt)
        if not os.path.isdir(self.DATA_VIZ_DIR):
            os.makedirs(self.DATA_VIZ_DIR)

        # load data
        collected_traj = dict()
        with h5py.File(f"{self.DATA_DIR}/{VISUALIZE_DATA_ID}.h5", "r") as root:
            for k in root.keys():
                collected_traj[k] = root[k][()]

        # Check exteroception
        for key in collected_traj.keys():
            if "images.rgb" in key:
                color_traj = collected_traj[key]  # uint8  # RGB
                frame_height, frame_width, _ = color_traj.shape[1:]
                cam_name = key.split(".")[-1]
                output_video_path = f"{self.DATA_VIZ_DIR}/{VISUALIZE_DATA_ID}_{cam_name}.mp4"

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
                video_writer = cv2.VideoWriter(output_video_path, fourcc, FREQUENCY, (frame_width, frame_height))

                for color_frame in color_traj:
                    bgr_frame = color_frame[..., ::-1]
                    video_writer.write(bgr_frame)

                video_writer.release()

        # Check proprioception
        for robot_id in self.robot_ids:
            for data_key in self.data_collector_config.data_to_collect:
                if data_key == "camera":
                    continue
                for key in self.data_collector_config.data_to_collect[data_key]:
                    if f"{key}_{robot_id}" in collected_traj:
                        traj = collected_traj[f"{key}_{robot_id}"]
                        n_steps = traj.shape[0]
                        if n_steps > 1:
                            time_traj = np.arange(n_steps) * 1 / FREQUENCY
                            plt.plot(time_traj, traj)
                            plt.xlabel("Time [s]")
                            plt.ylabel(key)
                            plt.savefig(f"{self.DATA_VIZ_DIR}/{VISUALIZE_DATA_ID}_{key}_{robot_id}.png")
                            plt.clf()
                            plt.close()
                        else:
                            with open(f"{self.DATA_VIZ_DIR}/{VISUALIZE_DATA_ID}_{key}_{robot_id}.txt", "w") as f:
                                f.write(str(traj[0]))
