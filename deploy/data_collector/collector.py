# base
import os
import h5py
import json
import cv2
import time
from threading import Thread
from typing import Dict
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# helper functions
from helper.extra_utils import ROBOT_STATE, KeyboardListener
from helper.controller_utils import Controller
from perception.realsense import RealsenseCamHandler
from data_collector.device.base import BaseDevice
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
            dagger_mode=kwargs.get("dagger_mode", False)
        )
        self.control_mode = self.data_collector.get_control_mode()
        
        # set empty variable
        self._collection_triggered = False
        self._collection_thread = None

    def exec_collection(self, mode: str):
        if not self._collection_triggered:
            assert mode in ["start", "current"]
            if mode == "start":
                # stop nn control + move to start joint position
                self.exec_home_movement(wait=True)
                
            self._collection_triggered = True
            
            self._collection_thread = Thread(target=self._collection_fn, daemon=True)
            self._collection_thread.start()
            self._collection_thread.join()
            
            # save trajectory if it's OK
            #command_listener = KeyboardListener(key_targets=["s", "e"])
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

    def collect_buffer(self):
        buffer_data = dict()
            
        # get proprioception
        for robot_id in self.robot_ids:
            if "proprio" in self.config.data_to_collect:
                control_dat = self.robot[robot_id].get_state()
                for k in self.config.data_to_collect["proprio"]:
                    buffer_data[f"{k}_{robot_id}"] = control_dat[k]
            if "gripper" in self.config.data_to_collect:
                gripper_state = self.robot[robot_id].get_gripper_state()
                if "gripper_position" in self.config.data_to_collect["proprio"]:
                    buffer_data[f"gripper_position_{robot_id}"] = gripper_state["gripper_pos"]
                if "grasp_state" in self.config.data_to_collect["proprio"]:
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
        # pre execution
        self.exec_start_movement()

        # initialize state variable
        prev_button = False
        is_recording = False
        value = {robot_id: self.robot[robot_id].get_state()["p"] for robot_id in self.robot_ids}

        # start
        while self._collection_triggered:
            control_start = time.time()
            buffer_data = self.collect_buffer() 
                    
            # get device data
            device_data = self.data_collector.get_device_input(
                **{f"p_{robot_id}": buffer_data[f"p_{robot_id}"] for robot_id in self.robot_ids})
            
            # device connection lost
            if not device_data["final_valid"]:
                print("Device input not valid or connection lost")
                break

            # finish recording
            if is_recording and (not prev_button) and device_data["final_button"]:
                print("Recording stopped")
                break
            
            # start recording
            if (not prev_button) and device_data["final_button"]:
                print("Recording started")
                is_recording = True

                for robot_id in self.robot_ids:
                    self.data_collector.device[robot_id].reset()
                
                device_data = self.data_collector.get_device_input(
                    **{f"p_{robot_id}": buffer_data[f"p_{robot_id}"] for robot_id in self.robot_ids})
                
            prev_button = device_data["final_button"]
                
            if is_recording:
                # get robot control
                value = {robot_id: self.task_config.extra_config.control_post_process_fn(device_data[robot_id]["control"]) \
                    for robot_id in self.robot_ids}
                
                # get gripper control (BINARY)
                gripper_command = {robot_id: (1. - np.round(device_data[robot_id]["trigger"])) for robot_id in self.robot_ids}
                
                # update buffer    
                for robot_id in self.robot_ids:                    
                    if "tele_abs_control" in self.config.data_to_collect["control"]:
                        buffer_data[f"tele_abs_control_{robot_id}"] = value[robot_id]
                    if "gripper_command" in self.config.data_to_collect["control"]:
                        buffer_data[f"gripper_command_{robot_id}"] = gripper_command[robot_id]
                    self.data_collector.update_data_buffer(**buffer_data)

                # execute control to robot
                self.robot_cluster.tele_move(
                    action=value,
                    mode=self.control_mode,
                    vel_scale={robot_id: self.robot_config.robot_params[robot_id]["control"]["vel_scale"] for robot_id in self.robot_config.robot_ids},
                    acc_scale={robot_id: self.robot_config.robot_params[robot_id]["control"]["acc_scale"] for robot_id in self.robot_config.robot_ids},
                )
                
                # execute control to gripper
                self.robot_cluster.move_gripper(mode="thread", value=gripper_command)
                
                # sync control frequency
                control_end = time.time()
                wait_time = self.robot_config.control_dt - (control_end - control_start)
                if wait_time > 0.:
                    time.sleep(wait_time)
                    
        # soft stop
        self.exec_soft_stop(
            last_action=value,
            control_period=self.robot_config.control_dt,
            mode=self.control_mode)
        
        # post execution
        self.exec_finish_movement()
        

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
                if self.task_config.data_config.device_type == "vive":
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
            
    def get_device_input(self, **kwargs):
        output = dict()
        output["final_button"] = False
        output["final_valid"] = True
        
        for robot_id in self.robot_ids:
            output[robot_id] = self.device[robot_id].get_input(robot_pose=kwargs[f"p_{robot_id}"])
            output["final_button"] = output["final_button"] or output[robot_id]["button"]
            output["final_valid"] = output["final_valid"] and output[robot_id]["valid"]
        return output
    
    def get_control_mode(self):
        from helper.extra_utils import ROBOT_CONTROL_MODE
        
        control_mode = self.device[self.robot_ids[0]].CONTROL_MODE
        if control_mode == ROBOT_CONTROL_MODE.TELE_JOINT_ABSOLUTE:
            return "joint_abs"
        else:
            return "task_abs"
            
    def init_data_buffer(self):
        """
        "q", "qdot", "p", "pdot",   # robot state (T, D)
        "gripper_position", "grasp_state",   # gripper state  (T, 1)
        "images.rgb.{cam_name}", "images.depth.{cam_name}", "images.intrinsics.{cam_name}",  # rgb (T, H, W, C) [rgb], depth (T, H, W) [float32], intrinsics (3, 3) [float32]
        "tele_abs_control"   # robot control  (T, 6)
        "gripper_command",   # gripper control (T, 1)
        """
        # Reset data
        self.data_types = []
        data_to_collect = []
        for k in self.data_collector_config.data_to_collect.keys():
            if k == "camera":
                continue
            else:
                data_to_collect.extend(self.data_collector_config.data_to_collect[k])
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
