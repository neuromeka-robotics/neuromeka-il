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

# communication
from communication.robot import Robot

# config
from helper.config_utils import ROBOT_CONFIG, TASK_CONFIG
from data_collector.config import DATA_COLLECTOR_ROBOT_CONFIG, DATA_COLLECTOR_TASK_CONFIG


class DataCollectionScheduler(Controller):
    ROBOT_CONFIG = DATA_COLLECTOR_ROBOT_CONFIG
    ROBOT_IDS = DATA_COLLECTOR_ROBOT_CONFIG.robot_ids
    TASK_CONFIG = DATA_COLLECTOR_TASK_CONFIG
    TASK_NAME = DATA_COLLECTOR_TASK_CONFIG.name
    
    def __init__(self, robot: Dict[int, Robot] | None = None, **kwargs):
        # set robot
        super(DataCollectionScheduler, self).__init__(robot=robot, **kwargs)
        
        # set camera
        self.camera: Dict[str, RealsenseCamHandler] = kwargs.get("camera", dict())
        for cam_name in self.TASK_CONFIG.camera_config.cam_names:
            if cam_name in self.camera.keys():
                if not self.camera[cam_name]._thread_running:
                    self.camera[cam_name].start()
            else:
                cam_config = self.TASK_CONFIG.camera_config.cam_params[cam_name]
                
                self.camera[cam_name] = RealsenseCamHandler(
                    serial_number=cam_config["serial"], 
                    align=True, 
                    clipping_distance_m=1., 
                    exposure=cam_config.get("exposure", None)
                )
                self.camera[cam_name].start()
        
        # set data collector
        self.data_collector = TeleopDataCollector(
            robot_config=self.ROBOT_CONFIG,
            task_config=self.TASK_CONFIG,
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
            command_listener = KeyboardListener(key_targets=["s", "e"])
            print("Click 's' to save / Click 'e' to not save")
            while True:
                command = command_listener.get_key_states()
                if command["updated"]:
                    if command["value"]["s"]:
                        print(f"Trajectory {self.data_collector.traj_num} saved.")
                        self.data_collector.save_data_buffer()
                        self.data_collector.visualize_last_data()
                        break
                    elif command["value"]["e"]:
                        print("Not saved")
                        break
                        
            self.data_collector.init_data_buffer()
                
            self._collection_triggered = False

    def _collection_fn(self):
        # pre execution
        self.exec_start_movement()

        # initialize state variable
        prev_button = False
        is_recording = False
        value = {robot_id: self.robot[robot_id].get_state()["p"] for robot_id in self.ROBOT_IDS}

        # start
        while self._collection_triggered:
            control_start = time.time()
            
            buffer_data = dict()
                
            # get proprioception
            for robot_id in self.ROBOT_IDS:
                control_dat = self.robot[robot_id].get_state()
                gripper_state = self.robot[robot_id].get_gripper_state()
                
                buffer_data[f"q_{robot_id}"] = control_dat["q"]
                buffer_data[f"qdot_{robot_id}"] = control_dat["qdot"]
                buffer_data[f"p_{robot_id}"] = control_dat["p"]
                buffer_data[f"pdot_{robot_id}"] = control_dat["pdot"]
                buffer_data[f"gripper_position_{robot_id}"] = gripper_state["gripper_pos"]
                buffer_data[f"grasp_state_{robot_id}"] = gripper_state["grasp_state"]
            
            # get exteroception
            for cam_name in self.TASK_CONFIG.camera_config.cam_names:
                cam_output = self.camera[cam_name].get_all()
                
                buffer_data[f"images.rgb.{cam_name}"] = cam_output["rgb"]
                if self.TASK_CONFIG.camera_config.cam_params[cam_name].get("enable_depth", False):
                    buffer_data[f"images.depth.{cam_name}"] = cam_output["depth"]
                buffer_data[f"images.intrinsics.{cam_name}"] = cam_output["intrinsics"]
                    
            # get device data
            device_data = self.data_collector.get_device_input(
                **{f"p_{robot_id}": buffer_data[f"p_{robot_id}"] for robot_id in self.ROBOT_IDS})
            
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

                for robot_id in self.ROBOT_IDS:
                    self.data_collector.device[robot_id].reset()
                
                device_data = self.data_collector.get_device_input(
                    **{f"p_{robot_id}": buffer_data[f"p_{robot_id}"] for robot_id in self.ROBOT_IDS})
                
            prev_button = device_data["final_button"]
                
            if is_recording:
                # get robot control
                value = {robot_id: self.TASK_CONFIG.extra_config.control_post_process_fn(device_data[robot_id]["control"]) \
                    for robot_id in self.ROBOT_IDS}
                
                # get gripper control (BINARY)
                gripper_command = {robot_id: (1. - np.round(device_data[robot_id]["trigger"])) for robot_id in self.ROBOT_IDS}
                
                # update buffer    
                for robot_id in self.ROBOT_IDS:                    
                    buffer_data[f"tele_abs_control_{robot_id}"] = value[robot_id]
                    buffer_data[f"gripper_command_{robot_id}"] = gripper_command[robot_id]
                    self.data_collector.update_data_buffer(**buffer_data)

                # execute control to robot
                self.robot_cluster.tele_move(
                    action=value,
                    mode=self.control_mode,
                    vel_scale={robot_id: self.ROBOT_CONFIG.robot_params[robot_id]["control"]["vel_scale"] for robot_id in self.ROBOT_CONFIG.robot_ids},
                    acc_scale={robot_id: self.ROBOT_CONFIG.robot_params[robot_id]["control"]["acc_scale"] for robot_id in self.ROBOT_CONFIG.robot_ids},
                )
                
                # execute control to gripper
                self.robot_cluster.move_gripper(mode="thread", value=gripper_command)
                
                # sync control frequency
                control_end = time.time()
                wait_time = self.ROBOT_CONFIG.control_dt - (control_end - control_start)
                if wait_time > 0.:
                    time.sleep(wait_time)
                    
        # soft stop
        self.exec_soft_stop(
            last_action=value,
            control_period=self.ROBOT_CONFIG.control_dt,
            mode=self.control_mode)
        
        # post execution
        self.exec_finish_movement()
        

class TeleopDataCollector:
    def __init__(self, 
                 robot_config: ROBOT_CONFIG, 
                 task_config: TASK_CONFIG, 
                 dagger_mode=False, 
                 device=None):
        self.ROBOT_CONFIG = robot_config
        self.ROBOT_IDS = robot_config.robot_ids
        self.TASK_CONFIG = task_config
        self.TASK_NAME = task_config.name
        
        self.device: Dict[int, BaseDevice] = dict()
        if device is None:
            for robot_id in self.ROBOT_IDS:
                if self.TASK_CONFIG.data_config.device_type == "vive":
                    from data_collector.device.vive import Vive
                    self.device[robot_id] = Vive()
                else:
                    raise ValueError
        else:
            self.device = device
        
        # check data colleciton progress
        self.DATA_DIR = os.path.join(self.TASK_CONFIG.data_config.data_dir, self.TASK_NAME)
        self.DATA_VIZ_DIR = os.path.join(self.TASK_CONFIG.data_config.data_viz_dir, self.TASK_NAME)
        
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
        print(f"[Task: {self.TASK_NAME}] Continue data saving from {self.traj_num}")
        print("==========================")
        
        # save data collection progress (DAGGER)
        if dagger_mode:
            collect_progress_file = os.path.join(self.TASK_CONFIG.data_config.data_dir, f"{self.TASK_NAME}_progress.json")
            if os.path.exists(collect_progress_file):
                with open(collect_progress_file, "r") as f:
                    collect_progress = json.load(f)
                
                collect_iters = []
                for k in collect_progress.keys():
                    collect_iters.append(int(k.split("_")[-1]))
                current_collect_iter = max(collect_iters) + 1
            else:
                collect_progress = dict()
                current_collect_iter = 1
            
            if collect_progress[f"iter_{current_collect_iter - 1}"] != self.traj_num:
                collect_progress[f"iter_{current_collect_iter}"] = self.traj_num  # the starting data point in current iteration
                with open(collect_progress_file, "w") as f:
                    json.dump(collect_progress, f)
            
        # initialize data buffer
        self.init_data_buffer()
        
    def __del__(self):
        for robot_id in self.ROBOT_IDS:
            self.device[robot_id].exit()
            
    def get_device_input(self, **kwargs):
        output = dict()
        output["final_button"] = False
        output["final_valid"] = True
        
        for robot_id in self.ROBOT_IDS:
            output[robot_id] = self.device[robot_id].get_input(robot_pose=kwargs[f"p_{robot_id}"])
            output["final_button"] = output["final_button"] or output[robot_id]["button"]
            output["final_valid"] = output["final_valid"] and output[robot_id]["valid"]
        return output
    
    def get_control_mode(self):
        from helper.extra_utils import ROBOT_CONTROL_MODE
        
        control_mode = self.device[self.ROBOT_IDS[0]].CONTROL_MODE
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
        for type in ["q", "qdot", "p", "pdot", "gripper_position", "grasp_state", "tele_abs_control", "gripper_command"]:
            for robot_id in self.ROBOT_IDS:
                self.data_types.append(f"{type}_{robot_id}")
                
        for cam_name in self.TASK_CONFIG.camera_config.cam_names:
            self.data_types.append(f"images.rgb.{cam_name}")
            if self.TASK_CONFIG.camera_config.cam_params[cam_name].get("enable_depth", False):
                self.data_types.append(f"images.depth.{cam_name}")
            self.data_types.append(f"images.intrinsics.{cam_name}")
        
        self.traj = dict()
        for data_type in self.data_types:
            self.traj[data_type] = []
        
        self.traj_len = 0
        
        # Reset device
        for robot_id in self.ROBOT_IDS:
            self.device[robot_id].reset()
        
    def update_data_buffer(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.data_types:
                if "intrinsics" in k:
                    if len(self.traj[k]) == 0:
                        self.traj[k] = v
                else:
                    self.traj[k].append(v)
            else:
                print(f"Ignoring key: {k}")

        self.traj_len += 1
        
    def save_data_buffer(self):
        for k in self.traj.keys():
            if len(self.traj[k]) > 0:
                if ("rgb" in k) or ("depth" in k):
                    self.traj[k] = np.asarray(self.traj[k])
                else:
                    self.traj[k] = np.asarray(self.traj[k]).astype(np.float32)
                
                if len(self.traj[k].shape) == 1:
                    self.traj[k] = self.traj[k][..., np.newaxis]  # change shapre from (T,) to (T, 1)
                    
                print(f"{k}: {self.traj[k].shape} ({self.traj[k].dtype})")

        with h5py.File(f"{self.DATA_DIR}/{self.traj_num}.h5", "w") as hf:
            for k, v in self.traj.items():
                if len(v) > 0:
                    hf.create_dataset(k, data=v)
        
        print(f"Data saved to {self.DATA_DIR}/{self.traj_num}.h5")
        self.traj_num += 1
        self.init_data_buffer()
    
    def visualize_last_data(self):
        # SET DATA ID TO VISUALIZE
        VISUALIZE_DATA_ID = self.traj_num - 1
        FREQUENCY = int(1 / self.ROBOT_CONFIG.control_dt)
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
        for robot_id in self.ROBOT_IDS:
            joint_pos_traj = collected_traj[f"q_{robot_id}"]
            joint_vel_traj = collected_traj[f"qdot_{robot_id}"]
            end_pose_traj = collected_traj[f"p_{robot_id}"]
            end_vel_traj = collected_traj[f"pdot_{robot_id}"]
            control_traj = collected_traj[f"tele_abs_control_{robot_id}"]
            gripper_position_traj = collected_traj[f"gripper_position_{robot_id}"]
            grasp_state_traj = collected_traj[f"grasp_state_{robot_id}"]
            gripper_command_traj = collected_traj[f"gripper_command_{robot_id}"]

            n_steps = joint_pos_traj.shape[0]
            time_traj = np.arange(n_steps) * 1 / FREQUENCY

            plt.plot(time_traj, joint_pos_traj)
            plt.xlabel("Time [s]")
            plt.ylabel("Joint position")
            plt.savefig(f"{self.DATA_VIZ_DIR}/{VISUALIZE_DATA_ID}_joint_pos_{robot_id}.png")
            plt.clf()
            plt.close()

            plt.plot(time_traj, joint_vel_traj)
            plt.xlabel("Time [s]")
            plt.ylabel("Joint velocity")
            plt.savefig(f"{self.DATA_VIZ_DIR}/{VISUALIZE_DATA_ID}_joint_vel_{robot_id}.png")
            plt.clf()
            plt.close()

            plt.plot(time_traj, end_pose_traj)
            plt.xlabel("Time [s]")
            plt.ylabel("End-effector position")
            plt.savefig(f"{self.DATA_VIZ_DIR}/{VISUALIZE_DATA_ID}_end_pos_{robot_id}.png")
            plt.clf()
            plt.close()

            plt.plot(time_traj, end_vel_traj)
            plt.xlabel("Time [s]")
            plt.ylabel("End-effector velocity")
            plt.savefig(f"{self.DATA_VIZ_DIR}/{VISUALIZE_DATA_ID}_end_vel_{robot_id}.png")
            plt.clf()
            plt.close()

            plt.plot(time_traj, control_traj)
            plt.xlabel("Time [s]")
            plt.ylabel("Control")
            plt.savefig(f"{self.DATA_VIZ_DIR}/{VISUALIZE_DATA_ID}_control_{robot_id}.png")
            plt.clf()
            plt.close()

            plt.plot(time_traj, gripper_command_traj)
            plt.xlabel("Time [s]")
            plt.ylabel("Gripper command")
            plt.savefig(f"{self.DATA_VIZ_DIR}/{VISUALIZE_DATA_ID}_gripper_command_{robot_id}.png")
            plt.clf()
            plt.close()

            plt.plot(time_traj, gripper_position_traj)
            plt.xlabel("Time [s]")
            plt.ylabel("Gripper position")
            plt.savefig(f"{self.DATA_VIZ_DIR}/{VISUALIZE_DATA_ID}_gripper_position_{robot_id}.png")
            plt.clf()
            plt.close()

            plt.plot(time_traj, grasp_state_traj)
            plt.xlabel("Time [s]")
            plt.ylabel("Grasp state")
            plt.savefig(f"{self.DATA_VIZ_DIR}/{VISUALIZE_DATA_ID}_grasp_state_{robot_id}.png")
            plt.clf()
            plt.close()