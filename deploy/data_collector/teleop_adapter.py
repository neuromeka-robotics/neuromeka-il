"""Default teleoperation behavior: one device per configured robot.

Adapters own device routing and robot commands. The collector owns timing,
recording and cameras. Subclasses can override this interface without changing
existing robot configurations or the collection loop.
"""

import numpy as np


class TeleopAdapter:
    @staticmethod
    def validate_config(collector_config):
        """Validate configuration without constructing an adapter or opening connections."""
        pass

    def __init__(self, collector_config):
        self.validate_config(collector_config)
        self.collector_config = collector_config
        self.robot_ids = list(collector_config.robot_config.robot_ids)
        self.device_ids = self.robot_ids.copy()
        self.post_process = collector_config.task_config.extra_config.control_post_process_fn

    def bind(self, robot_cluster, devices, device_mode):
        """Attach runtime connections after the collector has created its devices."""
        if device_mode not in ("task_abs", "joint_abs"):
            raise ValueError(f"Unsupported device output mode: {device_mode}")
        self.robot_cluster = robot_cluster
        self.devices = devices
        # This adapter forwards device commands directly to the robot.
        self.control_mode = device_mode

    def device_poses(self, buffer_data):
        return {f"p_{robot_id}": buffer_data[f"p_{robot_id}"] for robot_id in self.robot_ids}

    def reset(self, buffer_data):
        """Anchor devices when recording starts."""
        for device in self.devices.values():
            device.reset()

    def convert(self, device_data, buffer_data):
        """Return robot actions and binary gripper commands, keyed by robot ID."""
        actions = {robot_id: self.post_process(device_data[robot_id]["control"])
                   for robot_id in self.robot_ids}
        grippers = {robot_id: 1. - np.round(device_data[robot_id]["trigger"])
                    for robot_id in self.robot_ids}
        return actions, grippers

    def recorded_actions(self, actions):
        return actions

    def send(self, action, mode, vel_scale, acc_scale):
        """Also used to resend the final action during a soft stop."""
        self.robot_cluster.tele_move(
            action=action, mode=mode, vel_scale=vel_scale, acc_scale=acc_scale)

    @property
    def recording_metadata(self):
        # Leave the existing single/dual-robot HDF5 schema unchanged.
        return {}
