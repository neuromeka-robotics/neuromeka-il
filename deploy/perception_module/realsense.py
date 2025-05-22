import time
import numpy as np
import pyrealsense2 as rs
from threading import Thread, Lock


class RealsenseCam:
    def __init__(self):
        self.pipeline = None
        self.depth_image = None
        self.color_image = None
        self.ctx = None

    def __del__(self):
        if self.pipeline is not None:
            self.pipeline.stop()
            del self.pipeline

    def connect(self, serial_number=None, align=False, clipping_distance_m=None, exposure=None):
        # Configure camera
        self.pipeline = rs.pipeline()
        config = rs.config()
        if serial_number is not None:
            config.enable_device(serial_number)
        self.ctx = rs.context()

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 60)

        # Start camera stream
        profile = self.pipeline.start(config)
        
        # Set exposure
        if exposure is not None:
            assert isinstance(exposure, int)
            device_name = profile.get_device().get_info(rs.camera_info.name)
            
            if device_name == "Intel RealSense D405":
                sensor = profile.get_device().query_sensors()[0]
            else:  # ex: Intel RealSense D435
                sensor = profile.get_device().query_sensors()[1]  # 0: Depth / 1: RGB
            sensor.set_option(rs.option.exposure, exposure)

        # Get camera data (depth scale, camera intrinsics)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()  # mm to m
        if clipping_distance_m is not None:
            self.clipping_distance = clipping_distance_m / self.depth_scale  # [mm]
        else:
            self.clipping_distance = None
        # print("Depth Scale is: ", self.depth_scale)

        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        raw_intrinsics = color_profile.get_intrinsics()
        self.intrinsics = np.array(
            [
                [raw_intrinsics.fx, 0, raw_intrinsics.ppx],
                [0, raw_intrinsics.fy, raw_intrinsics.ppy],
                [0, 0, 1],
            ]
        )

        # Set RGB-Depth align function
        if align:
            align_to = rs.stream.color
            self.align_func = rs.align(align_to)
        else:
            self.align_func = None
        
    def update_data(self):
        frames = self.pipeline.wait_for_frames()  # 500
        if self.align_func:
            frames = self.align_func.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()  # RGB
        if not depth_frame or not color_frame:
            return False

        self.depth_image = np.asarray(depth_frame.get_data())  # (480, 640)
        if self.clipping_distance:
            self.depth_image[self.depth_image > self.clipping_distance] = 0.0
        self.depth_image = self.depth_image.astype(np.float32) * self.depth_scale
        self.color_image = np.asarray(color_frame.get_data())  # (480, 640, 3)
        return True

    def get_num_devices(self):
        devices = self.ctx.query_devices()
        return len(devices)


class RealsenseCamHandler:
    def __init__(self, serial_number=None, align=False, clipping_distance_m=None, dt=0.03, exposure=None):
        # Set variable
        self._thread = None
        self._cam_data_lock = Lock()
        self.dt = dt
        self.data = {
            "rgb": None,
            "depth": None,
            "intrinsics": None
        }
        
        # Set camera
        self.camera = RealsenseCam()
        self.camera.connect(serial_number, align, clipping_distance_m, exposure)
        time.sleep(2)
        
    def __del__(self):
        self.stop()
        
    def start(self):
        self._thread_running = True
        self._cam_updated = False
        self._thread = Thread(target=self._thread_callback, daemon=True)
        self._thread.start()
        
    def stop(self):
        print("stop called for camHandler")
        if self._thread is not None:
            self._thread_running = False
            self._thread.join()
            self._thread = None

        del self.camera
        
    def _thread_callback(self):
        while self._thread_running:
            time_start = time.time()
            self._update_measurement()
            duration = time.time() - time_start 
        
            wait_time = self.dt - duration
            if wait_time > 0.:
                time.sleep(wait_time)
        
    def _update_measurement(self):
        self.camera.update_data()
        self._cam_data_lock.acquire()
        self.data["rgb"] = self.camera.color_image.copy()
        self.data["depth"] = self.camera.depth_image.copy()
        self.data["intrinsics"] = self.camera.intrinsics.copy()
        self._cam_data_lock.release()
        self._cam_updated = True
        
    # getters
    def get_rgb_image(self):
        if not self._cam_updated:
            return None
        else:
            self._cam_data_lock.acquire()
            ouput = self.data["rgb"].copy()
            self._cam_data_lock.release()
            return ouput

    def get_all(self):
        if not self._cam_updated:
            return None
        else:
            output = dict()
            self._cam_data_lock.acquire()
            output["rgb"] = self.data["rgb"].copy()
            output["depth"] = self.data["depth"].copy()
            output["intrinsics"] = self.data["intrinsics"].copy()
            self._cam_data_lock.release()
            return output
        
    def get_num_devices(self):
        return self.camera.get_num_devices()
