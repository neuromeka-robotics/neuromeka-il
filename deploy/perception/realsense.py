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

    def connect(
        self,
        serial_number=None,
        align=False,
        clipping_distance_m=None,
        exposure=None,
        width=640,
        height=480,
        fps=60,
        enable_depth=True,
    ):
        # Configure camera
        self.pipeline = rs.pipeline()
        config = rs.config()
        if serial_number is not None:
            config.enable_device(serial_number)
        self.ctx = rs.context()

        if enable_depth:
            config.enable_stream(
                rs.stream.depth, width, height, rs.format.z16, fps
            )
        config.enable_stream(
            rs.stream.color, width, height, rs.format.rgb8, fps
        )

        # Start camera stream
        profile = self.pipeline.start(config)
        
        # Set exposure
        if exposure is not None:
            color_sensor = next(
                (
                    sensor
                    for sensor in profile.get_device().query_sensors()
                    if sensor.supports(rs.option.exposure)
                    and any(
                        stream.stream_type() == rs.stream.color
                        for stream in sensor.get_stream_profiles()
                    )
                ),
                None,
            )
            if color_sensor is None:
                self.pipeline.stop()
                self.pipeline = None
                raise RuntimeError(
                    "The RealSense color sensor does not support exposure"
                )
            if color_sensor.supports(rs.option.enable_auto_exposure):
                color_sensor.set_option(rs.option.enable_auto_exposure, 0.0)
            color_sensor.set_option(rs.option.exposure, float(exposure))
            applied_exposure = color_sensor.get_option(rs.option.exposure)
            print(f"RealSense color exposure: {applied_exposure:g}")

        # Get camera data (depth scale, camera intrinsics)
        self.enable_depth = enable_depth
        if self.enable_depth:
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()  # mm to m
        else:
            self.depth_scale = None
        if clipping_distance_m is not None and self.enable_depth:
            self.clipping_distance = clipping_distance_m / self.depth_scale
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
        self.dist_coeffs = np.asarray(raw_intrinsics.coeffs[:5], dtype=np.float64)

        # Set RGB-Depth align function
        if align and self.enable_depth:
            align_to = rs.stream.color
            self.align_func = rs.align(align_to)
        else:
            self.align_func = None
        
    def update_data(self):
        frames = self.pipeline.wait_for_frames()  # 500
        if self.align_func:
            frames = self.align_func.process(frames)
        depth_frame = frames.get_depth_frame() if self.enable_depth else None
        color_frame = frames.get_color_frame()  # RGB
        if not color_frame or (self.enable_depth and not depth_frame):
            return False

        if self.enable_depth:
            self.depth_image = np.asarray(depth_frame.get_data())
            if self.clipping_distance:
                self.depth_image[
                    self.depth_image > self.clipping_distance
                ] = 0.0
            self.depth_image = (
                self.depth_image.astype(np.float32) * self.depth_scale
            )
        else:
            self.depth_image = None
        self.color_image = np.asarray(color_frame.get_data())  # (480, 640, 3)
        return True

    def get_num_devices(self):
        devices = self.ctx.query_devices()
        return len(devices)

    def get_device_serial_numbers(self):
        devices = rs.context().query_devices()
        return [device.get_info(rs.camera_info.serial_number) for device in devices]


class RealsenseCamHandler:
    def __init__(
        self,
        serial_number=None,
        align=False,
        clipping_distance_m=None,
        dt=0.03,
        exposure=None,
        width=640,
        height=480,
        fps=60,
        enable_depth=True,
    ):
        # Set variable
        self._thread = None
        self._cam_data_lock = Lock()
        self.dt = dt
        self.data = {
            "rgb": None,
            "depth": None,
            "intrinsics": None,
            "dist_coeffs": None,
        }
        
        # Set camera
        self.camera = RealsenseCam()
        self.camera.connect(
            serial_number,
            align,
            clipping_distance_m,
            exposure,
            width,
            height,
            fps,
            enable_depth,
        )
        time.sleep(0.5)
        
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
        if not self.camera.update_data():
            return
        self._cam_data_lock.acquire()
        self.data["rgb"] = self.camera.color_image.copy()
        self.data["depth"] = (
            None
            if self.camera.depth_image is None
            else self.camera.depth_image.copy()
        )
        self.data["intrinsics"] = self.camera.intrinsics.copy()
        self.data["dist_coeffs"] = self.camera.dist_coeffs.copy()
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
            output["depth"] = (
                None
                if self.data["depth"] is None
                else self.data["depth"].copy()
            )
            output["intrinsics"] = self.data["intrinsics"].copy()
            output["dist_coeffs"] = self.data["dist_coeffs"].copy()
            self._cam_data_lock.release()
            return output
        
    def get_num_devices(self):
        return self.camera.get_num_devices()
