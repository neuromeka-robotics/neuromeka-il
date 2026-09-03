"""Estimate a marker-covered box pose and run live RealSense demos.

The reusable estimator has no RealSense or Open3D dependency.  It returns
``T_camera_box`` in metres using the OpenCV optical camera convention
(+x right, +y down, +z forward).

Examples, run from ``deploy``::

    python perception/aruco_box_pose.py camera
    python perception/aruco_box_pose.py robot \
        --calibration /opt/neuromeka/psf/calib_results/Thc_EIR8_260810.json \
        --robot-ip 192.168.0.10

For a fixed camera, the robot demo accepts a PSF ``Tbc`` artifact and does not
need ``--robot-ip``.  For a head-mounted ``Thc`` artifact, either provide a
live robot IP or a fixed ``--base-head-pose X Y Z RX RY RZ`` (mm, degrees).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2 as cv
import numpy as np


@dataclass
class MarkerPose:
    """Pose of a marker in the box frame (``T_box_marker``)."""

    position_m: np.ndarray
    rotation: np.ndarray

    @classmethod
    def from_config(cls, value: "MarkerPose | Mapping[str, Any]") -> "MarkerPose":
        if isinstance(value, cls):
            position = value.position_m
            rotation = value.rotation
        else:
            position = value.get("position_m", value.get("position", value.get("pos")))
            rotation = value.get("rotation", value.get("rot"))
        position_array = np.asarray(position, dtype=np.float64)
        rotation_array = np.asarray(rotation, dtype=np.float64)
        if position_array.shape != (3,) or not np.all(np.isfinite(position_array)):
            raise ValueError("Each marker position must contain three finite values")
        if rotation_array.shape != (3, 3) or not np.all(np.isfinite(rotation_array)):
            raise ValueError("Each marker rotation must be a finite 3x3 matrix")
        if not np.allclose(rotation_array.T @ rotation_array, np.eye(3), atol=1e-6):
            raise ValueError("Each marker rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation_array), 1.0, atol=1e-6):
            raise ValueError("Each marker rotation must have determinant +1")
        return cls(position_m=position_array.copy(), rotation=rotation_array.copy())


@dataclass
class ArucoBoxConfig:
    """Geometry and detector settings consumed by :class:`ArucoBoxPoseEstimator`."""

    marker_length_m: float
    marker_poses: Mapping[int, MarkerPose | Mapping[str, Any]]
    dictionary: str | int = "DICT_6X6_250"
    detector_parameters: Mapping[str, Any] = field(default_factory=dict)
    box_size_m: Sequence[float] = (0.307, 0.153, 0.215)
    translation_smoothing_alpha: float = 1.0
    rotation_smoothing_alpha: float = 1.0
    image_color_order: str = "rgb"

    def __post_init__(self) -> None:
        self.marker_length_m = float(self.marker_length_m)
        if not np.isfinite(self.marker_length_m) or self.marker_length_m <= 0.0:
            raise ValueError("marker_length_m must be a positive finite value")
        self.marker_poses = {
            int(marker_id): MarkerPose.from_config(pose)
            for marker_id, pose in self.marker_poses.items()
        }
        if not self.marker_poses:
            raise ValueError("marker_poses must contain at least one marker")
        self.box_size_m = tuple(float(value) for value in self.box_size_m)
        if len(self.box_size_m) != 3 or any(value <= 0.0 for value in self.box_size_m):
            raise ValueError("box_size_m must contain three positive values")
        for name in ("translation_smoothing_alpha", "rotation_smoothing_alpha"):
            alpha = float(getattr(self, name))
            if not 0.0 < alpha <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
            setattr(self, name, alpha)
        self.image_color_order = self.image_color_order.lower()
        if self.image_color_order not in {"rgb", "bgr"}:
            raise ValueError("image_color_order must be 'rgb' or 'bgr'")

    @classmethod
    def from_dict(cls, config: Mapping[str, Any]) -> "ArucoBoxConfig":
        values = dict(config)
        if "marker_length" in values and "marker_length_m" not in values:
            values["marker_length_m"] = values.pop("marker_length")
        return cls(**values)

    @property
    def dictionary_id(self) -> int:
        if isinstance(self.dictionary, str):
            if not hasattr(cv.aruco, self.dictionary):
                raise ValueError(f"Unknown ArUco dictionary: {self.dictionary}")
            return int(getattr(cv.aruco, self.dictionary))
        return int(self.dictionary)


@dataclass
class ArucoBoxDetection:
    """Full detection result; ``pose_camera_box`` is the controller-facing output."""

    pose_camera_box: np.ndarray
    raw_pose_camera_box: np.ndarray
    corners: list[np.ndarray]
    marker_ids: np.ndarray
    used_marker_ids: tuple[int, ...]
    rvecs: np.ndarray
    tvecs: np.ndarray


def marker_face_label(marker_pose: MarkerPose) -> str:
    """Describe the box face containing a marker from its configured position."""

    axis_index = int(np.argmax(np.abs(marker_pose.position_m)))
    axis_value = marker_pose.position_m[axis_index]
    axis_name = "XYZ"[axis_index]
    sign = "+" if axis_value >= 0.0 else "-"
    semantic_name = {
        (0, -1): "LEFT",
        (0, 1): "RIGHT",
        (1, -1): "FRONT",
        (1, 1): "BACK",
        (2, -1): "BOTTOM",
        (2, 1): "TOP",
    }[(axis_index, 1 if axis_value >= 0.0 else -1)]
    return f"{semantic_name} ({sign}{axis_name})"


class ArucoBoxPoseEstimator:
    """Fuse known ArUco marker poses into a box pose in the camera frame.

    A controller normally only needs::

        estimator = ArucoBoxPoseEstimator(config)
        T_camera_box = estimator.estimate(rgb, intrinsics, dist_coeffs)

    ``None`` is returned when none of the configured markers is visible.  The
    estimator never substitutes the previous pose for a missed detection.
    """

    def __init__(self, config: ArucoBoxConfig | Mapping[str, Any]):
        self.config = (
            config if isinstance(config, ArucoBoxConfig) else ArucoBoxConfig.from_dict(config)
        )
        self._dictionary = cv.aruco.getPredefinedDictionary(self.config.dictionary_id)
        self._parameters = cv.aruco.DetectorParameters()
        for name, value in self.config.detector_parameters.items():
            if not hasattr(self._parameters, name):
                raise ValueError(f"Unknown ArUco detector parameter: {name}")
            setattr(self._parameters, name, value)
        self._detector = (
            cv.aruco.ArucoDetector(self._dictionary, self._parameters)
            if hasattr(cv.aruco, "ArucoDetector")
            else None
        )
        self._previous_pose: np.ndarray | None = None

    def reset(self) -> None:
        """Clear smoothing state, for example at the start of a policy rollout."""

        self._previous_pose = None

    @staticmethod
    def _camera_parameters(
        camera_matrix: Any, dist_coeffs: Any | None
    ) -> tuple[np.ndarray, np.ndarray]:
        intrinsics = np.asarray(camera_matrix, dtype=np.float64)
        if intrinsics.shape != (3, 3) or not np.all(np.isfinite(intrinsics)):
            raise ValueError("camera_matrix must be a finite 3x3 matrix")
        distortion = (
            np.zeros(5, dtype=np.float64)
            if dist_coeffs is None
            else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
        )
        if not np.all(np.isfinite(distortion)):
            raise ValueError("dist_coeffs contains NaN or infinite values")
        return intrinsics, distortion

    def _grayscale(self, image: Any, color_order: str | None) -> np.ndarray:
        frame = np.asarray(image)
        if frame.ndim == 2:
            return frame
        if frame.ndim != 3 or frame.shape[2] not in {3, 4}:
            raise ValueError("image must be grayscale, RGB/RGBA, or BGR/BGRA")
        order = (color_order or self.config.image_color_order).lower()
        if order not in {"rgb", "bgr"}:
            raise ValueError("color_order must be 'rgb' or 'bgr'")
        if frame.shape[2] == 3:
            code = cv.COLOR_RGB2GRAY if order == "rgb" else cv.COLOR_BGR2GRAY
        else:
            code = cv.COLOR_RGBA2GRAY if order == "rgb" else cv.COLOR_BGRA2GRAY
        return cv.cvtColor(frame, code)

    def estimate(
        self,
        image: Any,
        camera_matrix: Any,
        dist_coeffs: Any | None = None,
        *,
        color_order: str | None = None,
    ) -> np.ndarray | None:
        """Return a 4x4 ``T_camera_box`` matrix, or ``None`` on a miss."""

        detection = self.detect(
            image, camera_matrix, dist_coeffs, color_order=color_order
        )
        return None if detection is None else detection.pose_camera_box.copy()

    def detect(
        self,
        image: Any,
        camera_matrix: Any,
        dist_coeffs: Any | None = None,
        *,
        color_order: str | None = None,
    ) -> ArucoBoxDetection | None:
        """Return pose plus marker details used by the visualization demos."""

        intrinsics, distortion = self._camera_parameters(camera_matrix, dist_coeffs)
        gray = self._grayscale(image, color_order)
        if self._detector is not None:
            corners, marker_ids, _ = self._detector.detectMarkers(gray)
        else:
            corners, marker_ids, _ = cv.aruco.detectMarkers(
                gray, self._dictionary, parameters=self._parameters
            )
        if marker_ids is None:
            return None
        marker_poses = self._estimate_marker_poses(
            corners, intrinsics, distortion
        )
        if marker_poses is None:
            return None
        rvecs, tvecs, valid_indices = marker_poses
        pose_marker_ids = np.asarray(marker_ids)[valid_indices]
        pose_corners = [corners[index] for index in valid_indices]
        fused = self.estimate_from_detections(pose_marker_ids, rvecs, tvecs)
        if fused is None:
            return None
        raw_pose, pose, used_ids = fused
        return ArucoBoxDetection(
            pose_camera_box=pose,
            raw_pose_camera_box=raw_pose,
            corners=pose_corners,
            marker_ids=np.asarray(pose_marker_ids, dtype=np.int32),
            used_marker_ids=used_ids,
            rvecs=np.asarray(rvecs, dtype=np.float64),
            tvecs=np.asarray(tvecs, dtype=np.float64),
        )

    def _estimate_marker_poses(
        self,
        corners: Sequence[np.ndarray],
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, list[int]] | None:
        """Estimate each marker pose with the supported OpenCV PnP API."""

        half_length = self.config.marker_length_m / 2.0
        # This is OpenCV's ARUCO_CCW_CENTER marker coordinate convention and
        # the required point order for SOLVEPNP_IPPE_SQUARE.
        object_points = np.asarray(
            [
                [-half_length, half_length, 0.0],
                [half_length, half_length, 0.0],
                [half_length, -half_length, 0.0],
                [-half_length, -half_length, 0.0],
            ],
            dtype=np.float64,
        )
        rvecs: list[np.ndarray] = []
        tvecs: list[np.ndarray] = []
        valid_indices: list[int] = []
        for index, marker_corners in enumerate(corners):
            image_points = np.ascontiguousarray(
                marker_corners, dtype=np.float64
            ).reshape(4, 2)
            success, rvec, tvec = cv.solvePnP(
                object_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv.SOLVEPNP_IPPE_SQUARE,
            )
            if not success:
                continue
            rvecs.append(np.asarray(rvec, dtype=np.float64).reshape(1, 3))
            tvecs.append(np.asarray(tvec, dtype=np.float64).reshape(1, 3))
            valid_indices.append(index)
        if not valid_indices:
            return None
        return np.asarray(rvecs), np.asarray(tvecs), valid_indices

    def estimate_from_detections(
        self, marker_ids: Any, rvecs: Any, tvecs: Any
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]] | None:
        """Fuse already-estimated marker poses (also useful for offline tests)."""

        ids = np.asarray(marker_ids).reshape(-1)
        rotations = np.asarray(rvecs, dtype=np.float64).reshape(-1, 3)
        translations = np.asarray(tvecs, dtype=np.float64).reshape(-1, 3)
        if len(ids) != len(rotations) or len(ids) != len(translations):
            raise ValueError("marker_ids, rvecs, and tvecs must have matching lengths")

        inferred_poses: list[np.ndarray] = []
        used_ids: list[int] = []
        for marker_id_raw, rvec, position_camera_marker in zip(
            ids, rotations, translations
        ):
            marker_id = int(marker_id_raw)
            marker_pose = self.config.marker_poses.get(marker_id)
            if marker_pose is None:
                continue
            rotation_camera_marker, _ = cv.Rodrigues(rvec)
            transform_camera_marker = np.eye(4, dtype=np.float64)
            transform_camera_marker[:3, :3] = rotation_camera_marker
            transform_camera_marker[:3, 3] = position_camera_marker

            transform_box_marker = np.eye(4, dtype=np.float64)
            transform_box_marker[:3, :3] = marker_pose.rotation
            transform_box_marker[:3, 3] = marker_pose.position_m
            inferred_poses.append(
                transform_camera_marker @ np.linalg.inv(transform_box_marker)
            )
            used_ids.append(marker_id)

        if not inferred_poses:
            return None

        average_rotation = np.mean(
            [pose[:3, :3] for pose in inferred_poses], axis=0
        )
        u_matrix, _, vt_matrix = np.linalg.svd(average_rotation)
        if np.linalg.det(u_matrix @ vt_matrix) < 0.0:
            u_matrix[:, -1] *= -1.0
        raw_pose = np.eye(4, dtype=np.float64)
        raw_pose[:3, :3] = u_matrix @ vt_matrix
        raw_pose[:3, 3] = np.mean(
            [pose[:3, 3] for pose in inferred_poses], axis=0
        )
        pose = self._smooth(raw_pose)
        return raw_pose, pose, tuple(used_ids)

    def _smooth(self, measured_pose: np.ndarray) -> np.ndarray:
        if self._previous_pose is None:
            smoothed = measured_pose.copy()
        else:
            smoothed = np.eye(4, dtype=np.float64)
            translation_alpha = self.config.translation_smoothing_alpha
            smoothed[:3, 3] = (
                (1.0 - translation_alpha) * self._previous_pose[:3, 3]
                + translation_alpha * measured_pose[:3, 3]
            )
            relative_rotation = (
                self._previous_pose[:3, :3].T @ measured_pose[:3, :3]
            )
            relative_rvec, _ = cv.Rodrigues(relative_rotation)
            step_rotation, _ = cv.Rodrigues(
                self.config.rotation_smoothing_alpha * relative_rvec
            )
            smoothed[:3, :3] = self._previous_pose[:3, :3] @ step_rotation
        self._previous_pose = smoothed.copy()
        return smoothed

    @staticmethod
    def draw_detection(
        image: np.ndarray,
        detection: ArucoBoxDetection,
        camera_matrix: Any,
        dist_coeffs: Any | None = None,
        axis_length_m: float = 0.03,
    ) -> np.ndarray:
        """Draw detected markers and their axes on ``image`` in place."""

        intrinsics, distortion = ArucoBoxPoseEstimator._camera_parameters(
            camera_matrix, dist_coeffs
        )
        cv.aruco.drawDetectedMarkers(
            image, detection.corners, detection.marker_ids
        )
        for rvec, tvec in zip(detection.rvecs, detection.tvecs):
            cv.drawFrameAxes(
                image, intrinsics, distortion, rvec, tvec, axis_length_m
            )
        return image


def tissue_box_aruco_config() -> ArucoBoxConfig:
    """Configuration matching ``aruco_vis_object_pose_live.py``."""

    width, length, height = 0.307, 0.153, 0.215
    return ArucoBoxConfig(
        marker_length_m=0.09,
        box_size_m=(width, length, height),
        translation_smoothing_alpha=0.25,
        rotation_smoothing_alpha=0.15,
        marker_poses={
            0: {
                "position_m": [0.0, -length / 2.0, 0.0],
                "rotation": [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
            },
            1: {
                "position_m": [-width / 2.0, 0.0, 0.0],
                "rotation": [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
            },
            2: {
                "position_m": [0.0, 0.0, height / 2.0],
                "rotation": np.eye(3),
            },
            3: {
                "position_m": [width / 2.0, 0.0, 0.0],
                "rotation": [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
            },
        },
    )


@dataclass
class PsfCameraCalibration:
    """Small reader for the PSF camera calibration JSON contract."""

    calibration_type: str
    transform_mount_camera_m: np.ndarray
    camera_serial: str | None = None

    @classmethod
    def load(cls, path: str | Path) -> "PsfCameraCalibration":
        calibration_path = Path(path).expanduser()
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        type_text = str(
            payload.get("calibration_type") or payload.get("key") or ""
        ).lower()
        type_by_lower = {"tbc": "Tbc", "thc": "Thc", "tec": "Tec"}
        calibration_type = type_by_lower.get(type_text)
        if calibration_type is None:
            for candidate in ("Tbc", "Thc", "Tec"):
                if candidate in payload:
                    calibration_type = candidate
                    break
        if calibration_type is None:
            raise ValueError(
                f"{calibration_path} does not contain a PSF Tbc/Thc/Tec calibration"
            )
        matrix_value = payload.get(calibration_type, payload.get("calibration_matrix"))
        matrix = np.asarray(matrix_value, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise ValueError(f"Invalid {calibration_type} matrix in {calibration_path}")
        if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
            raise ValueError(f"Invalid homogeneous row in {calibration_path}")
        return cls(
            calibration_type=calibration_type,
            transform_mount_camera_m=matrix,
            camera_serial=(
                str(payload["serial"]).strip() if payload.get("serial") else None
            ),
        )

    def base_to_camera(
        self, transform_base_mount_m: np.ndarray | None = None
    ) -> np.ndarray:
        """Return ``T_base_camera``, which maps camera points into base."""

        if self.calibration_type == "Tbc":
            return self.transform_mount_camera_m.copy()
        if transform_base_mount_m is None:
            mount = "head" if self.calibration_type == "Thc" else "end effector"
            raise ValueError(
                f"{self.calibration_type} requires the current base-to-{mount} pose"
            )
        transform = np.asarray(transform_base_mount_m, dtype=np.float64)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError("transform_base_mount_m must be a finite 4x4 matrix")
        return transform @ self.transform_mount_camera_m


def robot_pose_command_to_transform(pose_mm_deg: Sequence[float]) -> np.ndarray:
    """Convert Neuromeka ``[x,y,z,rx,ry,rz]`` (mm/deg) to an SE(3) matrix."""

    pose = np.asarray(pose_mm_deg, dtype=np.float64)
    if pose.shape != (6,) or not np.all(np.isfinite(pose)):
        raise ValueError("Robot mount pose must contain six finite values")
    rx, ry, rz = np.deg2rad(pose[3:])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rotation_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rotation_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rotation_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_z @ rotation_y @ rotation_x
    transform[:3, 3] = pose[:3] / 1000.0
    return transform


class _GeometryTracker:
    def __init__(self, visualizer: Any, create_geometries: Any):
        self.visualizer = visualizer
        self.create_geometries = create_geometries
        self.geometries: list[Any] = []
        self.pose: np.ndarray | None = None

    def update(self, pose: np.ndarray) -> None:
        if self.pose is None:
            self.geometries = self.create_geometries()
            for geometry in self.geometries:
                geometry.transform(pose)
                self.visualizer.add_geometry(geometry)
        else:
            delta = pose @ np.linalg.inv(self.pose)
            for geometry in self.geometries:
                geometry.transform(delta)
                self.visualizer.update_geometry(geometry)
        self.pose = pose.copy()


def _create_box_geometries(o3d: Any, config: ArucoBoxConfig) -> list[Any]:
    width, length, height = config.box_size_m
    x, y, z = width / 2.0, length / 2.0, height / 2.0
    corners = np.asarray(
        [
            [-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z],
            [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z],
        ]
    )
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([[0.2, 0.6, 1.0]] * len(lines))
    geometries: list[Any] = [line_set]

    front = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    front.translate([0.0, -y, 0.0])
    front.paint_uniform_color([1.0, 0.1, 0.1])
    geometries.append(front)
    for marker_pose in config.marker_poses.values():
        transform_box_marker = np.eye(4)
        transform_box_marker[:3, :3] = marker_pose.rotation
        transform_box_marker[:3, 3] = marker_pose.position_m
        marker_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
        marker_frame.transform(transform_box_marker)
        geometries.append(marker_frame)
    return geometries


def _load_config(path: str | None) -> ArucoBoxConfig:
    if path is None:
        return tissue_box_aruco_config()
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    return ArucoBoxConfig.from_dict(payload)


def _run_live_demo(args: argparse.Namespace) -> None:
    # These heavy, demo-only dependencies stay out of controller imports.
    import open3d as o3d
    import pyrealsense2 as rs

    box_config = _load_config(args.config)
    estimator = ArucoBoxPoseEstimator(box_config)

    calibration: PsfCameraCalibration | None = None
    if args.frame == "robot":
        if not args.calibration:
            raise ValueError("The robot demo requires --calibration with a PSF JSON file")
        calibration = PsfCameraCalibration.load(args.calibration)
        if (
            args.serial
            and calibration.camera_serial
            and args.serial != calibration.camera_serial
        ):
            raise ValueError(
                f"Requested camera {args.serial} does not match calibration camera "
                f"{calibration.camera_serial}"
            )

    robot_client: Any | None = None
    static_base_mount: np.ndarray | None = None
    if calibration and calibration.calibration_type != "Tbc":
        if args.robot_ip and args.base_head_pose:
            raise ValueError("Use either --robot-ip or --base-head-pose, not both")
        if args.robot_ip:
            from neuromeka import IndyDCP3

            robot_client = IndyDCP3(robot_ip=args.robot_ip)
        elif args.base_head_pose:
            static_base_mount = robot_pose_command_to_transform(args.base_head_pose)
        else:
            raise ValueError(
                f"A {calibration.calibration_type} robot demo requires --robot-ip "
                "or --base-head-pose"
            )

    serial = args.serial or (calibration.camera_serial if calibration else None)
    pipeline = rs.pipeline()
    stream_config = rs.config()
    if serial:
        stream_config.enable_device(serial)
    stream_config.enable_stream(
        rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps
    )
    profile = pipeline.start(stream_config)
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics_raw = color_profile.get_intrinsics()
    camera_matrix = np.array(
        [
            [intrinsics_raw.fx, 0.0, intrinsics_raw.ppx],
            [0.0, intrinsics_raw.fy, intrinsics_raw.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.asarray(intrinsics_raw.coeffs[:5], dtype=np.float64)

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(
        "ArUco Box Pose - Camera Frame"
        if args.frame == "camera"
        else "ArUco Box Pose - Robot Base Frame",
        width=1280,
        height=720,
    )
    visualizer.add_geometry(
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    )
    cv_to_open3d = np.eye(4)
    cv_to_open3d[:3, :3] = np.diag([1.0, -1.0, -1.0])
    if args.frame == "camera":
        camera_frustum = o3d.geometry.LineSet.create_camera_visualization(
            args.width, args.height, camera_matrix, cv_to_open3d, scale=0.1
        )
        visualizer.add_geometry(camera_frustum)

    box_tracker = _GeometryTracker(
        visualizer, lambda: _create_box_geometries(o3d, box_config)
    )
    camera_tracker = _GeometryTracker(
        visualizer,
        lambda: [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.12)],
    )

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())
            detection = estimator.detect(
                frame, camera_matrix, dist_coeffs, color_order="bgr"
            )
            if detection is not None:
                estimator.draw_detection(
                    frame, detection, camera_matrix, dist_coeffs
                )
                marker_descriptions = []
                for marker_id_raw, marker_corners in zip(
                    detection.marker_ids.reshape(-1), detection.corners
                ):
                    marker_id = int(marker_id_raw)
                    marker_pose = box_config.marker_poses.get(marker_id)
                    if marker_pose is None:
                        continue
                    face_label = marker_face_label(marker_pose)
                    marker_descriptions.append(f"{marker_id}:{face_label}")
                    center = np.rint(
                        np.asarray(marker_corners).reshape(-1, 2).mean(axis=0)
                    ).astype(int)
                    cv.putText(
                        frame,
                        f"ID {marker_id} {face_label}",
                        (int(center[0]) + 10, int(center[1]) - 10),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 255),
                        2,
                        cv.LINE_AA,
                    )
                transform_camera_box = detection.pose_camera_box
                if args.frame == "camera":
                    display_pose = cv_to_open3d @ transform_camera_box
                    label = "camera"
                else:
                    if robot_client is not None:
                        state = robot_client.get_robot_data()
                        static_base_mount = robot_pose_command_to_transform(
                            state["p"][:6]
                        )
                    transform_base_camera = calibration.base_to_camera(
                        static_base_mount
                    )
                    display_pose = transform_base_camera @ transform_camera_box
                    camera_tracker.update(transform_base_camera)
                    label = "robot base"
                box_tracker.update(display_pose)
                xyz = display_pose[:3, 3]
                cv.putText(
                    frame,
                    f"box in {label}: [{xyz[0]:+.3f}, {xyz[1]:+.3f}, {xyz[2]:+.3f}] m",
                    (20, 40),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                    cv.LINE_AA,
                )
                cv.putText(
                    frame,
                    "markers: " + ", ".join(marker_descriptions),
                    (20, 75),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                    cv.LINE_AA,
                )
            preview = cv.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
            if not visualizer.poll_events():
                break
            visualizer.update_renderer()
            cv.imshow("RealSense ArUco detections (ESC to quit)", preview)
            if cv.waitKey(1) & 0xFF == 27:
                break
    finally:
        pipeline.stop()
        cv.destroyAllWindows()
        visualizer.destroy_window()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "frame",
        nargs="?",
        default="camera",
        choices=("camera", "robot"),
        help="Visualize the pose in the camera frame or robot base frame",
    )
    parser.add_argument(
        "--config",
        help="Optional JSON ArUcoBoxConfig; the tissue-box setup is the default",
    )
    parser.add_argument("--serial", help="RealSense serial number")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--calibration",
        help="PSF Tbc/Thc/Tec calibration JSON (required for the robot demo)",
    )
    parser.add_argument(
        "--robot-ip",
        help="Read the live base-to-head pose from p[:6] for a Thc calibration",
    )
    parser.add_argument(
        "--base-head-pose",
        "--base-mount-pose",
        dest="base_head_pose",
        type=float,
        nargs=6,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        help="Fixed base-to-mount pose in robot command units (mm, XYZ degrees)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    _run_live_demo(_parse_args())
