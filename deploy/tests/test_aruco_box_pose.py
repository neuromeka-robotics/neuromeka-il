import json
import tempfile
import unittest
from pathlib import Path

import cv2 as cv
import numpy as np

from perception.aruco_box_pose import (
    ArucoBoxConfig,
    ArucoBoxPoseEstimator,
    PsfCameraCalibration,
    robot_pose_command_to_transform,
    tissue_box_aruco_config,
)


class ArucoBoxPoseEstimatorTest(unittest.TestCase):
    def test_detection_path_estimates_marker_pose_with_solve_pnp(self):
        estimator = ArucoBoxPoseEstimator(
            {
                "marker_length_m": 0.09,
                "marker_poses": {
                    7: {"position_m": [0, 0, 0], "rotation": np.eye(3)}
                },
            }
        )
        camera_matrix = np.asarray(
            [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]
        )
        expected_rvec = np.asarray([0.2, -0.1, 0.05], dtype=np.float64)
        expected_tvec = np.asarray([0.03, -0.02, 0.7], dtype=np.float64)
        half_length = estimator.config.marker_length_m / 2.0
        object_points = np.asarray(
            [
                [-half_length, half_length, 0.0],
                [half_length, half_length, 0.0],
                [half_length, -half_length, 0.0],
                [-half_length, -half_length, 0.0],
            ]
        )
        image_points, _ = cv.projectPoints(
            object_points,
            expected_rvec,
            expected_tvec,
            camera_matrix,
            np.zeros(5),
        )

        class FakeDetector:
            def detectMarkers(self, _image):
                return [image_points.reshape(1, 4, 2)], np.asarray([[7]]), []

        estimator._detector = FakeDetector()
        detection = estimator.detect(
            np.zeros((480, 640, 3), dtype=np.uint8),
            camera_matrix,
            np.zeros(5),
        )

        self.assertIsNotNone(detection)
        expected_pose = np.eye(4)
        expected_pose[:3, :3], _ = cv.Rodrigues(expected_rvec)
        expected_pose[:3, 3] = expected_tvec
        np.testing.assert_allclose(
            detection.pose_camera_box, expected_pose, atol=1e-8
        )

    def test_multimarker_fusion_recovers_camera_box_pose(self):
        config = tissue_box_aruco_config()
        config.translation_smoothing_alpha = 1.0
        config.rotation_smoothing_alpha = 1.0
        estimator = ArucoBoxPoseEstimator(config)

        rotation_camera_box, _ = cv.Rodrigues(
            np.asarray([0.2, -0.1, 0.3], dtype=np.float64)
        )
        expected = np.eye(4)
        expected[:3, :3] = rotation_camera_box
        expected[:3, 3] = [0.12, -0.04, 0.75]

        marker_ids = []
        rvecs = []
        tvecs = []
        for marker_id, marker_pose in config.marker_poses.items():
            transform_box_marker = np.eye(4)
            transform_box_marker[:3, :3] = marker_pose.rotation
            transform_box_marker[:3, 3] = marker_pose.position_m
            transform_camera_marker = expected @ transform_box_marker
            rvec, _ = cv.Rodrigues(transform_camera_marker[:3, :3])
            marker_ids.append([marker_id])
            rvecs.append(rvec.reshape(1, 3))
            tvecs.append(transform_camera_marker[:3, 3].reshape(1, 3))

        result = estimator.estimate_from_detections(
            np.asarray(marker_ids), np.asarray(rvecs), np.asarray(tvecs)
        )

        self.assertIsNotNone(result)
        raw_pose, pose, used_ids = result
        np.testing.assert_allclose(raw_pose, expected, atol=1e-9)
        np.testing.assert_allclose(pose, expected, atol=1e-9)
        self.assertEqual(used_ids, (0, 1, 2, 3))

    def test_unconfigured_markers_return_no_box_pose(self):
        estimator = ArucoBoxPoseEstimator(tissue_box_aruco_config())
        result = estimator.estimate_from_detections(
            [[42]], [[[0.0, 0.0, 0.0]]], [[[0.0, 0.0, 1.0]]]
        )
        self.assertIsNone(result)

    def test_mapping_config_accepts_original_pos_and_rot_names(self):
        estimator = ArucoBoxPoseEstimator(
            {
                "marker_length": 0.1,
                "marker_poses": {
                    "7": {"pos": [0, 0, 0], "rot": np.eye(3).tolist()}
                },
            }
        )
        self.assertIsInstance(estimator.config, ArucoBoxConfig)
        self.assertIn(7, estimator.config.marker_poses)


class PsfCameraCalibrationTest(unittest.TestCase):
    def test_thc_composes_base_head_with_head_camera(self):
        transform_head_camera = np.eye(4)
        transform_head_camera[:3, 3] = [0.1, -0.02, 0.03]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Thc.json"
            path.write_text(
                json.dumps(
                    {
                        "calibration_type": "Thc",
                        "Thc": transform_head_camera.tolist(),
                        "serial": "camera-1",
                    }
                ),
                encoding="utf-8",
            )
            calibration = PsfCameraCalibration.load(path)

        transform_base_head = robot_pose_command_to_transform(
            [1000.0, 2000.0, 3000.0, 0.0, 0.0, 90.0]
        )
        expected = transform_base_head @ transform_head_camera
        np.testing.assert_allclose(
            calibration.base_to_camera(transform_base_head), expected, atol=1e-12
        )
        self.assertEqual(calibration.camera_serial, "camera-1")

    def test_tbc_is_already_base_camera(self):
        transform_base_camera = np.eye(4)
        transform_base_camera[:3, 3] = [0.4, -0.3, 1.2]
        calibration = PsfCameraCalibration(
            calibration_type="Tbc",
            transform_mount_camera_m=transform_base_camera,
        )
        np.testing.assert_array_equal(
            calibration.base_to_camera(), transform_base_camera
        )


if __name__ == "__main__":
    unittest.main()
