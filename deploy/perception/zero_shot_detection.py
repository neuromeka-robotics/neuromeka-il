"""Process 2: DINOv2 prompts -> streaming SAM2 mask, with a RealSense demo.

Run from deploy: python perception/zero_shot_detection.py --help
All public image inputs are RGB uint8. Masks are H x W x 1 float32.
"""
import argparse
from contextlib import nullcontext
import os
from pathlib import Path
import random
import sys
import time

import cv2
import numpy as np
import torch
import zmq

DEFAULT_SAM2_REPO = Path(__file__).resolve().parents[1] / "third_party/sam2"
LOCAL_WEIGHTS = Path(__file__).resolve().parents[3] / "ETC/weights"
DEFAULT_CHECKPOINT = LOCAL_WEIGHTS / "sam2.1_hiera_small.pt"


def add_sam2_arguments(parser):
    parser.add_argument("--sam2-repo", default=os.environ.get("SAM2_REPO", str(DEFAULT_SAM2_REPO)))
    parser.add_argument("--sam2-checkpoint", default=os.environ.get("SAM2_CHECKPOINT", str(DEFAULT_CHECKPOINT)))
    parser.add_argument("--sam2-config", default=os.environ.get(
        "SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_s.yaml"))


def build_predictor(checkpoint=None, model_cfg=None, sam2_repo=None, *, video=False):
    checkpoint = checkpoint or os.environ.get("SAM2_CHECKPOINT", str(DEFAULT_CHECKPOINT))
    if not checkpoint or not Path(checkpoint).expanduser().is_file():
        raise FileNotFoundError("Set SAM2_CHECKPOINT or --sam2-checkpoint to your existing SAM2 weights")
    repo = Path(sam2_repo or os.environ.get("SAM2_REPO", DEFAULT_SAM2_REPO)).expanduser()
    if repo.is_dir():
        sys.path.insert(0, str(repo))
    try:
        from sam2.build_sam import build_sam2_camera_predictor, build_sam2_video_predictor
    except ImportError as exc:
        raise ImportError("Streaming SAM2 is required; see deploy/perception/README.md") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("The streaming SAM2 backend requires CUDA; no CUDA device is available")
    builder = build_sam2_video_predictor if video else build_sam2_camera_predictor
    return builder(model_cfg or os.environ.get("SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_s.yaml"),
                   str(Path(checkpoint).expanduser()), device="cuda",
                   # Avoid requiring the optional compiled connected-components extension.
                   apply_postprocessing=False)


def inference_context():
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast("cuda", dtype=dtype)
    return nullcontext()


class DetectionClient:
    """REQ/REP client; recreate REQ after a timeout so subsequent calls work."""
    def __init__(self, host="127.0.0.1", port=5555, timeout=5000):
        self.address = f"tcp://{host}:{port}"
        self.timeout = timeout
        self.context = zmq.Context()
        self.socket = None
        self._connect()

    def _connect(self):
        if self.socket is not None:
            self.socket.close(linger=0)
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout)
        self.socket.connect(self.address)

    def _send_request(self, request):
        try:
            self.socket.send_pyobj(request)
            response = self.socket.recv_pyobj()
        except zmq.ZMQError as exc:
            self._connect()
            raise ConnectionError(f"Detection server unavailable at {self.address} ({self.timeout} ms)") from exc
        if not isinstance(response, dict) or response.get("status") != "success":
            raise RuntimeError(f"Detection server error: {response}")
        return response

    def register(self, target_key, target_img_np):
        return self._send_request(dict(endpoint="register", target_key=target_key, target_img_np=target_img_np))

    def detect(self, target_key, scene_img_np):
        return self._send_request(dict(endpoint="detect", target_key=target_key, scene_img_np=scene_img_np))

    def close(self):
        if self.socket is not None:
            self.socket.close(linger=0)
            self.socket = None
        if not self.context.closed:
            self.context.term()


def largest_component(mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return np.zeros_like(mask, dtype=np.uint8)
    return (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)


class DinoV2MaskLabeler:
    def __init__(self, checkpoint=None, model_cfg=None, sam2_repo=None, *,
                 host="127.0.0.1", port=5555, timeout=5000, min_mask_area=20,
                 max_mask_fraction=0.9, predictor=None, client=None):
        self.mask_predictor = predictor if predictor is not None else build_predictor(checkpoint, model_cfg, sam2_repo)
        self.detection_client = client
        self.client_params = dict(host=host, port=port, timeout=timeout)
        self.min_mask_area = min_mask_area
        self.max_mask_fraction = max_mask_fraction
        self.is_init = False
        self._has_frame = False
        self._target = None
        self.registered_target = set()
        self._shape = None

    def reset(self):
        if self._has_frame:
            self.mask_predictor.reset_state()
        self.is_init = False
        self._has_frame = False
        self._shape = None

    def close(self):
        self.reset()
        if self.detection_client is not None:
            self.detection_client.close()

    def _result(self, logits, shape, apply_contour_filter=True):
        mask = (logits[0] > 0).detach().cpu().numpy().reshape(shape)
        if apply_contour_filter:
            mask = largest_component(mask)
        area = int(mask.sum())
        valid = self.min_mask_area <= area <= self.max_mask_fraction * mask.size
        if not valid:
            self.reset()
            mask = np.zeros(shape, dtype=np.uint8)
        else:
            self.is_init = True
        return {"mask": mask[..., None].astype(np.float32), "valid": valid, "area": area}

    @torch.inference_mode()
    def initialize_points(self, rgb_img, points, labels, apply_contour_filter=True):
        self.reset()
        self._shape = rgb_img.shape[:2]
        with inference_context():
            self.mask_predictor.load_first_frame(rgb_img)
            self._has_frame = True
            _, _, logits = self.mask_predictor.add_new_prompt(
                frame_idx=0, obj_id=1, points=np.asarray(points, dtype=np.float32),
                labels=np.asarray(labels, dtype=np.int32))
        return self._result(logits, rgb_img.shape[:2], apply_contour_filter)

    @torch.inference_mode()
    def track(self, rgb_img, apply_contour_filter=True):
        if not self.is_init or self._shape != rgb_img.shape[:2]:
            raise RuntimeError("Initialize the tracker on an image with the same size first")
        with inference_context():
            _, logits = self.mask_predictor.track(rgb_img)
        return self._result(logits, rgb_img.shape[:2], apply_contour_filter)

    def __call__(self, rgb_img, target, apply_contour_filter=True, obj_center_limit=None):
        if rgb_img.ndim != 3 or rgb_img.shape[2] != 3 or rgb_img.dtype != np.uint8:
            raise ValueError("Expected uint8 H x W x 3 RGB image")
        target_id = (target["name"], str(Path(target["image_path"]).expanduser().resolve()))
        if target_id != self._target:
            self.registered_target.clear()
        if target_id != self._target or self._shape not in (None, rgb_img.shape[:2]):
            self.reset()
            self._target = target_id
        if self.is_init:
            return self.track(rgb_img, apply_contour_filter)
        if self.detection_client is None:
            self.detection_client = DetectionClient(**self.client_params)
        try:
            if target_id not in self.registered_target:
                bgr = cv2.imread(target_id[1])
                if bgr is None:
                    raise FileNotFoundError(target_id[1])
                self.detection_client.register(target["name"], cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                self.registered_target.add(target_id)
            objects = self.detection_client.detect(target["name"], rgb_img)["object_coords"]
        except (ConnectionError, RuntimeError):
            # The next attempt also works after the server has restarted.
            self.registered_target.clear()
            self.reset()
            raise
        candidates = []
        for coords in objects:
            points = np.asarray(coords).reshape(-1, 2)
            if not len(points):
                continue
            center_xy = points.mean(axis=0)[::-1]
            if obj_center_limit is not None and not (
                np.all(center_xy >= obj_center_limit["min"]) and np.all(center_xy <= obj_center_limit["max"])):
                continue
            candidates.append(points)
        if not candidates:
            # Stay uninitialized; retry detection on the next image.
            return {"mask": np.zeros((*rgb_img.shape[:2], 1), dtype=np.float32), "valid": False, "area": 0}
        # points = max(candidates, key=len)
        points = random.choice(candidates)
        # Repeatable interior prompts; all prompts remain inside the DINO component.
        center = points.mean(axis=0)
        order = np.argsort(np.linalg.norm(points - center, axis=1))
        samples = order[np.linspace(0, max(0, len(order) // 2 - 1), min(3, len(order)), dtype=int)]
        return self.initialize_points(rgb_img, points[samples, ::-1], np.ones(len(samples)), apply_contour_filter)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_sam2_arguments(parser)
    parser.add_argument("--reference", required=True, help="Required path to the reference image of the target object")
    parser.add_argument("--serial", default=os.environ.get("REALSENSE_SERIAL"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--timeout", type=int, default=5000)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", type=int, default=0, help="Stop after N frames; 0 runs until quit")
    parser.add_argument("--save-dir", type=Path, help="Save last RGB, mask and overlay when done")
    args = parser.parse_args()
    # Camera-only program: no robot imports or connections.
    import pyrealsense2 as rs
    tracker = DinoV2MaskLabeler(args.sam2_checkpoint, args.sam2_config, args.sam2_repo,
                              host=args.host, port=args.port, timeout=args.timeout)
    target = {"name": Path(args.reference).stem, "image_path": args.reference}
    pipeline, config = rs.pipeline(), rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    started = False
    try:
        pipeline.start(config)
        started = True
        count = 0
        print("q: quit, r: reset detection", flush=True)
        while not args.frames or count < args.frames:
            frame = pipeline.wait_for_frames(5000).get_color_frame()
            if not frame:
                continue
            rgb = np.asanyarray(frame.get_data()).copy()
            begin = time.monotonic()
            result = tracker(rgb, target)
            overlay = rgb.copy()
            mask = result["mask"][..., 0].astype(bool)
            overlay[mask] = (0.5 * overlay[mask] + np.array([0, 127, 0])).astype(np.uint8)
            count += 1
            if count % 30 == 0 or count == 1:
                print(f"frame={count} valid={result['valid']} area={result['area']} inference={time.monotonic()-begin:.3f}s", flush=True)
            if not args.headless:
                cv2.imshow("Zero-shot SAM2 (q quit, r reset)", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(1) & 0xff
                if key == ord("q"):
                    break
                if key == ord("r"):
                    tracker.reset()
        if args.save_dir and count:
            args.save_dir.mkdir(parents=True, exist_ok=True)
            for name, data in [("rgb", rgb[..., ::-1]), ("overlay", overlay[..., ::-1]), ("mask", mask.astype(np.uint8)*255)]:
                cv2.imwrite(str(args.save_dir / f"{name}.png"), data)
    finally:
        if started:
            pipeline.stop()
        tracker.close()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
