"""Process 1: DINOv2 feature matching (the camera and SAM2 live in process 2)."""
import argparse
import logging
import os
from pathlib import Path

import zmq


class RequestHandler:
    def __init__(self, detector):
        self.detector = detector

    def handle(self, data):
        try:
            endpoint = data["endpoint"]
            if endpoint == "ping":
                return {"status": "success"}
            key = data["target_key"]
            if endpoint == "register":
                self.detector.register_target(data["target_img_np"], key)
                return {"status": "success"}
            if endpoint == "detect":
                return {"status": "success", "object_coords":
                        self.detector.detect(data["scene_img_np"], key)}
            raise ValueError(f"Unknown endpoint: {endpoint}")
        except Exception as exc:
            logging.exception("Detection request failed")
            return {"status": "error", "message": str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Trusted clients only: protocol uses pickle")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bg-threshold", type=float, default=-1.)
    parser.add_argument("--erosion-count", type=int, default=3)
    parser.add_argument("--min-size", type=int, default=20)
    parser.add_argument("--target-multiplier", type=int, default=25)
    parser.add_argument("--img-multiplier", type=int, default=50)
    parser.add_argument("--standard-array")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    local_assets = Path(__file__).resolve().parents[3] / "ETC"
    local_repo = local_assets / "facebookresearch_dinov2_main"
    local_weights = local_assets / "weights/dinov2_vitb14_pretrain.pth"
    parser.add_argument("--dino-repo", default=os.environ.get("DINO_REPO", str(local_repo) if local_repo.is_dir() else None),
                        help="Local DINOv2 checkout; otherwise use torch hub")
    parser.add_argument("--dino-weights", default=os.environ.get("DINO_WEIGHTS", str(local_weights) if local_weights.is_file() else None),
                        help="Existing ViT-B/14 checkpoint; otherwise use torch hub cache/download")
    args = parser.parse_args()
    if __package__:
        from .dinov2_detector import DinoV2ObjectDetection
    else:
        from dinov2_detector import DinoV2ObjectDetection
    detector = DinoV2ObjectDetection(
        standard_array_path=args.standard_array, threshold=args.threshold,
        background_threshold=args.bg_threshold, final_erosion_count=args.erosion_count,
        segment_min_size=args.min_size, target_multiplier=args.target_multiplier,
        img_multiplier=args.img_multiplier, device=args.device,
        repo=args.dino_repo, weights=args.dino_weights)
    handler = RequestHandler(detector)
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    try:
        socket.bind(f"tcp://{args.host}:{args.port}")
        print(f"DINOv2 ready on tcp://{args.host}:{args.port}", flush=True)
        while True:
            try:
                response = handler.handle(socket.recv_pyobj())
            except Exception as exc:
                response = {"status": "error", "message": str(exc)}
            socket.send_pyobj(response)
    except KeyboardInterrupt:
        pass
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    main()
