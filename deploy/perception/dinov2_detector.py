"""Feature-matching detector migrated from ETC/nrmk_perception.

RGB inputs; connected components contain (row, column) coordinates. DINOv2
ViT-B/14 and the bundled foreground projection reproduce the legacy detector.
"""
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_erosion, label
from torchvision import transforms


class DinoV2ObjectDetection:
    def __init__(self, *, standard_array_path=None, threshold=0.5,
                 background_threshold=-1., final_erosion_count=3,
                 segment_min_size=20, target_multiplier=25, img_multiplier=50,
                 device=None, repo=None, weights=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if weights and not Path(weights).expanduser().is_file():
            raise FileNotFoundError(weights)
        source = "local" if repo else "github"
        self.model = torch.hub.load(
            repo or "facebookresearch/dinov2", "dinov2_vitb14", source=source,
            pretrained=weights is None).to(self.device).eval()
        if weights:
            state = torch.load(Path(weights).expanduser(), map_location="cpu", weights_only=True)
            self.model.load_state_dict(state)
        self.patch_size = self.model.patch_size
        path = standard_array_path or Path(__file__).parent / "assets/standard_array.npy"
        self.standard_array = torch.as_tensor(
            np.load(path, allow_pickle=False), dtype=torch.float32, device=self.device).flatten()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.threshold = threshold
        self.background_threshold = background_threshold
        self.erosion = final_erosion_count
        self.min_size = segment_min_size
        self.target_multiplier = target_multiplier
        self.img_multiplier = img_multiplier
        self.targets_features = {}
        if not 0 < threshold <= 1 or self.erosion < 0 or self.min_size < 1:
            raise ValueError("Use 0 < threshold <= 1, erosion >= 0 and min-size >= 1")
        if min(target_multiplier, img_multiplier) < 1:
            raise ValueError("Resize multipliers must be positive")

    @torch.inference_mode()
    def _features(self, rgb, multiplier, upscale=False):
        if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Expected H x W x 3 RGB image")
        h, w = rgb.shape[:2]
        if min(h, w) < self.patch_size:
            raise ValueError("Image must be at least one DINOv2 patch (14 pixels)")
        scale = multiplier * self.patch_size / min(h, w)
        if not upscale:
            scale = min(1., scale)
        rh, rw = int(h * scale), int(w * scale)
        image = Image.fromarray(rgb.astype(np.uint8)).resize((rw, rh), Image.Resampling.BICUBIC)
        ch, cw = rh - rh % self.patch_size, rw - rw % self.patch_size
        image = image.crop((0, 0, cw, ch))
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        tokens = self.model.get_intermediate_layers(tensor, n=1, return_class_token=False)[0][0]
        grid = (ch // self.patch_size, cw // self.patch_size)
        foreground = (tokens @ self.standard_array > self.background_threshold).reshape(grid)
        foreground = binary_erosion(foreground.cpu().numpy()).ravel()
        return tokens, foreground, grid, (rh, rw), (ch, cw)

    def register_target(self, rgb, target_key):
        tokens, foreground, *_ = self._features(rgb, self.target_multiplier, upscale=True)
        if not foreground.any():
            raise ValueError("No foreground in reference; use a clean, tightly framed reference image")
        # Replacement makes registration idempotent after a client timeout/retry.
        self.targets_features[target_key] = F.normalize(tokens[foreground], dim=-1)

    @torch.inference_mode()
    def detect(self, rgb, target_key):
        if target_key not in self.targets_features:
            raise KeyError(f"Target {target_key!r} is not registered; register it again")
        tokens, foreground, grid, resized_hw, cropped_hw = self._features(rgb, self.img_multiplier)
        scores = torch.zeros(len(tokens), device=self.device)
        if foreground.any():
            scores[foreground] = (F.normalize(tokens[foreground], dim=-1)
                                  @ self.targets_features[target_key].T).amax(dim=-1)
        coarse = (scores.reshape(grid).cpu().numpy() >= self.threshold).astype(np.uint8)
        cropped = cv2.resize(coarse, cropped_hw[::-1], interpolation=cv2.INTER_NEAREST)
        if self.erosion:
            cropped = binary_erosion(cropped, iterations=self.erosion).astype(np.uint8)
        # Preserve cropped-out right/bottom strips instead of stretching detections.
        resized = np.zeros(resized_hw, dtype=np.uint8)
        resized[:cropped_hw[0], :cropped_hw[1]] = cropped
        mask = cv2.resize(resized, rgb.shape[1::-1], interpolation=cv2.INTER_NEAREST)
        components, count = label(mask)
        objects = [np.argwhere(components == i) for i in range(1, count + 1)]
        return [points for points in objects if len(points) >= self.min_size]
