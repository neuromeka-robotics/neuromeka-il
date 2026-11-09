"""Prepare masked RGB and s_t^-1 s_0 labels from Nami SpaceMouse recordings."""
import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation


def goal_corrections(poses, target_index=0):
    """Input mm/degrees; output translation in current TCP frame (m), rotation matrices."""
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 6 or not len(poses) or not np.all(np.isfinite(poses)):
        raise ValueError("poses must be a nonempty, finite T x 6 array")
    rotations = Rotation.from_euler("xyz", poses[:, 3:], degrees=True).as_matrix()
    inverse = rotations.transpose(0, 2, 1)
    translation = np.einsum("tij,tj->ti", inverse, (poses[target_index, :3] - poses[:, :3]) / 1000.)
    relative_rotation = inverse @ rotations[target_index]
    return translation.astype(np.float32), relative_rotation.astype(np.float32)


def select_task_poses(values, arm_index):
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError("Expected T x D task poses")
    if values.shape[1] == 6:
        return values
    if values.shape[1] == 18 and arm_index in (0, 1, 2):
        return values[:, 6 * arm_index:6 * (arm_index + 1)]
    raise ValueError(f"Expected six task values or Nami's 18, got {values.shape}")


def process_episode(raw_path, mask_path, output_path, *, camera="wrist", mask_name="target",
                    robot_id=0, arm_index=1, target_frame="first", min_mask_area=20):
    """Write a new processed episode atomically; leave raw recordings and masks intact."""
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to replace {output_path}; choose an empty output directory")
    temp_path = output_path.with_suffix(".h5.tmp")
    try:
        with h5py.File(raw_path, "r") as raw, h5py.File(mask_path, "r") as labeled:
            images = raw[f"images.rgb.{camera}"]
            poses = select_task_poses(raw[f"p_{robot_id}"][:], arm_index)
            masks = labeled[mask_name]
            n, height, width, _ = images.shape
            if len(poses) != n or masks.shape not in ((n, height, width), (n, height, width, 1)):
                raise ValueError(f"Frame/mask alignment mismatch in {raw_path}: images={images.shape}, masks={masks.shape}, poses={poses.shape}")
            if "source_episode" in labeled.attrs and labeled.attrs["source_episode"] != Path(raw_path).name:
                raise ValueError("Mask source episode does not match recording")
            if "camera" in labeled.attrs and labeled.attrs["camera"] != camera:
                raise ValueError("Mask camera does not match recording")
            if "arm_indices" in raw.attrs and arm_index not in raw.attrs["arm_indices"]:
                raise ValueError(f"Arm {arm_index} was not recorded in {raw_path}")
            target_index = 0 if target_frame == "first" else n - 1
            dp, dr = goal_corrections(poses, target_index)
            valid = np.array([min_mask_area <= np.count_nonzero(masks[i]) < height * width * 0.9
                              for i in range(n)])
            if not valid[target_index]:
                raise ValueError(f"Goal frame {target_index} has no valid mask in {mask_path}; relabel it")
            indices = np.flatnonzero(valid)
            if len(indices) < 2:
                raise ValueError("Need at least two valid masked frames")
            rotations = Rotation.from_euler("xyz", poses[:, 3:], degrees=True).as_matrix().astype(np.float32)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(temp_path, "w") as out:
                rgb_out = out.create_dataset(f"observation.images.rgb.{camera}",
                    shape=(len(indices), height, width, 3), dtype=np.uint8,
                    chunks=(1, height, width, 3), compression="lzf")
                for dest, source in enumerate(indices):
                    mask = np.asarray(masks[source]).reshape(height, width, 1) > 0
                    rgb_out[dest] = images[source] * mask
                out["observation.end_position"] = (poses[indices, :3] / 1000.).astype(np.float32)
                out["observation.end_orientation"] = rotations[indices, None]
                # Keep ordinary task labels for compatibility with ImageLoadDataset.
                # The visual_servo mode trains only on the explicit goal corrections below.
                out["action.end_pos"] = (poses[indices, :3] / 1000.).astype(np.float32)
                out["action.end_ori"] = rotations[indices, None]
                out["action.visual_servoing.end_pos"] = dp[indices]
                out["action.visual_servoing.end_ori"] = dr[indices, None]
                out["is_success"] = ((np.linalg.norm(dp[indices], axis=1) < .005)
                    & (Rotation.from_matrix(dr[indices]).magnitude() < np.deg2rad(.5)))[:, None].astype(np.float32)
                out.attrs.update(source_episode=Path(raw_path).name, camera=camera,
                                 arm_index=arm_index, target_frame=target_index,
                                 label_equation="inverse(current_pose) @ goal_pose")
                out.attrs["source_frame_indices"] = indices
        temp_path.rename(output_path)
        return len(indices)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="syringe_visual_servo")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--mask-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--camera", default="wrist")
    parser.add_argument("--mask-name", default="target")
    parser.add_argument("--robot-id", type=int, default=0)
    parser.add_argument("--arm-index", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--target-frame", choices=("first", "last"), default="first")
    args = parser.parse_args()
    data_dir = args.data_dir or root / "data" / args.task
    mask_dir = args.mask_dir or root / "data_mask" / args.task
    output_dir = args.output_dir or root / "processed_data" / args.task
    episodes = sorted(data_dir.glob("*.h5"), key=lambda path: int(path.stem))
    if not episodes:
        raise FileNotFoundError(f"No numbered H5 episodes in {data_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")
    # Check every mask path before starting a partially labeled dataset.
    mask_paths = [mask_dir / f"{path.stem}_{args.camera}.h5" for path in episodes]
    for path in mask_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Label this episode with generate_mask.py first: {path}")
    for index, (episode, mask_path) in enumerate(zip(episodes, mask_paths)):
        count = process_episode(episode, mask_path, output_dir / f"{index}.h5",
            camera=args.camera, mask_name=args.mask_name, robot_id=args.robot_id,
            arm_index=args.arm_index, target_frame=args.target_frame)
        print(f"{episode.name}: {count} valid frames -> {index}.h5")


if __name__ == "__main__":
    main()
