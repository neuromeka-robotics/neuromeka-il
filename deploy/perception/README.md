# Zero-shot masks and learned visual servoing

This ports the deprecated DINOv2 feature matcher, SAM2 tracker, manual mask
labeling, and ACT visual servoing into the current repository. The controller
follows the [paper's visual-servoing method](https://arxiv.org/html/2604.22235v1):
collect perturbations from a goal pose, train on `inverse(current_pose) @ goal_pose`,
and predict another correction after reaching the previous target. The learned
policy consumes masked RGB, with one action per inference. Nami commands use xyz
and yaw with fixed roll/pitch.

## Environment and model files

Run from the repository root:

```bash
conda activate env_il
bash deploy/perception/setup_sam2.sh
```

The streaming backend is [Gy920's SAM2 fork](https://github.com/Gy920/segment-anything-2-real-time),
tested at `e3623cda3f748df999f76db663f08da7a884a820`. Its source is under
`deploy/third_party/sam2` (git-ignored); the helper imports it directly, so compiling
the optional SAM2 CUDA extension is unnecessary. This backend requires CUDA.
`setup_sam2.sh` installs the small extra Python dependencies and never fetches weights.

For this workspace, existing files were copied from `192.168.0.141` into the sibling
`ETC` directory:

```text
ETC/weights/sam2.1_hiera_small.pt
ETC/weights/sam2.1_hiera_large.pt
ETC/weights/dinov2_vitb14_pretrain.pth
ETC/facebookresearch_dinov2_main/
```

The tools automatically use these paths when present. Override with
`SAM2_REPO`, `SAM2_CHECKPOINT`, `SAM2_CONFIG`, `DINO_REPO`, and `DINO_WEIGHTS`, or the
corresponding CLI arguments. Match the SAM2 architecture to its checkpoint:

```bash
# Optional: use large SAM2 instead of the small default.
export SAM2_CHECKPOINT="$(realpath ../ETC/weights/sam2.1_hiera_large.pt)"
export SAM2_CONFIG=configs/sam2.1/sam2.1_hiera_l.yaml
```

Without local DINOv2 paths, the server uses Torch Hub, which may download its source
and weights. The supplied foreground projection is the legacy `standard_array.npy`,
originally from [DINOv2](https://dl.fbaipublicfiles.com/dinov2/arrays/standard.npy).
An example reference is copied from `ETC/syringe.jpeg` to `deploy/perception/assets/syringe.jpeg`.
The standalone demo requires an explicit `--reference` image path; there is no default.

The original environment's PyTorch 2.6/CUDA 12.4 cannot execute on this machine's
RTX 5070 Ti. `env_il` was updated to PyTorch 2.7.1/CUDA 12.8, torchvision 0.22.1,
and torchaudio 2.7.1. For a fresh RTX 50-series installation, apply this after the
base environment setup:

```bash
python -m pip install -r deploy/perception/requirements-rtx50.txt
```

The original base environment files remain usable on their supported GPUs.
[PyTorch's release notes](https://pytorch.org/blog/pytorch-2-7/) describe Blackwell support.

## 1. Verify syringe detection with two processes

Terminal 1, from `deploy`:

```bash
conda activate env_il
cd deploy
bash start_object_detect.sh
```

Terminal 2:

```bash
conda activate env_il
cd deploy
python perception/zero_shot_detection.py --reference perception/assets/syringe.jpeg --serial 233522079515
```

Use your wrist camera's serial; this machine also has `254622075364`. The standalone
demo uses only the camera and has no robot connection. Press `r` to select again
with DINOv2; press `q` to quit. A randomly selected matching component supplies SAM2 prompts.
Images use RGB throughout. An empty/oversized mask resets tracking and the next
frame retries detection. Server errors invalidate registration and REQ sockets
recover after timeouts.

Useful options:

```bash
# Terminal 1, from deploy: tune feature similarity and erosion for your scene.
bash start_object_detect.sh --threshold 0.5 --erosion-count 3

# Terminal 2: a bounded headless camera test, run from deploy.
python perception/zero_shot_detection.py --reference perception/assets/syringe.jpeg --serial 233522079515 \
  --headless --frames 30 --save-dir ../logs/vision_check
```

Pass the target image path using `--reference /path/to/object.jpeg`. For rollout,
set `ZERO_SHOT_REFERENCE` explicitly as well. Keep masks consistent between labeling and rollout
(for example, whether the syringe's packaging belongs to the target).
The ZMQ server binds to localhost by default and retains the legacy pickle
protocol; expose it only to trusted clients. Both processes must use the same port.

## 2. Collect Nami left-arm xyz+yaw perturbations

Install/start the SpaceMouse system driver once if absent:

```bash
sudo apt install libspnav-dev spacenavd
sudo systemctl start spacenavd
```

The Python `spnav` package is already in `env_il`. SpaceMouse input also requires
the system library `libspnav` and a running `spacenavd` service.

Review [visual_servo/config.py](../middle_level_controller/visual_servo/config.py)
and the `nami_visual_servo` preset in [data_collector/config.py](../data_collector/config.py).
Robot IP (`192.168.0.120`) and wrist camera serial (`233522079515`) are set directly
in `visual_servo/config.py`, beside inline `TODO` comments. Edit those entries to
change hardware; no environment variables are needed for either setting.
Logical robot ID `0` is the full Nami connection. **Nami task index `1` is the left
arm** (`0` is the head, `2` the right arm). Only task index `1` receives commands.
The preset uses 20 Hz task-absolute teleoperation and locks SpaceMouse roll/pitch;
`device_params.fix` also accepts `x`, `y`, `z`, or `yaw` to constrain other axes.
Mouse sensitivity matches `ETC/nrmk_il`: `max_value=100`, `deadzone=0.4`, linear
translation gain `0.002 m/s` and rotation gain `0.1 rad/s` per normalized input
unit. The driver polls at 20 Hz. At input `1`, this gives `0.1 mm` and about
`0.286 degrees` per control step. Nami uses its base XYZ axes, with yaw around
base Z; the old Dual Nuri axis permutation is not used.

In `visual_servo/config.py`, `locked_joint_indices: [0, 1, 2, 3]` selects the
torso/head joints. Before each collection or visual-servo rollout, startup
unconditionally calls `set_locked_joint(1)`, `(2)`, `(3)`, and `(4)` individually.
It does not query lock state. Set `locked_joint_indices: []` to issue
`set_locked_joint(0)` and disable all locking instead. These controller commands
are separate from the SpaceMouse roll/pitch constraints.

```bash
conda activate env_il
cd deploy
python collect_data.py nami_visual_servo
```

1. Manually position the robot at the desired goal view before recording.
2. Press `3` to collect from the current pose without a home move. Press `1` to
   move to the saved q22 `home_pos`, or `2` to move there and then collect.
3. Press SpaceMouse button 0 to start recording **at the goal**, then move around
   it in xyz and yaw. The first recorded pose remains the episode's target.
4. Press button 0 again to stop; `s` saves, `e` discards. Gripper control is disabled.
5. Vary the object's position/orientation and repeat, starting each episode at its
   desired goal view.

Raw episodes are `train/data/syringe_visual_servo/<episode>.h5`. They contain Nami's
full q22/p18 state and the selected left-arm command. Preprocessing explicitly
selects the left task pose from p18. The detector server is unnecessary during
collection and manual labeling.

## 3. Label masks after collection

```bash
conda activate env_il
cd train
python generate_mask.py --task syringe_visual_servo --port 5800
```

Open `http://127.0.0.1:5800`. Load a raw H5 episode (recommended for exact frame
alignment), or upload the collector's `<episode>_wrist.mp4` from `data_viz`.
Select the `target` mask and click positive/negative points or two box corners.
Negative boxes erase an area from the selected frame's mask.
Press **Track**, inspect the preview and frame slider, correct prompts on any
frame, then track again and **Save masks**. Multiple named masks and renaming are
supported; add all objects before tracking. **Clear all prompts** resets the entire current episode's annotations.

Outputs are `train/data_mask/syringe_visual_servo/<episode>_wrist.h5`, with a
`target` dataset shaped `(T, H, W, 1)` and a browser-playable preview MP4.
Keep the mask name `target`, or pass the same name to preprocessing with
`--mask-name`. Existing mask files are protected unless `--overwrite` is supplied.
Loading another episode clears previous session annotations; save first.

## 4. Preprocess and train ACT

```bash
conda activate env_il
cd train
python preprocess_visual_servo.py --task syringe_visual_servo
python imitate.py --config-path=config/visual_servo --config-name=syringe
```

Preprocessing checks frame counts, spatial dimensions and labeling metadata,
blackens the background, filters invalid masks, and keeps the original first pose
as the target even if intermediate frames are filtered. Translation labels are
meters in the current TCP frame; rotation labels are relative matrices, trained
through ACT's 6D rotation head. It never uses the SpaceMouse command as the
visual-servo label. Output episodes are numbered contiguously under
`train/processed_data/syringe_visual_servo`.

Use an empty processed output directory; this command does not delete existing
training data. `--target-frame last` supports separately collected goal-last
episodes, but the default and the paper's collection procedure are goal-first.
The ACT config uses image-only input and `chunk_size=n_action_steps=1`.
Weights/config/statistics go to `train/weights/syringe_visual_servo/<timestamp>`.

## 5. Roll out through task_demo.py

Keep process 1 (`bash start_object_detect.sh`) running. In the rollout terminal:

```bash
conda activate env_il
export VISUAL_SERVO_MODEL_DIR=/absolute/path/to/train/weights/syringe_visual_servo/TIMESTAMP
export ZERO_SHOT_REFERENCE=/absolute/path/to/object.jpeg
cd deploy
python task_demo.py visual_servo
```

Press `2` to run, `0` to stop, `q` to quit. The controller applies the learned TCP
relative correction to the current left-arm pose, keeps roll/pitch fixed, and holds
each estimated goal until reached before querying ACT again. SAM2 tracks on every
camera update. Correction bounds, goal tolerances, timeouts, and mask size/ROI
settings are in `visual_servo/config.py`. Defaults stop at predicted correction
norms below 5 mm and 0.5 degrees of yaw. This task finishes at visual alignment;
add any syringe-specific grasp or subsequent FSM motion separately.

Tracking loss holds the current pose and retries detection; repeated loss, stale
camera frames, inference/server errors, timeout, and explicit stop exit teleoperation.
`NN_controller` exposes `exec_nn_control`, `exec_nn_control_stop`, `control_state`,
and `close` for a larger FSM as well. Automatic DAGGER collection is disabled in
this preset; collect another goal-first episode and retrain when needed.

## Validation

```bash
python tests/test_visual_servo.py -v  # repository root, env_il
```

Regression checks cover SE(3) goal labels, mask/frame alignment, RGB registration,
tracker loss and recovery, ZMQ timeout recovery, SpaceMouse constraints, left-arm
routing, correction bounds, goal-to-goal refinement, and stop-on-error behavior.
Hardware validation used the copied checkpoints, the syringe reference, and 30
live frames from camera `233522079515`. Manual labeling was exercised through
SAM2 propagation, H5 saving, preview encoding and a local Gradio HTTP check.
No robot motion was executed. Train/deploy integration was checked with synthetic
data; the syringe policy still needs real goal-first recordings and training.
