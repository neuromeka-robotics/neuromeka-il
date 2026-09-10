# Deployment

The `deploy` directory contains the necessary code to collect demonstration data and run the trained policy with real Neuromeka robots (e.g., [Indy7](https://en.neuromeka.com/indy)).

## Real-world configurations
Data collection and policy deployment both require the configuration of necessary components. The base configurations are located in `helper/config_utils.py`. The specific configurations required are listed below.

- **ROBOT_CONFIG**: Robot configuration (e.g., robot ip, robot home position in joint-space, control parameters)
- **TASK_CONFIG**: Task configuration (e.g., task name + CAMERA_CONFIG / MODEL_CONFIG / DATA_CONFIG / EXTRA_CONFIG)
    - **CAMERA_CONFIG**: Camera configuration. Current code only includes [realsense camera](https://www.intel.com/content/www/us/en/ark/products/series/85364/intel-realsense-cameras.html) (e.g., serial number, camera-specific parameters)
    - **MODEL_CONFIG**: Imitation learning model configuration (e.g., model type, model path)
    - **DATA_CONFIG**: Data collection configuration (e.g., path to save data, path to save visualization data, teleoperation device)
    - **EXTRA_CONFIG**: Extra configuration. Currently, it includes movement functions that should be defined in a task-specific manner.

**Configurations are intended to be defined for each task** by overriding the base configurations. Check configuration examples listed below.
- `data_collector/config.py`: Configuration for data collection. Define `DATA_COLLECTOR_ROBOT_CONFIG` and `DATA_COLLECTOR_TASK_CONFIG`.
- `middle_level_controller/act_il/config.py`: Configuration for task `act_il` controller. Define `CUSTOM_ROBOT_CONFIG` and `CUSTOM_TASK_CONFIG`.
- `middle_level_controller/act_il_remote/config.py`: Configuration for task `act_il_remote` controller. Define `CUSTOM_ROBOT_CONFIG` and `CUSTOM_TASK_CONFIG`.

## Data collection
Most imitation learning requires collecting demonstration data by teleoperating real-world robots. The codebase supports teleoperating Neuromeka robots with [VIVE Pro 2](https://www.vive.com/us/product/vive-pro2-full-kit/overview/). Follow below three steps.

Before using VIVE teleoperation or calibration:

- Install SteamVR by following Steam's official instructions: <https://help.steampowered.com/en/faqs/view/18A4-1E10-8A94-3DDA>.
- Start SteamVR and confirm the VIVE base stations and tracker/controller are connected and tracking before running calibration or data collection.

Before using SpaceMouse teleoperation, install and start the SpaceMouse daemon:
```bash
sudo apt install libspnav-dev spacenavd
sudo systemctl start spacenavd
```

### 1. Calibrate VIVE Pro 2
Attach the VIVE tracker/controller rigidly to the robot end effector before calibration. It may be offset from the TCP and does not need to be axis-aligned, but it must not slip or rotate relative to the robot during calibration. Keep it visible to the VIVE base stations and make sure there is at least 100 mm of free motion in robot +X, +Y, and -Z from the current pose.

Preview the fixed cuboid calibration path first:
```bash
python calibrate_vive.py
```

If the preview is safe, execute the calibration motion:
```bash
python calibrate_vive.py --execute
```

After execution, inspect the printed calibration result:

- `calib_uvw`: Euler angles in radians. After the diagnostics look good, copy this value to `DATA_CONFIG.device_params["calib_uvw"]`.
- `Estimated VIVE-to-robot rotation R_RV`: the 3x3 rotation matrix mapping VIVE-frame motion deltas into robot-frame motion deltas. Use it to sanity-check axis directions.
- `4x4 display matrix`: the same rotation shown in homogeneous-matrix form with zero translation. Teleoperation uses relative motion, so this translation term is intentionally not calibrated.
- `Diagnostics`:
  - `num_samples`: number of valid paired robot/VIVE samples used; more samples are better.
  - `singular_values`: indicates whether the calibration motion spans enough independent directions; values should not collapse to a single dominant direction.
  - `rms_error_mm`: fit error in millimeters; lower is better. A large value can indicate VIVE tracking loss, a slipping mount, collision/dragging cables, or insufficient robot motion.
  - `determinant`: should be close to `1.0`; a negative or far-from-one value indicates an invalid rotation estimate.

Only update `DATA_CONFIG.device_params["calib_uvw"]` after these checks look reasonable. If the diagnostics look bad, fix the VIVE visibility/mounting or robot clearance and rerun calibration instead of copying the result.

### 2. Configure robot and task
Configure robot and task in `data_collector/config.py`.

### 3. Collect data
Run data collector.
```bash
python collect_data.py [CONFIG_NAME]
```
The high-level command is assigned by keyboard. The keyboard commands in the present setting are as follows:

|  Keyboard  |              Command             |                     Description                    |
|:----------:|:--------------------------------:|:--------------------------------------------------:|
| 1          |         MOVE_TO_TASK_HOME        |Move to task-specific robot home position           |
| 2          |  EXECUTE_START_STATE_COLLECTION  |Collect data after moving to robot home position    |
| 3          | EXECUTE_CURRENT_STATE_COLLECTION |Collect data from current robot position            |

With `ik_type="pink"`, click anywhere on the VIVE trackpad to start/stop recording. With `ik_type="rl_constraint"`, use the upper trackpad to start/stop recording and the lower trackpad to switch Pink/RL. The gripper command is given by pressing the trigger on the back of the Vive controller. When teleoperation finishes, press `s` to save or `e` to discard.

### Plane-constraint reflex teleoperation

`lift_box` is configured with `ik_type="rl_constraint"`, loading Pink and
`data_collector/models/dual_arm_plane_800.onnx` at startup. Run from `deploy`:

```bash
python collect_data.py lift_box
```

Press keyboard `2` to prepare collection from home or `3` from the current pose.
Then use either VIVE controller:

| Trackpad click | Action |
| --- | --- |
| Upper (`y > 0.3`) | Start/stop recording |
| Lower (`y < -0.3`) | Toggle Pink ↔ RL while recording |
| Center | No action |

Each recording starts in Pink. Release the trackpad between clicks; dragging a
held click into another region does not generate another action. Every Pink → RL
switch captures **current measured joints** as the policy's initial-joint reference.
This is independent of home and recording-start posture. Switching back to Pink
re-anchors both VIVE controllers to the current measured robot poses. The first
Pink solve uses those measured poses as targets and the latest measured joints as
its initial guess, so the old accumulated VIVE targets are not executed. Pink
needs no background solves. Switching modes keeps recording and policy history
active; Pink → RL preserves the VIVE anchors.

For the first hardware test, `lift_box` has `rl_constraint_dry_run=True`. Pink
commands execute normally. In RL mode, each projected q22 command is printed in
degrees, while the last command actually sent is repeated to hold the existing
target with compliance enabled. History and RL-entry references still update
normally. Saved `joint_abs_control_0` and the stopping command use the actual hold
target, never the printed projection. Set `rl_constraint_dry_run=False` in
`data_collector/config.py` when ready to execute RL commands.

The policy uses current/previous desired palm poses, ten measured joint-position
samples at 20 Hz (oldest first), and the RL-entry joint reference: 182 float inputs.
Joint and desired-pose history update on every recording tick, including Pink ticks,
and survive all mode switches. History starts filled with the first measured joints.
The palm material point is recomputed from measured palm orientation at RL entry,
matching the simulator's episode-initial inner box support point. Both desired
poses are expressed using that same point and the simulation world translation.

The ONNX actor includes trained observation normalization. Its 14 outputs are
measured-joint offsets scaled by 0.02 radians and clipped to the training URDF's
joint limits, then converted to DCP q22 degrees. The exported metadata defines
joint order, limits, timing, palm geometry and frame conventions. No Genesis,
RSL-RL, or policy architecture import is needed for this wrapper. It uses the CPU
provider of `onnxruntime` (tested with 1.23.2), NumPy and SciPy.

Both modes send and save the exact final q22 command as `joint_abs_control_0`.
With joint locking enabled they share the same non-arm command reference. The
0.1 s delay modeled in training is not added again to the real control path.
Contact is inferred from motion history; no contact flag or plane pose is an input,
and the learned behavior does not enforce an analytical plane constraint.

To export/evaluate another checkpoint, run from `nrmk-genesis`:

```bash
.venv/bin/python experiments/dual_arm_plane/eval.py \
  run_name=20260909-182947 checkpoint=model_800.pt \
  cpu=true viewer=false episodes=1
```

This exports `model_800.onnx` beside the checkpoint, checks PyTorch/ONNX numerical
parity, and runs ONNX in simulation. Use `export_only=true` to skip simulation,
`cpu=false viewer=true` for visual GPU evaluation, or `export_onnx=false` to reuse
the existing export. Copy the single ONNX file into `data_collector/models/` and
set `rl_constraint_model_path` in `data_collector/config.py`. Keep `control_dt=0.05`
for this checkpoint. Set `ik_type="pink"` to use the original whole-trackpad controls.

By default, raw data are stored in `train/data/TASK_NAME` as `*.h5` files, and the corresponding visualizations are saved in `train/data_viz/TASK_NAME`.

Example real-world data are available in `unit_test/example/data`, with corresponding visualizations in `unit_test/example/data_viz`.


## Deploy controller trained with imitation learning
After training imitation learning models with the code in `train` or other third-party libraries, they can be evaluated in the real-world by following the three steps outlined below.

### 1. Add task controller
Because of the potential differences between task and model design, controllers are intended to be defined for each task. Specifically, each task controller should be listed below `middle_level_controller`, with the folder name matching the task name. Then, `config.py`, `model.py`, and `controller.py` should be added below the task folder, with each file having the following roles:

- `config.py`: Configuration for the corresponding task controller. Define `CUSTOM_ROBOT_CONFIG` and `CUSTOM_TASK_CONFIG`.
- `model.py`: Model wrapper to include model loading, input preprocessing, model inference, and output postprocessing. Define `NN_policy`.
- `controller.py`: Policy wrapper to connect with robots and sensors. Define `NN_controller`.

We offer a general implementation to deploy any task trained with [ACT (i.e., Action Chunking Transformer)](https://arxiv.org/abs/2304.13705) in `middle_level_controller/act_il` to aid in the above implementation for each task. 

To start, duplicate `act_il` inside `middle_level_controller` directory and rename it to the your task name. Then you need to adjust its configuration to match your task.

### 2. Evaluate task controller
Run task controller.
```bash
python task_demo.py [TASK_NAME]
```
`[TASK_NAME]` should be what you defined under `middle_level_controller` in step 1.
The high-level command is assigned by keyboard. The keyboard commands in the present setting are as follows:

|  Keyboard  |              Command             |                               Description                                 |
|:----------:|:--------------------------------:|:-------------------------------------------------------------------------:|
| 1          |         MOVE_TO_TASK_HOME        | Move to task-specific robot home position                                 |
| 2          |            EXECUTE_TASK          | Run task controller                                                       |
| 0          |          EXECUTE_NN_STOP         | Stop task controller                                                      |
| 3          |    EXECUTE_START_STATE_DAGGER    | Stop task controller and collect data after moving to robot home position |
| 4          |   EXECUTE_CURRENT_STATE_DAGGER   | Stop task controller and collect data from current robot position         |

During model evaluation, we support [DAGGER](https://arxiv.org/abs/1011.0686), which allows a human to intervene when the model enters a potential failure mode and collect additional data for retraining. In our experience, DAGGER is critical for improving the performance of models trained with imitation learning. 

To enable DAGGER during evaluation, set `DATA_CONFIG` appropriately in `CUSTOM_TASK_CONFIG`. If `DATA_CONFIG` is *None*, DAGGER will be disabled.

## Deploy controller trained with third-party models
There are many imitation learning models developed by researchers beyond ACT. To evaluate third-party models that are not included in `train`—and to avoid conda environment conflicts—we provide a server-client example in `middle_level_controller/act_il_remote`.

In this setup, two processes need to be run: the robot controller (client) and the model (server).

The robot controller can be launched with `python task_demo.py [TASK_NAME]`, same as before.

To run the model, follow the three steps outlined below.
### 1. Implement model server
Add a new file named `run_nn_server.py` below `middle_level_controller/TASK_NAME`. In `run_nn_server.py`, implement a `RequestHandler` that loads your model and processes incoming requests. Use `middle_level_controller/act_il_remote` as a reference example.

### 2. Add additional dependencies in conda environment of the third-party model
Install only the minimal extra dependencies directly in the existing conda environment of the third-party model to run the server.
```bash
conda activate MODEL_CONDA_ENV
pip install pyzmq
```

### 3. Run model server
```bash
conda activate MODEL_CONDA_ENV
python middle_level_controller/TASK_NAME/run_nn_server.py
```
Make sure the port number matches between the server and the client. 

Additionally, to debug the server, execute `run_nn_server.py` with the `--fake_client` flag. This will generate random data and send it to the server for testing.

## Test trained controllers with Neuromeka Conty (Android App)
You can also execute trained controllers using Neuromeka Conty.

First, start the servicer by following the instructions in the notebook `conty/run_mimic_servicer.ipynb`.

Then, in the Conty Android app, go to the Program → MoveMimic tab and follow the sequence provided there.

## Chaining multiple imitation learning controllers
Although imitation learning is a powerful method, it has limitations when dealing with long-horizon tasks that consist of multiple sub-tasks. To address this, we provide an example showing how to chain multiple imitation learning controllers using a [Finite-State Machine](https://en.wikipedia.org/wiki/Finite-state_machine).

The code snippet below illustrates the idea. You can implement your own FSM if needed.

```bash
python fsm_demo.py
```
