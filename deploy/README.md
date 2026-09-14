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

Set `ik_type="rl_constraint"` and `rl_constraint_model_path` in
`data_collector/config.py` to load Pink and the ONNX projector together. The
model may live anywhere; an absolute path to the exported file is simplest. Run
from `deploy`:

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

With `rl_constraint_dry_run=True`, Pink commands execute normally. In RL mode,
each projected q22 command is printed in degrees and the last command actually
sent is repeated. History still updates, and saved `joint_abs_control_0` contains
the actual hold target. With `rl_constraint_dry_run=False`, projected commands
are sent without per-tick printing.

The simplified Genesis training configuration uses 34 current-frame inputs:
14 encoder joints, 18 desired palm-pose values, and two compliance flags.
It uses offsets `[0]`, no joint-target error, no measured palm pose, and no
previous actions. Deployment selects this layout from ONNX metadata (or the
unambiguous input count for Genesis exports missing the measured-pose flag).
Desired commands are used from the first tick. A newly trained/exported model
is required; changing training defaults does not convert an existing checkpoint.

The older `compliant_plane_history_v3` policy runs at 50 Hz. Its original
observation has 25 synchronized history samples of encoder joints, controller
target error (`last_sent_command - q`), desired palm pose, measured palm pose, and independent
left/right compliance-mode flags: 1650 floats. Histories use the exact saved
offset order and update during both Pink and RL. Desired and measured palm poses
refer to the fixed inner collision-face points in the robot-base frame, with
orientations represented by the first two rotation-matrix columns. The runtime
also accepts the shared-mode v2 contract and retains compatibility with old
182-input v5 exports.

RL uses encoder joints from the ordinary robot state and local URDF FK for
measured gripper-base poses, followed by the saved palm offset. Target error
uses the last command actually sent, including Pink and dry-run hold commands.
Reset repeats the measured pose as both desired and actual with zero target
error in every history slot; subsequent ticks use the requested target.
The last sent target does not reproduce PACE's delayed applied target. The
real encoder value is used directly: deployment adds no simulated encoder bias
and no explicit observation or action delay. The output command is
`q_encoder + saved_action_scale * action`; the saved ONNX metadata supplies the
scale, joint order, timing, geometry, and observation options. No Genesis,
RSL-RL, or policy architecture import is needed at runtime.

Both modes send and save the exact final q22 command as `joint_abs_control_0`.
With joint locking enabled they share the same non-arm command reference.
Contact is inferred from motion history; no contact flag or plane pose is an input,
and the learned behavior does not enforce an analytical plane constraint.

Export a current checkpoint with the Genesis environment, from `neuromeka-il`:

```bash
/home/user/yunho/nrmk-genesis/.venv/bin/python deploy/export_rl_constraint.py \
  /path/to/run/model_N.pt --output /path/to/run/model_N.onnx
```

The exporter loads the saved architecture only while converting the checkpoint,
embeds the deployment contract, validates the ONNX graph, and compares 32 ONNX
outputs against PyTorch. Set the resulting path in the collector configuration.
The runtime rejects a model when its saved frequency, shape, or interface does
not match the controller. For v3 teleoperation, set the `lift_box` robot
`control_dt` to `0.02`.

The Genesis evaluator can export the same contract and run the ONNX actor in the
actual training environment before hardware use:

```bash
cd /home/user/yunho/nrmk-genesis
.venv/bin/python experiments/dual_arm_plane/eval.py \
  checkpoint=/path/to/run/model_N.pt policy_backend=onnx \
  export_onnx=true verify_onnx=true cpu=true viewer=false episodes=1
```

To exercise the same fixed box-lift trajectory through this projector, set
`IK_TYPE="rl_constraint"` and `RL_CONSTRAINT_MODEL_PATH` in
`middle_level_controller/box_lift_open_loop/config.py`, then run:

```bash
python task_demo.py box_lift_open_loop
```

Press `1` to move home, `2` to start the trajectory, and `0` to stop. `IK_TYPE`
selects Pink replay or RL projection for the entire run; there is no runtime
mode switch. The 20 Hz recorded joint commands are held at a 50 Hz loop without
interpolation. Pink mode therefore sends only exact recorded commands. In RL
mode, FK of those commands supplies desired TCP history and the policy projects
the arm joints. Set `RL_COMPLIANCE_INTERACTIVE=True` to start each run with
policy compliance `[1, 1]` and press `r` during execution to toggle both channels
between `[1, 1]` and `[0, 0]`. Each change prints the new command and takes effect
at a control tick. With the flag false, `RL_COMPLIANCE_COMMAND` supplies the fixed
value and `r` is disabled. This changes the policy input while the IK backend
remains selected by `IK_TYPE`.
Robot compliance is enabled for both modes and remains enabled after the run.
Set `HOLD_FIRST_TARGET=True` to repeat the first recorded command for the entire
run as a stationary-target diagnostic.

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
