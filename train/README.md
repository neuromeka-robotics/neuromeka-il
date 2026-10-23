# Training
The `train` directory contains the necessary code to train policies with imitation learning.

## Training configurations
The codebase supports the following configurations.

(1) Algorithm
- ACT (i.e., Action Chunking Transformer) [[Paper](https://arxiv.org/abs/2304.13705)]

(2) Robot mode
- Single robot arm
- Single robot arm with gripper
- Dual robot arm
- Dual robot arm with gripper

(3) Control mode (i.e., output of neural network)
- Task space
- Relative delta task space [[Paper](https://arxiv.org/abs/2402.10329)]

Configurations can be controlled via *yaml* files listed below `config`. 

To create your own config, create a new directory under `config`, and copy one of the existing *yaml* files under `config/fake_data_example` or `config/real_data_example` as a template. You can then adjust parameters such as `task_name`, `batch_size`, or `lr`.

If you want to use RGB images as input, you can also crop the image by setting `extra.image_crop.[observation_name]` to `[[x_min, x_max], [y_min, y_max]]`. Refer to `config/fake_data_example/single_robot_task.yaml` for an example. You can experiment with different crop areas using the notebook in `deploy/helper/notebook/data_example.ipynb`.

## Usage
The collected data should first be placed under `data` as follows.
```
|- data
|---TASK_NAME
|----- 0.h5
|----- 1.h5
|---- ...
```
Then, follow these **three** steps:

1. Activate conda environment.
```
conda activate env_il
```
2. Preprocess the data. The preprocessed files will be saved under `processed_data`.
```
python preprocess.py --task TASK_NAME
```
3. Train the neural networks.
```
python imitate.py --config-path=config/CONFIG_DIRECTORY --config-name=CONFIG_FILE
```

## Usage examples
We provide a pipeline with randomly generated data to help users understand the process and required data format.

Before running the pipeline, ensure your SSD has at least 1GB of free space.

First, create and preprocess synthetic data. This will generate four datasets (*test_single_robot*, *test_single_robot_gripper*, *test_dual_robot*, *test_dual_robot_gripper*) under `data` and `processed_data`.
```
bash unit_test/make_fake_data.sh
```
Then, train neural networks with example configurations.
```
# Example 1: Single robot + Task space action
python imitate.py --config-path=config/fake_data_example --config-name=single_robot_task.yaml 

# Example 2: Single robot + Relative delta task space action
python imitate.py --config-path=config/fake_data_example --config-name=single_robot_relative_delta.yaml

# Example 3: Single robot with gripper + Task space action
python imitate.py --config-path=config/fake_data_example --config-name=single_robot_gripper_task.yaml

# Example 4: Single robot with gripper + Relative delta task space action
python imitate.py --config-path=config/fake_data_example --config-name=single_robot_gripper_relative_delta.yaml

# Example 5: Dual robot + Task space action
python imitate.py --config-path=config/fake_data_example --config-name=dual_robot_task.yaml

# Example 6: Dual robot + Relative delta task space action
python imitate.py --config-path=config/fake_data_example --config-name=dual_robot_relative_delta.yaml

# Example 7: Dual robot with gripper + Task space action
python imitate.py --config-path=config/fake_data_example --config-name=dual_robot_gripper_task.yaml

# Example 8: Dual robot with gripper + Relative delta task space action
python imitate.py --config-path=config/fake_data_example --config-name=dual_robot_gripper_relative_delta.yaml 
```

We also provide configuration files for training imitation policies with real-world data in `config/real_data_example`.

## Real data examples
Compatible real-world data are located in unit_test/example/data, with corresponding visualizations in unit_test/example/data_viz.

## EIR nominal-command correction

Training uses [`config/correction/eir_nominal_command.yaml`](config/correction/eir_nominal_command.yaml).
Its only data setting is `data.processed_dir`, pointing to the folder containing
normalized H5 episodes, `split.json`, and `normalization.json`. Raw recordings,
IK settings and split fractions are not needed for training.

`train.output_dir` specifies the checkpoint parent, with one subfolder per
`train.run_name` (or `--run`). `train.device` defaults to `cuda:0`.
Paths support `~`; relative training paths resolve from the YAML's directory.
The supplied config uses `joint_task_nominal/processed/teleop_common` with all
episodes in the training split. Data must use `next_step_immediate` timing.

Generation/preprocessing use [`config/correction/eir_data.yaml`](config/correction/eir_data.yaml).
That separate file holds the raw joint directory, task files (0.h5 and 1.h5),
timing/IK robot config, output locations and validation fraction. Sources resolve
relative to `data.root`; labels and processed paths resolve relative to `data.work_dir`.
To regenerate, choose a new `data.work_dir`, then update the training YAML's
`data.processed_dir` to the resulting dataset. Existing outputs are never overwritten.
The generic task-space/image ACT workflows above are unchanged.

```bash
conda activate pace
cd ~/neuromeka-il/train

bash run_correction.sh generate --dry-run  # Inspect all paths and commands.
bash run_correction.sh generate           # IK, minimum duration, timing repair, labels.
bash run_correction.sh preprocess         # Split by episode, fit train-only normalization.
bash run_correction.sh train --run eir_nominal_run4
```

`--config PATH` overrides the stage's config: preparation YAML for generate/preprocess,
training YAML for train. `all` uses both default configs; with custom paths, use
`all --config TRAIN_YAML --data-config DATA_YAML`. Their processed paths must match.
`--run NAME` and `--device DEVICE` apply only to train/all; `--dry-run` prints commands.
Epochs, batch size and W&B settings are in `train`;
set `train.wandb.enabled: true` to enable logging.
No dataset directory flags are needed. Running `train` uses the saved split and
normalization unchanged; the preparation YAML's `data.validation_fraction` affects preprocessing only.
At 0.05 with eight episodes, one whole episode is held out, not exactly 5% of frames.

The generate stage preserves the existing recording-gap inference based on
position/velocity consistency. Joint recordings undergo timing repair directly.
Task recordings undergo arm-only IK, contiguous-success segmentation (minimum
5 s), then timing repair. The original files are unchanged. Logs and source
hashes live under the dataset work directory. The normalized data and future
checkpoints also live there. During inferred collector pauses, inserted teleop
targets hold the previous dispatched target rather than anticipating the next one.

### Training and command timing

ACT receives three independent input keys, each with 14 active arm joints:
`observation.nominal_q`, `observation.nominal_v`, and
`observation.nominal_teleop_error`. Outputs are
`action.command_delta_q` and `action.command_delta_v`, each 14-dimensional.
No images or pose/orientation transforms are used. Inputs and outputs are
normalized by preprocessing, so policy normalization dictionaries stay empty.

`policy.n_obs_steps` selects history length (11 in this config). For H > 1,
each key has shape `[batch, H, 14]`, ordered oldest to current; ACT receives `3*H`
tokens with learned key and time embeddings. Eleven samples span 50 ms before the
current tick at dt=0.005 s. History repeats the first frame at episode start and
never crosses episode boundaries; future action chunk masking is unchanged.
Replay buffers every physics tick, including while executing a queued action,
and resets history on a new episode/loop. Processed data and statistics with
`next_step_immediate` timing can be reused when changing `n_obs_steps`;
the model must be retrained. `n_obs_steps: 1` uses a single frame.
Observation history currently supports state tokens without images or a VAE.

```python
# Input history ends at k, from independent nominal state:
dq, dv = network(q_nom[k-H+1:k+1], v_nom[k-H+1:k+1], teleop_error[k-H+1:k+1])
v_nom[k+1] = v_nom[k] + dt * a_PD(teleop[k], q_nom[k], v_nom[k])
q_nom[k+1] = q_nom[k] + dt * v_nom[k+1]

# Labels (qdes and qdotdes independently linearly interpolated):
command_delta_q[k] = qdes[k+1] - q_nom[k+1]
command_delta_v[k] = qdotdes[k+1] - v_nom[k+1]

# Predicted endpoint targets, applied immediately during physics interval k:
q_cmd[k] = q_nom[k+1] + dq
v_cmd[k] = v_nom[k+1] + dv
```

Only nominal state feeds the next nominal step. Command corrections never feed
back. Exports store only these five fields with `nominal_command_timing:
next_step_immediate`. Regenerate labels and normalization before training new
weights. N reference states produce N-1 transition rows: the terminal reference
supervises the last label, with no terminal observation. A chunk may predict the
next recorded boundary, but transitions beyond it are masked because they need
the next teleop input. Raw data have 18 columns; normalized training data have 14.

### Replay

From the pace-sim2real checkout:

```bash
python genesis_bridge/replay_teleop.py \
  --config genesis_bridge/configs/eir_train.yaml \
  --data ~/Desktop/pace_eir_data/joint_task_nominal/postprocessed \
  --correction neural \
  --weights ~/Desktop/pace_eir_data/joint_task_nominal/weights/eir_nominal_run4
```

Replay modes are `ideal` (logged qdes/qdotdes at k+1), `ideal_correction` (nominal PD +
saved residuals), `none` (nominal PD), and `neural` (nominal PD + ACT residuals).
For `ideal_correction`, add `--labels` pointing to the YAML's **raw** labels_dir;
use the same postprocessed source recordings. This checks the actual saved labels.
Perfect saved residuals reproduce ideal commands; physical motor tracking remains
separate. Neural replay takes architecture, `n_obs_steps` and `n_action_steps` from the weight
folder's config, and loads its normalization and checkpoint. Each predicted
endpoint command is applied immediately during step k, with no extra tick of
buffering. Command RMSE compares to reference[k+1]; physical state RMSE keeps
the measured and simulated timestamps aligned.

`--prepare-only` evaluates commands without Genesis; `--headless` runs physics
without a viewer. Replay saves no outputs. Command q/v RMSE and simulated robot
position RMSE are reported separately.

## Training plot examples
[Wandb](https://wandb.ai/site/) can be enabled in the configuration file (`logging: True`) to visualize training progress. Below is an example training result for a single robot arm task:

<img width=600 src='unit_test/example/log_plot.png'>

