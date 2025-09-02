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

## Training plot examples
[Wandb](https://wandb.ai/site/) can be enabled in the configuration file (`logging: True`) to visualize training progress. Below is an example training result for a single robot arm task:

<img width=600 src='unit_test/example/log_plot.png'>

