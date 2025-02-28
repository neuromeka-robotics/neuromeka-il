# Neuromeka Imitation Learning

## License
TBD

**Authors: Neuroemka AI team**

## Intro

There are two contents in this directory (i.e., `train`).
- Training neural networks

Each purpose requires a unique setup and usage.

## Setup (Training)

### Dependencies
For training neural networks, the conda environment (name: `il`) can be created as follows.
```
# Create environment and install dependencies
conda env create -f environment.yaml -n il

# Activate environment
conda activate il
```

### Configuration
Training configurations are listed below `config`. Create a new folder and configuration files similar to those in `config/yunho`.

## Usage (Training)
First, activate conda environment.
```
conda activate il
```
Second, pre-process data
```
python process/reformat_data.py --task TASK_NAME
```
Third, train neural networks
```
# Example 1: Train task w/o gripper
python imitate.py --config-path=config/yunho --config-name=configuration_wo_gripper.yaml

# Example 2: Train task w/ gripper
python imitate.py --config-path=config/yunho --config-name=configuration_w_gripper.yaml

# Example 3: Train task w/o gripper + disable wandb logging
python imitate.py --config-path=config/yunho --config-name=configuration_wo_gripper.yaml base.logging=False

# Example 4: Train visual servoing task
python imitate.py --config-path=config/yunho --config-name=act_visual_servo.yaml

# Example 5: Train mask predictor 
# Before executing the command below, generate mask labels first.
python train_mask_predictor.py --config-path=config/yunho --config-name=mask_predictor.yaml

# Example 5: Train task monitor (i.e., task selection and success detection)
python train_task_monitor.py --config-path=config/yunho --config-name=task_monitor.yaml

# Example 6. Train feature-matching based anomaly detector
# Before executing the command below, generate centroid features first. 
# Check 'test_feature_retrieval.py' to generate centroid features.
python train_anomaly_detector.py --config-path=config/yunho --config-name=feature_anomaly_detector.yaml
```
```
bash make_fake_data.sh

python imitate.py --config-path=config/example --config-name=single_robot_task.yaml 
python imitate.py --config-path=config/example --config-name=single_robot_relative_delta.yaml
python imitate.py --config-path=config/example --config-name=single_robot_gripper_task.yaml
python imitate.py --config-path=config/example --config-name=single_robot_gripper_relative_delta.yaml
python imitate.py --config-path=config/example --config-name=dual_robot_task.yaml
python imitate.py --config-path=config/example --config-name=dual_robot_relative_delta.yaml
python imitate.py --config-path=config/example --config-name=dual_robot_gripper_task.yaml
python imitate.py --config-path=config/example --config-name=dual_robot_gripper_relative_delta.yaml 
```