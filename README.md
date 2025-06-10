# Neuromeka Imitation Learning

This repository contains code for training and deploying imitation learning policies for Neuromeka robots. The policy is based on [Action Chunking with Transformers (ACT)](https://arxiv.org/abs/2304.13705).

## Repository Structure

-   `train/`: Contains code for training imitation learning policies.
-   `deploy/`: Contains code for deploying trained policies on a robot.

## Installation

1.  Clone this repository:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  Create a conda environment using the provided `train/environment.yaml` file. This will install all the necessary dependencies.
    ```bash
    conda env create -f train/environment.yaml
    conda activate neuromeka-il
    ```

## Training

The training pipeline uses `pytorch` for model implementation and `hydra` for configuration management.

### 1. Prepare Your Dataset

The training script expects a dataset with a specific structure.
-   Use `train/preprocess.py` to process your raw robot demonstration data into the required format.
-   For testing purposes, you can generate a synthetic dataset by running:
    ```bash
    bash train/make_fake_data.sh
    ```
    This will create a `data/` directory with some sample episodes.

### 2. Configure Your Training Run

Training configurations are located in `train/config/`. You can find example configurations in `train/config/example/`.
To create a new training run, you can either create a new `.yaml` file or override parameters from the command line.

### 3. Run Training

To start a training run, use `train/imitate.py`. You need to specify the configuration to use.

For example, to train with the `single_robot_task` configuration:
```bash
python train/imitate.py --config-name=example/single_robot_task
```

Trained model weights (`.pth` files) and logs will be saved in an output directory managed by Hydra (e.g., `outputs/YYYY-MM-DD/HH-MM-SS/`).

## Deployment

The `deploy` directory contains the necessary code to run the trained policy on a real Neuromeka robot.

### 1. Copy Trained Weights

Copy the trained policy weights (`.pth` file) from the training output directory to `deploy/weights/<your_task_name>/`. You might need to create this directory.

### 2. Configure Deployment Settings

-   **Robot Configuration**: Update robot IP addresses and other parameters in `deploy/config/robot.py`.
-   **Model Configuration**: Check `deploy/config/model.py` for policy-related settings.

### 3. Run the Mimic Server

The `MimicServer` is a gRPC server that loads the policy and exposes methods to control the robot.

To start the server:
```bash
python deploy/MimicServer.py
```

You can also use the `deploy/RunMimicServer.ipynb` notebook for an interactive way to start and test the server.

### 4. Control the Robot

Once the server is running, you can use a gRPC client to send commands. The available RPCs are defined in `deploy/communication/impl/mimic.proto` and include:
-   `SetRobotAddress`
-   `GetSkillList`
-   `GetSkillHome`
-   `RunSkill`
-   `StopSkill`

## License

This project is licensed under the terms of the license agreement found in `train/LICENSE`.
