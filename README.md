# Neuromeka Imitation Learning

This repository contains an implementation for training neural network controllers with imitation learning. The codebase was used to control either a single or dual [Indy7](https://en.neuromeka.com/indy) robots manufactured by Neuromeka. ([Demo video](https://youtu.be/xl4yk2qT7DA?si=70NDDoPU6yNK84tE)) The default policy is based on [Action Chunking with Transformers (ACT)](https://arxiv.org/abs/2304.13705).

## Repository Structure

-   `train`: Contains code for 
    -   Preprocessing demonstration data
    -   Training imitation learning policies
-   `deploy`: Contains code for 
    -   Collecting demonstration data via teleoperation
    -   Deploying trained policies on real robots
    -   Extra functionalities useful for improving the policy's performance (e.g., [DAGGER](https://arxiv.org/abs/1011.0686), Finite-State-Machine).

## Installation

1.  Clone this repository:
    ```bash
    git clone https://github.com/neuromeka-robotics/neuromeka-il.git
    cd neuromeka-il
    ```

2.  Create a conda environment using the provided `environment.yaml` or `environment_train.yaml` file. The required conda environment (name: `env_il`) can be created as follows. The implementation was tested with NVIDIA RTX 3060 + CUDA 11.7 and NVIDIA RTX 4090 + CUDA 12.1.
    ```bash
    # Option 1: Generate conda environment for both TRAIN and DEPLOY
    conda env create -f environment.yaml -n env_il

    # OPTION 2: Generate conda environment only for TRAIN
    conda env create -f environment_train.yaml -n env_il
    ```

3. Install `train/nrmk_il` as a Python package. `train/nrmk_il` includes the core algorithmic components of imitation learning. The package will be installed with a symbolic link, so any changes in the repository will be reflected during execution.
    ```bash
    conda activate env_il
    cd train/nrmk_il
    pip install -e .
    ```

## Further instruction
Check [train/README.md](train/README.md) and [deploy/README.md](deploy/README.md) to learn more.

The recommended steps to follow are outlined below:
1. [Collect data](deploy/README.md#data-collection)
2. [Train model](train/README.md)
3. [Evaluate model in real-world](deploy/README.md#deploy-controller-trained-with-imitation-learning)

For a detailed walkthrough, check out the [tutorial slides](/Neuromeka_IL_tutorial.pdf), which explain how to use this repository.

## Credits
The algorithm code in `train/nrmk_il/src/nrmk_il/policies` are modified version from [LeRobot](https://github.com/huggingface/lerobot), which is licensed under the Apache-2.0 license.

## Authors
[Neuromeka AI team](https://ai.neuromeka.com/)

## License

This project is licensed under the terms of the license agreement found in `LICENSE`.
