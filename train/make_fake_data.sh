#!/bin/bash

# Generate data
echo -e "=========================\nGenerating synthetic data\n========================="
python helper/generate_synthetic_data.py --task test_single_robot --robot single_robot
python helper/generate_synthetic_data.py --task test_single_robot_gripper --robot single_robot_gripper
python helper/generate_synthetic_data.py --task test_dual_robot --robot dual_robot
python helper/generate_synthetic_data.py --task test_dual_robot_gripper --robot dual_robot_gripper

# Preprocess data
echo -e "=========================\nPreprocessing synthetic data\n========================="
python preprocess.py --task test_single_robot
python preprocess.py --task test_single_robot_gripper
python preprocess.py --task test_dual_robot
python preprocess.py --task test_dual_robot_gripper