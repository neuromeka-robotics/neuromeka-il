# Deployment

The `deploy` directory contains the necessary code to collect demonstration data and run the trained policy with real Neuromeka robots (e.g., [Indy7](https://en.neuromeka.com/indy)).

## Real-world configurations
Data collection and policy deployment both require the configuration of necessary components. The base configurations are located in `helper/config_utils.py`. The specific configurations required are listed below.

- **ROBOT_CONFIG**: Robot configuration (e.g., robot ip, robot home position in joint-space, control parameters)
- **TASK_CONFIG**: Task configuration (e.g., task name + CAMERA_CONFIG / MODEL_CONFIG / DATA_CONFIG / EXTRA_CONFIG)
    - **CAMERA_CONFIG**: Camera configuration Current code only includes [realsense camera](https://www.intel.com/content/www/us/en/ark/products/series/85364/intel-realsense-cameras.html) (e.g., serial number, camera-specific parameters)
    - **MODEL_CONFIG**: Imitation learning model configuration (e.g., model type, model path)
    - **DATA_CONFIG**: Data collection configuration (e.g., path to save data, path to save visualization data, teleoperation device)
    - **EXTRA_CONFIG**: Extra configuration. Currently, it includes movement functions that should be defined in a task-specific manner.

**Configurations are intended to be defined for each task** by overriding the base configurations. Check configuration examples listed below.
- `data_collector/config.py`: Configuration for data collection. Define `DATA_COLLECTOR_ROBOT_CONFIG` and `DATA_COLLECTOR_TASK_CONFIG`.
- `middle_level_controller/act_il/config.py`: Configuration for task `act_il` controller. Define `CUSTOM_ROBOT_CONFIG` and `CUSTOM_TASK_CONFIG`.
- `middle_level_controller/act_il_remote/config.py`: Configuration for task `act_il_remote` controller. Define `CUSTOM_ROBOT_CONFIG` and `CUSTOM_TASK_CONFIG`.

## Data collection
Most imitation learning requires collecting demonstration data by teleoperating real-world robots. The codebase supports teleoperating Neuromeka robots with [VIVE Pro 2](https://www.vive.com/us/product/vive-pro2-full-kit/overview/). Follow below three steps.

### 1. Calibrate VIVE Pro 2
Calibrate vive with Neuromeka Conty. Then, copy-paste the calibration result from the Control Box PC (path: `/home/user/release/IndyDeployment/TeleOp/Calibs/*.json`) to DATA_CONFIG.device_params["calib_uvw"].

### 2. Configure robot and task
Configure robot and task in `data_collector/config.py`.

### 3. Collect data
Run data collector.
```bash
python collect_data.py
```
The high-level command is assigned to the robot by keyboard. The keyboard commands in the present setting are as follows:

|  Keyboard  |              Command             |                     Description                    |
|:----------:|:--------------------------------:|:--------------------------------------------------:|
| 1          |         MOVE_TO_TASK_HOME        |Move to task-specific robot home position           |
| 2          |  EXECUTE_START_STATE_COLLECTION  |Collect data after moving to robot home position    |
| 3          | EXECUTE_CURRENT_STATE_COLLECTION |Collect data from current robot position            |

Teleoperation begins when the menu button on Vive is clicked once, and ends when it is clicked again. The gripper command is given by pressing the trigger button on the back of the Vive controller. When the teleoperation is completed, the user is asked whether or not to save the data. To save, click 's'; to not save, press 'e'.

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

### 2. Set task name
In `task_demo.py`, specify the task name to evaluate as `demo_task = TASK_NAME`. `TASK_NAME` should be an existing folder beneath `middle_level_controller`.

### 3. Evaluate task controller
Run task controller.
```bash
python task_demo.py
```
The high-level command is assigned to the robot by keyboard. The keyboard commands in the present setting are as follows:

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

The robot controller can be launched with `python task_demo.py`, same as before.

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
