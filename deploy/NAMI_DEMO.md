# NAMI teleoperation demo

Use the installed deployment environment. Run Python commands from `neuromeka-il/deploy`.
Before the demo, set `NAMI_HOME_POS` to a validated 22-joint home pose in
[`data_collector/config.py`](data_collector/config.py), and set the robot IP inside `make_nami_config()`.

Use the [NAMI notebook](helper/notebook/nami_robot_example.ipynb) to check the robot connection, read its state, and try interactive commands.

## 1. Start SteamVR

```bash
steam
```

In **Library**, launch **SteamVR**. Confirm both base stations, the headset, and both
controllers are detected. Otherwise, check controller power, headset connections,
and power. The headset distributor/link box should have its green LED on; press
its power button if needed.

## 2. Calibrate VIVE after moving the base stations

Use the [NAMI notebook](helper/notebook/nami_robot_example.ipynb) to open or close the gripper when mounting the VIVE controller.

Securely grip/mount `controller_1` in NAMI's left hand so it cannot slip.
Replace the device name below if your mounted controller has a different name. Leave 100 mm clearance
in robot +X, +Y, and −Z. Preview, then execute:

```bash
python calibrate_vive.py --config nami_task --arm-index 1 --device-name controller_1
python calibrate_vive.py --config nami_task --arm-index 1 --device-name controller_1 --execute
```

After checking the printed diagnostics, copy the three `calib_uvw` values (radians)
into `make_nami_config()` → `DATA_CONFIG` → `device_params["calib_uvw"]` in
[`data_collector/config.py`](data_collector/config.py). Both arms share this calibration.

## 3. Run teleoperation

```bash
python collect_data.py nami_task
```

1. Press keyboard **1** to move home; wait until motion finishes.
2. Press keyboard **2** to move home and prepare teleoperation.
3. Click the VIVE trackpad button once to start control; click again to stop.
4. Press keyboard **e** to discard the recording.
5. Press **1** to return home, then **2** to repeat.

`NAMI_HOME_POS=None` skips home movement.
