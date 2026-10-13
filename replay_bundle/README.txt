EIR recorded-joint replay in Viser

Use Python 3.10, 3.11, or 3.12 (tested with Python 3.10).
Unzip the complete folder, open a terminal in it, and run:

    python -m pip install -r requirements.txt
    python replay_collection.py

Open http://127.0.0.1:8080 and press Play. Ctrl+C exits the server.
If port 8080 is occupied, add --port 8081 and use the printed address.
An internet connection is needed to install the dependencies; replay then
runs locally. No robot SDK, robot connection, or original repository is needed.

The default recording is data/3.h5 (211 frames). To open another recording:

    python replay_collection.py path/to/episode.h5 --dt 0.05

Blue is the measured robot; translucent orange arms are the saved commands.
Playback supports pause, speed adjustment, frame scrubbing, and single-frame
stepping. Joint traces compare measured/sent positions and consecutive command
changes. Choose History = All and the final frame to inspect the whole episode.

Every saved row is preserved; no smoothing or IK is applied. Slow processing
slows playback instead of jumping ahead. A browser can still miss visual
refreshes; pause and step to inspect each frame individually.

Time assumes 0.05 seconds per row (20 Hz). The recording has no per-sample
timestamps. Measured state is read before the command in the same row.
Home movements and unlogged soft-stop repeats are absent from this recording.
The 3D view shows 18 active joints; the four dummy joints are ignored and
unrecorded gripper fingers remain static.

Contents:
    replay_collection.py       Portable viewer, independent of the repository
    requirements.txt           Tested Python dependency versions
    data/3.h5                  Unmodified copy of the original recording
    urdf/EIR_w_gripper.urdf     EIR model with portable relative mesh paths
    meshes/                    All 24 original visual GLB files

The HDF5 copy is complete, including camera images. The viewer reads only
q_0 and joint_abs_control_0 and never modifies the input recording.
Keep the entire folder together when sharing or moving it.
