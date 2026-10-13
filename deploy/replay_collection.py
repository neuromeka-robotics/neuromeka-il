#!/usr/bin/env python3
"""Portable EIR joint replay. Share the complete replay_bundle folder or ZIP.

Python 3.10-3.12 setup (no repository checkout, Neuromeka SDK, or Pink needed):
    python -m pip install -r requirements.txt
    python replay_collection.py
    python replay_collection.py data/3.h5 --dt 0.05

Open http://127.0.0.1:8080. The default episode and robot model are resolved
relative to this script, so the folder can be moved to another computer.
No robot connection is created, and the input HDF5 file is opened read-only.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import h5py
import numpy as np

DCP_ACTIVE_JOINT_NAMES = (
    "Joint_L0",
    "Joint_L1",
    "Joint_L2_U",
    "Joint_L3_U",
    "Joint_L2_L",
    "Joint_L3_L",
    "Joint_L4_L",
    "Joint_L5_L",
    "Joint_L6_L",
    "Joint_L7_L",
    "Joint_L8_L",
    "Joint_L2_R",
    "Joint_L3_R",
    "Joint_L4_R",
    "Joint_L5_R",
    "Joint_L6_R",
    "Joint_L7_R",
    "Joint_L8_R",
)


def joint_values(q):
    """Map IndyDCP q22 degrees to URDF names, ignoring the four dummy joints."""
    q = np.asarray(q, dtype=float)
    if q.shape != (22,) or not np.isfinite(q).all():
        raise ValueError("Viewer expects 22 finite IndyDCP joint values in degrees")
    return dict(zip(DCP_ACTIVE_JOINT_NAMES, np.deg2rad(q[:18])))


def _keep_arm_geometry(model):
    """Remove non-arm meshes while preserving the complete joint/frame tree."""
    arm_roots = {model.joint_map[name].child for name in ("Joint_L2_L", "Joint_L2_R")}
    for scene in (model.scene, model.collision_scene):
        if scene is None:
            continue
        parents = scene.graph.transforms.parents
        hidden = []
        for geometry in scene.geometry:
            ancestor = geometry
            while ancestor not in arm_roots and ancestor in parents:
                ancestor = parents[ancestor]
            if ancestor not in arm_roots:
                hidden.append(geometry)
        scene.delete_geometry(hidden)


def bundle_path(relative: str) -> Path:
    """Find adjacent bundle assets, also supporting the repository's source script."""
    script_directory = Path(__file__).resolve().parent
    adjacent = script_directory / relative
    repository_bundle = script_directory.parent / "replay_bundle" / relative
    return adjacent if adjacent.exists() or not repository_bundle.exists() else repository_bundle


def load_episode(path: Path, robot_id: int = 0):
    """Read only joint datasets; preserve every recorded row in its original order."""
    keys = (f"q_{robot_id}", f"joint_abs_control_{robot_id}")
    with h5py.File(path, "r") as episode:
        for key in keys:
            if key not in episode:
                raise ValueError(
                    f"Missing dataset {key!r}; this viewer requires a joint_abs recording.")
        measured, commanded = (np.asarray(episode[key], dtype=float) for key in keys)
    for key, values in zip(keys, (measured, commanded)):
        if values.ndim != 2 or values.shape[1] != 22 or not len(values):
            raise ValueError(f"{key} must have shape (N, 22), N > 0; got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"{key} contains NaN or infinite joint values")
    if measured.shape != commanded.shape:
        raise ValueError(
            f"State/command shapes differ: {measured.shape} versus {commanded.shape}; "
            "refusing to truncate or misalign the recording.")
    return measured, commanded


def make_robots(server, urdf_path):
    """Two independent URDFs, with the command shown as an arm overlay."""
    from viser.extras import ViserUrdf
    from yourdfpy import URDF

    urdf_path = Path(urdf_path).expanduser().resolve()
    robots, roots = [], []
    for name, color in (
        ("measured", (0.2, 0.55, 1., 1.)),
        ("commanded", (1., 0.55, 0.1, 0.4)),
    ):
        model = URDF.load(
            str(urdf_path),
            filename_handler=lambda fname: str(
                urdf_path.parent / Path(fname.removeprefix("package://"))),
            load_meshes=True,
            build_scene_graph=True,
            load_collision_meshes=False,
            build_collision_scene_graph=False,
        )
        if not set(DCP_ACTIVE_JOINT_NAMES).issubset(model.joint_map):
            raise ValueError("Viewer URDF is missing EIR joints")
        if name == "commanded":
            _keep_arm_geometry(model)
        roots.append(server.scene.add_frame(f"/{name}", show_axes=False))
        robots.append(ViserUrdf(
            server, model, root_node_name=f"/{name}",
            load_meshes=True, load_collision_meshes=False,
            mesh_color_override=color, collision_mesh_color_override=color))
    server.scene.add_grid("/grid", width=3., height=3.)

    @server.on_client_connect
    def _(client):
        client.camera.position = (2.5, -2.5, 1.8)
        client.camera.look_at = (0., 0., 0.7)

    return robots, roots


class EpisodeReplay:
    def __init__(self, server, robots, roots, measured, commanded, *, dt, name):
        self.server, self.robots, self.roots = server, robots, roots
        self.measured, self.commanded = measured, commanded
        self.dt = dt
        self.count = len(measured)
        self.times = np.arange(self.count, dtype=float) * dt
        self._last_view = None
        self._last_playback = None
        self._next_frame = 0.

        gui = server.gui
        gui.configure_theme(control_width="large")
        gui.add_markdown(
            f"**{name}** — {self.count} saved frames  \n"
            "**Blue:** measured robot. **Orange:** commanded arms.  \n"
            f"Time assumes **{dt:g} s/frame ({1. / dt:g} Hz)**; "
            "this file has no per-frame timestamps. Measured state is read "
            "before the command in the same frame. No smoothing or IK.")
        with gui.add_folder("Playback"):
            self.playing = gui.add_checkbox("Play", False, disabled=self.count == 1)
            self.speed = gui.add_dropdown(
                "Speed", ("0.1x", "0.25x", "0.5x", "1x", "2x"), initial_value="1x")
            self.loop = gui.add_checkbox("Loop", False)
            self.frame = gui.add_slider(
                "Frame (zero-based)", min=0, max=max(1, self.count - 1),
                step=1, initial_value=0, disabled=self.count == 1)
            previous = gui.add_button("Previous frame", disabled=self.count == 1)
            following = gui.add_button("Next frame", disabled=self.count == 1)
            restart = gui.add_button("Back to start")
            self.status = gui.add_markdown("")
        self.show_measured = gui.add_checkbox("Show measured", True)
        self.show_commanded = gui.add_checkbox("Show commanded", True)

        with gui.add_folder("Joint traces"):
            self.joint = gui.add_dropdown(
                "Joint", DCP_ACTIVE_JOINT_NAMES,
                initial_value=DCP_ACTIVE_JOINT_NAMES[4])
            self.window = gui.add_dropdown(
                "History", ("2 s", "5 s", "10 s", "All"), initial_value="5 s")
            gui.add_markdown(
                "Plots end at the selected frame. Every saved sample is included. "
                "Alternating command steps reveal target jitter; "
                "peak-to-peak also includes intentional motion.")
            self.metrics = gui.add_markdown("")
            empty = np.empty(0, dtype=float)
            series = (
                {"label": "Time (s)"},
                {"label": "Measured (deg)", "stroke": "#3399ff", "width": 2},
                {"label": "Sent (deg)", "stroke": "#ff8c1a", "width": 2},
            )
            self.position_plot = gui.add_uplot(
                (empty, empty, empty), series,
                title="Joint position", scales={"x": {"time": False}}, aspect=1.6)
            self.step_plot = gui.add_uplot(
                (empty, empty),
                ({"label": "Time (s)"},
                 {"label": "Sent change (deg/frame)", "stroke": "#ff8c1a", "width": 2}),
                title="Command change between consecutive frames",
                scales={"x": {"time": False}}, aspect=1.8)
        with gui.add_folder("All joint values"):
            self.table = gui.add_markdown("")

        @self.frame.on_update
        def _(event):
            if event.client_id is not None:
                self.playing.value = False

        @previous.on_click
        def _(_event):
            self.seek(int(self.frame.value) - 1)

        @following.on_click
        def _(_event):
            self.seek(int(self.frame.value) + 1)

        @restart.on_click
        def _(_event):
            self.seek(0)

        self.tick(time.monotonic())

    def seek(self, frame):
        self.playing.value = False
        self.frame.value = max(0, min(self.count - 1, frame))

    def tick(self, now):
        """Advance at most one recorded frame, so slow rendering cannot skip rows."""
        speed = float(self.speed.value.removesuffix("x"))
        playback = (self.playing.value, speed)
        if playback != self._last_playback:
            self._next_frame = now + self.dt / speed
            self._last_playback = playback
        index = int(self.frame.value)
        if self.playing.value and now >= self._next_frame:
            if index < self.count - 1:
                index += 1
            elif self.loop.value:
                index = 0
            else:
                self.playing.value = False
            self.frame.value = index
            self._next_frame = now + self.dt / speed

        view = (index, self.joint.value, self.window.value,
                self.show_measured.value, self.show_commanded.value, self.playing.value)
        if view != self._last_view:
            with self.server.atomic():
                self.render(index)
            self._last_view = view

    def render(self, index):
        measured, commanded = self.measured[index], self.commanded[index]
        self.roots[0].visible = self.show_measured.value
        self.roots[1].visible = self.show_commanded.value
        self.robots[0].update_cfg(joint_values(measured))
        self.robots[1].update_cfg(joint_values(commanded))
        state = "Playing" if self.playing.value else "Paused"
        self.status.content = (
            f"**{state}** — frame **{index} / {self.count - 1}**, "
            f"assumed time **{self.times[index]:.3f} / {self.times[-1]:.3f} s**")

        joint = DCP_ACTIVE_JOINT_NAMES.index(self.joint.value)
        start = 0 if self.window.value == "All" else max(
            0, index - int(float(self.window.value.split()[0]) / self.dt))
        stop = index + 1
        times = self.times[start:stop]
        actual = self.measured[start:stop, joint]
        sent = self.commanded[start:stop, joint]
        self.position_plot.data = (times, actual, sent)
        changes = np.diff(sent)
        self.step_plot.data = (times[1:], changes)
        max_step = float(np.max(np.abs(changes))) if len(changes) else 0.
        self.metrics.content = (
            f"**{self.joint.value}** — measured **{measured[joint]:.4f}°**, "
            f"sent **{commanded[joint]:.4f}°**  \n"
            f"In plotted window: measured peak-to-peak **{np.ptp(actual):.4f}°**, "
            f"sent peak-to-peak **{np.ptp(sent):.4f}°**, "
            f"max sent step **{max_step:.4f}°**.")
        self.table.content = (
            "| Joint | Measured (°) | Sent (°) | Sent − measured (°) |\n"
            "| --- | ---: | ---: | ---: |\n" + "\n".join(
                f"| {name} | {q:.4f} | {cmd:.4f} | {cmd - q:+.4f} |"
                for name, q, cmd in zip(DCP_ACTIVE_JOINT_NAMES, measured, commanded)))


def positive_float(value):
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("episode", type=Path, nargs="?", default=bundle_path("data/3.h5"),
                        help="Saved collection .h5 file (default: bundled data/3.h5)")
    parser.add_argument("--robot-id", type=int, default=0)
    parser.add_argument("--dt", type=positive_float, default=0.05,
                        help="Assumed recording interval in seconds (default: 0.05)")
    parser.add_argument("--urdf", type=Path, default=bundle_path("urdf/EIR_w_gripper.urdf"),
                        help="EIR URDF (default: bundled urdf/EIR_w_gripper.urdf)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    try:
        measured, commanded = load_episode(args.episode.expanduser(), args.robot_id)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    import viser

    urdf_path = args.urdf.expanduser().resolve()
    if not urdf_path.is_file():
        parser.error(f"URDF not found: {urdf_path}. Keep the complete replay bundle together or pass --urdf.")
    server = viser.ViserServer(host=args.host, port=args.port)
    try:
        robots, roots = make_robots(server, urdf_path)
        replay = EpisodeReplay(
            server, robots, roots, measured, commanded,
            dt=args.dt, name=args.episode.name)
        print(
            f"Loaded {len(measured)} paired frames from {args.episode}.\n"
            f"Assuming {args.dt:g} s/frame; open http://{args.host}:{server.get_port()} "
            "and press Play. Ctrl+C exits.", flush=True)
        while True:
            replay.tick(time.monotonic())
            time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
