#!/usr/bin/env python3
"""Compare the final recorded states of two box-lift result CSVs in Viser."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np

from data_collector.pink_ik import DCP_ACTIVE_JOINT_NAMES
from data_collector.visualizer import _keep_arm_geometry, joint_values
from helper.eir_pink_ik_visualizer import (
    DEFAULT_CONFIG_PATH, _has_unfetched_lfs_meshes, _load_eir_settings,
)


RESULT_DIR = Path(__file__).resolve().parent / "middle_level_controller/box_lift_open_loop/result"
# Select two logs here, relative to RESULT_DIR or as absolute paths.
LEFT_CSV = "pink_traj_1_0.125/joint_states_20260915_114241_796188.csv"
RIGHT_CSV = "rl_constraint_traj_1_0.125/joint_states_20260915_113921_417806.csv"


@dataclass
class FinalSample:
    path: Path
    index: int
    elapsed: float
    mode: str
    measured: list[float]
    commanded: list[float]

    @property
    def arm_error_norm(self):
        return float(np.linalg.norm(
            np.asarray(self.commanded[4:18]) - self.measured[4:18]))


def load_final_sample(path: str | Path) -> FinalSample:
    """Read the last data row by column name, independent of column ordering."""
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = RESULT_DIR / path
    measured_columns = [f"q_{name}_deg" for name in DCP_ACTIVE_JOINT_NAMES]
    commanded_columns = [f"command_{name}_deg" for name in DCP_ACTIVE_JOINT_NAMES]
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        header = reader.fieldnames or []
        if len(set(header)) != len(header):
            raise ValueError(f"{path}: duplicate CSV columns")
        required = ["sample_index", "elapsed_time_s", *measured_columns, *commanded_columns]
        missing = [name for name in required if name not in header]
        if missing:
            raise ValueError(f"{path}: missing result columns: {', '.join(missing)}")
        final = None
        for row in reader:
            final = row
    if final is None:
        raise ValueError(f"{path}: no recorded samples")
    try:
        if None in final or any(value is None for value in final.values()):
            raise ValueError("row width does not match header")
        index = int(final["sample_index"])
        elapsed = float(final["elapsed_time_s"])
        # Dummy joints do not affect the URDF; retain the q22 viewer interface.
        measured = [float(final[name]) for name in measured_columns] + [0.] * 4
        commanded = [float(final[name]) for name in commanded_columns] + [0.] * 4
        joint_values(measured)
        joint_values(commanded)
        if index < 0 or not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError("invalid sample index or elapsed time")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: invalid final sample: {exc}") from exc
    return FinalSample(path, index, elapsed, final.get("ik_mode", "unknown"), measured, commanded)


def create_comparison(left, right, *, config=DEFAULT_CONFIG_PATH,
                      host="127.0.0.1", port=8081, spacing=1.8):
    """Build a static scene from saved data, with no robot connection."""
    import viser
    from viser.extras import ViserUrdf
    from yourdfpy import URDF

    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("Spacing must be positive and finite")
    urdf_path = _load_eir_settings(Path(config).expanduser().resolve())["visual_urdf_path"]
    if _has_unfetched_lfs_meshes(urdf_path):
        raise ValueError(f"Fetch the visual mesh Git LFS files for {urdf_path} before viewing")

    def add_run(server, side, sample, y):
        root = f"/{side}"
        server.scene.add_frame(root, position=(0., y, 0.), show_axes=False)
        server.scene.add_label(f"{root}/label", text=f"{side.title()}: {sample.path.parent.name}",
                               position=(0., 0., 1.8))
        roots = []
        for kind, q, color in (("measured", sample.measured, None),
                               ("commanded", sample.commanded, (1., .55, .1, .35))):
            model = URDF.load(
                str(urdf_path),
                filename_handler=lambda fname: str(
                    urdf_path.parent / Path(fname.removeprefix("package://"))),
                load_meshes=True, build_scene_graph=True,
                load_collision_meshes=False, build_collision_scene_graph=False)
            if kind == "commanded":
                _keep_arm_geometry(model)
            name = f"{root}/{kind}"
            roots.append(server.scene.add_frame(name, show_axes=False))
            robot = ViserUrdf(server, model, root_node_name=name,
                              load_meshes=True, load_collision_meshes=False,
                              mesh_color_override=color)
            robot.update_cfg(joint_values(q))
        with server.gui.add_folder(side.title()):
            server.gui.add_markdown(
                f"**{sample.path.parent.name}**  \n`{sample.path.name}`  \n"
                f"Final sample: **{sample.index}** · Time: **{sample.elapsed:.3f} s**  \n"
                f"Mode: **{sample.mode}**  \n"
                f"14-arm-joint error norm: **{sample.arm_error_norm:.3f}°**")
            show_measured = server.gui.add_checkbox("Show measured robot", True)
            show_commanded = server.gui.add_checkbox("Show sent command", True)

            @show_measured.on_update
            def _(_event):
                roots[0].visible = show_measured.value

            @show_commanded.on_update
            def _(_event):
                roots[1].visible = show_commanded.value

    server = viser.ViserServer(host=host, port=port)
    try:
        server.scene.add_grid("/grid", width=3., height=spacing + 2.)
        server.gui.add_markdown(
            "## Final replay samples\n"
            "**Original visual mesh:** measured robot  \n"
            "**Orange arms:** command sent to `tele_move`  \n"
            "Each side shows its own final recorded time step.")
        add_run(server, "left", left, -spacing / 2)
        add_run(server, "right", right, spacing / 2)

        @server.on_client_connect
        def _(client):
            client.camera.position = (max(3.5, spacing * 1.8), 0., 2.)
            client.camera.look_at = (0., 0., .8)

        return server
    except Exception:
        server.stop()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", default=LEFT_CSV, help="Left result CSV, relative to result/ or absolute.")
    parser.add_argument("--right", default=RIGHT_CSV, help="Right result CSV, relative to result/ or absolute.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--spacing", type=float, default=1.8, help="Distance between robot bases in metres.")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    try:
        left, right = load_final_sample(args.left), load_final_sample(args.right)
        server = create_comparison(left, right, config=args.config, host=args.host,
                                   port=args.port, spacing=args.spacing)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    try:
        for side, sample in (("Left", left), ("Right", right)):
            print(f"{side}: {sample.path} | sample {sample.index}, {sample.elapsed:.3f} s "
                  f"| arm error norm {sample.arm_error_norm:.3f} deg")
        print(f"Open http://{args.host}:{server.get_port()} — Ctrl+C to exit.", flush=True)
        while True:
            time.sleep(1.)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
