"""Read-only EIR measured, raw, and sent-command overlays in a separate process."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

from data_collector.pink_ik import DCP_ACTIVE_JOINT_NAMES


def joint_values(q):
    """Map IndyDCP q22 degrees to URDF names, ignoring the four dummy joints."""
    q = np.asarray(q, dtype=float)
    if q.shape != (22,) or not np.isfinite(q).all():
        raise ValueError("Viewer expects 22 finite IndyDCP joint values in degrees")
    return dict(zip(DCP_ACTIVE_JOINT_NAMES, np.deg2rad(q[:18])))


class CollectionVisualizer:
    """A latest-sample mailbox: publishing never waits for the renderer.

    The child has no robot connection. Only small numeric snapshots cross the
    process boundary; URDF loading, FK and browser traffic stay in the child.
    """

    def __init__(self, config_path, *, host="127.0.0.1", port=8080, hz=15.):
        if not np.isfinite(hz) or hz <= 0:
            raise ValueError("Viser update frequency must be positive and finite")
        if not 1 <= port <= 65535:
            raise ValueError("Viser port must be between 1 and 65535")
        context = mp.get_context("spawn")
        # timestamp, command kind (0/1/2), measured q22, sent q22,
        # three task-target validity flags, three mm/degree task poses,
        # raw-command validity flag, raw CSV q22.
        self._sample = context.RawArray("d", 90)
        self._lock = context.Lock()
        self._stop = context.Event()
        self._command = None
        self._raw_command = None
        self._task_commands = {}
        self._disabled = False
        self._process = context.Process(
            target=_run_viewer,
            args=(str(config_path), host, port, hz,
                  self._sample, self._lock, self._stop),
            name="collection-viser",
            daemon=True,
        )
        self._process.start()

    def publish(self, measured, command=None, *, raw_command=None,
                task_commands=None, reset=False):
        """Copy an existing state and, optionally, the command actually sent.

        raw_command is the selected CSV target before policy projection.
        Omitted commands retain their last targets, including dry-run holds.
        Visualization failure disables publishing without interrupting control.
        """
        if self._disabled:
            return
        try:
            if not self._process.is_alive():
                raise RuntimeError("renderer exited")
            joint_values(measured)
            if reset:
                self._command = None
                self._raw_command = None
                self._task_commands = {}
            if command is not None:
                joint_values(command)
                self._command = list(command)
                self._task_commands = {}
            if raw_command is not None:
                joint_values(raw_command)
                self._raw_command = list(raw_command)
            if task_commands is not None:
                for arm, pose in task_commands.items():
                    pose = np.asarray(pose, dtype=float)
                    if arm not in (0, 1, 2) or pose.shape != (6,) or not np.isfinite(pose).all():
                        raise ValueError("Viewer expects a finite 6D pose for arm 0, 1 or 2")
                    self._task_commands[arm] = pose.tolist()
                self._command = None
                self._raw_command = None
            kind = 2 if self._task_commands else int(self._command is not None)
            values = [time.monotonic(), kind]
            values.extend(measured)
            values.extend(self._command if self._command is not None else measured)
            values.extend(float(arm in self._task_commands) for arm in range(3))
            for arm in range(3):
                values.extend(self._task_commands.get(arm, [0.] * 6))
            values.append(float(self._raw_command is not None))
            values.extend(self._raw_command if self._raw_command is not None else measured)
            if self._lock.acquire(False):
                try:
                    self._sample[:] = values
                finally:
                    self._lock.release()
        except Exception as exc:
            self._disabled = True
            print(f"[viser] Visualization disabled: {exc}", flush=True)

    def close(self):
        self._disabled = True
        self._stop.set()
        self._process.join(timeout=2.)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.)


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


def _make_scene(config_path, host, port):
    import viser
    from viser.extras import ViserUrdf
    from yourdfpy import URDF

    from helper.eir_pink_ik_visualizer import (
        _has_unfetched_lfs_meshes, _load_eir_settings,
    )

    settings = _load_eir_settings(Path(config_path).expanduser().resolve())
    urdf_path = settings["visual_urdf_path"]
    collisions = _has_unfetched_lfs_meshes(urdf_path)
    if collisions:
        print("[viser] Visual meshes are LFS pointers; measured mesh unavailable. "
              "Using collision geometry for command shadows.",
              flush=True)

    def load_robot():
        return URDF.load(
            str(urdf_path),
            filename_handler=lambda fname: str(
                urdf_path.parent / Path(fname.removeprefix("package://"))),
            load_meshes=not collisions,
            build_scene_graph=not collisions,
            load_collision_meshes=False,
            build_collision_scene_graph=collisions,
        )

    # Independent URDF instances prevent one overlay's FK from changing the other.
    models = [load_robot(), load_robot(), load_robot()]
    for model in models:
        if not set(DCP_ACTIVE_JOINT_NAMES).issubset(model.joint_map):
            raise ValueError("Viewer URDF is missing EIR joints")
    for model in models[1:]:
        _keep_arm_geometry(model)
    server = viser.ViserServer(host=host, port=port)
    try:
        server.scene.add_grid("/grid", width=3., height=3.)
        actual_root = server.scene.add_frame("/measured", show_axes=False, visible=False)
        target_root = server.scene.add_frame("/commanded", show_axes=False, visible=False)
        raw_root = server.scene.add_frame("/raw_command", show_axes=False, visible=False)
        robots = []
        for model, root, color in zip(
                models, ("/measured", "/commanded", "/raw_command"),
                (None, (1., 0.55, 0.1, 0.35),
                 (0.15, 0.9, 0.3, 0.35))):
            robots.append(ViserUrdf(
                server, model, root_node_name=root,
                load_meshes=not collisions,
                load_collision_meshes=collisions and root != "/measured",
                mesh_color_override=color, collision_mesh_color_override=color,
            ))
        server.gui.add_markdown(
            "**Visual mesh:** measured robot (original materials)  \n"
            "**Green arms:** raw CSV command (before projection)  \n"
            "**Orange arms:** sent command (after projection in RL mode)  \n"
            "Joint mode: exact sent joints. Task mode: local IK estimate; "
            "axes show exact sent Cartesian targets.  \n"
            "Read-only view. Joint values and errors are in degrees.")
        show_actual = server.gui.add_checkbox("Show measured", True)
        show_target = server.gui.add_checkbox("Show commanded", True)
        show_raw = server.gui.add_checkbox("Show raw CSV command", True)
        status = server.gui.add_markdown("Waiting for control samples. Start execution in the terminal.")
        with server.gui.add_folder("Joint tracking errors"):
            errors = server.gui.add_markdown("No command sent yet.")

        @server.on_client_connect
        def _(client):
            client.camera.position = (2.5, -2.5, 1.8)
            client.camera.look_at = (0., 0., 0.7)

        print(f"[viser] Open http://{host}:{server.get_port()}", flush=True)
        return (server, robots, actual_root, target_root,
                show_actual, show_target, status, errors, raw_root, show_raw)
    except Exception:
        server.stop()
        raise


def _run_viewer(config_path, host, port, hz, sample, lock, stop):
    server = None
    try:
        (server, robots, actual_root, target_root, show_actual,
         show_target, status, errors, raw_root, show_raw) = _make_scene(config_path, host, port)
        previous_stamp = 0.
        next_status = 0.
        solver = None
        target_frames = {}
        estimate_ok = True
        commanded = None
        while not stop.wait(1. / hz):
            if not lock.acquire(False):
                continue
            try:
                snapshot = list(sample)
            finally:
                lock.release()
            stamp, kind = snapshot[:2]
            if not stamp:
                continue
            measured = snapshot[2:24]
            if stamp != previous_stamp:
                commanded = snapshot[24:46]
                if kind == 2:
                    # Visualization-only IK: no extra RPCs and no feedback into control.
                    from data_collector.pink_ik import PinkTeleopIK
                    from scipy.spatial.transform import Rotation

                    if solver is None:
                        solver = PinkTeleopIK(config_path)
                        solver._settings = dict(solver._settings)
                        solver._settings["max_iterations"] = min(
                            20, int(solver._settings["max_iterations"]))
                    targets = {
                        arm: snapshot[49 + 6 * arm:55 + 6 * arm]
                        for arm in range(3) if snapshot[46 + arm]
                    }
                    result = solver.solve_multi(
                        targets=targets, init_jpos=measured,
                        lock_non_selected_joints=True)
                    commanded = list(result["jpos"]) + [0.] * 4
                    estimate_ok = result["success"]
                    for arm, pose in targets.items():
                        if arm not in target_frames:
                            target_frames[arm] = server.scene.add_frame(
                                f"/task_targets/arm_{arm}",
                                axes_length=0.12, axes_radius=0.004)
                        target_frames[arm].position = np.asarray(pose[:3]) / 1000.
                        xyzw = Rotation.from_euler("xyz", pose[3:], degrees=True).as_quat()
                        target_frames[arm].wxyz = xyzw[[3, 0, 1, 2]]
            now = time.monotonic()
            age = now - stamp
            with server.atomic():
                actual_root.visible = show_actual.value
                target_root.visible = show_target.value and bool(kind)
                raw_root.visible = show_raw.value and bool(snapshot[67])
                for arm, frame in target_frames.items():
                    frame.visible = show_target.value and kind == 2 and bool(snapshot[46 + arm])
                if stamp != previous_stamp:
                    robots[0].update_cfg(joint_values(measured))
                    robots[1].update_cfg(joint_values(commanded))
                    if snapshot[67]:
                        robots[2].update_cfg(joint_values(snapshot[68:90]))
                    previous_stamp = stamp
                if now < next_status:
                    continue
                next_status = now + 0.2
                state = "LIVE" if age < 0.5 else "STALE / control inactive"
                text = f"**{state}** — sample age {age:.2f} s"
                if kind:
                    if kind == 2:
                        text += "  \n**Shadow: local IK estimate, not the controller's joint target.**"
                        if not estimate_ok:
                            text += "  \nIK did not converge within the viewer's iteration budget."
                    delta = np.asarray(commanded[:18]) - measured[:18]
                    worst = int(np.argmax(np.abs(delta)))
                    metric = "estimated difference" if kind == 2 else "error"
                    text += (f"  \nMax {metric}: **{abs(delta[worst]):.2f}°** "
                             f"({DCP_ACTIVE_JOINT_NAMES[worst]})"
                             f"  \nRMS {metric}: **{np.sqrt(np.mean(delta**2)):.2f}°**")
                    errors.content = (
                        ("| Joint | Measured | IK estimate | Difference |\n" if kind == 2
                         else "| Joint | Measured | Sent | Error |\n") +
                        "| --- | ---: | ---: | ---: |\n" + "\n".join(
                            f"| {name} | {q:.2f} | {cmd:.2f} | {err:+.2f} |"
                            for name, q, cmd, err in zip(
                                DCP_ACTIVE_JOINT_NAMES, measured, commanded, delta)))
                else:
                    text += "  \nWaiting for the first command."
                    errors.content = "No command sent in this run yet."
                status.content = text
    except Exception as exc:
        print(f"[viser] Viewer stopped: {exc}", flush=True)
    finally:
        if server is not None:
            server.stop()
