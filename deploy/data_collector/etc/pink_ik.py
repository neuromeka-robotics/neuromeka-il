"""Robot-independent, fixed-base Pink IK and an interactive Viser playground.

Joint positions are named scalars in radians (revolute/continuous) or metres
(prismatic). Targets are Pinocchio SE3 transforms in the URDF base frame.
"""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml


class PinkIK:
    """Solve frame targets using selected joints, holding other joints fixed.

    Supports scalar revolute, continuous, and prismatic joints. Mimic mechanisms
    can be positioned but are held fixed during IK; their drivers cannot be
    selected as IK joints. No collision constraints are applied.
    """

    def __init__(self, urdf_path: str | Path, joint_names, settings=None):
        import pinocchio as pin
        import pink
        from pink.exceptions import NoSolutionFound
        from pink.limits import ConfigurationLimit
        from pink.tasks import FrameTask, PostureTask
        from qpsolvers import available_solvers

        self.pin = pin
        self._pink = pink
        self._no_solution_found = NoSolutionFound
        self._configuration_limit = ConfigurationLimit
        self._frame_task_type = FrameTask
        self._posture_task_type = PostureTask
        self.settings = {
            "solver": "daqp", "position_cost": 1., "orientation_cost": 1.,
            "posture_cost": 1e-4, "lm_damping": 1e-6, "damping": 1e-8,
            "position_tolerance_m": 1e-4, "orientation_tolerance_rad": 1e-3,
            "max_iterations": 100, "integration_dt": 1.,
        }
        self.settings.update(settings or {})
        for key, value in self.settings.items():
            if key == "solver":
                continue
            if not np.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"Pink IK {key} must be finite and non-negative")
        for key in ("max_iterations", "integration_dt", "position_tolerance_m",
                    "orientation_tolerance_rad"):
            if float(self.settings[key]) <= 0:
                raise ValueError(f"Pink IK {key} must be positive")
        if self.settings["solver"] not in available_solvers:
            raise ValueError(
                f"QP solver {self.settings['solver']!r} is not installed; "
                f"available solvers: {available_solvers}")

        urdf_path = Path(urdf_path).expanduser().resolve()
        if not urdf_path.is_file():
            raise FileNotFoundError(f"Pink IK URDF does not exist: {urdf_path}")
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.joint_names = tuple(joint_names)
        self._model_names = tuple(self.model.names)[1:]
        for name in self._model_names:
            joint = self.model.joints[self.model.getJointId(name)]
            if joint.nv != 1 or joint.nq not in (1, 2):
                raise ValueError(f"Joint {name!r} must be revolute, continuous, or prismatic")
        self._validate_joint_names(self.joint_names)
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("Select at least one IK joint, with no duplicate names")

        self._mimics = {}
        for joint in ET.parse(urdf_path).getroot().findall("joint"):
            mimic = joint.find("mimic")
            if mimic is not None:
                self._mimics[joint.attrib["name"]] = (
                    mimic.attrib["joint"], float(mimic.get("multiplier", 1.)),
                    float(mimic.get("offset", 0.)))
        mimic_joints = set(self._mimics) | {value[0] for value in self._mimics.values()}
        if mimic_joints.intersection(self.joint_names):
            raise ValueError("Mimic joints and their drivers must be excluded from IK joints")

    def _validate_joint_names(self, names):
        unknown = set(names) - set(self._model_names)
        if unknown:
            raise ValueError(f"Unknown URDF joints: {sorted(unknown)}")

    def validate_frames(self, frames):
        for frame in frames:
            if not self.model.existFrame(frame):
                raise ValueError(f"URDF has no frame named {frame!r}")

    def _joint_values(self, model, q):
        values = {}
        for name in tuple(model.names)[1:]:
            joint = model.joints[model.getJointId(name)]
            values[name] = float(q[joint.idx_q]) if joint.nq == 1 else float(
                np.arctan2(q[joint.idx_q + 1], q[joint.idx_q]))
        return values

    def _complete_positions(self, positions):
        self._validate_joint_names(positions)
        values = self._joint_values(self.model, self.pin.neutral(self.model))
        values.update({name: float(value) for name, value in positions.items()})
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError("Joint positions must be finite")

        def mimic_value(name, ancestors=()):
            if name not in self._mimics:
                return values[name]
            if name in ancestors:
                raise ValueError(f"Cyclic mimic joint: {name}")
            source, multiplier, offset = self._mimics[name]
            return multiplier * mimic_value(source, (*ancestors, name)) + offset

        for name in self._mimics:
            values[name] = mimic_value(name)
        return values

    def _configuration(self, model, positions):
        q = self.pin.neutral(model)
        for name in tuple(model.names)[1:]:
            joint = model.joints[model.getJointId(name)]
            value = positions[name]
            if joint.nq == 1:
                q[joint.idx_q] = value
            else:
                q[joint.idx_q:joint.idx_q + 2] = (np.cos(value), np.sin(value))
        return q

    def forward(self, joint_positions, frames):
        """Return frame-name → SE3 poses, expressed in the URDF base frame."""
        frames = tuple(frames)
        self.validate_frames(frames)
        q = self._configuration(self.model, self._complete_positions(joint_positions))
        data = self.model.createData()
        self.pin.framesForwardKinematics(self.model, data, q)
        return {frame: data.oMf[self.model.getFrameId(frame)].copy() for frame in frames}

    def solve(self, targets, joint_positions, locked_joints=(), joint_reference=None):
        """Return convergence, named joint positions, and per-frame errors.

        On non-convergence, return the last valid iterate. Locks use the supplied
        reference, or the initial joint positions when no reference is supplied.
        """
        if not targets:
            raise ValueError("At least one task target is required")
        self.validate_frames(targets)
        unknown_locks = set(locked_joints) - set(self.joint_names)
        if unknown_locks:
            raise ValueError(f"Unknown IK joints to lock: {sorted(unknown_locks)}")
        values = self._complete_positions(joint_positions)
        if joint_reference is not None:
            for name in locked_joints:
                values[name] = float(joint_reference[name])
            values = self._complete_positions(values)
        full_q = self._configuration(self.model, values)
        fixed = (set(self._model_names) - set(self.joint_names)) | set(locked_joints)
        model = self.model
        if fixed:
            model = self.pin.buildReducedModel(
                model, [model.getJointId(name) for name in sorted(fixed)], full_q)
        initial_q = self._configuration(model, values)
        configuration = self._pink.Configuration(
            model, model.createData(), initial_q, copy_data=False)
        frame_tasks = {}
        for frame, target in targets.items():
            if (not np.all(np.isfinite(target.homogeneous))
                    or not np.allclose(target.rotation.T @ target.rotation, np.eye(3), atol=1e-6)
                    or not np.isclose(np.linalg.det(target.rotation), 1.)):
                raise ValueError(f"Target for {frame!r} must be a finite rigid transform")
            task = self._frame_task_type(
                frame, position_cost=float(self.settings["position_cost"]),
                orientation_cost=float(self.settings["orientation_cost"]),
                lm_damping=float(self.settings["lm_damping"]))
            task.set_target(target)
            frame_tasks[frame] = task
        tasks = list(frame_tasks.values())
        if float(self.settings["posture_cost"]) > 0:
            posture = self._posture_task_type(cost=float(self.settings["posture_cost"]))
            posture.set_target(initial_q)
            tasks.append(posture)

        max_iterations = int(self.settings["max_iterations"])
        dt = float(self.settings["integration_dt"])
        for iterations in range(max_iterations + 1):
            errors = {}
            for frame, task in frame_tasks.items():
                error = task.compute_error(configuration)
                errors[frame] = {
                    "position_error_m": float(np.linalg.norm(error[:3])),
                    "orientation_error_rad": float(np.linalg.norm(error[3:])),
                }
            failed = [frame for frame, error in errors.items() if (
                error["position_error_m"] >= float(self.settings["position_tolerance_m"])
                or error["orientation_error_rad"] >= float(self.settings["orientation_tolerance_rad"]))]
            if not failed or iterations == max_iterations or model.nv == 0:
                break
            try:
                velocity = self._pink.solve_ik(
                    configuration, tasks=tasks, dt=dt, solver=self.settings["solver"],
                    damping=float(self.settings["damping"]),
                    limits=(self._configuration_limit(model),))
            except self._no_solution_found:
                break
            if not np.all(np.isfinite(velocity)):
                break
            configuration.integrate_inplace(velocity, dt)

        values.update(self._joint_values(model, configuration.q))
        result = {"success": not failed, "joint_positions": values,
                  "iterations": iterations, "per_frame_errors": errors}
        if failed:
            result.update(failed_frames=failed, error=(
                f"did not converge after {iterations} iterations; " + ", ".join(
                    f"{frame}: pos={error['position_error_m']:.6g}m "
                    f"ori={error['orientation_error_rad']:.6g}rad"
                    for frame, error in errors.items())))
        return result


class IKPlayground:
    """Drag frame targets, adjust joint positions, and toggle IK joint locks."""

    def __init__(self, solver, urdf_path, config, host="127.0.0.1", port=8080):
        from functools import partial
        from threading import RLock
        import viser
        import viser.transforms as tf
        import yourdfpy
        from viser.extras import ViserUrdf

        self.solver = solver
        self._tf = tf
        self._lock = RLock()
        self.frames = tuple(config["frames"])
        if not self.frames or len(set(self.frames)) != len(self.frames):
            raise ValueError("Configure at least one frame, with no duplicates")
        solver.validate_frames(self.frames)
        self.initial = solver._complete_positions(config.get("initial_positions", {}))
        self.positions = dict(self.initial)
        initial_locks = set(config.get("locked_joints", ()))
        if initial_locks - set(solver.joint_names):
            raise ValueError("locked_joints must be selected IK joints")
        urdf_path = Path(urdf_path)
        urdf = yourdfpy.URDF.load(
            urdf_path, load_collision_meshes=False,
            filename_handler=partial(yourdfpy.filename_handler_magic, dir=urdf_path.parent))
        self.server = viser.ViserServer(host=host, port=port, label="Pink IK playground")
        try:
            self.robot = ViserUrdf(self.server, urdf, root_node_name="/robot")
        except Exception:
            self.server.stop()
            raise
        self.server.scene.set_up_direction("+z")
        self.server.gui.add_markdown(
            "Drag a target to solve IK. Joint sliders set the pose and re-anchor all targets. "
            "Angles are radians; distances are metres. No collision checking.")
        self.status = self.server.gui.add_markdown("Ready.")
        self.reset_button = self.server.gui.add_button("Reset pose and targets")
        self.reset_button.on_click(lambda _: self.reset())
        self.targets = {}
        self.enabled = {}
        with self.server.gui.add_folder("Target frames"):
            for frame in self.frames:
                self.enabled[frame] = self.server.gui.add_checkbox(frame, initial_value=True)
                target = self.server.scene.add_transform_controls(
                    f"/targets/{frame}", scale=0.15)
                self.server.scene.add_label(f"/targets/{frame}/label", frame)
                target.on_update(self._on_target_update)
                self.targets[frame] = target
        self.locks = {}
        with self.server.gui.add_folder("Lock IK joints", expand_by_default=False):
            for name in solver.joint_names:
                self.locks[name] = self.server.gui.add_checkbox(
                    name, initial_value=name in initial_locks)
        self.sliders = {}
        with self.server.gui.add_folder("Joint positions", expand_by_default=False):
            for name, (lower, upper) in self.robot.get_actuated_joint_limits().items():
                lower = -np.pi if lower is None or not np.isfinite(lower) else lower
                upper = np.pi if upper is None or not np.isfinite(upper) else upper
                slider = self.server.gui.add_slider(
                    name, min=lower, max=upper, step=0.001,
                    initial_value=self.positions[name])
                slider.on_update(lambda event, name=name: self._on_joint_update(event, name))
                self.sliders[name] = slider
        self._show_pose(reanchor=True)
        bounds = urdf.scene.bounds
        if bounds is None:
            bounds = np.array([[-0.5, -0.5, 0.], [0.5, 0.5, 1.]])
        center = bounds.mean(axis=0)
        extent = max(float(np.max(bounds[1] - bounds[0])), 0.5)
        self.server.scene.add_grid(
            "/ground", width=extent * 2, height=extent * 2,
            position=(center[0], center[1], bounds[0, 2]))

        @self.server.on_client_connect
        def set_camera(client):
            client.camera.position = center + extent * np.array([1.2, 1.2, 0.8])
            client.camera.look_at = center

    def _show_pose(self, reanchor=False):
        with self.server.atomic():
            self.robot.update_cfg(np.array([
                self.positions[name] for name in self.robot.get_actuated_joint_names()]))
            for name, slider in self.sliders.items():
                slider.value = self.positions[name]
            if reanchor:
                for frame, pose in self.solver.forward(self.positions, self.frames).items():
                    self.targets[frame].position = pose.translation
                    self.targets[frame].wxyz = self._tf.SO3.from_matrix(pose.rotation).wxyz

    def reset(self):
        with self._lock:
            self.positions = dict(self.initial)
            self._show_pose(reanchor=True)
            self.status.content = "Reset to initial pose."

    def _on_joint_update(self, event, name):
        # Programmatic slider updates must not launch another solve/re-anchor.
        if event.client is None:
            return
        with self._lock:
            self.positions[name] = float(self.sliders[name].value)
            self.positions = self.solver._complete_positions(self.positions)
            self._show_pose(reanchor=True)
            self.status.content = "Joint pose changed; targets re-anchored."

    def _on_target_update(self, event):
        if event.client is None:
            return
        with self._lock:
            targets = {
                frame: self.solver.pin.SE3(
                    self._tf.SO3(target.wxyz).as_matrix(), np.asarray(target.position))
                for frame, target in self.targets.items() if self.enabled[frame].value}
            if not targets:
                self.status.content = "Enable at least one target frame."
                return
            locked = [name for name, checkbox in self.locks.items() if checkbox.value]
            try:
                result = self.solver.solve(targets, self.positions, locked_joints=locked)
            except (ValueError, RuntimeError) as exc:
                self.status.content = f"IK error: {exc}"
                return
            self.positions = result["joint_positions"]
            self._show_pose()
            state = "Converged" if result["success"] else "Not converged (last iterate shown)"
            self.status.content = f"**{state}** — {result['iterations']} iterations\n\n" + "\n\n".join(
                f"{frame}: {error['position_error_m'] * 1000:.3f} mm, "
                f"{np.rad2deg(error['orientation_error_rad']):.3f}°"
                for frame, error in result["per_frame_errors"].items())


def main():
    import time

    # EIR example: edit these settings to try another model or set of targets.
    config_path = (
        Path(__file__).resolve().parents[4]
        / "robot_interface/robot_interface/config/eir.yaml"
    )
    model = "gripper"  # EIR_w_gripper.urdf
    frames = ("left_tcp", "right_tcp")
    locked_joints = ("Joint_L0", "Joint_L1", "Joint_L2_U", "Joint_L3_U")
    host, port = "127.0.0.1", 8080

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    urdf_path = config_path.parent.parent / config["robot"]["urdf_paths"][model]
    solver = PinkIK(urdf_path, config["joints"], config["ik"])
    initial_positions = {
        gripper["joint_name"]: gripper["open_position"]
        for gripper in config.get("grippers", {}).values()
        if solver.model.existJointName(gripper["joint_name"])}
    initial_positions.update(config.get("default_pose", {}))
    playground = IKPlayground(solver, urdf_path, {
        "frames": frames,
        "locked_joints": locked_joints,
        "initial_positions": initial_positions,
    }, host, port)
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        playground.server.stop()


if __name__ == "__main__":
    main()
