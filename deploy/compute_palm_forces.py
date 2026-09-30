#!/usr/bin/env python3
"""Estimate palm forces from saved DCP torques using the EIR Pinocchio model."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from data_collector.pink_ik import DCP_ACTIVE_JOINT_NAMES
from helper.eir_pink_ik_visualizer import DEFAULT_CONFIG_PATH, _load_eir_settings


RESULT_DIR = Path(__file__).resolve().parent / "middle_level_controller/box_lift_open_loop/result"
FORCE_COLUMNS = (
    "tau_ext_force_left_norm_N",
    "tau_ext_force_right_norm_N",
    "tau_ext_force_mean_norm_N",
)
JTS_FORCE_COLUMNS = (
    "tau_jts_gravity_force_left_norm_N",
    "tau_jts_gravity_force_right_norm_N",
    "tau_jts_gravity_force_mean_norm_N",
)


class PalmForceEstimator:
    """Fit seven external joint torques to a six-component wrench at each TCP."""

    def __init__(self, urdf_path):
        import pinocchio as pin

        self.pin = pin
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        self.neutral = pin.neutral(self.model)
        if any(not self.model.existJointName(name) for name in DCP_ACTIVE_JOINT_NAMES):
            raise ValueError("The URDF must contain the 18 named DCP joints")
        joints = [self.model.joints[self.model.getJointId(name)]
                  for name in DCP_ACTIVE_JOINT_NAMES]
        if any(j.nq != 1 or j.nv != 1 for j in joints):
            raise ValueError("The URDF must contain the 18 scalar DCP joints")
        self.q_indices = [joint.idx_q for joint in joints]
        self.v_indices = [joint.idx_v for joint in joints]
        self.arms = []
        for side, indices in (("left", range(4, 11)), ("right", range(11, 18))):
            frame = self.model.getFrameId(f"{side}_tcp")
            if frame >= self.model.nframes:
                raise ValueError(f"Missing {side}_tcp in {urdf_path}")
            self.arms.append((frame, list(indices), [joints[i].idx_v for i in indices]))

    def gravity(self, q_deg):
        """Gravity torque in DCP joint order (Nm), using radians internally."""
        if not np.isfinite(q_deg).all():
            return np.full(len(self.q_indices), np.nan)
        q = self.neutral.copy()
        q[self.q_indices] = np.deg2rad(q_deg)
        return self.pin.computeGeneralizedGravity(
            self.model, self.data, q)[self.v_indices].copy()

    def estimate(self, q_deg, tau_ext):
        """Return left/right force norms in N; NaN marks unavailable arm torque."""
        result = np.full(2, np.nan)
        if not np.isfinite(q_deg).all():
            return result
        q = self.neutral.copy()
        q[self.q_indices] = np.deg2rad(q_deg)
        self.pin.computeJointJacobians(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)
        for arm, (frame, indices, velocity_indices) in enumerate(self.arms):
            torque = tau_ext[indices]
            if not np.isfinite(torque).all():
                continue
            jacobian = self.pin.getFrameJacobian(
                self.model, self.data, frame, self.pin.LOCAL_WORLD_ALIGNED)[:, velocity_indices]
            wrench = np.linalg.lstsq(jacobian.T, torque, rcond=None)[0]
            result[arm] = np.linalg.norm(wrench[:3])
        return result


def _number(value):
    return float(value) if value is not None and value.strip() else np.nan


def _read_csv(csv_path):
    path = Path(csv_path).expanduser()
    if not path.is_absolute() and not path.is_file():
        path = RESULT_DIR / path
    path = path.resolve()
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        header = reader.fieldnames or []
        rows = list(reader)
    if len(set(header)) != len(header):
        raise ValueError(f"{path}: duplicate CSV columns")
    if not rows or "elapsed_time_s" not in header:
        raise ValueError(f"{path}: expected replay samples and elapsed_time_s")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"{path}: a row does not match the CSV header")
    elapsed = np.array([_number(row["elapsed_time_s"]) for row in rows])
    if not np.isfinite(elapsed).all():
        raise ValueError(f"{path}: invalid elapsed times")
    return path, header, rows, elapsed


def _torque_samples(header, rows, source):
    q_columns = [f"q_{name}_deg" for name in DCP_ACTIVE_JOINT_NAMES]
    torque_columns = [f"{source}_{name}" for name in DCP_ACTIVE_JOINT_NAMES]
    if any(name not in header for name in q_columns + torque_columns[4:18]):
        raise ValueError(
            f"Missing measured-joint/{source} columns. Record a new run with torque "
            "logging; old joint-error logs cannot be converted.")
    positions = np.array([[_number(row[name]) for name in q_columns] for row in rows])
    torques = np.array([[_number(row.get(name, "")) for name in torque_columns] for row in rows])
    return positions, torques


def _sensor_bias(estimator, bias_csv, sensor_sign):
    """Mean signed sensor-minus-gravity residual in a stationary no-contact log."""
    _, header, rows, _ = _read_csv(bias_csv)
    positions, torques = _torque_samples(header, rows, "tau_jts")
    residuals = sensor_sign * torques - np.array([estimator.gravity(q) for q in positions])
    bias = np.zeros(len(DCP_ACTIVE_JOINT_NAMES))
    for joint in range(4, 18):
        valid = np.isfinite(residuals[:, joint])
        if not valid.any():
            raise ValueError(f"No usable bias samples for {DCP_ACTIVE_JOINT_NAMES[joint]}")
        bias[joint] = residuals[valid, joint].mean()
    return bias


def process_csv(csv_path, *, config=DEFAULT_CONFIG_PATH, torque_source="tau_jts",
                bias_csv=None, sensor_sign=1):
    if torque_source not in ("tau_jts", "tau_ext"):
        raise ValueError("torque_source must be tau_jts or tau_ext")
    if sensor_sign not in (-1, 1):
        raise ValueError("sensor_sign must be +1 or -1")
    if torque_source == "tau_ext" and (bias_csv is not None or sensor_sign != 1):
        raise ValueError("Sensor bias/sign options apply only to tau_jts")
    path, header, rows, elapsed = _read_csv(csv_path)
    force_columns = JTS_FORCE_COLUMNS if torque_source == "tau_jts" else FORCE_COLUMNS
    prefix = "tau_jts_gravity" if torque_source == "tau_jts" else "tau_ext"

    recompute = False
    if any(name in header for name in force_columns):
        try:
            answer = input(f"{path}: {prefix} force columns already exist. Recompute and replace them? [y/N] ")
        except EOFError:
            answer = ""
        recompute = answer.strip().lower() == "y"
        if recompute:
            # Remove only in memory. Keep the original file until recomputation
            # succeeds and the complete replacement CSV is ready.
            header = [name for name in header if name not in force_columns]
            for row in rows:
                for name in force_columns:
                    row.pop(name, None)

    missing = [name for name in force_columns if name not in header]
    if any(name in missing for name in force_columns[:2]):
        positions, torques = _torque_samples(header, rows, torque_source)
        settings = _load_eir_settings(Path(config).expanduser().resolve())
        estimator = PalmForceEstimator(settings["kinematics_urdf_path"])
        bias = np.zeros(len(DCP_ACTIVE_JOINT_NAMES))
        if torque_source == "tau_jts":
            if bias_csv is not None:
                bias = _sensor_bias(estimator, bias_csv, sensor_sign)
                print(f"Sensor bias from stationary no-contact CSV: {bias_csv}")
            else:
                print("Sensor bias assumed zero (uncalibrated estimate); use --bias-csv for calibration.")
            print(f"tau_jts assumed Nm, sensor sign {sensor_sign:+d}; subtracting URDF gravity only. "
                  "Moving samples also contain uncompensated dynamics.")
        estimates = []
        for q, torque in zip(positions, torques):
            if torque_source == "tau_jts":
                torque = sensor_sign * torque - estimator.gravity(q) - bias
            estimates.append(estimator.estimate(q, torque))
        estimates = np.asarray(estimates)
        if not np.isfinite(estimates).any():
            raise ValueError(f"{path}: no samples with usable {torque_source}; CSV left unchanged")
        for arm, name in enumerate(force_columns[:2]):
            if name in missing:
                for row, value in zip(rows, estimates[:, arm]):
                    row[name] = repr(float(value)) if np.isfinite(value) else ""
    if force_columns[2] in missing:
        for row in rows:
            mean = np.mean([_number(row[name]) for name in force_columns[:2]])
            row[force_columns[2]] = repr(float(mean)) if np.isfinite(mean) else ""

    forces = np.array([[_number(row[name]) for name in force_columns] for row in rows])
    if not np.isfinite(forces).any():
        raise ValueError(f"{path}: no finite force estimates to plot")
    if missing:
        # Replace only after a complete write, preserving all original columns.
        temporary = None
        try:
            with NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", newline="",
                                    encoding="utf-8", delete=False) as target:
                temporary = Path(target.name)
                writer = csv.DictWriter(target, fieldnames=header + missing)
                writer.writeheader()
                writer.writerows(rows)
            temporary.chmod(path.stat().st_mode & 0o777)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        action = "Recomputed and replaced" if recompute else "Added"
        print(f"{action} {', '.join(missing)} in {path}")
    else:
        print(f"Force columns already exist; CSV unchanged: {path}")

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(11, 5), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    for index, label in enumerate(("Left palm", "Right palm", "Two-palm mean")):
        axis.plot(elapsed, forces[:, index], label=label)
    axis.set(xlabel="Elapsed time (s)", ylabel="Estimated force norm (N)",
             title=f"{'tau_jts − gravity (quasi-static)' if torque_source == 'tau_jts' else 'DCP tau_ext'} force estimate\n{path.stem}")
    axis.grid(True, alpha=.3)
    axis.legend()
    plot_path = path.with_name(f"{path.stem}_{prefix}_force_norm.png")
    figure.savefig(plot_path, dpi=150)
    print(f"Saved {plot_path}")
    for index, label in enumerate(("Left", "Right", "Two-palm mean")):
        valid = np.isfinite(forces[:, index])
        mean = float(forces[valid, index].mean()) if valid.any() else float("nan")
        final = valid & (elapsed >= elapsed.max() - 1.)
        final_mean = float(forces[final, index].mean()) if final.any() else float("nan")
        print(f"{label}: {mean:.6g} N average ({valid.sum()}/{len(rows)} valid samples); "
              f"final 1 s: {final_mean:.6g} N ({final.sum()} valid samples)")
    return plot_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", help="CSV paths, or paths relative to box_lift_open_loop/result/")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="EIR URDF configuration")
    parser.add_argument("--torque-source", choices=("tau_jts", "tau_ext"), default="tau_jts",
                        help="Default: tau_jts with Pinocchio gravity compensation")
    parser.add_argument("--bias-csv", type=Path,
                        help="Stationary no-contact CSV used to calibrate tau_jts bias (all rows)")
    parser.add_argument("--sensor-sign", type=int, choices=(-1, 1), default=1,
                        help="Sign converting tau_jts to the URDF joint convention (default: +1)")
    args = parser.parse_args()
    failed = False
    for path in args.csv:
        try:
            process_csv(path, config=args.config, torque_source=args.torque_source,
                        bias_csv=args.bias_csv, sensor_sign=args.sensor_sign)
        except (OSError, ValueError) as exc:
            print(f"Failed: {exc}")
            failed = True
    raise SystemExit(int(failed))


if __name__ == "__main__":
    main()
