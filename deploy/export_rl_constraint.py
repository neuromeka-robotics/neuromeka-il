#!/usr/bin/env python3
"""Export a trusted new dual_arm_plane checkpoint with its deployment contract.

Run with the nrmk-genesis Python environment. Only the resulting ONNX file is
needed by the deployment environment; simulation/model imports stay in this CLI.
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
from pathlib import Path
import xml.etree.ElementTree as ET


def deployment_contract(task, urdf):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from nrmk_genesis.envs.proprioception import compliant_observation_dim

    cfg = task.env
    if cfg.observe_constraint_normal or cfg.observe_palm_plane_distance:
        raise ValueError("Deployment requires an actor without privileged plane geometry inputs")
    root = ET.parse(urdf).getroot()
    references = []
    for side, sign in (("left", -1.), ("right", 1.)):
        collision = root.find(f"link[@name='{side}_gripper_base_link']/collision")
        origin = collision.find("origin")
        attrs = {} if origin is None else origin.attrib
        center = np.fromstring(attrs.get("xyz", "0 0 0"), sep=" ")
        rotation = Rotation.from_euler("xyz", np.fromstring(attrs.get("rpy", "0 0 0"), sep=" "))
        size = np.fromstring(collision.find("geometry/box").attrib["size"], sep=" ")
        references.append((center + rotation.apply([0., sign * size[1] / 2, 0.])).tolist())
    return {
        "interface": "dual_arm_plane",
        "policy_interface": cfg.policy_interface,
        "pose_frame": cfg.pose_frame,
        "orientation_representation": cfg.orientation_representation,
        "history_order": "term_major_offset_order",
        "joint_pos_history_offsets_steps": list(cfg.joint_pos_history_offsets_steps),
        "observe_joint_target_error_history": cfg.observe_joint_target_error_history,
        "include_previous_actions": cfg.include_previous_actions,
        "include_last_last_action": cfg.include_last_last_action,
        "observe_constraint_normal": cfg.observe_constraint_normal,
        "observe_palm_plane_distance": cfg.observe_palm_plane_distance,
        "joint_names": [n for n, c in task.robot.joint_cfg.items() if not c.locked],
        "num_obs": compliant_observation_dim(cfg, task.robot.num_actions),
        "control_dt": cfg.action_dt,
        "action_scale": cfg.action_scale,
        "action_semantics": "encoder_joint_delta_rad",
        "palm_reference_local_m": references,
        "applied_target_source": "controller_qdes",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--genesis-root", type=Path,
                        default=Path(__file__).resolve().parents[2] / "nrmk-genesis")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, help="override a relocated saved URDF path")
    args = parser.parse_args()
    sys.path[:0] = [str(args.genesis_root), str(args.genesis_root / "experiments")]
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from tensordict import TensorDict
    from rsl_rl.utils import resolve_callable
    from dual_arm_plane.config import validate_policy_interface

    checkpoint = args.checkpoint.resolve()
    config_file = checkpoint.parent / "cfgs.pkl"
    if not config_file.is_file():
        config_file = checkpoint.parent.parent / "cfgs.pkl"
    with config_file.open("rb") as stream:
        task, train = pickle.load(stream)
    validate_policy_interface(task)
    urdf = args.urdf or Path(task.robot.urdf_file)
    if not urdf.is_file():
        urdf = args.genesis_root / "robot_interface/robot_interface/assets/eir/urdf" / urdf.name
    contract = deployment_contract(task, urdf)
    obs = TensorDict({"policy": torch.zeros(1, contract["num_obs"])}, batch_size=[1])
    actor_config = copy.deepcopy(train["actor"])
    actor_type = resolve_callable(actor_config.pop("class_name"))
    actor = actor_type(obs, train["obs_groups"], "actor", len(contract["joint_names"]), **actor_config)
    actor.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["actor_state_dict"])
    actor.eval()
    model = actor.as_onnx(verbose=False).cpu().eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, model.get_dummy_inputs(), args.output,
                      opset_version=18, input_names=["obs"], output_names=["actions"],
                      dynamo=False, dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}})
    graph = onnx.load(args.output)
    contract["checkpoint"] = str(checkpoint)
    onnx.helper.set_model_props(graph, {"dual_arm_plane": json.dumps(contract)})
    onnx.checker.check_model(graph)
    onnx.save(graph, args.output)
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    samples = torch.randn(32, contract["num_obs"], generator=torch.Generator().manual_seed(1))
    if actor.obs_normalization:
        samples = samples * actor.obs_normalizer._std + actor.obs_normalizer._mean
    with torch.inference_mode():
        expected = actor(TensorDict({"policy": samples}, batch_size=[32])).numpy()
    actual, = session.run(["actions"], {"obs": samples.numpy()})
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
    print(f"Exported {args.output}: {contract['num_obs']} inputs at {1 / contract['control_dt']:g} Hz; "
          f"max PyTorch/ONNX action error {np.max(np.abs(actual - expected)):.3g}")


if __name__ == "__main__":
    main()
