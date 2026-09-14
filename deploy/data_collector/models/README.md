# RL constraint models

This directory is a convenient optional location for exported ONNX actors. Large
model files should remain outside Git unless explicitly versioned.

Export a checkpoint with the Genesis environment:

```bash
/home/user/yunho/nrmk-genesis/.venv/bin/python deploy/export_rl_constraint.py \
  /path/to/run/model_N.pt \
  --output deploy/data_collector/models/dual_arm_plane.onnx
```

Then set the absolute ONNX path as `rl_constraint_model_path` in
`deploy/data_collector/config.py`, or as `RL_CONSTRAINT_MODEL_PATH` in the
open-loop box-lift configuration.
