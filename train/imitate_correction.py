"""Train nominal-command ACT on pre-normalized q/v residual chunks."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random
import shutil
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent / "nrmk_il/src"))

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from correction_config import DEFAULT_CONFIG, load_settings, require_empty

from nrmk_il.dataset.correction_dataset import CorrectionDataset, policy_features, select_normalization
from nrmk_il.policies import ACTConfig, ACTPolicy


def logging_context(settings, output, normalizer):
    """Opt-in W&B metrics; the context closes the run on success or failure."""
    options = settings["train"].get("wandb", {})
    if not options.get("enabled", False):
        return nullcontext()
    import wandb

    return wandb.init(
        project=options.get("project", "eir-correction"),
        entity=options.get("entity"),
        name=options.get("name") or output.name,
        mode=options.get("mode", "online"),
        dir=str(output.resolve()),
        config={**settings, "dataset": {
            "physics_dt_s": normalizer["physics_dt_s"],
            "joint_names": normalizer["joint_names"],
        }},
    )


def run_epoch(policy, loader, device, optimizer=None, grad_clip_norm=1.0):
    policy.train(optimizer is not None)
    total, count = 0., 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.set_grad_enabled(optimizer is not None):
            losses, _ = policy(batch)
            loss = losses["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite correction loss")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm, error_if_nonfinite=True)
                optimizer.step()
        weight = int((~batch["is_pad"]).sum())
        total += loss.item() * weight
        count += weight
    return total / count if count else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run", help="Override train.run_name; all directories come from YAML")
    parser.add_argument("--device", help="Override train.device")
    args = parser.parse_args()
    settings, paths = load_settings(args.config, args.run)
    train = settings["train"]
    data, output = paths["processed"], paths["weights"]
    device = args.device or train.get("device", "cuda:0")
    train["device"] = device
    require_empty(output)
    epochs = train["epochs"]
    if epochs < 1:
        parser.error("epochs must be positive")
    seed = train.get("seed", 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    normalizer = json.loads((data / "normalization.json").read_text())
    state_keys, outputs = policy_features(settings["policy"])
    normalizer = select_normalization(normalizer, settings["policy"])
    config = ACTConfig(**settings["policy"])
    joints = len(normalizer["joint_names"])
    if (list(config.state_token_keys or []) != state_keys
            or config.input_shapes != {key: [joints] for key in state_keys}
            or config.output_shapes != {key: [joints] for key in outputs}):
        parser.error("Configuration must match the selected input/output keys and dataset joints")
    if config.input_normalization_modes or config.output_normalization_modes:
        parser.error("Processed correction data are already normalized; normalization modes must be empty")
    if config.temporal_ensemble_momentum is not None:
        parser.error("ACT temporal ensembling is not implemented; use null")
    if config.use_vae:
        parser.error("This deterministic correction trainer uses use_vae: false")
    training = CorrectionDataset(data, "train", config.chunk_size, state_keys=state_keys, action_keys=outputs,
                                 n_obs_steps=config.n_obs_steps)
    validation = CorrectionDataset(data, "validation", config.chunk_size, state_keys=state_keys, action_keys=outputs,
                                   n_obs_steps=config.n_obs_steps)
    if not len(training):
        parser.error("Training split is empty")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(data / "split.json", output / "split.json")
    (output / "normalization.json").write_text(json.dumps(normalizer, indent=2) + "\n")
    settings["train"]["epochs"] = epochs
    (output / "config.yaml").write_text(yaml.safe_dump(settings, sort_keys=False))
    loaders = [DataLoader(dataset, batch_size=train["batch_size"], shuffle=(i == 0),
                          num_workers=train.get("num_workers", 0))
               for i, dataset in enumerate((training, validation))]
    policy = ACTPolicy(config).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=train["lr"], weight_decay=train["weight_decay"])
    history, best = [], float("inf")
    with logging_context(settings, output, normalizer) as wandb_run:
        for epoch in range(epochs):
            training_loss = run_epoch(policy, loaders[0], device, optimizer, train.get("grad_clip_norm", 1.))
            validation_loss = run_epoch(policy, loaders[1], device)
            row = {"epoch": epoch + 1, "train_l1_normalized": training_loss,
                   "validation_l1_normalized": validation_loss}
            history.append(row)
            checkpoint = {"model": policy.state_dict(), "policy_config": settings["policy"],
                          "normalization": normalizer, "epoch": epoch + 1,
                          "optimizer": optimizer.state_dict(), "metrics": row}
            torch.save(checkpoint, output / "last.pt")
            score = validation_loss if validation_loss is not None else training_loss
            if score < best:
                best = score
                torch.save(checkpoint, output / "best.pt")
            (output / "metrics.json").write_text(json.dumps(history, indent=2) + "\n")
            print(json.dumps(row), flush=True)
            if wandb_run is not None:
                metrics = {"epoch": epoch + 1, "train/l1_normalized": training_loss,
                           "learning_rate": optimizer.param_groups[0]["lr"]}
                if validation_loss is not None:
                    metrics["validation/l1_normalized"] = validation_loss
                wandb_run.log(metrics, step=epoch + 1)
                best_key = "best_validation_l1_normalized" if validation_loss is not None else "best_train_l1_normalized"
                wandb_run.summary[best_key] = best


if __name__ == "__main__":
    main()
