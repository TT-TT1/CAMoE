#!/usr/bin/env python3
"""Train and evaluate CAMoE on one dataset/setting/seed."""

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np
import torch
from torch import Tensor

from camoe import CAMoE, CAMoEConfig, CAMoELoss, sentiment_metrics
from camoe.data import make_dataloaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--feature-path", type=Path, required=True)
    parser.add_argument("--dataset", choices=("mosi", "mosei"), required=True)
    parser.add_argument("--setting", choices=("aligned", "unaligned"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def load_experiment_config(path: Path, dataset: str, setting: str) -> Tuple[CAMoEConfig, Dict]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    model_values = dict(raw["model"])
    model_values.update(raw["datasets"][dataset][setting].get("model_overrides", {}))
    return CAMoEConfig.from_dict(model_values), dict(raw["training"])


def move_batch(batch: Mapping[str, object], device: torch.device) -> Dict[str, object]:
    return {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def model_forward(model: CAMoE, batch: Mapping[str, object]) -> Dict[str, object]:
    return model(
        batch["U_T"],
        batch["U_A"],
        batch["U_V"],
        batch["lengths_T"],
        batch["lengths_A"],
        batch["lengths_V"],
    )


def train_epoch(
    model: CAMoE,
    objective: CAMoELoss,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    loss_sum = 0.0
    sample_count = 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = model_forward(model, batch)
        losses = objective(output, batch["y"])
        losses["L_total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        batch_size = int(batch["y"].numel())
        loss_sum += float(losses["L_total"].detach()) * batch_size
        sample_count += batch_size
    return loss_sum / sample_count


@torch.no_grad()
def evaluate(
    model: CAMoE,
    objective: CAMoELoss,
    loader: Iterable,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    predictions = []
    targets = []
    loss_sum = 0.0
    sample_count = 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        output = model_forward(model, batch)
        losses = objective(output, batch["y"])
        batch_size = int(batch["y"].numel())
        loss_sum += float(losses["L_total"]) * batch_size
        sample_count += batch_size
        predictions.append(output["y_hat"].cpu())
        targets.append(batch["y"].cpu())
    result = sentiment_metrics(torch.cat(predictions), torch.cat(targets))
    result["loss"] = loss_sum / sample_count
    return result


def append_seed_result(path: Path, row: Mapping[str, object]) -> None:
    columns = ("dataset", "setting", "seed", "model", "acc7", "acc2", "f1", "mae")
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row[key] for key in columns})


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    config, training = load_experiment_config(
        args.config, args.dataset, args.setting
    )
    device = torch.device(args.device)
    generator = torch.Generator().manual_seed(args.seed)
    loaders = make_dataloaders(
        args.feature_path,
        batch_size=int(training["batch_size"]),
        expected_dims=(config.d_text, config.d_audio, config.d_visual),
        num_workers=args.num_workers,
        generator=generator,
    )
    model = CAMoE(config).to(device)
    objective = CAMoELoss(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )

    run_dir = args.output_dir / args.dataset / args.setting / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best.pt"
    best_validation_mae = float("inf")
    stale_epochs = 0
    history = []
    max_epochs = int(training["epochs"])
    patience = int(training["early_stopping_patience"])
    for epoch in range(1, max_epochs + 1):
        train_loss = train_epoch(
            model,
            objective,
            loaders["train"],
            optimizer,
            device,
            float(training["gradient_clip"]),
        )
        validation = evaluate(model, objective, loaders["valid"], device)
        history.append({"epoch": epoch, "train_loss": train_loss, "valid": validation})
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.6f} "
            f"valid_mae={validation['mae']:.6f}"
        )
        if validation["mae"] < best_validation_mae:
            best_validation_mae = validation["mae"]
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": config.to_dict(),
                    "dataset": args.dataset,
                    "setting": args.setting,
                    "seed": args.seed,
                    "epoch": epoch,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_result = evaluate(model, objective, loaders["test"], device)
    result = {
        "dataset": args.dataset,
        "setting": args.setting,
        "seed": args.seed,
        "model": "CAMoE",
        **test_result,
        "best_epoch": checkpoint["epoch"],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    with (run_dir / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    with (run_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    append_seed_result(args.output_dir / "seed_level_results.csv", result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
