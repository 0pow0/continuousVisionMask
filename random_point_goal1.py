#!/usr/bin/env python3
"""
Random Attribution Baseline for PointGoal1
------------------------------------------

Uses randomly generated attribution scores (independent of the actor) to feed
the existing Insertion/Deletion metrics. Serves as a sanity baseline alongside
the learned attribution (point_goal1.py) and SHAP baseline.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.distributions import Distribution

from actor import Actor
from metrics import Insertion, Deletion
from point_goal1 import get_dataloaders_vector, kl_divergence_gaussian, set_seed


@dataclass
class RandomConfig:
    data_path: str = "./data/vectors.npy"
    actor_ckpt: str | None = None
    actor_hidden: tuple[int, ...] = (64, 64)
    batch_size: int = 256
    num_workers: int = 8
    eval_batches: int | None = None
    insertion_fraction: float = 1.0
    seed: int = 42
    attr_strategy: str = "normal"
    normalize_attr: bool = False
    out_dir: str = "./random_outputs"
    plot_metrics: bool = True


def load_actor_state(
    input_dim: int,
    action_dim: int,
    hidden_sizes: Iterable[int],
    ckpt_path: str,
    device: torch.device,
) -> Actor:
    actor = Actor(obs_dim=input_dim, act_dim=action_dim, hidden_sizes=list(hidden_sizes))
    if ckpt_path is None:
        raise ValueError("actor_ckpt must be provided.")
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and any(k.startswith("actor.") for k in state.keys()):
        state = {
            k.split("actor.", 1)[1]: v for k, v in state.items() if k.startswith("actor.")
        }
    if not isinstance(state, dict):
        raise ValueError("Actor checkpoint must resolve to a state_dict.")

    missing, unexpected = actor.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys in actor checkpoint: {unexpected}")
    if missing:
        print(f"Warning: missing actor parameters in checkpoint: {missing}")

    actor.to(device)
    actor.eval()
    for param in actor.parameters():
        param.requires_grad = False
    return actor


def make_output_transform(target_mean: torch.Tensor, target_std: torch.Tensor):
    target_mean = target_mean.detach()
    target_std = target_std.detach()

    def transform(output):
        primary = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(primary, Distribution):
            raise TypeError("Insertion/Deletion output transform expects a Distribution.")
        pred_mean = primary.mean
        pred_std = primary.stddev
        return kl_divergence_gaussian(
            pred_mean=pred_mean,
            pred_std=pred_std,
            target_mean=target_mean.to(pred_mean.device),
            target_std=target_std.to(pred_mean.device),
        )

    return transform


def generate_random_attr(
    obs: torch.Tensor,
    strategy: str = "normal",
    normalize: bool = False,
) -> torch.Tensor:
    if strategy == "normal":
        attr = torch.randn_like(obs)
    elif strategy == "uniform":
        attr = torch.rand_like(obs)
    else:
        raise ValueError("attr_strategy must be 'normal' or 'uniform'")
    attr = attr.abs()
    if normalize:
        flat = attr.view(attr.size(0), -1)
        denom = flat.sum(dim=1, keepdim=True).clamp_min(1e-8)
        flat = flat / denom
        attr = flat.view_as(attr)
    return attr


def evaluate(cfg: RandomConfig):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, input_dim, action_dim = get_dataloaders_vector(
        cfg.data_path, cfg.batch_size, cfg.num_workers, seed=cfg.seed
    )
    if val_loader is None:
        raise RuntimeError("Validation loader is None; random baseline needs a split.")
    if action_dim is None:
        raise ValueError("Dataset must contain action labels to compute metrics.")

    actor = load_actor_state(
        input_dim=input_dim,
        action_dim=action_dim,
        hidden_sizes=cfg.actor_hidden,
        ckpt_path=cfg.actor_ckpt or "",
        device=device,
    )

    insertion = Insertion(enable_plots=cfg.plot_metrics)
    deletion = Deletion(enable_plots=cfg.plot_metrics)
    output_root = Path(cfg.out_dir)
    curves_dir = output_root / "curves"
    os.makedirs(curves_dir, exist_ok=True)

    aggregated = {
        "nll": 0.0,
        "mse_mean": 0.0,
        "mse_std": 0.0,
        "kl": 0.0,
    }
    total_examples = 0

    for batch_idx, batch in enumerate(val_loader):
        if cfg.eval_batches is not None and batch_idx >= cfg.eval_batches:
            break

        obs = batch[0].to(device, non_blocking=True)
        target_mean = batch[1].to(device, non_blocking=True)
        target_std = batch[2].to(device, non_blocking=True)
        batch_ids = batch[3] if len(batch) > 3 else None

        attr = generate_random_attr(
            obs,
            strategy=cfg.attr_strategy,
            normalize=cfg.normalize_attr,
        )

        dist = actor(obs * torch.sigmoid(attr))
        pred_mean = dist.mean
        pred_std = dist.stddev

        per_sample_kl = kl_divergence_gaussian(
            pred_mean=pred_mean,
            pred_std=pred_std,
            target_mean=target_mean,
            target_std=target_std,
        )
        kl_mean = per_sample_kl.mean()
        mse_mean = F.mse_loss(pred_mean, target_mean)
        mse_std = F.mse_loss(pred_std, target_std)
        bs = obs.size(0)

        aggregated["nll"] += kl_mean.item() * bs
        aggregated["kl"] += per_sample_kl.sum().item()
        aggregated["mse_mean"] += mse_mean.item() * bs
        aggregated["mse_std"] += mse_std.item() * bs

        output_transform = make_output_transform(target_mean, target_std)
        insertion.output_transform = output_transform
        deletion.output_transform = output_transform

        insertion_auc = insertion(
            actor,
            obs,
            attr,
            fraction=cfg.insertion_fraction,
            sample_ids=batch_ids,
        )
        deletion_auc = deletion(
            actor,
            obs,
            attr,
            fraction=cfg.insertion_fraction,
            sample_ids=batch_ids,
        )

        total_examples += bs

    normalized_insertion = insertion.flush(str(curves_dir), prefix="random_insertion")
    normalized_deletion = deletion.flush(str(curves_dir), prefix="random_deletion")

    if total_examples == 0:
        raise RuntimeError("Validation loader produced zero examples.")

    summary = {k: v / total_examples for k, v in aggregated.items()}
    def mean_auc(records):
        if not records:
            return 0.0
        return sum(item.get("auc", 0.0) for item in records.values()) / max(
            1, len(records)
        )

    summary["insertion_auc"] = mean_auc(normalized_insertion)
    summary["deletion_auc"] = mean_auc(normalized_deletion)
    os.makedirs(output_root, exist_ok=True)
    summary_path = output_root / "random_metrics.json"
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump({"config": asdict(cfg), "metrics": summary}, fp, indent=2)

    print("Random baseline metrics:")
    for key, value in summary.items():
        print(f"  {key}: {value:.6f}")
    print(f"Saved detailed curves and metrics under: {output_root.resolve()}")


def parse_args() -> RandomConfig:
    parser = argparse.ArgumentParser(
        description="Random attribution baseline for PointGoal1 insertion/deletion."
    )
    parser.add_argument("--data-path", type=str, default="./data/vectors.npy")
    parser.add_argument(
        "--actor-ckpt",
        type=str,
        required=True,
        help="Path to the pretrained actor checkpoint.",
    )
    parser.add_argument(
        "--actor-hidden",
        type=int,
        nargs="*",
        default=[64, 64],
        help="Hidden sizes used by the actor architecture.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=None,
        help="Optional limit on validation batches processed.",
    )
    parser.add_argument(
        "--insertion-fraction",
        type=float,
        default=1.0,
        help="Fraction of features to reveal/remove in insertion/deletion.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--attr-strategy",
        type=str,
        choices=["normal", "uniform"],
        default="normal",
        help="Distribution for random attributions.",
    )
    parser.add_argument(
        "--normalize-attr",
        action="store_true",
        help="Normalize attributions per-sample to sum to 1.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./random_outputs",
        help="Directory where curves and metrics will be stored.",
    )
    parser.add_argument(
        "--no-plot-metrics",
        action="store_true",
        help="Disable insertion/deletion plotting for faster runs.",
    )

    args = parser.parse_args()
    return RandomConfig(
        data_path=args.data_path,
        actor_ckpt=args.actor_ckpt,
        actor_hidden=tuple(args.actor_hidden),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        eval_batches=args.eval_batches,
        insertion_fraction=args.insertion_fraction,
        seed=args.seed,
        attr_strategy=args.attr_strategy,
        normalize_attr=args.normalize_attr,
        out_dir=args.out_dir,
        plot_metrics=not args.no_plot_metrics,
    )


if __name__ == "__main__":
    cfg = parse_args()
    os.makedirs(cfg.out_dir, exist_ok=True)
    evaluate(cfg)
