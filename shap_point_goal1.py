#!/usr/bin/env python3
"""
SHAP Baseline Evaluation for the PointGoal1 Actor
-------------------------------------------------

This script mirrors the evaluation pipeline in point_goal1.py but replaces the
learned attribution coming from the VQ-VAE encoder with SHAP values computed on
top of the frozen actor. The resulting attributions are fed into the same
Insertion/Deletion metrics so they can be compared apples-to-apples with the
model-based explanations.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution
from tqdm.auto import tqdm

try:
    import shap  # type: ignore
except ImportError as exc:  # pragma: no cover - makes failure mode explicit
    raise ImportError(
        "shap_point_goal1.py requires the 'shap' package. "
        "Install it with `pip install shap` and re-run."
    ) from exc

from actor import Actor
from metrics import Deletion, Insertion, QuantusAttributionMetrics
from point_goal1 import (
    get_dataloaders_vector,
    kl_divergence_gaussian,
    set_seed,
)


@dataclass
class SHAPConfig:
    data_path: str = "./data/vectors.npy"
    actor_ckpt: str | None = None
    actor_hidden: tuple[int, ...] = (64, 64)
    batch_size: int = 256
    num_workers: int = 8
    background_size: int = 512
    eval_batches: int | None = None
    insertion_fraction: float = 1.0
    seed: int = 42
    out_dir: str = "./shap_outputs"
    plot_metrics: bool = True


class ActorMeanHead(nn.Module):
    """Lightweight wrapper that exposes the actor's mean for SHAP."""

    def __init__(self, actor: Actor):
        super().__init__()
        self.actor = actor

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs).mean


def make_output_transform(target_mean: torch.Tensor, target_std: torch.Tensor):
    target_mean = target_mean.detach()
    target_std = target_std.detach()

    def transform(output):
        primary = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(primary, Distribution):
            raise TypeError(
                "Insertion/Deletion output transform expects a Distribution."
            )
        pred_mean = primary.mean
        pred_std = primary.stddev
        return kl_divergence_gaussian(
            pred_mean=pred_mean,
            pred_std=pred_std,
            target_mean=target_mean.to(pred_mean.device),
            target_std=target_std.to(pred_mean.device),
        )

    return transform


def aggregate_shap_values(
    shap_values: Iterable[torch.Tensor] | list,
    reference: torch.Tensor,
) -> torch.Tensor:
    """
    Collapses SHAP values (list per-action or tensor) into a single attribution
    matching the input shape. Absolute contributions are averaged across action
    dimensions so the attribution remains non-negative for ranking.
    """
    device = reference.device
    dtype = reference.dtype

    def reduce_dim(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 3:
            return tensor.mean(dim=-1)

    if isinstance(shap_values, list):
        tensors = [
            torch.as_tensor(sv, device=device, dtype=dtype) for sv in shap_values
        ]
        stacked = torch.stack(tensors, dim=1)  # [B, act_dim, D]
        attr = reduce_dim(stacked)
    else:
        tmp = torch.as_tensor(shap_values, device=device, dtype=dtype)
        attr = reduce_dim(tmp)

    if attr.shape != reference.shape:
        attr = attr.reshape_as(reference)
    return attr


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


@torch.no_grad()
def collect_background(
    loader: torch.utils.data.DataLoader,
    max_examples: int,
    device: torch.device,
) -> torch.Tensor:
    collected = []
    total = 0
    for batch in loader:
        chunk = batch[0]
        collected.append(chunk)
        total += chunk.size(0)
        if total >= max_examples:
            break
    if not collected:
        raise RuntimeError("Unable to collect background samples from the dataset.")
    background = torch.cat(collected, dim=0)[:max_examples]
    return background.to(device)


def evaluate(cfg: SHAPConfig):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, input_dim, action_dim = get_dataloaders_vector(
        cfg.data_path, cfg.batch_size, cfg.num_workers, seed=cfg.seed
    )
    if action_dim is None:
        raise ValueError("Dataset must contain action labels to compute metrics.")
    actor = load_actor_state(
        input_dim=input_dim,
        action_dim=action_dim,
        hidden_sizes=cfg.actor_hidden,
        ckpt_path=cfg.actor_ckpt or "",
        device=device,
    )

    background = collect_background(train_loader, cfg.background_size, device)
    explainer = shap.DeepExplainer(ActorMeanHead(actor), background)

    insertion = Insertion(enable_plots=cfg.plot_metrics)
    deletion = Deletion(enable_plots=cfg.plot_metrics)
    quantus_metrics = None
    background_dtype = background.dtype
    try:
        quantus_metrics = QuantusAttributionMetrics(
            sparseness_kwargs={"disable_warnings": True, "display_progressbar": False},
            lipschitz_kwargs={"disable_warnings": True, "display_progressbar": False},
            device="gpu" if device.type == "cuda" else "cpu",
        )
    except ImportError:
        print("Quantus not available; skipping Quantus metrics.")
        quantus_metrics = None

    def quantus_explain_fn(_, inputs, targets, **kwargs):
        inputs_tensor = torch.as_tensor(inputs, device=device, dtype=background_dtype)
        shap_vals = explainer.shap_values(inputs_tensor)
        attr_local = aggregate_shap_values(shap_vals, inputs_tensor)
        return attr_local.detach().cpu().numpy()

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

    if val_loader is None:
        raise RuntimeError("Validation loader is None; SHAP baseline needs a split.")

    total_batches = len(val_loader)
    if cfg.eval_batches is not None:
        total_batches = min(total_batches, cfg.eval_batches)
    progress = tqdm(
        val_loader,
        total=total_batches,
        desc="Evaluating SHAP baseline",
        leave=False,
    )
    quantus_summary: dict[str, float] = {}

    for batch_idx, batch in enumerate(progress):
        if batch_idx >= total_batches:
            break

        obs = batch[0].to(device, non_blocking=True)
        target_mean = batch[1].to(device, non_blocking=True)
        target_std = batch[2].to(device, non_blocking=True)
        batch_ids = batch[3] if len(batch) > 3 else None

        shap_values = explainer.shap_values(obs)
        attr = aggregate_shap_values(shap_values, obs)

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

        batch_size = obs.size(0)
        aggregated["nll"] += kl_mean.item() * batch_size
        aggregated["kl"] += per_sample_kl.sum().item()
        aggregated["mse_mean"] += mse_mean.item() * batch_size
        aggregated["mse_std"] += mse_std.item() * batch_size

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
        if quantus_metrics is not None:
            quantus_metrics(
                model=actor,
                x=obs,
                attr=attr,
                sample_ids=batch_ids,
                explain_func=quantus_explain_fn,
            )

        total_examples += batch_size

        progress.set_postfix(
            batches=f"{batch_idx+1}/{total_batches}",
            insertion=f"{(sum(insertion_auc)/batch_size):.4f}",
            deletion=f"{(sum(deletion_auc)/batch_size):.4f}",
        )

    progress.close()
    normalized_insertion = insertion.flush(str(curves_dir), prefix="shap_insertion")
    normalized_deletion = deletion.flush(str(curves_dir), prefix="shap_deletion")
    if quantus_metrics is not None:
        quantus_summary = quantus_metrics.summary()
        quantus_metrics.flush(curves_dir / "shap_quantus.pt")

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
    if quantus_summary:
        summary["quantus_sparseness"] = quantus_summary.get("sparseness_mean", 0.0)
        summary["quantus_lipschitz"] = quantus_summary.get("local_lipschitz_mean", 0.0)
    summary_path = output_root / "shap_metrics.json"
    os.makedirs(output_root, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump({"config": asdict(cfg), "metrics": summary}, fp, indent=2)

    print("SHAP baseline metrics:")
    for key, value in summary.items():
        print(f"  {key}: {value:.6f}")
    print(f"Saved detailed curves and metrics under: {output_root.resolve()}")


def parse_args() -> SHAPConfig:
    parser = argparse.ArgumentParser(
        description="SHAP baseline for the PointGoal1 actor (Insertion/Deletion)."
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
        "--background-size",
        type=int,
        default=512,
        help="Number of samples used as SHAP background.",
    )
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
        "--out-dir",
        type=str,
        default="./shap_outputs",
        help="Directory where curves and metrics will be stored.",
    )
    parser.add_argument(
        "--no-plot-metrics",
        action="store_true",
        help="Disable insertion/deletion plotting for faster runs.",
    )

    args = parser.parse_args()
    return SHAPConfig(
        data_path=args.data_path,
        actor_ckpt=args.actor_ckpt,
        actor_hidden=tuple(args.actor_hidden),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        background_size=args.background_size,
        eval_batches=args.eval_batches,
        insertion_fraction=args.insertion_fraction,
        seed=args.seed,
        out_dir=args.out_dir,
        plot_metrics=not args.no_plot_metrics,
    )


if __name__ == "__main__":
    cfg = parse_args()
    os.makedirs(cfg.out_dir, exist_ok=True)
    evaluate(cfg)
