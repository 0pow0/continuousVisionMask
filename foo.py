#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch


def describe(key, obj, indent=0, max_depth=None):
    pad = "  " * indent
    if max_depth is not None and indent > max_depth:
        print(f"{pad}{key}: ...")
        return
    if torch.is_tensor(obj):
        print(f"{pad}{key}: tensor dtype={obj.dtype} shape={tuple(obj.shape)}")
    elif isinstance(obj, dict):
        print(f"{pad}{key}: dict[{len(obj)}]")
        for sub_key, value in obj.items():
            describe(f"{key}.{sub_key}", value, indent + 1, max_depth)
    elif isinstance(obj, (list, tuple)):
        print(f"{pad}{key}: {type(obj).__name__}[{len(obj)}]")
        for idx, value in enumerate(obj):
            describe(f"{key}[{idx}]", value, indent + 1, max_depth)
    else:
        print(f"{pad}{key}: {type(obj)}")


def summarize_tensor(name: str, tensor: torch.Tensor):
    if tensor.ndim != 2:
        print(f"{name}: expected 2D tensor, got shape {tuple(tensor.shape)}")
        return
    mins = tensor.amin(dim=0)
    maxs = tensor.amax(dim=0)
    means = tensor.mean(dim=0)
    stds = tensor.std(dim=0, unbiased=False)
    print(f"{name}: shape={tuple(tensor.shape)}")
    for idx, (mn, mx, mu, sd) in enumerate(zip(mins, maxs, means, stds)):
        print(
            f"  dim {idx:>3}: min={mn.item(): .6f} max={mx.item(): .6f} "
            f"mean={mu.item(): .6f} std={sd.item(): .6f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Inspect a PyTorch checkpoint (.pt/.pth)."
    )
    parser.add_argument("checkpoint", type=Path, help="Path to the checkpoint file.")
    parser.add_argument(
        "--depth", type=int, default=None, help="Limit recursion depth."
    )
    parser.add_argument(
        "--map-location", default="cpu", help="torch.load map_location (default: cpu)."
    )
    args = parser.parse_args()

    print(f"Loading {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.map_location)
    print(f"Top-level type: {type(checkpoint)}")

    if isinstance(checkpoint, dict):
        print("Top-level keys:")
        for key in checkpoint.keys():
            print(f" - {key}")
        for key, value in checkpoint.items():
            describe(key, value, indent=0, max_depth=args.depth)
    else:
        print(checkpoint)
    for key in ("obs", "act", "act_mean", "act_std"):
        if key in checkpoint:
            summarize_tensor(key, checkpoint[key])
        else:
            print(f"{key}: not found in checkpoint")


if __name__ == "__main__":
    main()
