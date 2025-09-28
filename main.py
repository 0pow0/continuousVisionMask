#!/usr/bin/env python3
"""
VQ‑VAE (Vector‑Quantized Variational Autoencoder) – PyTorch Training Pipeline
----------------------------------------------------------------------------

Features
- Encoder/Decoder with residual blocks (CIFAR‑10 default, easy to adjust)
- Vector Quantizer with EMA updates (VQ‑VAE v2‑style stable codebook)
- Commitment loss + MSE reconstruction loss
- Mixed precision (torch.cuda.amp)
- Gradient clipping, cosine LR schedule, EMA of model weights (optional)
- Checkpointing, resuming, and sample reconstruction grids
- Clean configuration via argparse

Quickstart
----------
python vqvae_train.py \
  --data-root ./data \
  --epochs 100 \
  --batch-size 256 \
  --latent-dim 64 \
  --n-codes 512 \
  --img-size 32

Outputs
- checkpoints/epoch_XX.pt           (model+optimizer+amp state)
- samples/recon_epoch_XX.png        (reconstructions)
- samples/code_usage_epoch_XX.txt   (perplexity + usage stats)

Notes
- Default dataset is CIFAR‑10. You can switch to ImageFolder with --dataset imagefolder --imagefolder-root <path>.
- Images are scaled to [-1, 1].
- For 64×64 or 128×128, increase encoder depth (see --levels) and adjust --img-size.
"""
from __future__ import annotations
import math
import os
import random
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils

# ---------------------------
# Utils
# ---------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_image_grid(tensor, path, nrow=8, normalize=True, value_range=(-1, 1)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    grid = vutils.make_grid(tensor, nrow=nrow, normalize=normalize, value_range=value_range)
    vutils.save_image(grid, path)


# ---------------------------
# Model blocks (Vector data version)
# ---------------------------
class MLPResBlock(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(x)))


class Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims=(512, 512), n_res_blocks: int = 2, res_hidden: int = 256):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.GELU()]
            prev = h
        layers += [nn.Linear(prev, latent_dim)]
        self.net = nn.Sequential(*layers)
        self.res = nn.Sequential(*[MLPResBlock(latent_dim, res_hidden) for _ in range(n_res_blocks)])

    def forward(self, x):  # x: [B, input_dim]
        z = self.net(x)
        z = self.res(z)
        return z  # [B, latent_dim]


class Decoder(nn.Module):
    def __init__(self, output_dim: int, latent_dim: int, hidden_dims=(512, 512), n_res_blocks: int = 2, res_hidden: int = 256):
        super().__init__()
        self.res = nn.Sequential(*[MLPResBlock(latent_dim, res_hidden) for _ in range(n_res_blocks)])
        layers = []
        prev = latent_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.GELU()]
            prev = h
        layers += [nn.Linear(prev, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, z_q):  # [B, latent_dim]
        z = self.res(z_q)
        x = self.net(z)
        return x  # [B, output_dim]


class VectorQuantizerEMA(nn.Module):
    """
    VQ layer with exponential moving average (EMA) codebook updates.
    Works for 2D (B, D) and 4D (B, D, H, W) inputs. For vectors, pass (B, D).
    """
    def __init__(self, n_codes: int, dim: int, decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.n_codes = n_codes
        self.dim = dim
        self.decay = decay
        self.eps = eps

        self.codebook = nn.Parameter(torch.randn(n_codes, dim))
        self.register_buffer("cluster_size", torch.zeros(n_codes))
        self.register_buffer("embed_avg", torch.randn(n_codes, dim))
        nn.init.normal_(self.codebook, mean=0.0, std=1.0)

    @torch.no_grad()
    def _ema_update(self, encodings: torch.Tensor, z_e_flat: torch.Tensor):
        cluster_size = encodings.sum(0)
        embed_sum = encodings.t() @ z_e_flat
        self.cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
        n = self.cluster_size.sum()
        cluster_size = (self.cluster_size + self.eps) / (n + self.n_codes * self.eps) * n
        self.codebook.data.copy_(self.embed_avg / cluster_size.unsqueeze(1))

    def forward(self, z_e: torch.Tensor):
        # Accept [B, D] or [B, D, H, W]
        if z_e.dim() == 2:
            B, D = z_e.shape
            H = W = 1
            z_flat = z_e
            reshape_back = lambda t: t
        elif z_e.dim() == 4:
            B, D, H, W = z_e.shape
            z_flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, D)
            reshape_back = lambda t: t.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()
        else:
            raise ValueError("z_e must be [B,D] or [B,D,H,W]")

        codebook = self.codebook
        dist = (z_flat.pow(2).sum(1, keepdim=True)
                + codebook.pow(2).sum(1)
                - 2 * z_flat @ codebook.t())
        indices = torch.argmin(dist, dim=1)
        z_q_flat = codebook[indices]
        z_q = reshape_back(z_q_flat)

        with torch.no_grad():
            encodings = F.one_hot(indices, self.n_codes).type_as(z_flat)
            self._ema_update(encodings, z_flat)

        commitment_loss = F.mse_loss(z_e.detach().view(-1, D), z_q_flat)
        z_q_st = z_e + (z_q - z_e).detach()
        avg_probs = encodings.float().mean(0)
        perplexity = torch.exp(-(avg_probs * (avg_probs + 1e-10).log()).sum())
        usage = (avg_probs > 0).float().mean()
        return z_q_st, commitment_loss, perplexity, usage


class VQVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, n_codes: int = 512, decay: float = 0.99, beta: float = 0.25,
                 enc_hidden=(512, 512), dec_hidden=(512, 512)):
        super().__init__()
        self.encoder = Encoder(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=enc_hidden)
        self.quantizer = VectorQuantizerEMA(n_codes=n_codes, dim=latent_dim, decay=decay)
        self.decoder = Decoder(output_dim=input_dim, latent_dim=latent_dim, hidden_dims=dec_hidden)
        self.beta = beta

    def forward(self, x):  # x: [B, input_dim]
        z_e = self.encoder(x)
        z_q, commit_loss, perplexity, usage = self.quantizer(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, commit_loss, perplexity, usage


# ---------------------------
# Data (Vector datasets)
# ---------------------------

# ---------------------------

def get_dataloaders_vector(path: str, batch_size: int, num_workers: int, val_split: float = 0.05, seed: int = 42):
    """
    Load vectors from .npy, .pt, or .csv. Shapes expected: [N, D].
    CSV: assumes numeric without header, or set VQVAE_CSV_SKIP_HEADER=1 to skip first row.
    """
    import numpy as np
    ext = os.path.splitext(path)[1].lower()
    if ext in [".npy", ".npz"]:
        arr = np.load(path)
        if isinstance(arr, np.lib.npyio.NpzFile):
            # pick the first array
            arr = arr[list(arr.files)[0]]
    elif ext in [".pt", ".pth"]:
        arr = torch.load(path)
        if isinstance(arr, torch.Tensor):
            arr = arr.cpu().numpy()
    elif ext == ".csv":
        skip = int(os.environ.get("VQVAE_CSV_SKIP_HEADER", "0"))
        arr = np.loadtxt(path, delimiter=",", skiprows=skip)
    else:
        raise ValueError(f"Unsupported data file: {path}")

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array [N,D], got shape {arr.shape}")
    N, D = arr.shape
    x = torch.from_numpy(arr).float()

    # Normalize to roughly [-1,1]
    x_mean = x.mean(0, keepdim=True)
    x_std = x.std(0, keepdim=True) + 1e-6
    x = (x - x_mean) / x_std

    ds = torch.utils.data.TensorDataset(x)
    # Split
    g = torch.Generator().manual_seed(seed)
    n_val = max(1, int(val_split * N))
    n_train = N - n_val
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val], generator=g)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, D



# ---------------------------
# Training
# ---------------------------
@dataclass
class Config:
    data_path: str = "./data/vectors.npy"  # .npy/.npz/.pt/.csv file with shape [N,D]

    epochs: int = 100
    batch_size: int = 512
    lr: float = 2e-4
    weight_decay: float = 1e-6
    beta: float = 0.25  # commitment weight
    n_codes: int = 512
    latent_dim: int = 64
    enc_hidden: tuple = (512, 512)
    dec_hidden: tuple = (512, 512)
    decay: float = 0.99

    num_workers: int = 8
    grad_clip: float = 1.0
    amp: bool = True
    ema_decay: float = 0.0  # set >0.0 to enable model EMA (e.g., 0.999)

    out_dir: str = "./checkpoints"
    sample_dir: str = "./samples"
    resume: str | None = None
    seed: int = 42


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                v.copy_(self.shadow[k])


def cosine_warmup_lr(optimizer, base_lr, warmup_steps, total_steps, step):
    if step < warmup_steps:
        lr = base_lr * step / max(1, warmup_steps)
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        lr = 0.5 * base_lr * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def save_ckpt(path, model, opt, scaler, epoch, step, cfg: Config):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "step": step,
        "config": cfg.__dict__,
    }, path)


def load_ckpt(path, model, opt=None, scaler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])  # type: ignore
    if opt is not None and ckpt.get("optimizer"):
        opt.load_state_dict(ckpt["optimizer"])  # type: ignore
    if scaler is not None and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])  # type: ignore
    return ckpt


def train(cfg: Config):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, input_dim = get_dataloaders_vector(cfg.data_path, cfg.batch_size, cfg.num_workers, seed=cfg.seed)

    model = VQVAE(input_dim=input_dim, latent_dim=cfg.latent_dim, n_codes=cfg.n_codes,
                  decay=cfg.decay, beta=cfg.beta, enc_hidden=cfg.enc_hidden, dec_hidden=cfg.dec_hidden).to(device)

    ema = ModelEMA(model, cfg.ema_decay) if cfg.ema_decay > 0 else None

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = GradScaler(enabled=cfg.amp)

    start_epoch = 0
    global_step = 0
    if cfg.resume:
        ck = load_ckpt(cfg.resume, model, opt, scaler)
        start_epoch = ck.get("epoch", 0)
        global_step = ck.get("step", 0)
        print(f"Resumed from {cfg.resume}: epoch {start_epoch}, step {global_step}")

    total_steps = cfg.epochs * len(train_loader)
    warmup_steps = int(0.02 * total_steps)

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        running_recon, running_commit, running_ppl = 0.0, 0.0, 0.0
        for i, (imgs, _) in enumerate(train_loader):
            vecs = next(iter(train_loader))[0]  # to infer shape earlier if needed
        for i, (vecs,) in enumerate(train_loader):
            vecs = vecs.to(device, non_blocking=True)

            with autocast(enabled=cfg.amp):
                x_recon, commit_loss, perplexity, usage = model(vecs)
                recon_loss = F.mse_loss(x_recon, vecs)
                loss = recon_loss + cfg.beta * commit_loss

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()

            if ema is not None:
                ema.update(model)

            # LR schedule
            lr = cosine_warmup_lr(opt, cfg.lr, warmup_steps, total_steps, global_step)

            running_recon += recon_loss.item()
            running_commit += commit_loss.item()
            running_ppl += float(perplexity)

            if (i + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{cfg.epochs} Step {i+1}/{len(train_loader)} | "
                      f"LR {lr:.3e} | Recon {running_recon/(i+1):.6f} | Commit {running_commit/(i+1):.6f} | "
                      f"Perplexity {running_ppl/(i+1):.2f} | Usage {usage.item():.2f}")

            global_step += 1

        # Validation / Sampling
        model.eval()
        with torch.no_grad():
            # Use EMA weights for eval if enabled
            if ema is not None:
                shadow = ModelEMA(model, cfg.ema_decay)
                shadow.shadow = {k: v.clone() for k, v in ema.shadow.items()}
                shadow.copy_to(model)

            # Take a small batch for qualitative numeric inspection
            batch = next(iter(val_loader if val_loader is not None else train_loader))[0][:64].to(device)
            recon, commit_loss, perplexity, usage = model(batch)
            # Save a few example pairs to .pt (tensor) for quick inspection
            os.makedirs(cfg.sample_dir, exist_ok=True)
            torch.save({
                "inputs": batch.cpu(),
                "recons": recon.cpu(),
                "perplexity": float(perplexity),
                "usage": float(usage),
                "commit_loss": float(commit_loss.item()),
            }, os.path.join(cfg.sample_dir, f"recon_epoch_{epoch+1:03d}.pt"))
            with open(os.path.join(cfg.sample_dir, f"code_usage_epoch_{epoch+1:03d}.txt"), "w") as f:
                f.write(f"perplexity={float(perplexity):.2f} usage={float(usage):.4f}")

        # Save checkpoint
        save_ckpt(os.path.join(cfg.out_dir, f"epoch_{epoch+1:03d}.pt"), model, opt, scaler, epoch+1, global_step, cfg)(os.path.join(cfg.out_dir, f"epoch_{epoch+1:03d}.pt"), model, opt, scaler, epoch+1, global_step, cfg)


# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Train VQ‑VAE on vector data (.npy/.npz/.pt/.csv)")
    p.add_argument("--data-path", type=str, default="./data/vectors.npy")

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-6)

    p.add_argument("--beta", type=float, default=0.25, help="commitment loss weight")
    p.add_argument("--n-codes", type=int, default=512)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--enc-hidden", type=int, nargs="*", default=[512, 512])
    p.add_argument("--dec-hidden", type=int, nargs="*", default=[512, 512])
    p.add_argument("--decay", type=float, default=0.99, help="EMA decay for codebook")

    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--ema-decay", type=float, default=0.0, help=">0 to enable model EMA, e.g. 0.999")

    p.add_argument("--out-dir", type=str, default="./checkpoints")
    p.add_argument("--sample-dir", type=str, default="./samples")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    cfg = Config(
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        beta=args.beta,
        n_codes=args.n_codes,
        latent_dim=args.latent_dim,
        enc_hidden=tuple(args.enc_hidden),
        dec_hidden=tuple(args.dec_hidden),
        decay=args.decay,
        num_workers=args.num_workers,
        grad_clip=args.grad_clip,
        amp=not args.no_amp,
        ema_decay=args.ema_decay,
        out_dir=args.out_dir,
        sample_dir=args.sample_dir,
        resume=args.resume,
        seed=args.seed,
    )

    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(cfg.sample_dir, exist_ok=True)

    print(cfg)
    train(cfg)
    )

    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(cfg.sample_dir, exist_ok=True)

    print(cfg)
    train(cfg)
