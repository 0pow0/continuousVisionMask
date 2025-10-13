#!/usr/bin/env python3
"""
VQ‑VAE (Vector‑Quantized Variational Autoencoder) – PyTorch Training Pipeline
----------------------------------------------------------------------------

Features
- Encoder with residual blocks for vector observations
- Discrete latent codebook with EMA updates (VQ‑VAE v2‑style stable codebook)
- Straight-through index head (optional Gumbel-Softmax) for code selection
- Frozen pretrained actor consumes quantized latents; training matches its distribution to demonstrations
- Commitment loss keeps encoder latents aligned with chosen codes
- Optional Weights & Biases logging for metrics
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
- checkpoints/epoch_XX.pt           (model+optimizer state)
- samples/actor_epoch_XX.pt         (inputs, predicted vs target actions)
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
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils
from tqdm.auto import tqdm

from actor import Actor


# ---------------------------
# Utils
# ---------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_image_grid(tensor, path, nrow=8, normalize=True, value_range=(-1, 1)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    grid = vutils.make_grid(
        tensor, nrow=nrow, normalize=normalize, value_range=value_range
    )
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
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims=(512, 512),
        n_res_blocks: int = 2,
        res_hidden: int = 256,
    ):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.GELU()]
            prev = h
        layers += [nn.Linear(prev, latent_dim)]
        self.net = nn.Sequential(*layers)
        self.res = nn.Sequential(
            *[MLPResBlock(latent_dim, res_hidden) for _ in range(n_res_blocks)]
        )

    def forward(self, x):  # x: [B, input_dim]
        z = self.net(x)
        z = self.res(z)
        return z  # [B, latent_dim]


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
        cluster_size = (
            (self.cluster_size + self.eps) / (n + self.n_codes * self.eps) * n
        )
        self.codebook.data.copy_(self.embed_avg / cluster_size.unsqueeze(1))

    def forward(self, z_e: torch.Tensor):
        orig_dtype = z_e.dtype
        z_e_fp32 = z_e.float()
        if z_e_fp32.dim() != 2:
            raise ValueError("VectorQuantizerEMA expects inputs of shape [B, D]")
        B, D = z_e_fp32.shape
        codebook = self.codebook.float()
        distances = (
            z_e_fp32.pow(2).sum(dim=1, keepdim=True)
            + codebook.pow(2).sum(dim=1)
            - 2 * z_e_fp32 @ codebook.t()
        )
        indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(indices, self.n_codes).type(z_e_fp32.dtype)
        z_q = F.embedding(indices, codebook)

        with torch.no_grad():
            self._ema_update(encodings, z_e_fp32)

        commitment_loss = F.mse_loss(z_e_fp32.detach(), z_q)
        z_q_st = z_e_fp32 + (z_q - z_e_fp32).detach()
        z_q = z_q.to(orig_dtype)
        z_q_st = z_q_st.to(orig_dtype)
        avg_probs = encodings.mean(0)
        perplexity = torch.exp(-(avg_probs * (avg_probs + 1e-10).log()).sum())
        usage = (avg_probs > 1e-3).float().mean()
        return z_q_st, commitment_loss, perplexity, usage, indices, encodings.to(orig_dtype)


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.query_norm = nn.LayerNorm(dim)
        self.key_norm = nn.LayerNorm(dim)
        self.value_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        hidden_dim = max(1, int(dim * mlp_ratio))
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, query: torch.Tensor, key_value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.query_norm(query)
        k = self.key_norm(key_value)
        v = self.value_norm(key_value)
        attn_out, attn_weights = self.attn(q, k, v, need_weights=True)
        query = query + self.dropout(attn_out)
        ff_out = self.ffn(self.ffn_norm(query))
        return query + self.dropout(ff_out), attn_weights


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        hidden_dim = max(1, int(dim * mlp_ratio))
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + self.dropout(attn_out)
        ff_out = self.ffn(self.norm2(x))
        return x + self.dropout(ff_out)


class VQVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        action_dim: int,
        n_codes: int = 512,
        decay: float = 0.99,
        beta: float = 0.25,
        enc_hidden=(512, 512),
        actor_hidden=(64, 64),
        head_hidden: tuple[int, ...] = (),
        gumbel_tau: float = 1.0,
        st_tau: float = 1.0,
        actor_ckpt: str | None = None,
        attn_heads: int = 4,
        attn_layers: int = 2,
        attn_mlp_ratio: float = 2.0,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.quantizer = VectorQuantizerEMA(n_codes=n_codes, dim=input_dim, decay=decay)
        self.actor = Actor(
            obs_dim=input_dim,
            act_dim=action_dim,
            hidden_sizes=list(actor_hidden),
        )
        if actor_ckpt is None:
            raise ValueError(
                "actor_ckpt must be provided to load the frozen actor state."
            )
        state = torch.load(actor_ckpt, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and any(
            k.startswith("actor.") for k in state.keys()
        ):
            state = {
                k.split("actor.", 1)[1]: v
                for k, v in state.items()
                if k.startswith("actor.")
            }
        if not isinstance(state, dict):
            raise ValueError(
                "Actor checkpoint must be a state_dict or a dict containing 'state_dict'."
            )
        missing, unexpected = self.actor.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys in actor checkpoint: {unexpected}")
        if missing:
            print(f"Warning: missing actor parameters in checkpoint: {missing}")
        self.actor.eval()
        for param in self.actor.parameters():
            param.requires_grad = False

        self.beta = beta
        self.action_dim = action_dim
        self.gumbel_tau = gumbel_tau
        self.st_tau = st_tau

    def forward(self, x):  # x: [B, input_dim]
        z_q, commit_loss, _, _, _, _ = self.quantizer(x)
        dist = self.actor(z_q.float())
        return dist, commit_loss


# ---------------------------
# Data (Vector datasets)
# ---------------------------

# ---------------------------


def get_dataloaders_vector(
    path: str,
    batch_size: int,
    num_workers: int,
    val_split: float = 0.05,
    seed: int = 42,
):
    """
    Load vectors from .npy, .pt, or .csv. Shapes expected: [N, D].
    CSV: assumes numeric without header, or set VQVAE_CSV_SKIP_HEADER=1 to skip first row.
    Requires continuous action distribution labels "act_mean" and "act_std" in the file.
    """
    x = torch.load(path)
    obs = x["obs"] if isinstance(x, dict) else x
    N, D = obs.shape

    if not isinstance(x, dict) or "act_mean" not in x or "act_std" not in x:
        raise ValueError("Dataset must contain 'act_mean' and 'act_std' tensors")

    act_mean = x["act_mean"]
    act_std = x["act_std"]
    if act_mean.shape[0] != N or act_std.shape[0] != N:
        raise ValueError("Action tensors must match obs length")

    if act_mean.shape != act_std.shape:
        raise ValueError("act_mean and act_std must share shape")

    tensors = (obs, act_mean, act_std)
    ds = torch.utils.data.TensorDataset(*tensors)
    # Split
    g = torch.Generator().manual_seed(seed)
    n_val = max(1, int(val_split * N))
    n_train = N - n_val
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val], generator=g)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    if act_mean.ndim == 1:
        label_dim = 1
    else:
        label_dim = act_mean.shape[1]
    return train_loader, val_loader, D, label_dim


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
    actor_hidden: tuple = (64, 64)
    decay: float = 0.99
    head_hidden: tuple = ()
    gumbel_tau: float = 1.0
    st_tau: float = 1.0
    actor_ckpt: str | None = None
    action_dim: int | None = None
    use_wandb: bool = False
    wandb_project: str = ""
    wandb_run_name: str | None = None

    num_workers: int = 8
    grad_clip: float = 1.0
    ema_decay: float = 0.0  # set >0.0 to enable model EMA (e.g., 0.999)

    out_dir: str = "./checkpoints"
    sample_dir: str = "./samples"
    resume: str | None = None
    seed: int = 42


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }

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


def save_ckpt(path, model, opt, epoch, step, cfg: Config):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "epoch": epoch,
            "step": step,
            "config": cfg.__dict__,
        },
        path,
    )


def load_ckpt(path, model, opt=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])  # type: ignore
    if opt is not None and ckpt.get("optimizer"):
        opt.load_state_dict(ckpt["optimizer"])  # type: ignore
    return ckpt


def kl_divergence_gaussian(
    pred_mean, pred_std, target_mean, target_std, min_std=1e-3, direction="target||pred"
):
    """
    计算两个对角高斯分布的 KL 散度，支持稳定性保护

    Args:
        pred_mean: (B, D) 预测分布的均值
        pred_scale: (B, D) 预测分布的 std
        target_mean: (B, D) 目标分布的均值
        target_scale: (B, D) 目标分布的 std
        min_std: 最小标准差下限，避免除零
        direction: "target||pred" 或 "pred||target"，选择 KL 方向
    """

    if direction == "target||pred":
        # KL(q||p): q = target, p = pred
        kl = (
            torch.log(pred_std / target_std)
            + (target_std.pow(2) + (target_mean - pred_mean).pow(2))
            / (2 * pred_std.pow(2))
            - 0.5
        )
    elif direction == "pred||target":
        # KL(p||q): p = pred, q = target
        kl = (
            torch.log(target_std / pred_std)
            + (pred_std.pow(2) + (pred_mean - target_mean).pow(2))
            / (2 * target_std.pow(2))
            - 0.5
        )
    else:
        raise ValueError("direction must be 'target||pred' or 'pred||target'")

    # 默认返回 batch 平均 KL
    return kl

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GaussianMSEKLLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1, gamma_max=1.0, min_std=1e-3, 
                 direction="target||pred", warmup_epochs=50, schedule="cosine"):
        """
        动态调权: MSE(mean) + MSE(std) + KL
        KL 的权重 gamma 会根据 schedule 从 0 -> gamma_max

        Args:
            alpha: mean MSE 权重
            beta: std MSE 权重
            gamma_max: KL 最大权重
            min_std: 避免除零的小常数
            direction: "target||pred" 或 "pred||target"
            warmup_epochs: KL 权重增长的训练轮数
            schedule: "linear" 或 "cosine"
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma_max = gamma_max
        self.min_std = min_std
        self.direction = direction
        self.warmup_epochs = warmup_epochs
        self.schedule = schedule
        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        """在每个 epoch 开头调用，用于更新 KL 权重"""
        self.current_epoch = epoch

    def _get_gamma(self):
        if self.schedule == "linear":
            return min(self.gamma_max, self.gamma_max * self.current_epoch / max(1, self.warmup_epochs))
        elif self.schedule == "cosine":
            progress = min(1.0, self.current_epoch / max(1, self.warmup_epochs))
            # 0 → gamma_max, 先慢后快
            return self.gamma_max * (1 - math.cos(progress * math.pi / 2))
        else:
            raise ValueError("schedule must be 'linear' or 'cosine'")

    def forward(self, pred_mean, pred_std, target_mean, target_std):
        pred_std = F.softplus(pred_std) + self.min_std
        target_std = target_std.clamp_min(self.min_std)

        mse_mean = F.mse_loss(pred_mean, target_mean)
        mse_std = F.mse_loss(pred_std, target_std)

        if self.direction == "target||pred":
            kl = (
                torch.log(pred_std / target_std)
                + (target_std.pow(2) + (target_mean - pred_mean).pow(2)) / (2 * pred_std.pow(2))
                - 0.5
            )
        elif self.direction == "pred||target":
            kl = (
                torch.log(target_std / pred_std)
                + (pred_std.pow(2) + (pred_mean - target_mean).pow(2)) / (2 * target_std.pow(2))
                - 0.5
            )
        else:
            raise ValueError("direction must be 'target||pred' or 'pred||target'")

        kl_mean = kl.mean()
        gamma = self._get_gamma()
        loss = self.alpha * mse_mean + self.beta * mse_std + gamma * kl_mean
        metrics = {
            "mse_mean": float(mse_mean.item()),
            "mse_std": float(mse_std.item()),
            "kl": float(kl_mean.item()),
            "gamma": float(gamma),
        }
        return loss, metrics


def train(cfg: Config):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, input_dim, action_dim = get_dataloaders_vector(
        cfg.data_path, cfg.batch_size, cfg.num_workers, seed=cfg.seed
    )
    if action_dim is None:
        raise ValueError(
            "Dataset must provide action labels (e.g., 'act') for training."
        )
    if not cfg.actor_ckpt:
        raise ValueError("Config requires actor_ckpt path to load the frozen actor.")
    cfg.action_dim = action_dim

    model = VQVAE(
        input_dim=input_dim,
        action_dim=action_dim,
        latent_dim=cfg.latent_dim,
        n_codes=cfg.n_codes,
        decay=cfg.decay,
        beta=cfg.beta,
        enc_hidden=cfg.enc_hidden,
        actor_hidden=cfg.actor_hidden,
        head_hidden=cfg.head_hidden,
        gumbel_tau=cfg.gumbel_tau,
        st_tau=cfg.st_tau,
        actor_ckpt=cfg.actor_ckpt,
    ).to(device)

    model.actor.to(device)

    ema = ModelEMA(model, cfg.ema_decay) if cfg.ema_decay > 0 else None

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    wandb_run = None
    if cfg.use_wandb:
        try:
            import wandb  # type: ignore
        except ImportError:
            print("wandb is not installed; disabling logging.")
        else:
            init_kwargs: dict[str, object] = {
                "project": cfg.wandb_project or "continuousVisionMask",
                "config": asdict(cfg),
            }
            if cfg.wandb_run_name:
                init_kwargs["name"] = cfg.wandb_run_name
            else:
                script_stem = Path(__file__).stem
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                init_kwargs["name"] = f"{script_stem}-{timestamp}"
            wandb_run = wandb.init(**init_kwargs)
    start_epoch = 0
    global_step = 0
    if cfg.resume:
        ck = load_ckpt(cfg.resume, model, opt)
        start_epoch = ck.get("epoch", 0)
        global_step = ck.get("step", 0)
        print(f"Resumed from {cfg.resume}: epoch {start_epoch}, step {global_step}")

    total_steps = cfg.epochs * len(train_loader)
    warmup_steps = int(0.02 * total_steps)

    kl_loss = GaussianMSEKLLoss(alpha=1.0, beta=1.0, min_std=1e-6)

    try:
        for epoch in range(start_epoch, cfg.epochs):
            model.train()
            kl_loss.set_epoch(epoch)
            running_nll, running_commit = 0.0, 0.0
            progress = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{cfg.epochs}",
                leave=False,
            )
            for i, batch in enumerate(progress):
                vecs = batch[0].to(device, non_blocking=True)
                target_mean = batch[1].to(device, non_blocking=True)
                target_std = batch[2].to(device, non_blocking=True)

                dist, commit_loss = model(vecs)
                pred_mean = dist.mean
                pred_std = dist.stddev

                kl_value, kl_info = kl_loss(
                    pred_mean=pred_mean,
                    pred_std=pred_std,
                    target_mean=target_mean,
                    target_std=target_std,
                )

                nll = kl_value
                loss = nll + cfg.beta * commit_loss

                opt.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()

                if ema is not None:
                    ema.update(model)

                # LR schedule
                lr = cosine_warmup_lr(opt, cfg.lr, warmup_steps, total_steps, global_step)

                running_nll += nll.item()
                running_commit += commit_loss.item()

                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/nll": float(nll.item()),
                            "train/commit": float(commit_loss.item()),
                            "train/lr": lr,
                            "train/mse_mean": kl_info.get("mse_mean", 0.0),
                            "train/mse_std": kl_info.get("mse_std", 0.0),
                            "train/kl": kl_info.get("kl", 0.0),
                            "train/gamma": kl_info.get("gamma", 0.0),
                        },
                        step=global_step,
                    )

                avg_nll = running_nll / (i + 1)
                avg_commit = running_commit / (i + 1)
                progress.set_postfix(
                    lr=f"{lr:.3e}",
                    nll=f"{avg_nll:.6f}",
                    commit=f"{avg_commit:.6f}",
                )

                global_step += 1
        progress.close()

        # Validation / Sampling
        model.eval()
        with torch.no_grad():
            # Use EMA weights for eval if enabled
            if ema is not None:
                shadow = ModelEMA(model, cfg.ema_decay)
                shadow.shadow = {k: v.clone() for k, v in ema.shadow.items()}
                shadow.copy_to(model)

            val_metrics = {}
            if val_loader is not None:
                total_nll = 0.0
                total_commit = 0.0
                total_examples = 0
                total_mse_mean = 0.0
                total_mse_std = 0.0
                total_kl = 0.0
                for v_batch in val_loader:
                    vecs_v = v_batch[0].to(device, non_blocking=True)
                    mean_v = v_batch[1].to(device, non_blocking=True)
                    std_v = v_batch[2].to(device, non_blocking=True)
                    dist_v, commit_loss_v = model(vecs_v)
                    pred_mean_v = dist_v.mean
                    pred_std_v = dist_v.stddev
                    kl_val, kl_info_val = kl_loss(
                        pred_mean=pred_mean_v,
                        pred_std=pred_std_v,
                        target_mean=mean_v,
                        target_std=std_v,
                    )
                    bs = vecs_v.size(0)
                    total_nll += kl_val.item() * bs
                    total_commit += commit_loss_v.item() * bs
                    total_examples += bs
                    total_mse_mean += kl_info_val["mse_mean"] * bs
                    total_mse_std += kl_info_val["mse_std"] * bs
                    total_kl += kl_info_val["kl"] * bs
                if total_examples > 0:
                    val_metrics = {
                        "val/nll": total_nll / total_examples,
                        "val/commit": total_commit / total_examples,
                    }
                    if total_examples > 0:
                        val_metrics["val/mse_mean"] = total_mse_mean / total_examples
                        val_metrics["val/mse_std"] = total_mse_std / total_examples
                        val_metrics["val/kl"] = total_kl / total_examples
                    if wandb_run is not None:
                        wandb_run.log(val_metrics, step=global_step)

            # Take a small batch for qualitative numeric inspection
            batch_full = next(
                iter(val_loader if val_loader is not None else train_loader)
            )
            batch_vecs = batch_full[0][:64].to(device)
            batch_mean = batch_full[1][:64].to(device)
            batch_std = batch_full[2][:64].to(device)
            dist, commit_loss = model(batch_vecs)
            dist_mean = dist.mean
            dist_std = dist.stddev
            # Save a few example pairs to .pt (tensor) for quick inspection
            os.makedirs(cfg.sample_dir, exist_ok=True)
            torch.save(
                {
                    "inputs": batch_vecs.cpu(),
                    "dist_mean": dist_mean.cpu(),
                    "dist_std": dist_std.cpu(),
                    "target_mean": batch_mean.cpu(),
                    "target_std": batch_std.cpu(),
                    "commit_loss": float(commit_loss.item()),
                },
                os.path.join(cfg.sample_dir, f"actor_epoch_{epoch+1:03d}.pt"),
            )

        # Save checkpoint
        save_ckpt(
            os.path.join(cfg.out_dir, f"epoch_{epoch+1:03d}.pt"),
            model,
            opt,
            epoch + 1,
            global_step,
            cfg,
        )
    finally:
        if wandb_run is not None:
            wandb_run.finish()


# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Train VQ‑VAE on vector data (.pt/)")
    p.add_argument("--data-path", type=str, default="./data/vectors.npy")

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-6)

    p.add_argument("--beta", type=float, default=0.25, help="commitment loss weight")
    p.add_argument("--n-codes", type=int, default=512)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--enc-hidden", type=int, nargs="*", default=[512, 512])
    p.add_argument("--actor-hidden", type=int, nargs="*", default=[64, 64])
    p.add_argument(
        "--head-hidden",
        type=int,
        nargs="*",
        default=[],
        help="hidden layer sizes for the index prediction head",
    )
    p.add_argument(
        "--actor-ckpt",
        type=str,
        required=True,
        help="Path to pretrained actor state_dict to freeze (expects Actor architecture).",
    )
    p.add_argument(
        "--gumbel-tau",
        type=float,
        default=1.0,
        help="temperature to use for Gumbel-Softmax sampling",
    )
    p.add_argument(
        "--st-tau",
        type=float,
        default=1.0,
        help="unused placeholder for compatibility",
    )
    p.add_argument("--decay", type=float, default=0.99, help="EMA decay for codebook")
    p.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    p.add_argument(
        "--wandb-project",
        type=str,
        default="",
        help="Weights & Biases project name",
    )
    p.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Optional Weights & Biases run name",
    )

    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument(
        "--ema-decay",
        type=float,
        default=0.0,
        help=">0 to enable model EMA, e.g. 0.999",
    )

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
        actor_hidden=tuple(args.actor_hidden),
        head_hidden=tuple(args.head_hidden),
        gumbel_tau=args.gumbel_tau,
        st_tau=args.st_tau,
        actor_ckpt=args.actor_ckpt,
        decay=args.decay,
        num_workers=args.num_workers,
        grad_clip=args.grad_clip,
        ema_decay=args.ema_decay,
        out_dir=args.out_dir,
        sample_dir=args.sample_dir,
        resume=args.resume,
        seed=args.seed,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )

    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(cfg.sample_dir, exist_ok=True)

    print(cfg)
    train(cfg)
