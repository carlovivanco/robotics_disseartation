"""
Diffusion Policy (Chi et al. 2023) for bolt screwdriving.

Architecture
------------
- Conditional 1-D temporal U-Net: denoises an action sequence conditioned on
  an observation window.
- DDPM training with a squaredcos_cap_v2 noise schedule.
- DDIM inference: reduces denoising steps from 100 → 10 with no quality loss.
- EMA weights for stable inference.

Inputs / Outputs
----------------
  Observation: (B, obs_horizon, obs_dim)   e.g. (B, 2, 64)
  Action:      (B, pred_horizon, act_dim)  e.g. (B, 16, 7)

References
----------
  Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
  RSS 2023.  https://diffusion-policy.cs.columbia.edu/
"""
from __future__ import annotations

import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import h5py
import pandas as pd
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
# Noise schedule helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _squaredcos_cap_v2(t: torch.Tensor, s: float = 0.008) -> torch.Tensor:
    """OpenAI improved cosine schedule (Nichol & Dhariwal 2021)."""
    return ((t / 1.0 + s) / (1.0 + s) * math.pi / 2).cos() ** 2


def make_beta_schedule(
    n_steps: int,
    schedule: str = "squaredcos_cap_v2",
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
) -> torch.Tensor:
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, n_steps)
    elif schedule == "squaredcos_cap_v2":
        steps = torch.linspace(0, 1, n_steps + 1)
        alphas_cumprod = _squaredcos_cap_v2(steps)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        return betas.clamp(0.0001, 0.9999)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


# ═══════════════════════════════════════════════════════════════════════════════
# 1-D U-Net building blocks
# ═══════════════════════════════════════════════════════════════════════════════

class SinusoidalPosEmb(nn.Module):
    """Diffusion timestep embedding (used to condition the U-Net on t)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=x.device) / (half - 1)
        )
        emb = x[:, None] * freqs[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResBlock1D(nn.Module):
    """
    1-D residual block with GroupNorm and FiLM conditioning.
    The diffusion timestep (and optionally observation) modulates activations
    via Feature-wise Linear Modulation: y = scale * x + shift.
    """

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, kernel_size: int = 5, n_groups: int = 8) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block1 = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding),
            nn.GroupNorm(n_groups, out_ch),
            nn.Mish(),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding),
            nn.GroupNorm(n_groups, out_ch),
            nn.Mish(),
        )
        # FiLM conditioning network
        self.cond_proj = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_ch * 2),
        )
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)   cond: (B, cond_dim)
        h = self.block1(x)
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
        h = h * (scale.unsqueeze(-1) + 1) + shift.unsqueeze(-1)
        h = self.block2(h)
        return h + self.residual(x)


class ConditionalUNet1D(nn.Module):
    """
    Temporal U-Net that predicts the noise ε added to the action sequence.

    Accepts:
      noisy_actions:  (B, act_dim, pred_horizon)
      timestep:       (B,)  integer diffusion step
      obs_cond:       (B, obs_horizon * obs_dim)  flattened observation window

    Returns:
      noise_pred:     (B, act_dim, pred_horizon)
    """

    def __init__(
        self,
        act_dim: int,
        obs_dim: int,
        obs_horizon: int,
        pred_horizon: int,
        down_dims: list[int] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        diffusion_step_embed_dim: int = 256,
    ) -> None:
        super().__init__()

        # Condition dimension = timestep emb + obs flattened
        self.obs_horizon = obs_horizon
        self.obs_dim = obs_dim
        step_emb_dim = diffusion_step_embed_dim
        obs_cond_dim = obs_horizon * obs_dim
        cond_dim = step_emb_dim + obs_cond_dim

        # Timestep embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(step_emb_dim),
            nn.Linear(step_emb_dim, step_emb_dim * 4),
            nn.Mish(),
            nn.Linear(step_emb_dim * 4, step_emb_dim),
        )

        # Encoder / down path
        channels = [act_dim] + list(down_dims)
        self.down_blocks = nn.ModuleList()
        self.down_sample = nn.ModuleList()
        for i in range(len(down_dims)):
            self.down_blocks.append(
                nn.ModuleList([
                    ResBlock1D(channels[i], channels[i+1], cond_dim, kernel_size, n_groups),
                    ResBlock1D(channels[i+1], channels[i+1], cond_dim, kernel_size, n_groups),
                ])
            )
            self.down_sample.append(
                nn.Conv1d(channels[i+1], channels[i+1], 3, stride=2, padding=1)
                if i < len(down_dims) - 1 else nn.Identity()
            )

        # Bottleneck
        mid_dim = down_dims[-1]
        self.mid_block1 = ResBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups)
        self.mid_block2 = ResBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups)

        # Decoder / up path
        self.up_blocks = nn.ModuleList()
        self.up_sample = nn.ModuleList()
        rev_channels = list(reversed(channels))
        for i in range(len(down_dims)):
            in_ch = rev_channels[i] + rev_channels[i+1]  # skip connection
            out_ch = rev_channels[i+1]
            self.up_blocks.append(
                nn.ModuleList([
                    ResBlock1D(in_ch, out_ch, cond_dim, kernel_size, n_groups),
                    ResBlock1D(out_ch, out_ch, cond_dim, kernel_size, n_groups),
                ])
            )
            self.up_sample.append(
                nn.ConvTranspose1d(rev_channels[i], rev_channels[i], 4, stride=2, padding=1)
                if i < len(down_dims) - 1 else nn.Identity()
            )

        self.final_conv = nn.Sequential(
            nn.GroupNorm(n_groups, down_dims[0]),
            nn.Mish(),
            nn.Conv1d(down_dims[0], act_dim, 1),
        )

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        obs_cond: torch.Tensor,
    ) -> torch.Tensor:
        # Build FiLM condition vector
        t_emb = self.time_mlp(timestep.float())
        cond = torch.cat([t_emb, obs_cond], dim=-1)

        # noisy_actions: (B, act_dim, T)
        x = noisy_actions
        skips = []

        for (r1, r2), ds in zip(self.down_blocks, self.down_sample):
            x = r1(x, cond)
            x = r2(x, cond)
            skips.append(x)
            x = ds(x)

        x = self.mid_block1(x, cond)
        x = self.mid_block2(x, cond)

        for (r1, r2), us in zip(self.up_blocks, self.up_sample):
            x = us(x)
            skip = skips.pop()
            # Align temporal dimension before cat (edge case with odd T)
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1])
            x = torch.cat([x, skip], dim=1)
            x = r1(x, cond)
            x = r2(x, cond)

        return self.final_conv(x)


# ═══════════════════════════════════════════════════════════════════════════════
# EMA helper
# ═══════════════════════════════════════════════════════════════════════════════

class EMA:
    """Exponential Moving Average of model weights for stable inference."""

    def __init__(self, model: nn.Module, power: float = 0.75, max_value: float = 0.9999) -> None:
        self.model = copy.deepcopy(model)
        self.power = power
        self.max_value = max_value
        self._step = 0

    def update(self, model: nn.Module) -> None:
        self._step += 1
        decay = min(self.max_value, 1 - (1 + self._step) ** -self.power)
        with torch.no_grad():
            for ema_p, p in zip(self.model.parameters(), model.parameters()):
                ema_p.lerp_(p, 1 - decay)


# ═══════════════════════════════════════════════════════════════════════════════
# Diffusion Policy
# ═══════════════════════════════════════════════════════════════════════════════

class DiffusionPolicy:
    """
    Inference wrapper around ConditionalUNet1D + DDPM/DDIM scheduler.

    Usage::

        policy = DiffusionPolicy.load("diffusion_model_best.pth", cfg, device)
        obs_window = deque(maxlen=cfg.obs_horizon)
        obs_window.append(obs)
        actions = policy.predict(obs_window)   # (action_horizon, act_dim)
        env.step(actions[0])
    """

    def __init__(self, net: ConditionalUNet1D, cfg, device: str = "cuda") -> None:
        self.net = net.to(device)
        self.cfg = cfg
        self.device = device

        # Build noise schedule
        betas = make_beta_schedule(
            cfg.n_diffusion_steps,
            schedule=cfg.beta_schedule,
            beta_start=cfg.beta_start,
            beta_end=cfg.beta_end,
        ).to(device)
        alphas = 1.0 - betas
        self.alphas_cumprod = alphas.cumprod(0)
        self.betas = betas

        # DDIM inference — pre-compute subset of timesteps
        self._ddim_ts = torch.linspace(
            0, cfg.n_diffusion_steps - 1, cfg.n_inference_steps, dtype=torch.long, device=device
        ).flip(0)

        # Normalisation stats (set by trainer after fitting)
        self.obs_mean: np.ndarray | None = None
        self.obs_std: np.ndarray | None = None
        self.act_mean: np.ndarray | None = None
        self.act_std: np.ndarray | None = None

    @torch.no_grad()
    def predict(self, obs_window: list[np.ndarray]) -> np.ndarray:
        """
        Run DDIM denoising to produce an action sequence.

        obs_window: list of `obs_horizon` observations (each shape obs_dim)
        Returns: (action_horizon, act_dim) numpy array (unnormalised)
        """
        cfg = self.cfg
        B = 1
        dev = self.device

        # Build obs condition
        obs_arr = np.stack(list(obs_window), axis=0)  # (obs_horizon, obs_dim)
        if self.obs_mean is not None:
            obs_arr = (obs_arr - self.obs_mean) / (self.obs_std + 1e-8)
        obs_cond = torch.tensor(obs_arr, dtype=torch.float32, device=dev).flatten().unsqueeze(0)

        # Start from pure noise
        x = torch.randn(B, cfg.action_dim, cfg.pred_horizon, device=dev)

        # DDIM reverse diffusion
        for i, t in enumerate(self._ddim_ts):
            t_batch = t.expand(B)
            alpha_bar_t = self.alphas_cumprod[t]
            alpha_bar_prev = self.alphas_cumprod[self._ddim_ts[i+1]] \
                if i + 1 < len(self._ddim_ts) else torch.tensor(1.0, device=dev)

            eps = self.net(x, t_batch, obs_cond)
            x0_pred = (x - eps * (1 - alpha_bar_t).sqrt()) / alpha_bar_t.sqrt()
            x0_pred = x0_pred.clamp(-1, 1)

            # DDIM step (no stochastic noise at inference)
            x = alpha_bar_prev.sqrt() * x0_pred + \
                (1 - alpha_bar_prev).sqrt() * eps

        # (B, act_dim, pred_horizon) → (pred_horizon, act_dim)
        actions = x[0].permute(1, 0).cpu().numpy()[:cfg.action_horizon]

        if self.act_mean is not None:
            actions = actions * (self.act_std + 1e-8) + self.act_mean
        return actions

    @classmethod
    def load(cls, path: str, cfg, device: str = "cuda") -> "DiffusionPolicy":
        net = ConditionalUNet1D(
            act_dim=cfg.action_dim,
            obs_dim=cfg.obs_dim,
            obs_horizon=cfg.obs_horizon,
            pred_horizon=cfg.pred_horizon,
            down_dims=list(cfg.network.down_dims),
            kernel_size=cfg.network.kernel_size,
            n_groups=cfg.network.n_groups,
            diffusion_step_embed_dim=cfg.network.diffusion_step_embed_dim,
        )
        ckpt = torch.load(path, map_location=device, weights_only=True)
        net.load_state_dict(ckpt["net"])
        policy = cls(net, cfg, device)
        policy.obs_mean = ckpt.get("obs_mean")
        policy.obs_std = ckpt.get("obs_std")
        policy.act_mean = ckpt.get("act_mean")
        policy.act_std = ckpt.get("act_std")
        return policy


# ═══════════════════════════════════════════════════════════════════════════════
# Diffusion Policy Trainer
# ═══════════════════════════════════════════════════════════════════════════════

class DiffusionPolicyTrainer:
    """
    Full training pipeline for DiffusionPolicy:
      1. Load HDF5 demos
      2. Normalise observations and actions
      3. Build temporal sequence dataset (sliding windows)
      4. Train ConditionalUNet1D with DDPM loss
      5. Track EMA weights and save best checkpoint
    """

    def __init__(self, cfg, device: str = "cuda") -> None:
        self.cfg = cfg
        self.device = device

        net = ConditionalUNet1D(
            act_dim=cfg.action_dim,
            obs_dim=cfg.obs_dim,
            obs_horizon=cfg.obs_horizon,
            pred_horizon=cfg.pred_horizon,
            down_dims=list(cfg.network.down_dims),
            kernel_size=cfg.network.kernel_size,
            n_groups=cfg.network.n_groups,
            diffusion_step_embed_dim=cfg.network.diffusion_step_embed_dim,
        )
        self.policy = DiffusionPolicy(net, cfg, device)
        self.ema = EMA(net, power=cfg.ema.power, max_value=cfg.ema.max_value) \
            if cfg.ema.enabled else None

        # Build noise schedule tensors on device
        self.n_steps = cfg.n_diffusion_steps
        betas = make_beta_schedule(
            self.n_steps, cfg.beta_schedule, cfg.beta_start, cfg.beta_end
        ).to(device)
        alphas = 1.0 - betas
        self.alphas_cumprod = alphas.cumprod(0)
        self.sqrt_alphas_cumprod = self.alphas_cumprod.sqrt()
        self.sqrt_one_minus = (1 - self.alphas_cumprod).sqrt()

    # ── data loading ──────────────────────────────────────────────────────────

    def _load_demos(self) -> tuple[np.ndarray, np.ndarray]:
        """Load all (obs, action) pairs from the HDF5 demo file."""
        obs_list, act_list = [], []
        with h5py.File(self.cfg.dataset.hdf5_path, "r") as f:
            for key in f["data"].keys():
                demo = f["data"][key]
                obs_list.append(demo["observations"][:])
                act_list.append(demo["actions"][:])
        return np.concatenate(obs_list), np.concatenate(act_list)

    def _build_dataset(self):
        """
        Create temporal sliding-window sequences:
          obs_seq: (N, obs_horizon, obs_dim)
          act_seq: (N, pred_horizon, act_dim)
        """
        obs_raw, act_raw = self._load_demos()
        cfg = self.cfg

        # Compute normalisation stats
        self.obs_mean = obs_raw.mean(0)
        self.obs_std = obs_raw.std(0) + 1e-8
        self.act_mean = act_raw.mean(0)
        self.act_std = act_raw.std(0) + 1e-8

        if cfg.dataset.normalize:
            obs_raw = (obs_raw - self.obs_mean) / self.obs_std
            act_raw = (act_raw - self.act_mean) / self.act_std

        # Sliding window
        obs_seqs, act_seqs = [], []
        T = len(obs_raw)
        oh, ph = cfg.obs_horizon, cfg.pred_horizon
        for i in range(oh - 1, T - ph + 1):
            obs_seqs.append(obs_raw[i - oh + 1: i + 1])   # (obs_horizon, obs_dim)
            act_seqs.append(act_raw[i: i + ph])            # (pred_horizon, act_dim)

        obs_t = torch.tensor(np.stack(obs_seqs), dtype=torch.float32)
        act_t = torch.tensor(np.stack(act_seqs), dtype=torch.float32)

        # Optional observation augmentation
        if cfg.dataset.augment:
            obs_t = obs_t + torch.randn_like(obs_t) * 0.01

        dataset = TensorDataset(obs_t, act_t)
        val_n = max(1, int(len(dataset) * cfg.training.val_split))
        train_n = len(dataset) - val_n
        return random_split(dataset, [train_n, val_n])

    # ── DDPM forward process ──────────────────────────────────────────────────

    def _add_noise(
        self, x0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample q(x_t | x_0) = sqrt(ᾱ_t)·x0 + sqrt(1-ᾱ_t)·ε"""
        eps = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_m = self.sqrt_one_minus[t].view(-1, 1, 1)
        x_t = sqrt_ab * x0 + sqrt_one_m * eps
        return x_t, eps

    # ── training loop ─────────────────────────────────────────────────────────

    def train(self, logger=None) -> None:
        cfg = self.cfg
        train_ds, val_ds = self._build_dataset()

        train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size)

        opt = torch.optim.AdamW(
            self.policy.net.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg.training.n_epochs
        )
        # Linear warmup
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1e-4, end_factor=1.0,
            total_iters=cfg.training.lr_warmup_steps
        )
        combined_sched = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup, scheduler], milestones=[cfg.training.lr_warmup_steps]
        )

        best_val = float("inf")
        patience = 0
        history = []
        dev = self.device

        for epoch in range(cfg.training.n_epochs):
            # ── train ─────────────────────────────────────────────────────────
            self.policy.net.train()
            train_loss = 0.0
            for obs_seq, act_seq in train_loader:
                obs_seq = obs_seq.to(dev)
                act_seq = act_seq.to(dev)

                B = obs_seq.shape[0]
                t = torch.randint(0, self.n_steps, (B,), device=dev)

                # act_seq: (B, pred_horizon, act_dim) → (B, act_dim, pred_horizon)
                x0 = act_seq.permute(0, 2, 1)
                x_t, eps = self._add_noise(x0, t)

                obs_cond = obs_seq.flatten(start_dim=1)  # (B, obs_horizon * obs_dim)
                eps_pred = self.policy.net(x_t, t, obs_cond)

                loss = F.mse_loss(eps_pred, eps)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.net.parameters(), cfg.training.gradient_clip)
                opt.step()
                combined_sched.step()

                if self.ema is not None:
                    self.ema.update(self.policy.net)

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # ── validation ────────────────────────────────────────────────────
            self.policy.net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for obs_seq, act_seq in val_loader:
                    obs_seq = obs_seq.to(dev)
                    act_seq = act_seq.to(dev)
                    B = obs_seq.shape[0]
                    t = torch.randint(0, self.n_steps, (B,), device=dev)
                    x0 = act_seq.permute(0, 2, 1)
                    x_t, eps = self._add_noise(x0, t)
                    obs_cond = obs_seq.flatten(start_dim=1)
                    eps_pred = self.policy.net(x_t, t, obs_cond)
                    val_loss += F.mse_loss(eps_pred, eps).item()
            val_loss /= max(len(val_loader), 1)

            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            print(f"[Epoch {epoch+1}/{cfg.training.n_epochs}] "
                  f"train={train_loss:.5f}  val={val_loss:.5f}")

            if logger is not None:
                logger.log({"train/loss": train_loss, "val/loss": val_loss}, step=epoch)

            # ── save best ─────────────────────────────────────────────────────
            if val_loss < best_val:
                best_val = val_loss
                patience = 0
                self._save(cfg.model_path)
                print(f"  → Best model saved (val={best_val:.5f})")
            else:
                patience += 1
                if patience >= cfg.training.early_stopping_patience:
                    print("Early stopping triggered.")
                    break

        pd.DataFrame(history).to_csv("training_diffusion.csv", index=False)

        # Attach normalisation stats to policy for inference
        self.policy.obs_mean = self.obs_mean
        self.policy.obs_std = self.obs_std
        self.policy.act_mean = self.act_mean
        self.policy.act_std = self.act_std

    # ── save / load ───────────────────────────────────────────────────────────

    def _save(self, path: str) -> None:
        # Use EMA weights for inference if available
        net_state = self.ema.model.state_dict() if self.ema is not None \
            else self.policy.net.state_dict()
        torch.save(
            {
                "net": net_state,
                "obs_mean": getattr(self, "obs_mean", None),
                "obs_std": getattr(self, "obs_std", None),
                "act_mean": getattr(self, "act_mean", None),
                "act_std": getattr(self, "act_std", None),
            },
            path,
        )
