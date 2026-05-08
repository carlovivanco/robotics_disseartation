"""
train_il.py — Imitation Learning: Diffusion Policy or DAgger.

Replaces Imitation.py with:
  - Hydra configuration (mode flag instead of comment-toggling)
  - Diffusion Policy (DDPM + DDIM) as the primary IL method
  - DAgger as an optional interactive correction loop
  - W&B + TensorBoard logging
  - Reproducible seeding
  - Unified CLI for train / eval / dagger modes

Usage
-----
  # Train Diffusion Policy:
  python train_il.py mode=train il=diffusion

  # Evaluate trained Diffusion Policy:
  python train_il.py mode=eval il=diffusion

  # Run DAgger interactive loop:
  python train_il.py mode=dagger il=dagger

  # Train legacy BC (optional, for comparison):
  python train_il.py mode=train il=bc     # (falls back to Imitation.py logic)

  # Optuna sweep (Hydra multirun):
  python train_il.py --multirun il.training.learning_rate=1e-5,1e-4,3e-4
"""
from __future__ import annotations

import os
import collections
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from utils.seed import set_global_seed
from utils.logger import Logger


# ═══════════════════════════════════════════════════════════════════════════════
# Diffusion Policy — train
# ═══════════════════════════════════════════════════════════════════════════════

def train_diffusion(cfg: DictConfig, logger: Logger) -> None:
    from algorithms.diffusion_policy import DiffusionPolicyTrainer

    trainer = DiffusionPolicyTrainer(cfg.il, device=cfg.device)
    trainer.train(logger=logger)
    print(f"Diffusion Policy saved to {cfg.il.model_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Diffusion Policy — eval
# ═══════════════════════════════════════════════════════════════════════════════

def eval_diffusion(cfg: DictConfig) -> None:
    from algorithms.diffusion_policy import DiffusionPolicy
    import pandas as pd

    policy = DiffusionPolicy.load(cfg.il.model_path, cfg.il, device=cfg.device)
    env = _make_eval_env(cfg)
    obs_raw, _ = env.reset()

    obs_window = collections.deque(
        [_flatten(obs_raw)] * cfg.il.obs_horizon, maxlen=cfg.il.obs_horizon
    )

    steps, rewards = [], []
    done = False
    t = 0

    while not done:
        t += 1
        # DDIM: predict (action_horizon, act_dim) chunk
        actions = policy.predict(list(obs_window))

        # Execute the first action of the predicted chunk
        action = actions[0]
        obs_raw, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        obs_window.append(_flatten(obs_raw))
        steps.append(t)
        rewards.append(float(reward))
        env.render()
        print(f"Step {t:4d} | Reward {reward:.4f}")

    env.close()
    pd.DataFrame({"step": steps, "reward": rewards}).to_csv(
        "Evaluation_diffusion.csv", index=False
    )
    print(f"Mean reward: {np.mean(rewards):.3f} | Episode length: {t}")


# ═══════════════════════════════════════════════════════════════════════════════
# DAgger
# ═══════════════════════════════════════════════════════════════════════════════

def run_dagger(cfg: DictConfig, logger: Logger) -> None:
    from algorithms.dagger import DAgger, BCOracle, DRLOracle
    import h5py

    # Load normalisation stats from existing demos (needed for oracle)
    obs_list, act_list = [], []
    with h5py.File(cfg.il.dataset.hdf5_path, "r") as f:
        for k in f["data"].keys():
            obs_list.append(f["data"][k]["observations"][:])
            act_list.append(f["data"][k]["actions"][:])
    obs_all = np.concatenate(obs_list)
    act_all = np.concatenate(act_list)
    obs_mean, obs_std = obs_all.mean(0), obs_all.std(0) + 1e-8
    act_mean, act_std = act_all.mean(0), act_all.std(0) + 1e-8

    oracle_cfg = cfg.il.oracle
    if oracle_cfg.type == "bc":
        oracle = BCOracle(
            oracle_cfg.path, obs_mean, obs_std, act_mean, act_std, device=cfg.device
        )
    else:
        oracle = DRLOracle(oracle_cfg.path, algo="TQC", device=cfg.device)

    def env_factory():
        from envs.robosuite_wrapper import DomainRandomizedGymWrapper
        return DomainRandomizedGymWrapper(
            has_renderer=False,
            horizon=cfg.il.horizon,
            dr_cfg=cfg.env.domain_randomization,
            reward_cfg=cfg.env.reward,
        )

    dagger = DAgger(cfg.il, oracle=oracle, env_factory=env_factory, device=cfg.device)
    dagger.run(logger=logger)


# ═══════════════════════════════════════════════════════════════════════════════
# Legacy BC training (kept for comparison / ablation)
# ═══════════════════════════════════════════════════════════════════════════════

def train_bc_legacy(cfg: DictConfig, logger: Logger) -> None:
    """
    Mirrors the original Imitation.py training loop but driven by Hydra config.
    Useful as a baseline to compare against Diffusion Policy.
    """
    import h5py
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset, random_split
    import pandas as pd

    # Load demos
    obs_list, act_list = [], []
    with h5py.File(cfg.il.dataset.hdf5_path, "r") as f:
        for k in f["data"].keys():
            obs_list.append(f["data"][k]["observations"][:])
            act_list.append(f["data"][k]["actions"][:])
    obs = np.concatenate(obs_list)
    acts = np.concatenate(act_list)

    obs_mean, obs_std = obs.mean(0), obs.std(0) + 1e-8
    act_mean, act_std = acts.mean(0), acts.std(0) + 1e-8
    obs_n = (obs - obs_mean) / obs_std
    acts_n = (acts - act_mean) / act_std

    obs_t = torch.tensor(obs_n, dtype=torch.float32)
    acts_t = torch.tensor(acts_n, dtype=torch.float32)
    dataset = TensorDataset(obs_t, acts_t)

    val_n = max(1, int(0.2 * len(dataset)))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_n, val_n])
    batch = cfg.il.training.batch_size
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch)

    dev = cfg.device
    net = nn.Sequential(
        nn.Linear(obs.shape[1], cfg.il.network.hidden_dim1),
        nn.LayerNorm(cfg.il.network.hidden_dim1), nn.ReLU(),
        nn.Linear(cfg.il.network.hidden_dim1, cfg.il.network.hidden_dim2),
        nn.LayerNorm(cfg.il.network.hidden_dim2), nn.ReLU(),
        nn.Linear(cfg.il.network.hidden_dim2, acts.shape[1]),
    ).to(dev)

    opt = optim.AdamW(net.parameters(), lr=cfg.il.training.learning_rate)
    loss_fn = nn.MSELoss()
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.il.training.n_epochs)

    best_val, patience_cnt = float("inf"), 0
    history = []

    for ep in range(cfg.il.training.n_epochs):
        net.train()
        tl = sum(
            (loss_fn(net(xb.to(dev)), yb.to(dev)).backward() or
             opt.zero_grad() or
             loss_fn(net(xb.to(dev)), yb.to(dev))).item()
            for xb, yb in train_loader
        ) / len(train_loader)

        # Simpler loop
        net.train()
        tl = 0.0
        for xb, yb in train_loader:
            pred = net(xb.to(dev))
            loss = loss_fn(pred, yb.to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item()
        tl /= len(train_loader)

        net.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                vl += loss_fn(net(xb.to(dev)), yb.to(dev)).item()
        vl /= max(len(val_loader), 1)
        sched.step()

        history.append({"epoch": ep, "train_loss": tl, "val_loss": vl})
        print(f"[BC Epoch {ep+1}] train={tl:.5f} val={vl:.5f}")
        if logger:
            logger.log({"train/loss": tl, "val/loss": vl}, step=ep)

        if vl < best_val:
            best_val = vl
            patience_cnt = 0
            torch.save(
                {"net": net.state_dict(), "obs_mean": obs_mean, "obs_std": obs_std,
                 "act_mean": act_mean, "act_std": act_std},
                "bc_model_best_SD.pth",
            )
        else:
            patience_cnt += 1
            if patience_cnt >= cfg.il.training.get("early_stopping_patience", 30):
                print("Early stopping.")
                break

    pd.DataFrame(history).to_csv("training_bc_legacy.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _make_eval_env(cfg: DictConfig):
    from envs.robosuite_wrapper import DomainRandomizedGymWrapper
    return DomainRandomizedGymWrapper(
        has_renderer=True,
        horizon=cfg.env.get("horizon", 2000),
        dr_cfg=cfg.env.domain_randomization,
        reward_cfg=cfg.env.reward,
        seed=cfg.seed,
    )


def _flatten(obs) -> np.ndarray:
    if isinstance(obs, dict):
        return np.concatenate([np.asarray(v).flatten() for v in obs.values()])
    return np.asarray(obs).flatten()


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    set_global_seed(cfg.seed)

    logger = Logger(cfg, cfg.output_dir)

    mode = cfg.mode
    il_name = cfg.il.name if hasattr(cfg, "il") else "diffusion"

    if mode == "train":
        if il_name == "diffusion":
            train_diffusion(cfg, logger)
        elif il_name == "dagger":
            run_dagger(cfg, logger)
        else:
            train_bc_legacy(cfg, logger)

    elif mode == "eval":
        if il_name == "diffusion":
            eval_diffusion(cfg)
        else:
            print("Use evaluate.py for other policy types.")

    elif mode == "dagger":
        run_dagger(cfg, logger)

    else:
        raise ValueError(f"Unknown mode: {mode}. Use train | eval | dagger.")

    logger.close()


if __name__ == "__main__":
    main()
