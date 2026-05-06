"""
DAgger — Dataset Aggregation (Ross, Gordon & Bagnell, AISTATS 2011)

Fixes Behavior Cloning's compounding-error problem by interactively
querying an oracle (expert) at states visited by the *learned* policy.

Algorithm
---------
For iteration i = 0 … N:
  1. Mix policy: π_i = β_i · π_expert + (1-β_i) · π_learned
  2. Roll out π_i in the environment to collect states
  3. Query π_expert for the *optimal* action at each visited state
  4. Aggregate dataset: D ← D ∪ {(s, a_expert)}
  5. Retrain π_learned on the full aggregated D

β decays each iteration (exponential schedule), so the policy transitions
from pure imitation → autonomous operation.

Oracle options
--------------
  'bc'   – pre-trained BehaviorCloningModel (from Imitation.py)
  'drl'  – pre-trained SB3 DRL model (TD3 / TQC)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import h5py
import pandas as pd
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
# Oracle wrappers
# ═══════════════════════════════════════════════════════════════════════════════

class BCOracle:
    """Wraps a pre-trained BehaviorCloningModel as a DAgger oracle."""

    def __init__(self, model_path: str, obs_mean: np.ndarray, obs_std: np.ndarray,
                 act_mean: np.ndarray, act_std: np.ndarray, device: str = "cuda") -> None:
        self.device = device
        self.obs_mean = obs_mean
        self.obs_std = obs_std
        self.act_mean = act_mean
        self.act_std = act_std

        # Import lazily to avoid circular deps
        from Imitation import BehaviorCloningModel, get_activation
        # Infer dims from saved weights
        ckpt = torch.load(model_path, map_location=device, weights_only=True)
        obs_dim = ckpt[list(ckpt.keys())[0]].shape[1]
        act_dim = ckpt[list(ckpt.keys())[-1]].shape[0]
        model = BehaviorCloningModel(
            obs_dim=obs_dim, act_dim=act_dim,
            hidden_dim1=256, hidden_dim2=1024,
            dropout_prob=0.0, activation=get_activation("relu"),
        )
        model.load_state_dict(ckpt)
        model.eval()
        self.model = model.to(device)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        obs_norm = (obs - self.obs_mean) / (self.obs_std + 1e-8)
        t = torch.tensor(obs_norm, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            a_norm = self.model(t).cpu().numpy().flatten()
        return a_norm * (self.act_std + 1e-8) + self.act_mean


class DRLOracle:
    """Wraps a Stable-Baselines3 policy (TQC/TD3/SAC) as a DAgger oracle."""

    def __init__(self, model_path: str, vecnorm_path: str | None = None,
                 algo: str = "TQC", device: str = "cuda") -> None:
        if algo == "TQC":
            from sb3_contrib import TQC as Algo
        elif algo == "TD3":
            from stable_baselines3 import TD3 as Algo
        else:
            from stable_baselines3 import SAC as Algo
        self.model = Algo.load(model_path, device=device)
        self._vecnorm = None
        if vecnorm_path is not None:
            from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
            # VecNormalize needs a venv to load — use a dummy single-env wrapper
            self._vecnorm_path = vecnorm_path

    def predict(self, obs: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(action).flatten()


# ═══════════════════════════════════════════════════════════════════════════════
# DAgger policy (MLP, same architecture as BC)
# ═══════════════════════════════════════════════════════════════════════════════

class _MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden1: int = 256, hidden2: int = 1024) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden1),
            nn.LayerNorm(hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════════════════
# DAgger trainer
# ═══════════════════════════════════════════════════════════════════════════════

class DAgger:
    """
    DAgger trainer.

    Parameters
    ----------
    cfg : OmegaConf / dataclass
        dagger.yaml config node
    oracle : BCOracle | DRLOracle
        The expert that labels visited states
    env_factory : callable
        Returns a fresh (unwrapped) environment instance
    device : str
    """

    def __init__(self, cfg, oracle, env_factory, device: str = "cuda") -> None:
        self.cfg = cfg
        self.oracle = oracle
        self.env_factory = env_factory
        self.device = device

        # Dataset (grows across iterations)
        self._obs_buffer: list[np.ndarray] = []
        self._act_buffer: list[np.ndarray] = []

        # Normalisation stats (fit once from seed demos)
        self.obs_mean: np.ndarray | None = None
        self.obs_std: np.ndarray | None = None
        self.act_mean: np.ndarray | None = None
        self.act_std: np.ndarray | None = None

        self.policy: _MLPPolicy | None = None
        self.obs_dim: int | None = None
        self.act_dim: int | None = None

    # ── public entry point ────────────────────────────────────────────────────

    def run(self, logger=None) -> None:
        """Execute the full DAgger loop."""
        self._seed_from_hdf5()
        self._init_policy()

        history = []
        for it in range(self.cfg.n_iterations):
            beta = self._beta(it)
            print(f"\n[DAgger] Iteration {it+1}/{self.cfg.n_iterations}  β={beta:.3f}")

            # Collect rollouts using the mixed policy
            new_obs, new_acts = self._collect_rollouts(beta)
            self._obs_buffer.append(new_obs)
            self._act_buffer.append(new_acts)

            # Retrain on full aggregated dataset
            val_loss = self._retrain(it)
            history.append({"iteration": it, "val_loss": val_loss, "beta": beta})

            if logger is not None:
                logger.log({"dagger/val_loss": val_loss, "dagger/beta": beta}, step=it)

        pd.DataFrame(history).to_csv("training_dagger.csv", index=False)
        self._save(self.cfg.model_path)

    # ── initialisation ────────────────────────────────────────────────────────

    def _seed_from_hdf5(self) -> None:
        """Bootstrap dataset from existing HDF5 demonstrations."""
        obs_list, act_list = [], []
        with h5py.File(self.cfg.dataset.hdf5_path, "r") as f:
            for key in f["data"].keys():
                obs_list.append(f["data"][key]["observations"][:])
                act_list.append(f["data"][key]["actions"][:])

        obs = np.concatenate(obs_list)
        acts = np.concatenate(act_list)

        self.obs_mean = obs.mean(0)
        self.obs_std = obs.std(0) + 1e-8
        self.act_mean = acts.mean(0)
        self.act_std = acts.std(0) + 1e-8

        self._obs_buffer.append(obs)
        self._act_buffer.append(acts)

        self.obs_dim = obs.shape[1]
        self.act_dim = acts.shape[1]
        print(f"[DAgger] Seeded with {len(obs)} transitions from HDF5 demos.")

    def _init_policy(self) -> None:
        cfg = self.cfg.network
        self.policy = _MLPPolicy(
            self.obs_dim, self.act_dim,
            cfg.hidden_dim1, cfg.hidden_dim2
        ).to(self.device)

    def _beta(self, iteration: int) -> float:
        if self.cfg.beta_schedule == "exponential":
            return self.cfg.beta_start * (self.cfg.beta_decay ** iteration)
        elif self.cfg.beta_schedule == "linear":
            return max(0.0, self.cfg.beta_start - iteration / self.cfg.n_iterations)
        return 0.0  # constant policy execution

    # ── rollout collection ────────────────────────────────────────────────────

    def _collect_rollouts(self, beta: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Run the mixed policy π = β·expert + (1-β)·learned in the environment,
        record states, and query the expert for the labelled actions.
        """
        obs_list, act_list = [], []

        for ep in range(self.cfg.rollouts_per_iter):
            env = self.env_factory()
            obs_raw, _ = env.reset()
            done = False
            step = 0

            while not done and step < self.cfg.horizon:
                obs_flat = self._flatten(obs_raw)

                # Expert action (always queried for labelling)
                a_expert = self.oracle.predict(obs_flat)

                # Decide which action to *execute*
                if np.random.random() < beta:
                    a_exec = a_expert
                else:
                    a_exec = self._policy_predict(obs_flat)

                obs_list.append(obs_flat)
                act_list.append(a_expert)  # DAgger key: label with expert, not executed

                obs_raw, _, terminated, truncated, _ = env.step(a_exec)
                done = terminated or truncated
                step += 1

            env.close()

        return np.array(obs_list), np.array(act_list)

    def _policy_predict(self, obs: np.ndarray) -> np.ndarray:
        obs_norm = (obs - self.obs_mean) / (self.obs_std + 1e-8)
        t = torch.tensor(obs_norm, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            a_norm = self.policy(t).cpu().numpy().flatten()
        return a_norm * (self.act_std + 1e-8) + self.act_mean

    # ── retraining ────────────────────────────────────────────────────────────

    def _retrain(self, iteration: int) -> float:
        cfg_t = self.cfg.training
        obs_all = np.concatenate(self._obs_buffer)
        act_all = np.concatenate(self._act_buffer)

        # Normalise
        obs_n = (obs_all - self.obs_mean) / (self.obs_std + 1e-8)
        act_n = (act_all - self.act_mean) / (self.act_std + 1e-8)

        obs_t = torch.tensor(obs_n, dtype=torch.float32)
        act_t = torch.tensor(act_n, dtype=torch.float32)

        dataset = TensorDataset(obs_t, act_t)
        val_n = max(1, int(0.1 * len(dataset)))
        train_n = len(dataset) - val_n
        from torch.utils.data import random_split
        train_ds, val_ds = random_split(dataset, [train_n, val_n])

        loader = DataLoader(train_ds, batch_size=cfg_t.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=cfg_t.batch_size)

        opt = optim.AdamW(self.policy.parameters(), lr=cfg_t.learning_rate)
        loss_fn = nn.MSELoss()

        best_val = float("inf")
        patience = 0
        dev = self.device

        for epoch in range(cfg_t.n_epochs):
            self.policy.train()
            for xb, yb in loader:
                pred = self.policy(xb.to(dev))
                loss = loss_fn(pred, yb.to(dev))
                opt.zero_grad()
                loss.backward()
                opt.step()

            self.policy.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    val_loss += loss_fn(self.policy(xb.to(dev)), yb.to(dev)).item()
            val_loss /= max(len(val_loader), 1)

            if val_loss < best_val:
                best_val = val_loss
                patience = 0
            else:
                patience += 1
                if patience >= cfg_t.early_stopping_patience:
                    break

        print(f"  Retrained on {len(obs_all)} transitions → best val={best_val:.5f}")
        return best_val

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _flatten(obs) -> np.ndarray:
        if isinstance(obs, dict):
            return np.concatenate([np.asarray(v).flatten() for v in obs.values()])
        return np.asarray(obs).flatten()

    def _save(self, path: str) -> None:
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "obs_mean": self.obs_mean,
                "obs_std": self.obs_std,
                "act_mean": self.act_mean,
                "act_std": self.act_std,
            },
            path,
        )
        print(f"[DAgger] Model saved to {path}")
