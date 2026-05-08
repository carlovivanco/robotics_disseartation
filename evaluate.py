"""
evaluate.py — Unified evaluation script for DRL and IL policies.

Loads any saved model (TQC, Diffusion Policy, DAgger/BC) and runs N episodes
in the robosuite environment, reporting success rate, mean reward, and
episode length. Results are saved to a structured CSV.

Usage
-----
  # Evaluate TQC:
  python evaluate.py --policy drl \
      --model_path logs_OSC/BoltSD_T/TD3/best_model.zip \
      --vecnorm_path vec_normalize_SD_OSC2.pkl

  # Evaluate Diffusion Policy:
  python evaluate.py --policy diffusion \
      --model_path diffusion_model_best.pth

  # Evaluate DAgger / BC:
  python evaluate.py --policy bc \
      --model_path bc_model_best_SD.pth

  # With rendering:
  python evaluate.py --policy drl --model_path ... --render
"""
from __future__ import annotations

import argparse
import collections
import numpy as np
import pandas as pd
import torch

from envs.robosuite_wrapper import DomainRandomizedGymWrapper
from utils.seed import set_global_seed


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _flatten(obs) -> np.ndarray:
    if isinstance(obs, dict):
        return np.concatenate([np.asarray(v).flatten() for v in obs.values()])
    return np.asarray(obs).flatten()


def _make_env(render: bool = False, horizon: int = 2000, seed: int = 42):
    return DomainRandomizedGymWrapper(
        has_renderer=render,
        offscreen_renderer=False,
        horizon=horizon,
        dr_cfg=None,      # no DR during evaluation for fair comparison
        reward_cfg=None,
        seed=seed,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Policy loaders
# ═══════════════════════════════════════════════════════════════════════════════

class DRLAgent:
    def __init__(self, model_path: str, vecnorm_path: str | None, device: str) -> None:
        try:
            from sb3_contrib import TQC
            self.model = TQC.load(model_path, device=device)
        except Exception:
            from stable_baselines3 import TD3
            self.model = TD3.load(model_path, device=device)

        self._vecnorm = None
        if vecnorm_path is not None:
            import os
            if os.path.exists(vecnorm_path):
                from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
                # We need a dummy venv just to load VecNormalize stats
                # Actual normalisation is applied manually below
                self._mean = None
                self._std = None
                try:
                    import pickle
                    with open(vecnorm_path, "rb") as f:
                        vn = pickle.load(f)
                    self._mean = vn.obs_rms.mean
                    self._std = np.sqrt(vn.obs_rms.var + vn.epsilon)
                except Exception:
                    print("[Warning] Could not load VecNormalize stats — using raw obs.")

    def predict(self, obs: np.ndarray) -> np.ndarray:
        if self._mean is not None:
            obs = (obs - self._mean) / (self._std + 1e-8)
            obs = np.clip(obs, -10.0, 10.0)
        action, _ = self.model.predict(obs[None], deterministic=True)
        return np.asarray(action).flatten()


class DiffusionAgent:
    def __init__(self, model_path: str, obs_horizon: int = 2, pred_horizon: int = 16,
                 action_horizon: int = 8, device: str = "cuda") -> None:
        from algorithms.diffusion_policy import DiffusionPolicy

        # Build a minimal config object
        class Cfg:
            pass
        cfg = Cfg()
        cfg.obs_dim = 64
        cfg.action_dim = 7
        cfg.obs_horizon = obs_horizon
        cfg.pred_horizon = pred_horizon
        cfg.action_horizon = action_horizon
        cfg.n_diffusion_steps = 100
        cfg.n_inference_steps = 10
        cfg.beta_start = 0.0001
        cfg.beta_end = 0.02
        cfg.beta_schedule = "squaredcos_cap_v2"

        class Net:
            down_dims = [256, 512, 1024]
            kernel_size = 5
            n_groups = 8
            diffusion_step_embed_dim = 256
        cfg.network = Net()

        self.policy = DiffusionPolicy.load(model_path, cfg, device)
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        self._obs_window: collections.deque | None = None
        self._action_queue: list[np.ndarray] = []

    def reset(self) -> None:
        self._obs_window = None
        self._action_queue = []

    def predict(self, obs: np.ndarray) -> np.ndarray:
        if self._obs_window is None:
            self._obs_window = collections.deque(
                [obs] * self.obs_horizon, maxlen=self.obs_horizon
            )
        self._obs_window.append(obs)

        # Re-plan when action queue is empty
        if not self._action_queue:
            actions = self.policy.predict(list(self._obs_window))
            self._action_queue = list(actions)

        return self._action_queue.pop(0)


class BCAgent:
    def __init__(self, model_path: str, device: str = "cuda") -> None:
        ckpt = torch.load(model_path, map_location=device, weights_only=True)
        import torch.nn as nn

        # Try to infer dims from checkpoint
        keys = list(ckpt.keys() if "net" not in ckpt else ckpt["net"].keys())
        if "net" in ckpt:
            state = ckpt["net"]
        elif "policy" in ckpt:
            state = ckpt["policy"]
        else:
            state = ckpt

        first_key = list(state.keys())[0]
        last_key = list(state.keys())[-1]
        obs_dim = state[first_key].shape[-1]
        act_dim = state[last_key].shape[0]

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 1024), nn.LayerNorm(1024), nn.ReLU(),
            nn.Linear(1024, act_dim),
        ).to(device)
        self.net.load_state_dict(state)
        self.net.eval()
        self.device = device

        self.obs_mean = ckpt.get("obs_mean")
        self.obs_std = ckpt.get("obs_std")
        self.act_mean = ckpt.get("act_mean")
        self.act_std = ckpt.get("act_std")

    def predict(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_mean is not None:
            obs = (obs - self.obs_mean) / (self.obs_std + 1e-8)
        t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            a = self.net(t).cpu().numpy().flatten()
        if self.act_mean is not None:
            a = a * (self.act_std + 1e-8) + self.act_mean
        return a

    def reset(self) -> None:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation loop
# ═══════════════════════════════════════════════════════════════════════════════

def run_evaluation(
    agent,
    n_episodes: int = 10,
    horizon: int = 2000,
    render: bool = False,
    seed: int = 42,
    output_csv: str = "Evaluation_improved.csv",
) -> dict:
    records = []
    successes = 0

    for ep in range(n_episodes):
        env = _make_env(render=render, horizon=horizon, seed=seed + ep)
        obs_raw, _ = env.reset()
        if hasattr(agent, "reset"):
            agent.reset()

        ep_reward = 0.0
        done = False
        t = 0

        while not done:
            obs = _flatten(obs_raw)
            action = agent.predict(obs)
            obs_raw, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            done = terminated or truncated
            t += 1
            if render:
                env.render()

        success = terminated  # terminated = task success; truncated = timeout
        successes += int(success)
        records.append({
            "episode": ep,
            "total_reward": ep_reward,
            "length": t,
            "success": int(success),
        })
        print(
            f"Ep {ep+1:3d}/{n_episodes} | "
            f"Reward {ep_reward:8.2f} | "
            f"Steps {t:5d} | "
            f"{'✓ SUCCESS' if success else '✗ timeout'}"
        )
        env.close()

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)

    summary = {
        "success_rate": successes / n_episodes,
        "mean_reward": df["total_reward"].mean(),
        "std_reward": df["total_reward"].std(),
        "mean_length": df["length"].mean(),
    }
    print("\n─── Summary ────────────────────────────────────")
    for k, v in summary.items():
        print(f"  {k:<20}: {v:.4f}")
    print(f"  Results saved to {output_csv}")
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained bolt-screwdriving policy.")
    parser.add_argument("--policy", choices=["drl", "diffusion", "bc", "dagger"],
                        required=True, help="Policy type to evaluate.")
    parser.add_argument("--model_path", required=True, help="Path to saved model checkpoint.")
    parser.add_argument("--vecnorm_path", default=None,
                        help="VecNormalize stats pkl (DRL only).")
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=2000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="Evaluation_improved.csv")
    parser.add_argument("--obs_horizon", type=int, default=2,
                        help="Observation window size (Diffusion Policy).")
    args = parser.parse_args()

    set_global_seed(args.seed)

    if args.policy == "drl":
        agent = DRLAgent(args.model_path, args.vecnorm_path, args.device)
    elif args.policy == "diffusion":
        agent = DiffusionAgent(
            args.model_path, obs_horizon=args.obs_horizon, device=args.device
        )
    elif args.policy in ("bc", "dagger"):
        agent = BCAgent(args.model_path, device=args.device)
    else:
        raise ValueError(f"Unknown policy: {args.policy}")

    run_evaluation(
        agent,
        n_episodes=args.n_episodes,
        horizon=args.horizon,
        render=args.render,
        seed=args.seed,
        output_csv=args.output,
    )


if __name__ == "__main__":
    main()
