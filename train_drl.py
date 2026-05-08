"""
train_drl.py — TQC-based Deep Reinforcement Learning training & evaluation.

Replaces DRL_BoltSD.py with:
  - Hydra configuration (no more comment-toggling)
  - TQC (Truncated Quantile Critics) instead of TD3
  - W&B + TensorBoard logging
  - Reproducible seeding
  - Domain randomization via robosuite_wrapper
  - Potential-based reward shaping
  - IsaacLab backend support (env=isaaclab)

Usage
-----
  # Train with robosuite (CPU, 8 envs):
  python train_drl.py mode=train env=robosuite algorithm=tqc

  # Train with IsaacLab (GPU, 4096 envs):
  python train_drl.py mode=train env=isaaclab algorithm=tqc

  # Evaluate saved model:
  python train_drl.py mode=eval env=robosuite algorithm=tqc

  # Hyperparameter sweep (Hydra multirun):
  python train_drl.py --multirun algorithm.learning_rate=1e-4,3e-4,1e-3

  # TensorBoard:
  tensorboard --logdir=outputs/ --port 6006
"""
from __future__ import annotations

import os
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd

from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnNoModelImprovement,
    BaseCallback,
    CallbackList,
)
from stable_baselines3.common.evaluation import evaluate_policy

try:
    from sb3_contrib import TQC
    TQC_AVAILABLE = True
except ImportError:
    from stable_baselines3 import TD3 as TQC  # type: ignore[assignment]
    TQC_AVAILABLE = False
    print("[Warning] sb3-contrib not installed — falling back to TD3.")

from stable_baselines3 import SAC, TD3

from utils.seed import set_global_seed
from utils.logger import Logger
from envs.robosuite_wrapper import make_vec_env, make_robosuite_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# ═══════════════════════════════════════════════════════════════════════════════
# W&B callback
# ═══════════════════════════════════════════════════════════════════════════════

class WandbCallback(BaseCallback):
    """Logs SB3 episode metrics to W&B / TensorBoard every rollout."""

    def __init__(self, logger: Logger, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._logger = logger

    def _on_rollout_end(self) -> None:
        if len(self.model.ep_info_buffer) == 0:
            return
        mean_r = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
        mean_l = np.mean([ep["l"] for ep in self.model.ep_info_buffer])
        self._logger.log(
            {"rollout/ep_rew_mean": mean_r, "rollout/ep_len_mean": mean_l},
            step=self.num_timesteps,
        )

    def _on_step(self) -> bool:
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# Dynamic exploration callback (adapted from DRL_BoltSD.py)
# ═══════════════════════════════════════════════════════════════════════════════

class DynamicExplorationCallback(BaseCallback):
    """
    Increases action noise when rewards plateau and freezes early layers
    at reward milestones — exactly as in the original DRL_BoltSD.py,
    now driven from Hydra config instead of hard-coded constants.
    """

    def __init__(self, cfg, verbose: int = 1) -> None:
        super().__init__(verbose)
        self.thresholds = list(cfg.thresholds)
        self.patience = cfg.patience
        self.noise_scale = cfg.noise_scale
        self.max_sigma = cfg.max_sigma
        self.min_sigma = cfg.min_sigma
        self._reached: set = set()
        self._no_improve = 0
        self._last_mean = -np.inf

    def _on_rollout_end(self) -> None:
        if not self.model.ep_info_buffer:
            return
        mean_r = np.mean([ep["r"] for ep in self.model.ep_info_buffer])

        for i, thr in enumerate(self.thresholds):
            if thr not in self._reached and mean_r >= thr:
                self._reached.add(thr)
                self._no_improve = 0
                self.model.save(f"checkpoint_{thr}.zip")
                self._freeze_layers(i + 1)

        if self._reached:
            if mean_r <= self._last_mean + 10:
                self._no_improve += 1
            else:
                self._no_improve = 0
            if self._no_improve >= self.patience:
                self._boost_noise()
                self._no_improve = 0

        self._last_mean = mean_r

    def _boost_noise(self) -> None:
        if hasattr(self.model, "action_noise") and isinstance(
            self.model.action_noise, NormalActionNoise
        ):
            self.model.action_noise.sigma = np.clip(
                self.model.action_noise.sigma * self.noise_scale,
                self.min_sigma,
                self.max_sigma,
            )

    def _freeze_layers(self, n: int) -> None:
        policy = self.model.policy
        frozen = 0
        nets = []
        if hasattr(policy, "mlp_extractor"):
            nets = list(policy.mlp_extractor.policy_net.children()) + \
                   list(policy.mlp_extractor.value_net.children())
        elif hasattr(policy, "actor"):
            nets = list(policy.actor.children())
        for layer in nets:
            if isinstance(layer, torch.nn.Linear):
                for p in layer.parameters():
                    p.requires_grad = False
                frozen += 1
                if frozen >= n:
                    break
        if self.verbose:
            print(f"[Callback] Frozen {frozen} layers.")

    def _on_step(self) -> bool:
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# Algorithm factory
# ═══════════════════════════════════════════════════════════════════════════════

def _build_model(cfg: DictConfig, train_env, log_dir: str):
    n_actions = train_env.action_space.shape[0]
    act_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=cfg.algorithm.action_noise.sigma * np.ones(n_actions),
    )

    common = dict(
        policy="MlpPolicy",
        env=train_env,
        tensorboard_log=os.path.join(log_dir, "tensorboard"),
        verbose=1,
        learning_rate=cfg.algorithm.learning_rate,
        buffer_size=cfg.algorithm.buffer_size,
        batch_size=cfg.algorithm.batch_size,
        tau=cfg.algorithm.tau,
        gamma=cfg.algorithm.gamma,
        train_freq=cfg.algorithm.train_freq,
        gradient_steps=cfg.algorithm.gradient_steps,
        learning_starts=cfg.algorithm.learning_starts,
        device=cfg.device,
        policy_kwargs=dict(
            net_arch=list(cfg.algorithm.policy_kwargs.net_arch),
            activation_fn=torch.nn.ReLU,
        ),
    )

    if cfg.algorithm.name == "TQC" and TQC_AVAILABLE:
        common["policy_kwargs"]["n_quantiles"] = cfg.algorithm.policy_kwargs.n_quantiles
        common["policy_kwargs"]["n_critics"] = cfg.algorithm.policy_kwargs.n_critics
        common["top_quantiles_to_drop_per_net"] = cfg.algorithm.top_quantiles_to_drop_per_net
        model = TQC(action_noise=act_noise, **common)
    elif cfg.algorithm.name == "SAC":
        model = SAC(**common)
    else:
        common["action_noise"] = act_noise
        model = TD3(**common)

    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def train(cfg: DictConfig) -> None:
    log_dir = os.path.join(cfg.output_dir, "model")
    os.makedirs(log_dir, exist_ok=True)

    logger = Logger(cfg, cfg.output_dir)

    # Build envs
    if cfg.env.name == "isaaclab":
        train_env, eval_env = _build_isaaclab_envs(cfg)
    else:
        train_env = make_vec_env(
            n_envs=cfg.env.n_envs,
            horizon=cfg.env.horizon,
            dr_cfg=cfg.env.domain_randomization,
            reward_cfg=cfg.env.reward,
            norm_obs=cfg.env.norm_obs,
            norm_reward=cfg.env.norm_reward,
            clip_obs=cfg.env.clip_obs,
            training=True,
            seed=cfg.seed,
        )
        eval_env = make_vec_env(
            n_envs=cfg.env.n_envs,
            horizon=cfg.env.horizon,
            dr_cfg=cfg.env.domain_randomization,
            reward_cfg=cfg.env.reward,
            norm_obs=cfg.env.norm_obs,
            norm_reward=False,
            clip_obs=cfg.env.clip_obs,
            training=False,
            seed=cfg.seed + 1000,
        )

    model_path = os.path.join(log_dir, "best_model.zip")
    if os.path.exists(model_path):
        print(f"Resuming from {model_path}")
        model = TQC.load(model_path, env=train_env)
    else:
        model = _build_model(cfg, train_env, log_dir)

    # Callbacks
    stop_cb = StopTrainingOnNoModelImprovement(max_no_improvement_evals=20, min_evals=200, verbose=1)
    explore_cb = DynamicExplorationCallback(cfg.algorithm.exploration, verbose=1)
    wandb_cb = WandbCallback(logger)
    after_eval = CallbackList([stop_cb, explore_cb])

    eval_cb = EvalCallback(
        eval_env,
        eval_freq=cfg.algorithm.eval_freq,
        n_eval_episodes=cfg.algorithm.n_eval_episodes,
        deterministic=True,
        best_model_save_path=log_dir,
        log_path=log_dir,
        callback_after_eval=after_eval,
        verbose=1,
    )

    model.learn(
        total_timesteps=cfg.algorithm.total_timesteps,
        callback=CallbackList([eval_cb, wandb_cb]),
    )

    train_env.save(cfg.env.vecnorm_path)
    logger.close()
    train_env.close()
    eval_env.close()
    print(f"Training complete. Best model → {model_path}")


def _build_isaaclab_envs(cfg: DictConfig):
    from envs.isaaclab_bolt_env import BoltScrewdrivingEnv, BoltScrewdrivingEnvCfg
    env_cfg = BoltScrewdrivingEnvCfg()
    env_cfg.num_envs = cfg.env.n_envs
    env_cfg.use_domain_rand = cfg.env.domain_randomization.enabled
    train_env = BoltScrewdrivingEnv(env_cfg)
    eval_env = BoltScrewdrivingEnv(env_cfg)
    return train_env, eval_env


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(cfg: DictConfig) -> None:
    log_dir = os.path.join(cfg.output_dir, "model")
    model_path = os.path.join(log_dir, "best_model.zip")

    eval_env = DummyVecEnv([
        make_robosuite_env(
            has_renderer=True,
            horizon=cfg.env.horizon,
            dr_cfg=cfg.env.domain_randomization,
            reward_cfg=cfg.env.reward,
            seed=cfg.seed,
        )
    ])
    if os.path.exists(cfg.env.vecnorm_path):
        eval_env = VecNormalize.load(cfg.env.vecnorm_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False

    model = TQC.load(model_path, env=eval_env)
    obs = eval_env.reset()
    done = [False]
    steps, rewards = [], []
    t = 0
    while not done[0]:
        t += 1
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        eval_env.render()
        rewards.append(float(reward[0]))
        steps.append(t)
        print(f"Step {t:4d} | Reward {reward[0]:.4f}")

    eval_env.close()
    pd.DataFrame({"step": steps, "reward": rewards}).to_csv(
        "Evaluation_DRL_improved.csv", index=False
    )
    print(f"Mean reward: {np.mean(rewards):.2f} | Episode length: {t}")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    set_global_seed(cfg.seed)

    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}. Use 'train' or 'eval'.")


if __name__ == "__main__":
    main()
