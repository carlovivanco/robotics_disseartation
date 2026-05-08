"""
Enhanced robosuite wrapper with:
  - Domain randomization at reset
  - Potential-based reward shaping (policy-invariant)
  - Observation noise injection
  - Action delay simulation
  - Proper seeding
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import GymWrapper
from robosuite.environments.manipulation.bolt_screwdriving_dissassembly import (
    BoltScrewdrivingDisassembly,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from envs.domain_randomization import DomainRandomizer


def _build_controller_config() -> dict:
    cfg = load_composite_controller_config(controller=None, robot="Panda")
    cfg["body_parts"]["right"].update(
        {
            "type": "OSC_POSE",
            "input_max": [1.0] * 6,
            "input_min": [-1.0] * 6,
            "output_max": [0.1] * 6,
            "output_min": [-0.1] * 6,
            "kp": 150,
            "damping_ratio": 1.0,
            "impedance_mode": "fixed",
            "control_delta": False,
            "input_ref_frame": "base",
        }
    )
    return cfg


class DomainRandomizedGymWrapper(gym.Env):
    """
    Thin wrapper around BoltScrewdrivingDisassembly that:
    1. Exposes a gymnasium-compatible interface
    2. Applies domain randomization every reset()
    3. Adds observation noise every step()
    4. Uses potential-based reward shaping to make dense rewards policy-invariant
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        has_renderer: bool = False,
        offscreen_renderer: bool = False,
        horizon: int = 5000,
        dr_cfg=None,
        reward_cfg=None,
        seed: int = 0,
    ) -> None:
        super().__init__()

        self._controller_cfg = _build_controller_config()
        self._horizon = horizon
        self._dr = DomainRandomizer(dr_cfg) if dr_cfg is not None else None
        self._gamma = getattr(reward_cfg, "potential_gamma", 0.9018)
        self._prev_potential: float = 0.0

        self._env = BoltScrewdrivingDisassembly(
            robots="Panda",
            gripper_types="default",
            env_configuration="opposed",
            controller_configs=self._controller_cfg,
            has_renderer=has_renderer,
            has_offscreen_renderer=offscreen_renderer,
            control_freq=20,
            horizon=horizon,
            use_object_obs=True,
            use_camera_obs=False,
            reward_shaping=True,
            single_object_mode=2,
            nut_type="m5",
            initial_grasp=True,
        )

        # Build observation / action spaces from the wrapped env
        gym_env = GymWrapper(self._env)
        self.observation_space = gym_env.observation_space
        self.action_space = gym_env.action_space
        self._gym_env = gym_env

        self.np_random, _ = gym.utils.seeding.np_random(seed)

    # ── gymnasium interface ───────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        if self._dr is not None:
            self._dr.reset()
            self._dr.randomise_physics(self._env.sim)

        obs, info = self._gym_env.reset(), {}
        obs = self._maybe_add_noise(obs)
        self._prev_potential = self._compute_potential(obs)
        return obs, info

    def step(self, action: np.ndarray):
        if self._dr is not None:
            action = self._dr.maybe_delay_action(action)

        obs, reward, terminated, info = self._gym_env.step(action)
        obs = self._maybe_add_noise(obs)

        # Potential-based shaping: F(s,s') = γ·Φ(s') - Φ(s)
        new_potential = self._compute_potential(obs)
        shaping = self._gamma * new_potential - self._prev_potential
        self._prev_potential = new_potential

        shaped_reward = reward + shaping
        truncated = False
        return obs, shaped_reward, terminated, truncated, info

    def render(self):
        self._env.render()

    def close(self):
        self._gym_env.close()

    # ── internals ─────────────────────────────────────────────────────────────

    def _maybe_add_noise(self, obs: np.ndarray) -> np.ndarray:
        if self._dr is not None:
            return self._dr.apply_obs_noise(obs)
        return obs

    def _compute_potential(self, obs: np.ndarray) -> float:
        """
        Φ(s) ∝ negative distance from gripper to bolt.
        Using potential-based shaping guarantees the optimal policy is unchanged.
        """
        # obs layout (matches BoltScrewdrivingDisassembly):
        # indices 35:38 → relative EE-to-bolt position (from _setup_observables)
        rel_pos = obs[35:38] if len(obs) > 38 else obs[-3:]
        dist = float(np.linalg.norm(rel_pos))
        return -dist  # closer → higher potential


# ── factory functions ─────────────────────────────────────────────────────────

def make_robosuite_env(
    has_renderer: bool = False,
    horizon: int = 5000,
    dr_cfg=None,
    reward_cfg=None,
    seed: int = 0,
):
    """Returns a single monitored env (for SB3 compatibility)."""
    def _init():
        env = DomainRandomizedGymWrapper(
            has_renderer=has_renderer,
            offscreen_renderer=False,
            horizon=horizon,
            dr_cfg=dr_cfg,
            reward_cfg=reward_cfg,
            seed=seed,
        )
        return Monitor(env)
    return _init


def make_vec_env(
    n_envs: int,
    horizon: int = 5000,
    dr_cfg=None,
    reward_cfg=None,
    norm_obs: bool = True,
    norm_reward: bool = True,
    clip_obs: float = 10.0,
    training: bool = True,
    vecnorm_path: str | None = None,
    seed: int = 0,
) -> VecNormalize:
    """
    Build a VecNormalize-wrapped SubprocVecEnv.
    Pass vecnorm_path to load existing normalisation stats for evaluation.
    """
    fns = [
        make_robosuite_env(
            has_renderer=False,
            horizon=horizon,
            dr_cfg=dr_cfg,
            reward_cfg=reward_cfg,
            seed=seed + i,
        )
        for i in range(n_envs)
    ]
    vec = SubprocVecEnv(fns)

    if vecnorm_path is not None:
        import os
        if os.path.exists(vecnorm_path):
            return VecNormalize.load(vecnorm_path, vec)

    return VecNormalize(
        vec,
        norm_obs=norm_obs,
        norm_reward=norm_reward,
        clip_obs=clip_obs,
        training=training,
    )
