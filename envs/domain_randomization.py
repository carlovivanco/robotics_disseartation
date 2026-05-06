"""
Domain randomization utilities for robosuite environments.

Applied at every env.reset() to randomise physics parameters, bridging
the sim-to-real gap without modifying the core environment class.
"""
from __future__ import annotations

import numpy as np


class DomainRandomizer:
    """
    Randomises MuJoCo model parameters (friction, mass, damping, gravity)
    in-place on a robosuite environment's sim object at each episode reset.
    """

    def __init__(self, cfg) -> None:
        self.enabled = cfg.enabled
        self.friction_range = tuple(cfg.friction_range)
        self.mass_mult_range = tuple(cfg.mass_multiplier_range)
        self.damping_range = tuple(cfg.damping_range)
        self.obs_noise_std = cfg.obs_noise_std
        self.action_delay_prob = cfg.action_delay_prob
        self.gravity_range = tuple(cfg.gravity_range)
        self._last_action: np.ndarray | None = None

    # ── called once per episode ───────────────────────────────────────────────

    def randomise_physics(self, sim) -> None:
        """Perturb MuJoCo model parameters in-place."""
        if not self.enabled:
            return

        model = sim.model

        # Friction — all geom pairs
        for i in range(model.ngeom):
            scale = np.random.uniform(*self.friction_range)
            model.geom_friction[i, 0] *= scale   # sliding friction

        # Mass — all bodies except world (id 0)
        for i in range(1, model.nbody):
            mult = np.random.uniform(*self.mass_mult_range)
            model.body_mass[i] *= mult

        # Joint damping
        for i in range(model.njnt):
            model.dof_damping[i] *= np.random.uniform(*self.damping_range)

        # Gravity
        gz = np.random.uniform(*self.gravity_range)
        model.opt.gravity[:] = [0.0, 0.0, -gz]

    # ── called every step ────────────────────────────────────────────────────

    def apply_obs_noise(self, obs: np.ndarray) -> np.ndarray:
        if not self.enabled or self.obs_noise_std <= 0:
            return obs
        return obs + np.random.randn(*obs.shape) * self.obs_noise_std

    def maybe_delay_action(self, action: np.ndarray) -> np.ndarray:
        """With small probability repeat the previous action (latency sim)."""
        if not self.enabled or self._last_action is None:
            self._last_action = action.copy()
            return action
        if np.random.random() < self.action_delay_prob:
            return self._last_action
        self._last_action = action.copy()
        return action

    def reset(self) -> None:
        self._last_action = None
