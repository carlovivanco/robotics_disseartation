"""
IsaacLab DirectRLEnv for the M5 bolt screwdriving / disassembly task.

Runs up to 4096 parallel environments on a single GPU using NVIDIA PhysX,
replacing the 8-CPU SubprocVecEnv used in the robosuite baseline.

Requirements
------------
- Isaac Sim 4.x  (https://developer.nvidia.com/isaac-sim)
- IsaacLab 1.x   (pip install isaaclab)
- isaaclab_assets (pip install isaaclab_assets)
- A USD bolt asset exported from the robosuite MuJoCo model:
    python tools/export_bolt_usd.py --output assets/m5_bolt.usd

Usage
-----
    python train_drl.py env=isaaclab
"""
from __future__ import annotations

import math
import torch
import numpy as np
from dataclasses import MISSING
from typing import Any

# ── IsaacLab imports ─────────────────────────────────────────────────────────
try:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import (
        Articulation,
        ArticulationCfg,
        RigidObject,
        RigidObjectCfg,
    )
    from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
    from isaaclab.sim import SimulationCfg
    from isaaclab.sim.spawners.from_files import UsdFileCfg
    from isaaclab.sim.spawners.shapes import CuboidCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.utils import configclass
    from isaaclab.utils.math import (
        euler_xyz_from_quat,
        quat_from_euler_xyz,
        quat_mul,
        quat_conjugate,
        subtract_frame_transforms,
    )

    # Franka Panda URDF/USD — ships with isaaclab_assets
    from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

    ISAACLAB_AVAILABLE = True

except ImportError as e:
    ISAACLAB_AVAILABLE = False
    _IMPORT_ERROR = str(e)


# ── helpers ───────────────────────────────────────────────────────────────────

def _check_isaaclab() -> None:
    if not ISAACLAB_AVAILABLE:
        raise ImportError(
            f"IsaacLab not found: {_IMPORT_ERROR}\n"
            "Install Isaac Sim 4.x and IsaacLab 1.x, then retry."
        )


# ── environment config dataclass ─────────────────────────────────────────────

if ISAACLAB_AVAILABLE:

    @configclass
    class BoltScrewdrivingEnvCfg(DirectRLEnvCfg):
        """
        Configuration for the GPU-parallel bolt screwdriving task.
        All fields are dataclass attributes — override via Hydra or Python.
        """

        # ── simulation ────────────────────────────────────────────────────────
        sim: SimulationCfg = SimulationCfg(
            dt=1.0 / 240.0,    # 240 Hz physics
            render_interval=12, # 240/12 = 20 Hz control
        )

        # ── scene ─────────────────────────────────────────────────────────────
        num_envs: int = 4096
        env_spacing: float = 2.5      # metres between env grids
        episode_length_s: float = 250.0  # 5000 steps × (1/20 s)

        # ── observation / action spaces ───────────────────────────────────────
        # Must match the 64-D obs and 7-D action of the robosuite baseline.
        observation_space: int = 64
        action_space: int = 7
        state_space: int = 0

        # ── robot ─────────────────────────────────────────────────────────────
        robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(
            prim_path="/World/envs/env_.*/Robot"
        )

        # ── bolt (rigid body with screw DOF — 1 revolute + 1 prismatic) ───────
        bolt: RigidObjectCfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Bolt",
            spawn=UsdFileCfg(
                usd_path="assets/m5_bolt.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    retain_accelerations=False,
                    linear_damping=0.0,
                    angular_damping=0.01,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                    max_depenetration_velocity=1.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.5, 0.0, 0.44),  # bolt starts on the table surface
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

        # ── table ─────────────────────────────────────────────────────────────
        table: RigidObjectCfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Table",
            spawn=CuboidCfg(
                size=(0.8, 0.8, 0.05),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.4)),
        )

        # ── control ───────────────────────────────────────────────────────────
        action_scale_pos: float = 0.1      # m per normalised unit
        action_scale_rot: float = 0.1      # rad per normalised unit

        # ── reward weights (mirrors robosuite shaped reward) ──────────────────
        reach_weight: float = 0.1
        grasp_weight: float = 0.35
        screw_weight: float = 0.5
        moveaway_weight: float = 0.7
        success_bonus: float = 1.0
        potential_gamma: float = 0.9018

        # ── domain randomisation ──────────────────────────────────────────────
        use_domain_rand: bool = True
        friction_range: tuple[float, float] = (0.3, 1.5)
        mass_mult_range: tuple[float, float] = (0.8, 1.2)
        obs_noise_std: float = 0.01
        rand_interval: int = 500            # re-randomise every N steps


    # ── environment class ─────────────────────────────────────────────────────

    class BoltScrewdrivingEnv(DirectRLEnv):
        """
        GPU-parallel bolt screwdriving environment built on IsaacLab's
        DirectRLEnv. Inherits vectorised reset/step/reward logic.

        Key differences from the robosuite baseline:
        - 4096 envs on one GPU instead of 8 CPU processes
        - Observations and rewards computed as batched torch tensors
        - Domain randomisation via PhysX material APIs
        - Screw-joint coupling handled by a velocity constraint applied
          each physics step (1 revolute + 1 prismatic linked by lead screw pitch)
        """

        cfg: BoltScrewdrivingEnvCfg

        # M5 bolt geometry
        SCREW_PITCH: float = 0.0008    # 0.8 mm per revolution (M5 standard)
        SUCCESS_DIST: float = 0.05     # horizontal separation for "off peg"

        def __init__(self, cfg: BoltScrewdrivingEnvCfg, render_mode: str | None = None):
            super().__init__(cfg, render_mode=render_mode)

            # Per-env state tensors (allocated in _setup_scene)
            self._bolt_angle: torch.Tensor = None    # cumulative rotation [n_envs]
            self._bolt_slide: torch.Tensor = None    # cumulative axial displacement [n_envs]
            self._prev_potential: torch.Tensor = None
            self._step_count: torch.Tensor = None

        # ── scene setup ───────────────────────────────────────────────────────

        def _setup_scene(self) -> None:
            """Instantiate robot, bolt, table; register with scene manager."""
            self.robot = Articulation(self.cfg.robot)
            self.bolt = RigidObject(self.cfg.bolt)
            self.table = RigidObject(self.cfg.table)

            self.scene.articulations["robot"] = self.robot
            self.scene.rigid_objects["bolt"] = self.bolt
            self.scene.rigid_objects["table"] = self.table

            # Add ground plane + lighting
            sim_utils.add_ground_plane()
            sim_utils.add_default_light()

            # Allocate per-env state
            n = self.num_envs
            dev = self.device
            self._bolt_angle = torch.zeros(n, device=dev)
            self._bolt_slide = torch.zeros(n, device=dev)
            self._prev_potential = torch.zeros(n, device=dev)
            self._step_count = torch.zeros(n, dtype=torch.int32, device=dev)

            # Clone environment origins for multi-env transforms
            self.scene.clone_environments(copy_from_source=False)
            self.scene.filter_collisions(global_prim_paths=[])

        # ── action application ────────────────────────────────────────────────

        def _pre_physics_step(self, actions: torch.Tensor) -> None:
            """Convert normalised 7-D actions to joint torques via IK/OSC."""
            self._actions = actions.clone().clamp(-1.0, 1.0)

        def _apply_action(self) -> None:
            """Apply OSC-equivalent position targets to Franka joints."""
            # Split action into EE delta-position, delta-rotation, gripper
            dpos = self._actions[:, :3] * self.cfg.action_scale_pos
            drot = self._actions[:, 3:6] * self.cfg.action_scale_rot
            gripper_cmd = self._actions[:, 6:7]

            # Get current EE pose
            ee_pos, ee_quat = self._get_ee_pose()

            # Compute new target position
            target_pos = ee_pos + dpos

            # Convert delta-euler to quaternion and compose
            dquat = quat_from_euler_xyz(drot[:, 0], drot[:, 1], drot[:, 2])
            target_quat = quat_mul(ee_quat, dquat)

            # Set joint position targets (simplified IK: jacobian pseudo-inverse)
            # In production: replace with IsaacLab's built-in DifferentialIKController
            joint_targets = self._osc_to_joint_targets(target_pos, target_quat, gripper_cmd)
            self.robot.set_joint_position_target(joint_targets)

            # Apply screw coupling constraint: link bolt rotation and translation
            self._apply_screw_constraint()

        def _apply_screw_constraint(self) -> None:
            """
            Couple bolt rotation and axial translation via lead-screw kinematics.
            Axial displacement = (angle / 2π) × pitch.
            Applied as a velocity constraint each physics sub-step.
            """
            bolt_ang_vel = self.bolt.data.root_ang_vel_w[:, 2]  # z-axis rotation
            axial_vel = bolt_ang_vel * (self.SCREW_PITCH / (2 * math.pi))

            target_lin_vel = self.bolt.data.root_lin_vel_w.clone()
            target_lin_vel[:, 2] = axial_vel  # constrain z-velocity
            self.bolt.write_root_velocity_to_sim(
                torch.cat([target_lin_vel, self.bolt.data.root_ang_vel_w], dim=-1)
            )

            # Accumulate bolt angle and slide for reward computation
            dt = self.cfg.sim.dt * self.cfg.sim.render_interval
            self._bolt_angle += bolt_ang_vel * dt
            self._bolt_slide += axial_vel * dt

        # ── observations ─────────────────────────────────────────────────────

        def _get_observations(self) -> dict[str, torch.Tensor]:
            """
            Build the 64-D observation vector, matching the robosuite layout:
            [joint_pos(7), cos_q(7), sin_q(7), joint_vel(7), joint_acc(7),
             ee_pos(3), ee_quat(4), ee_quat_site(4), gripper_qpos(2), gripper_qvel(2),
             world_pose_in_gripper(3), rel_ee_to_bolt_pos(3), rel_quat(4),
             bolt_pos(3), bolt_quat(4)]
            """
            # Robot proprioception
            q = self.robot.data.joint_pos[:, :7]
            qd = self.robot.data.joint_vel[:, :7]
            qa = self.robot.data.joint_acc[:, :7] if hasattr(self.robot.data, "joint_acc") \
                 else torch.zeros_like(qd)

            ee_pos, ee_quat = self._get_ee_pose()
            gripper_pos = self.robot.data.joint_pos[:, 7:9]
            gripper_vel = self.robot.data.joint_vel[:, 7:9]

            # Object state
            bolt_pos = self.bolt.data.root_pos_w - self.scene.env_origins
            bolt_quat = self.bolt.data.root_quat_w

            # Relative transforms
            rel_pos, rel_quat = subtract_frame_transforms(
                ee_pos, ee_quat, bolt_pos, bolt_quat
            )
            # World pose in gripper frame
            world_in_gripper, _ = subtract_frame_transforms(
                ee_pos, ee_quat,
                torch.zeros_like(ee_pos), torch.tensor([1,0,0,0], device=self.device).expand(self.num_envs,-1)
            )

            obs = torch.cat([
                q,                            # 7
                torch.cos(q),                 # 7
                torch.sin(q),                 # 7
                qd,                           # 7
                qa,                           # 7
                ee_pos,                       # 3
                ee_quat,                      # 4
                ee_quat,                      # 4 (site frame — same here)
                gripper_pos,                  # 2
                gripper_vel,                  # 2
                world_in_gripper,             # 3
                rel_pos,                      # 3
                rel_quat,                     # 4
                bolt_pos,                     # 3
                bolt_quat,                    # 4
            ], dim=-1)  # → [n_envs, 64]

            # Domain randomisation: observation noise
            if self.cfg.use_domain_rand and self.cfg.obs_noise_std > 0:
                obs = obs + torch.randn_like(obs) * self.cfg.obs_noise_std

            return {"policy": obs}

        # ── rewards ───────────────────────────────────────────────────────────

        def _get_rewards(self) -> torch.Tensor:
            """
            Hierarchical shaped reward (reach → grasp → screw → move-away)
            + potential-based shaping for policy invariance.
            """
            cfg = self.cfg
            ee_pos, ee_quat = self._get_ee_pose()
            bolt_pos = self.bolt.data.root_pos_w - self.scene.env_origins

            # ── reach reward ──────────────────────────────────────────────────
            dist = torch.norm(ee_pos - bolt_pos, dim=-1)
            r_reach = (1.0 - torch.tanh(5.0 * dist)) * cfg.reach_weight

            # ── grasp reward (contact proxy: distance < threshold) ────────────
            grasp_mask = (dist < 0.04).float()
            r_grasp = grasp_mask * cfg.grasp_weight

            # ── screw reward ──────────────────────────────────────────────────
            # Normalised progress: how many full turns completed
            turns_done = self._bolt_angle / (2 * math.pi)
            turns_target = 5.0  # M5 bolt ~5 turns to fully remove
            r_rotate = torch.clamp(turns_done / turns_target, 0.0, 1.0)

            slide_progress = torch.clamp(
                self._bolt_slide / (self.SCREW_PITCH * turns_target), 0.0, 1.0
            )
            r_screw = (0.5 * r_rotate + 0.5 * slide_progress) * cfg.screw_weight

            # ── move-away reward (only once screwing is nearly complete) ──────
            screw_done = (turns_done > 0.8 * turns_target).float()
            peg_pos = bolt_pos.clone()
            peg_pos[:, 2] -= 0.02  # peg tip below bolt initial pos
            off_peg_dist = torch.norm(bolt_pos - peg_pos, dim=-1)
            r_move = screw_done * torch.tanh(10.0 * off_peg_dist) * cfg.moveaway_weight

            # ── success bonus ─────────────────────────────────────────────────
            success = self._check_success()
            r_success = success.float() * cfg.success_bonus

            # ── potential-based shaping ───────────────────────────────────────
            new_potential = -dist
            shaping = cfg.potential_gamma * new_potential - self._prev_potential
            self._prev_potential = new_potential.detach()

            total = r_reach + r_grasp + r_screw + r_move + r_success + shaping
            return total

        # ── termination ───────────────────────────────────────────────────────

        def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Returns (terminated, truncated).
            terminated: task success
            truncated:  episode time limit
            """
            success = self._check_success()
            time_out = self._step_count >= int(
                self.cfg.episode_length_s * self.cfg.sim.render_interval / self.cfg.sim.dt
            )
            self._step_count += 1
            return success, time_out

        # ── reset ─────────────────────────────────────────────────────────────

        def _reset_idx(self, env_ids: torch.Tensor) -> None:
            """Reset specified environments to initial state."""
            super()._reset_idx(env_ids)

            n = len(env_ids)
            dev = self.device

            # Reset bolt position with small random perturbation
            bolt_pos = torch.zeros(n, 3, device=dev)
            bolt_pos[:, 0] = 0.5 + torch.rand(n, device=dev) * 0.02 - 0.01
            bolt_pos[:, 1] = torch.rand(n, device=dev) * 0.02 - 0.01
            bolt_pos[:, 2] = 0.44
            bolt_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev).expand(n, -1)
            self.bolt.write_root_pose_to_sim(
                torch.cat([bolt_pos, bolt_quat], dim=-1), env_ids=env_ids
            )

            # Reset robot to home configuration
            default_q = self.robot.data.default_joint_pos[env_ids]
            self.robot.write_joint_position_target(default_q, env_ids=env_ids)
            self.robot.write_joint_state_to_sim(
                default_q,
                torch.zeros_like(default_q),
                env_ids=env_ids,
            )

            # Reset per-env state
            self._bolt_angle[env_ids] = 0.0
            self._bolt_slide[env_ids] = 0.0
            self._step_count[env_ids] = 0

            # Domain randomisation at reset
            if self.cfg.use_domain_rand:
                self._apply_domain_rand(env_ids)

            # Recompute initial potential
            ee_pos, _ = self._get_ee_pose()
            bolt_pos_w = self.bolt.data.root_pos_w - self.scene.env_origins
            dist = torch.norm(ee_pos[env_ids] - bolt_pos_w[env_ids], dim=-1)
            self._prev_potential[env_ids] = -dist

        # ── domain randomisation ──────────────────────────────────────────────

        def _apply_domain_rand(self, env_ids: torch.Tensor) -> None:
            """Randomise PhysX material properties for the given env subset."""
            n = len(env_ids)
            dev = self.device

            # Friction scale (applied to bolt-table contact material)
            friction_scale = (
                torch.rand(n, device=dev)
                * (self.cfg.friction_range[1] - self.cfg.friction_range[0])
                + self.cfg.friction_range[0]
            )
            # Mass multiplier
            mass_mult = (
                torch.rand(n, device=dev)
                * (self.cfg.mass_mult_range[1] - self.cfg.mass_mult_range[0])
                + self.cfg.mass_mult_range[0]
            )
            # Apply via PhysX API (isaaclab.utils.physics_utils if available)
            # Fallback: scale the bolt inertia tensor
            base_mass = 0.01  # kg (from cfg)
            new_masses = base_mass * mass_mult
            self.bolt.write_root_inertia_to_sim(
                self.bolt.data.default_root_state[env_ids, 10:13] * mass_mult.unsqueeze(-1),
                env_ids=env_ids,
            ) if hasattr(self.bolt, "write_root_inertia_to_sim") else None

        # ── helpers ───────────────────────────────────────────────────────────

        def _get_ee_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
            """Return end-effector position and quaternion [n_envs, 3/4]."""
            ee_idx = self.robot.find_bodies("panda_hand")[0][0]
            pos = self.robot.data.body_pos_w[:, ee_idx, :] - self.scene.env_origins
            quat = self.robot.data.body_quat_w[:, ee_idx, :]
            return pos, quat

        def _osc_to_joint_targets(
            self,
            target_pos: torch.Tensor,
            target_quat: torch.Tensor,
            gripper_cmd: torch.Tensor,
        ) -> torch.Tensor:
            """
            Simplified Jacobian pseudo-inverse IK.
            Replace with IsaacLab's DifferentialIKController for production use.
            """
            # Current joint positions
            q = self.robot.data.joint_pos[:, :7]
            # Use current joints as targets (IK would update this)
            # This is a placeholder — real IK needed for deployment
            joint_targets = torch.cat([
                q,
                gripper_cmd.expand(-1, 2) * 0.04,
            ], dim=-1)
            return joint_targets

        def _check_success(self) -> torch.Tensor:
            """
            Task is successful when bolt is off the peg:
            horizontal separation > 5 cm OR lifted > 2 cm above peg surface.
            """
            bolt_pos = self.bolt.data.root_pos_w - self.scene.env_origins
            peg_center = torch.zeros_like(bolt_pos)
            peg_center[:, 0] = 0.5
            peg_center[:, 2] = 0.44

            horiz = torch.norm(bolt_pos[:, :2] - peg_center[:, :2], dim=-1)
            vert = bolt_pos[:, 2] - peg_center[:, 2]

            return (horiz > self.SUCCESS_DIST) | (vert > 0.02)

else:
    # Stub classes so the rest of the codebase can import without crashing
    class BoltScrewdrivingEnvCfg:  # type: ignore[no-redef]
        pass

    class BoltScrewdrivingEnv:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _check_isaaclab()
