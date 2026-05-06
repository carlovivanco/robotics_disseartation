# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a robotics dissertation implementing Deep Reinforcement Learning (DRL) and Imitation Learning (IL) for automated bolt screwdriving/disassembly on a Franka Panda robot. The pipeline covers: human demonstration collection → behavior cloning → RL fine-tuning → sim-to-real deployment via ROS2.

**Simulator**: robosuite (locally modified v1.5.1) + MuJoCo ≥ 3.3.0  
**Task environment**: `BoltScrewdrivingDisassembly` — Panda arm unscrews an M5 bolt using OSC_POSE controller  
**Key algorithms**: TD3 (primary), SAC/PPO (alternate), Behavior Cloning

## Setup & Environment

```bash
# Install main dependencies
pip install -r requirements1.txt

# Install local robosuite fork (required — the stock robosuite package lacks the bolt envs)
pip install -e .

# Vision module (separate environment recommended)
pip install -r Vision_M/requirements.txt
```

The expected conda env name referenced in code is `mujoco_franka`.

## Common Commands

```bash
# Collect human demonstrations (keyboard/spacemouse/dualsense)
python data_collector.py --environment BoltScrewdrivingDisassembly --robots Panda --device keyboard --renderer mjviewer

# Train DRL — comment/uncomment the train vs test blocks in DRL_BoltSD.py
python DRL_BoltSD.py

# Train IL (Behavior Cloning) — comment/uncomment train vs eval blocks in Imitation.py
python Imitation.py

# Monitor training
tensorboard --logdir=./tensorboard --port 6006

# Vision segmentation (from Vision_M/)
python Vision_M/vision_seg.py

# Real robot (ROS2 Humble required)
source /opt/ros/humble/setup.bash && ros2 run <package> franka_policy_node
```

## Architecture

### Training & Inference Toggle Pattern

Both `DRL_BoltSD.py` and `Imitation.py` use a **comment-based train/test toggle** — there is no CLI flag. Scroll to the bottom of each file and comment/uncomment the relevant block to switch modes.

### DRL Pipeline (`DRL_BoltSD.py`)

- 8 parallel `SubprocVecEnv` environments + `VecNormalize` (obs + reward normalization)
- `VecNormalize` stats saved to `vec_normalize_SD_OSC2.pkl` — must be loaded alongside the policy for evaluation
- Custom callbacks:
  - `DynamicExplorationCallback` — increases TD3 action noise when rewards plateau
  - Checkpoint callback — saves at reward thresholds `[450, 1950, 4250]` and freezes early layers
  - `StopTrainingOnNoModelImprovement` — halts after 20 evals without improvement
- Best model saved to `logs_OSC/BoltSD_T/TD3/best_model.zip`
- Evaluation logged to `Evaluation_DRL_SD2.csv`

### IL Pipeline (`Imitation.py`)

- Loads HDF5 demos from `demos/demo_SD_2/demo.hdf5`
- Z-score normalization (obs + actions) with stats saved inline for inference
- MLP architecture: two hidden layers (default `[256, 1024]`)
- Best weights: `bc_model_best_SD.pth`; final: `bc_model_SD.pth`
- Training curves: `training_IL_SD.csv`; evaluation: `Evaluation_IL_SD.csv`
- Optuna integration exists but is commented out (20 trials)

### Vision Module (`Vision_M/`)

- `vision_seg.py` — YOLOv8/YOLOv11m-seg for bolt segmentation + centroid extraction; supports webcam and RealSense
- `vision.py` — YOLOv11 detection + ViT (`vit_base_patch16_224`) for bolt grade classification (8 classes); includes RealSense 3D coordinate extraction

### Real Robot Deployment (`Integration.py`)

- ROS2 node subscribing to Franka joint states and EE pose; publishes `PoseStamped` equilibrium targets at 20 Hz
- Observation: 50D proprioception + 14D object state = 64D total
- Action: 6D EE delta (pos ×0.1, angle ×0.2) + 1D gripper

### Custom Environment (`robosuite/environments/manipulation/bolt_screwdriving_dissassembly.py`)

- Controller: `OSC_POSE`, Kp=150, 20 Hz, 7D action space (6D pose + 1D gripper)
- Observation: ~64D (joint pos/vel/acc, EE pos+quat, gripper state, object-relative pose)
- Horizon: 5000 steps (DRL) / 2000 steps (IL eval)
- `single_object_mode=2` forces a constant M5 nut; `use_latch=True` enables initial grasp

## Key Hyperparameters (Optuna-tuned)

| | TD3 (DRL) | BC (IL) |
|---|---|---|
| Learning rate | 0.00015546 | 0.000358 |
| Network | [512, 256] actor/critic | [256, 1024] |
| Batch size | 512 | 128 |
| Gamma | 0.9018 | — |
| Tau | 0.01417 | — |
| Optimizer | Adam | RMSprop |

## Output Files

| File | Source | Contents |
|------|--------|----------|
| `logs_OSC/BoltSD_T/TD3/best_model.zip` | DRL training | Best TD3 policy |
| `vec_normalize_SD_OSC2.pkl` | DRL training | VecNormalize stats (required for inference) |
| `bc_model_best_SD.pth` | IL training | Best BC weights |
| `Evaluation_DRL_SD2.csv` | DRL test | Per-step timestep + reward |
| `Evaluation_IL_SD.csv` | IL eval | Per-step timestep + reward |
| `training_IL_SD.csv` | IL training | Epoch-level train/val loss |
| `tensorboard/BoltSD_OSC/TD3/` | DRL training | TensorBoard scalars |
