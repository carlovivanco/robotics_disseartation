# AI-Enhanced Intelligent Systems and Robotics for Flexible Manufacturing Systems

Automated M5 bolt screwdriving / disassembly with a Franka Panda robot using
Deep Reinforcement Learning and Imitation Learning.

---

## Project Overview

| Component | Original | Improved (this branch) |
|-----------|----------|------------------------|
| Simulator | robosuite + MuJoCo (8 CPU envs) | **robosuite** OR **IsaacLab** (4096 GPU envs) |
| DRL algorithm | TD3 | **TQC** (Truncated Quantile Critics) |
| IL algorithm | Behavior Cloning | **Diffusion Policy** + **DAgger** |
| Configuration | Comment-toggle in code | **Hydra** YAML + CLI flags |
| Logging | TensorBoard only | **W&B** + TensorBoard + CSV |
| Sim-to-Real | None | **Domain Randomization** (friction, mass, gravity, obs noise) |
| Reward shaping | Hand-crafted | + **Potential-based shaping** (policy-invariant) |
| Reproducibility | No seed management | **Global seed** (Python, NumPy, PyTorch, CUDA) |

---

## Directory Structure

```
robotics_disseartation/
├── configs/                   ← Hydra YAML configs
│   ├── config.yaml            ← Main entrypoint
│   ├── algorithm/
│   │   ├── tqc.yaml           ← TQC hyperparameters (default)
│   │   └── sac.yaml           ← SAC alternative
│   ├── env/
│   │   ├── robosuite.yaml     ← CPU simulation (default)
│   │   └── isaaclab.yaml      ← GPU simulation (4096 envs)
│   └── il/
│       ├── diffusion.yaml     ← Diffusion Policy (default)
│       └── dagger.yaml        ← DAgger
│
├── envs/
│   ├── domain_randomization.py  ← Physics parameter randomisation
│   ├── robosuite_wrapper.py     ← Enhanced env (DR + potential shaping)
│   └── isaaclab_bolt_env.py     ← IsaacLab DirectRLEnv (GPU)
│
├── algorithms/
│   ├── diffusion_policy.py    ← DDPM 1D U-Net + DDIM inference + EMA
│   └── dagger.py              ← Dataset Aggregation (interactive IL)
│
├── utils/
│   ├── seed.py                ← Global seed management
│   └── logger.py              ← W&B + TensorBoard + CSV logger
│
├── train_drl.py               ← New DRL training script (Hydra-driven)
├── train_il.py                ← New IL training script (Hydra-driven)
├── evaluate.py                ← Unified evaluation (any policy type)
│
├── DRL_BoltSD.py              ← Original DRL script (kept for reference)
├── Imitation.py               ← Original IL script (kept for reference)
├── Integration.py             ← ROS2 real-robot deployment node
└── data_collector.py          ← Human demonstration collection
```

---

## 1. Environment Setup

### Base environment (robosuite, required for all modes)

```bash
conda create -n bolt_rl python=3.10 -y
conda activate bolt_rl

# Install robosuite fork (custom bolt environment)
pip install -e .

# Base dependencies
pip install -r requirements1.txt

# New improved-pipeline dependencies
pip install -r requirements_improvements.txt
```

### IsaacLab (optional — NVIDIA GPU required, RTX 3090+ recommended)

IsaacLab provides 4096 parallel GPU environments vs 8 CPU environments,
reducing training time by ~100×.

```bash
# 1. Install Isaac Sim 4.x via the NVIDIA Omniverse Launcher
#    https://developer.nvidia.com/isaac-sim

# 2. Install IsaacLab inside Isaac Sim's Python environment
#    (replace <isaac_sim_path> with your actual installation path)
<isaac_sim_path>/python.sh -m pip install isaaclab
<isaac_sim_path>/python.sh -m pip install isaaclab_assets

# 3. Export the M5 bolt USD asset from the robosuite MuJoCo model
mkdir -p assets
python tools/export_bolt_usd.py --output assets/m5_bolt.usd
# (if the export tool is not available, use the MuJoCo USD exporter:
#  https://mujoco.readthedocs.io/en/latest/programming/usd.html)
```

---

## 2. Collect Human Demonstrations

```bash
# Keyboard control
python data_collector.py \
    --environment BoltScrewdrivingDisassembly \
    --robots Panda \
    --device keyboard \
    --renderer mjviewer

# PS5 DualSense controller
python data_collector.py \
    --environment BoltScrewdrivingDisassembly \
    --robots Panda \
    --device dualsense

# Output: demos/demo_SD_2/demo.hdf5
```

---

## 3. Deep Reinforcement Learning (TQC)

### Train

```bash
# Robosuite backend (8 CPU envs — original setup, improved algorithm)
python train_drl.py mode=train env=robosuite algorithm=tqc

# IsaacLab backend (4096 GPU envs — ~100× faster)
python train_drl.py mode=train env=isaaclab algorithm=tqc

# Override any hyperparameter on the command line
python train_drl.py mode=train algorithm.learning_rate=3e-4 env.n_envs=16

# Hyperparameter sweep with Hydra multirun
python train_drl.py --multirun \
    algorithm.learning_rate=1e-4,3e-4,1e-3 \
    algorithm.batch_size=256,512
```

### Evaluate

```bash
python train_drl.py mode=eval env=robosuite

# Or use the unified evaluator:
python evaluate.py \
    --policy drl \
    --model_path outputs/<run>/model/best_model.zip \
    --vecnorm_path vec_normalize_SD_OSC.pkl \
    --n_episodes 20 \
    --render
```

### Monitor training

```bash
# TensorBoard
tensorboard --logdir=outputs/ --port 6006

# W&B (configure entity in configs/config.yaml)
# Go to https://wandb.ai/your-entity/bolt-screwdriving
```

---

## 4. Imitation Learning

### 4a. Diffusion Policy (recommended — state-of-the-art)

```bash
# Train
python train_il.py mode=train il=diffusion

# Evaluate
python train_il.py mode=eval il=diffusion

# Or with unified evaluator
python evaluate.py \
    --policy diffusion \
    --model_path diffusion_model_best.pth \
    --n_episodes 10 \
    --render
```

**Why Diffusion Policy?**  
Unlike Behavior Cloning, Diffusion Policy models the full distribution of
expert actions using a denoising diffusion process. This makes it far more
robust to multi-modal demonstrations and avoids the compounding-error problem
of standard BC.

### 4b. DAgger (interactive imitation learning)

```bash
# Run the full DAgger loop (requires a pre-trained BC oracle)
python train_il.py mode=dagger il=dagger

# The oracle can also be a pre-trained DRL model:
# edit configs/il/dagger.yaml → oracle.type: drl, oracle.path: ...
```

**Why DAgger?**  
DAgger collects data at states visited by the *learned* policy (not just the
expert), solving BC's distributional shift problem.

### 4c. Legacy Behavior Cloning (for ablation / comparison)

```bash
python train_il.py mode=train il=bc
```

---

## 5. Real Robot Deployment (ROS2)

```bash
# Source ROS2 Humble
source /opt/ros/humble/setup.bash

# Launch the policy node (edit MODEL_PATH in Integration.py)
ros2 run <package> franka_policy_node

# Or run directly:
python Integration.py
```

The `Integration.py` node:
- Subscribes to Franka joint states and EE pose at 20 Hz
- Loads a TQC/TD3 policy (with VecNormalize stats)
- Publishes equilibrium pose targets

---

## 6. Key Configuration Options

All settings live in `configs/`. Override anything via CLI:

```bash
# Change simulator
python train_drl.py env=isaaclab

# Disable W&B
python train_drl.py logging.use_wandb=false

# Change seed
python train_drl.py seed=123

# Change IL method
python train_il.py il=dagger

# Full example: TQC + IsaacLab + no wandb
python train_drl.py \
    mode=train \
    env=isaaclab \
    algorithm=tqc \
    logging.use_wandb=false \
    seed=42
```

---

## 7. Output Files

| File | Script | Contents |
|------|--------|----------|
| `outputs/<run>/model/best_model.zip` | `train_drl.py` | Best TQC policy |
| `vec_normalize_SD_OSC.pkl` | `train_drl.py` | Observation normalisation stats |
| `diffusion_model_best.pth` | `train_il.py` | Diffusion Policy weights + norm stats |
| `dagger_model_best.pth` | `train_il.py` | DAgger MLP policy |
| `training_diffusion.csv` | `train_il.py` | Per-epoch train/val loss |
| `training_dagger.csv` | `train_il.py` | Per-iteration DAgger val loss |
| `Evaluation_DRL_improved.csv` | `evaluate.py` | Per-step reward (DRL) |
| `Evaluation_diffusion.csv` | `evaluate.py` | Per-step reward (Diffusion Policy) |
| `outputs/<run>/tensorboard/` | `train_drl.py` | TensorBoard events |
| `outputs/<run>/metrics.csv` | both | All logged metrics |

---

## 8. Domain Randomisation (Sim-to-Real)

Enabled by default (`env.domain_randomization.enabled: true`). Randomises at
every episode reset:

| Parameter | Range | Purpose |
|-----------|-------|---------|
| Coulomb friction | [0.3, 1.5] | Bolt-surface contact variability |
| Bolt mass | ×[0.8, 1.2] | Manufacturing tolerances |
| Joint damping | ×[0.5, 2.0] | Actuator variability |
| Gravity | [9.5, 10.1] m/s² | IMU / calibration noise |
| Observation noise | σ=0.01 | Sensor noise simulation |
| Action delay | p=0.05 | Network / actuator latency |

Disable for faster training on clean simulation:
```bash
python train_drl.py env.domain_randomization.enabled=false
```

---

## 9. Vision Segmentation

```bash
cd Vision_M
pip install -r requirements.txt
python vision_seg.py   # webcam or RealSense
```

See `Vision_M/vision_seg.py` for webcam vs RealSense configuration.

---

## 10. Troubleshooting

**`sb3_contrib` not found → falls back to TD3**  
```bash
pip install sb3-contrib
```

**IsaacLab import error**  
IsaacLab must be installed inside Isaac Sim's Python environment. See Section 1.

**W&B authentication**  
```bash
wandb login
# or set WANDB_API_KEY environment variable
```

**CUDA out of memory (IsaacLab)**  
Reduce `env.n_envs` in `configs/env/isaaclab.yaml` (default 4096).
For an RTX 3080 try 2048; for an RTX 4090 try 8192.
