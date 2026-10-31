"""Roll out the trained TD3 policy on BoltScrewdrivingDisassembly and export an H.264 clip.

Same env/controller config as make_screwdriving_env in DRL_BoltSD.py (copied, because importing
DRL_BoltSD runs its __main__ test block). Offscreen render, no window needed.

Usage (from repo root):
  MUJOCO_GL=egl python scripts/export_sim_video.py \
      --model logs_OSC/BoltSD_T/TD3/best_model.zip --vecnorm vec_normalize_SD_OSC2.pkl \
      --out portfolio_export/franka-sim.mp4
"""
import argparse
import os
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")

_GL = os.environ["MUJOCO_GL"]

import robosuite.utils.binding_utils as binding_utils

# robosuite overwrites MUJOCO_GL with "wgl" when macros.MUJOCO_GPU_RENDERING is on (unless osmesa/glx);
# restore the backend requested by the user (egl/osmesa) without editing the repo.
binding_utils._MUJOCO_GL = _GL.lower().strip()
os.environ["MUJOCO_GL"] = _GL

import imageio.v2 as imageio
import numpy as np
from robosuite.controllers import load_composite_controller_config
from robosuite.environments.manipulation.bolt_screwdriving_dissassembly import BoltScrewdrivingDisassembly
from robosuite.wrappers import GymWrapper
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

W, H, CONTROL_FREQ = 1280, 720, 20


def controller_config():
    cfg = load_composite_controller_config(controller=None, robot="Panda")
    cfg["body_parts"]["right"].update({
        "type": "OSC_POSE",
        "input_max": [1.0] * 6, "input_min": [-1.0] * 6,
        "output_max": [0.1] * 6, "output_min": [-0.1] * 6,
        "kp": 150, "damping_ratio": 1.0, "impedance_mode": "fixed",
        "control_delta": False, "input_ref_frame": "base",
    })
    return cfg


def make_env(camera_h, camera_w):
    env = BoltScrewdrivingDisassembly(
        robots="Panda", gripper_types="default", env_configuration="opposed",
        controller_configs=controller_config(),
        has_renderer=False, has_offscreen_renderer=True,
        control_freq=CONTROL_FREQ, horizon=5000,
        use_object_obs=True, use_camera_obs=False, reward_shaping=True,
        single_object_mode=2, nut_type="m5", initial_grasp=True,
        camera_heights=camera_h, camera_widths=camera_w,
    )
    return env, Monitor(GymWrapper(env))


def bolt_joints(env):
    """(slide, rotate) of the M5 bolt; slide > 0 means the bolt moved up (unscrewing)."""
    slide, rotate = env.nuts[0].joints[:2]
    return float(env.sim.data.get_joint_qpos(slide)), float(env.sim.data.get_joint_qpos(rotate))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="logs_OSC/BoltSD_T/TD3/best_model.zip")
    p.add_argument("--vecnorm", default="vec_normalize_SD_OSC2.pkl")
    p.add_argument("--out", default="portfolio_export/franka-sim.mp4")
    p.add_argument("--camera", default="frontview")
    p.add_argument("--max-steps", type=int, default=800, help="policy steps to simulate (20 Hz)")
    p.add_argument("--start-s", type=float, default=None,
                   help="clip start (s). Default: 1 s before the bolt rotation first changes")
    p.add_argument("--duration", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    # Load the policy before creating the GL context: building torch optimizers after an
    # OSMesa context exists segfaults in some setups.
    model = TD3.load(a.model, device="cpu")
    raw, gym_env = make_env(H, W)
    venv = DummyVecEnv([lambda: gym_env])
    venv = VecNormalize.load(a.vecnorm, venv)
    venv.training, venv.norm_reward = False, False
    np.random.seed(a.seed)

    obs = venv.reset()
    _, rot0 = bolt_joints(raw)
    frames, rot, total = [], [], 0.0
    for _ in range(a.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = venv.step(action)
        total += float(reward[0])
        frames.append(raw.sim.render(width=W, height=H, camera_name=a.camera)[::-1])
        rot.append(bolt_joints(raw)[1] - rot0)
        if done[0]:
            break
    if not frames:
        raise RuntimeError("no frames rendered")

    if a.start_s is None:
        hit = next((i for i, m in enumerate(rot) if abs(m) > 0.05), 0)
        start = max(0, hit - CONTROL_FREQ)
    else:
        start = int(a.start_s * CONTROL_FREQ)
    n = int(a.duration * CONTROL_FREQ)
    clip = frames[start:start + n]
    slide, rotate = bolt_joints(raw)
    print(f"steps simulated: {len(frames)}; clip steps {start}..{start + len(clip)} "
          f"({len(clip) / CONTROL_FREQ:.1f}s)")
    print(f"episode reward so far: {total:.1f}; final bolt slide={slide:.4f} m, "
          f"rotate change={rot[-1]:.3f} rad, success={raw._check_success()}")
    print("CHECK THE VIDEO: this script exports whatever the policy does; it does not guarantee unscrewing.")
    if len(clip) < 5 * CONTROL_FREQ:
        print("WARNING: clip shorter than 5 s; raise --max-steps or lower --start-s")

    raw_path = a.out + ".raw.mp4"
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    imageio.mimsave(raw_path, clip, fps=CONTROL_FREQ, macro_block_size=1)
    dur = len(clip) / CONTROL_FREQ
    fade = 0.4  # fade from/to black so the loop point is clean
    vf = f"fps=30,fade=t=in:st=0:d={fade},fade=t=out:st={dur - fade}:d={fade}"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw_path, "-vf", vf, "-an",
                    "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", a.out], check=True)
    os.remove(raw_path)
    print(f"wrote {a.out}\ncheckpoint: {a.model} + {a.vecnorm}")


if __name__ == "__main__":
    main()
