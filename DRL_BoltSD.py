"""
DRL_BoltSD.py

Deep Reinforcement Learning for Bolt Screwdriving Task using Robosuite and Stable-Baselines3.
Includes hyperparameter tuning with Optuna, training, evaluation, and model management.

Author: [Your Name]
Date: [Today's Date]
"""

# === Imports ===
from stable_baselines3 import SAC, PPO, TD3, DDPG, A2C
import robosuite as suite
import pandas as pd
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import GymWrapper
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnNoModelImprovement,
    BaseCallback,
    CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.utils import get_schedule_fn
import os
import optuna
from optuna.pruners import MedianPruner
import numpy as np
import torch
from robosuite.environments.manipulation.bolt_screwdriving_dissassembly import BoltScrewdrivingDisassembly

# === Controller Configuration ===
# Load and modify the composite controller for the Panda robot
composite_controller_config = load_composite_controller_config(controller=None, robot="Panda")
controller_config = composite_controller_config["body_parts"]["right"]
controller_config["type"] = "OSC_POSE"
controller_config.update({
    "type": "OSC_POSE",
    "input_max": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "input_min": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
    "output_max": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    "output_min": [-0.1, -0.1, -0.1, -0.1, -0.1, -0.1],
    "kp": 150,
    "damping_ratio": 1.0,
    "impedance_mode": "fixed",
    "control_delta": False,
    "input_ref_frame": "base",  # Use robot base frame for control
})

def make_screwdriving_env(composite_controller_config, has_renderer, offscreen_renderer):
    """
    Factory function to create a monitored and Gym-wrapped BoltScrewdrivingDisassembly environment.

    Args:
        composite_controller_config (dict): Controller configuration.
        has_renderer (bool): Enable on-screen rendering.
        offscreen_renderer (bool): Enable offscreen rendering.

    Returns:
        function: Environment initializer for vectorized environments.
    """
    def _init():
        env = BoltScrewdrivingDisassembly(
            robots="Panda",
            gripper_types="default",
            env_configuration="opposed",
            controller_configs=composite_controller_config,
            has_renderer=has_renderer,
            has_offscreen_renderer=offscreen_renderer,
            control_freq=20,
            horizon=5000,
            use_object_obs=True,
            use_camera_obs=False,
            reward_shaping=True,
            single_object_mode=2,
            nut_type="m5",
            initial_grasp=True,
        )
        return Monitor(GymWrapper(env))
    return _init

# === Utility: Freeze Layers ===
def freeze_policy_layers(model, num_layers_to_freeze=1, verbose=True):
    """
    Freezes the first N linear layers of the actor and critic networks in the policy.

    Args:
        model (BaseAlgorithm): The trained model whose layers will be frozen.
        num_layers_to_freeze (int): Number of linear layers to freeze.
        verbose (bool): Enable logging.
    """
    if verbose:
        print(f"[Freezing] Freezing {num_layers_to_freeze} linear layers.")

    policy = model.policy
    frozen = 0
    if hasattr(policy, "mlp_extractor"):
        all_layers = list(policy.mlp_extractor.policy_net.children()) + list(policy.mlp_extractor.value_net.children())
        for layer in all_layers:
            if isinstance(layer, torch.nn.Linear):
                for param in layer.parameters():
                    param.requires_grad = False
                frozen += 1
                if frozen >= num_layers_to_freeze:
                    break
    if verbose:
        print(f"[Freezing] Layers frozen: {frozen}")

# === Custom Callback for Exploration & Freezing ===
class DynamicExplorationCallback(BaseCallback):
    """
    Callback that adjusts exploration dynamically and freezes layers
    based on reward thresholds during training.
    """
    def __init__(self, thresholds, patience=10, noise_scale=1.2, verbose=1):
        super().__init__(verbose)
        self.thresholds = thresholds
        self.patience = patience
        self.noise_scale = noise_scale
        self.reached = set()
        self.no_improve = 0
        self.last_mean = -np.inf

    def _on_rollout_end(self):
        """
        Called at the end of each rollout to evaluate reward improvements and take actions accordingly.
        """
        if len(self.model.ep_info_buffer) == 0:
            return

        mean_r = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
        if self.verbose > 0:
            print(f"[Callback] Mean reward: {mean_r:.2f}")

        # Freeze layers when certain thresholds are reached
        for i, thr in enumerate(self.thresholds):
            if thr not in self.reached and mean_r >= thr:
                self.reached.add(thr)
                self.no_improve = 0
                self.model.save(f"checkpoint_reached_{thr}.zip")
                if self.verbose > 0:
                    print(f"[Callback] Threshold {thr} reached. Model saved.")
                freeze_policy_layers(self.model, num_layers_to_freeze=i+1, verbose=self.verbose)

        # Increase exploration if no improvement
        if self.reached:
            if mean_r <= self.last_mean + 10:
                self.no_improve += 1
            else:
                self.no_improve = 0

            if self.no_improve >= self.patience:
                self._increase_noise()
                self.no_improve = 0

        self.last_mean = mean_r

    def _increase_noise(self):
        """
        Dynamically increase the action noise to promote exploration.
        """
        if hasattr(self.model, "action_noise") and isinstance(self.model.action_noise, NormalActionNoise):
            old_sigma = self.model.action_noise.sigma.copy()
            self.model.action_noise.sigma = np.clip(
                self.model.action_noise.sigma * self.noise_scale, 0.1, 1.0
            )
            if self.verbose > 0:
                print(f"[Callback] Increased exploration noise: sigma {old_sigma} -> {self.model.action_noise.sigma}")

    def _on_step(self) -> bool:
        return True

def DRL_Training(cfg):
    """
    Trains the TD3 agent on the screwdriving environment.

    Args:
        cfg (dict): Controller configuration.

    Returns:
        dict: Trained models keyed by algorithm name.
    """
    num_envs = 8
    n_actions = 7

    train_env = SubprocVecEnv([make_screwdriving_env(composite_controller_config, False, False) for _ in range(num_envs)])
    eval_env = SubprocVecEnv([make_screwdriving_env(composite_controller_config, False, False) for _ in range(num_envs)])

    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.)
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)

    act_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.25 * np.ones(n_actions))

    models_dict = {"TD3": TD3}
    trained_models = {}

    for name, algo in models_dict.items():
        log_dir = f"./logs_OSC/BoltSD/{name}/"
        os.makedirs(log_dir, exist_ok=True)

        stop_cb = StopTrainingOnNoModelImprovement(max_no_improvement_evals=20, min_evals=200, verbose=1)
        explore_cb = DynamicExplorationCallback(thresholds=[450, 1950, 4250], patience=20, verbose=1)
        callback_after_eval = CallbackList([stop_cb, explore_cb])
        eval_cb = EvalCallback(eval_env, eval_freq=20_000, n_eval_episodes=20,
                               deterministic=True, callback_after_eval=callback_after_eval,
                               best_model_save_path=log_dir, log_path=log_dir, verbose=1)

        print(f"\n===== Training {name} =====")

        kwargs = dict(
            policy="MlpPolicy",
            env=train_env,
            tensorboard_log=f"./tensorboard/BoltSD_OSC/{name}/",
            verbose=1,
            policy_kwargs=dict(
                net_arch=dict(pi=[512, 256], qf=[512, 256]),
                activation_fn=torch.nn.ReLU,
            ),
            learning_rate=0.00015546357086099763,
            buffer_size=2_000_000,
            batch_size=512,
            tau=0.014172157215208286,
            gamma=0.9018495089302186,
            train_freq=(2, "step"),
            gradient_steps=16,
            learning_starts=10_000,
            action_noise=act_noise,
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            device="cuda",
        )

        model_path = f"{log_dir}/best_model.zip"
        if os.path.exists(model_path):
            print(f"Loading existing {name} model …")
            model = algo.load(model_path, env=train_env)
        else:
            print("New Training")
            model = algo(**kwargs)

        model.learn(total_timesteps=15_000_000, callback=eval_cb)
        trained_models[name] = model
        # Reload best model before final evaluation
        model = algo.load(model_path, env=eval_env)
        mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
        print(f"{name} mean reward: {mean_r:.1f} ± {std_r:.1f}")

    train_env.close()
    eval_env.close()
    train_env.save("vec_normalize_SD_OSC.pkl")
    return trained_models

def linear_schedule(initial_value):
    """
    Returns a callable for linearly decaying learning rate.

    Args:
        initial_value (float): Initial learning rate.

    Returns:
        function: Schedule function for learning rate.
    """
    return get_schedule_fn(initial_value)

def DRL_Test(model, composite_controller_config):
    """
    Test a trained model visually using rendering.

    Args:
        model (BaseAlgorithm): Trained RL model.
        composite_controller_config (dict): Controller configuration for the robot.
    """
    env_eval = DummyVecEnv([make_screwdriving_env(composite_controller_config, has_renderer=True, offscreen_renderer=False)])
    env_eval = VecNormalize.load("vec_normalize_SD_OSC2.pkl", env_eval)

    # Ensure it's in eval mode
    env_eval.training = False
    env_eval.norm_reward = False
    obs_ev = env_eval.reset()
    done = [False]
    time_steps = []
    time_step = 0
    rewards = []
    while not done[0]:
        time_step += 1
        time_steps.append(time_step)
        action, _ = model.predict(obs_ev, deterministic=True)
        obs_ev, reward, done, info = env_eval.step(action)
        env_eval.render()
        print(f'Reward: {reward}, Done: {done[0]}')
        rewards.append(reward)

    env_eval.close()
    pd.DataFrame({"Time_steps": time_steps, "Reward": rewards}).to_csv("Evaluation_DRL_SD2.csv", index=False)
    print("Evaluation data saved to Evaluation_DRL_SD2.csv")

def objective(trial):
    """
    Optuna objective function for hyperparameter tuning of TD3.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.

    Returns:
        float: Mean reward for the trial.
    """
    num_envs = 8
    n_actions = 7  # Update based on your action space

    # Create training and evaluation environments
    train_env = SubprocVecEnv([make_screwdriving_env(composite_controller_config, False, False) for _ in range(num_envs)])
    eval_env  = SubprocVecEnv([make_screwdriving_env(composite_controller_config, False, False) for _ in range(num_envs)])

    # Normalize observations and rewards
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.)
    eval_env  = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)

    # Action noise for TD3
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # Suggest TD3 hyperparameters
    learning_rate  = trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True)
    buffer_size    = trial.suggest_int("buffer_size", 500000, 3000000, step=500000)
    batch_size     = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])
    tau            = trial.suggest_float("tau", 0.005, 0.015)
    gamma          = trial.suggest_float("gamma", 0.9, 0.99)
    train_freq     = trial.suggest_categorical("train_freq", [1, 2, 4, 8])
    gradient_steps = trial.suggest_categorical("gradient_steps", [1, 4, 8, 16])
    hidden_size_options = [[256, 256], [512, 256], [1024, 512, 256], [400, 300]]
    pi_layers = trial.suggest_categorical("pi_layers", hidden_size_options)
    qf_layers = trial.suggest_categorical("qf_layers", hidden_size_options)

    # Define policy_kwargs with separate pi/qf architectures
    policy_kwargs = dict(
        net_arch=dict(pi=pi_layers, qf=qf_layers),
        activation_fn=torch.nn.ReLU
    )

    # Initialize the TD3 model
    model = TD3(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        action_noise=action_noise,
        policy_kwargs=policy_kwargs,
        verbose=0,
        device="cuda",
        tensorboard_log=None
    )
    stop_cb = StopTrainingOnNoModelImprovement(max_no_improvement_evals=20, min_evals=200, verbose=1)
    explore_cb = DynamicExplorationCallback(thresholds=[450, 1950, 4250], patience=10, verbose=1)
    callback_after_eval = CallbackList([stop_cb, explore_cb])
    eval_cb = EvalCallback(eval_env, eval_freq=20_000, n_eval_episodes=15,
                               deterministic=True, callback_after_eval=callback_after_eval, verbose=1)

    try:
        model.learn(total_timesteps=1_000_000, callback=eval_cb)
        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10)
        print(f'Mean Reward: {mean_reward}')
    except Exception as e:
        return -9999  # Penalize failed trials

    train_env.close()
    eval_env.close()

    return mean_reward

if __name__ == "__main__":
    """
    Main entry point for Optuna tuning, training, and evaluation.
    """
    print("NEW::::::::::::::::::::::::::::::::::::")
    # Optuna hyperparameter tuning
    #study = optuna.create_study(direction="maximize", pruner=MedianPruner(n_warmup_steps=5))
    #study.optimize(objective, n_trials=20)
    #print("Best Hyperparameters:")
    #print(study.best_params)
    #print(f"Best mean reward: {study.best_value:.2f}")
    #df = study.trials_dataframe()
    #df.to_csv("optuna_results.csv", index=False)
    #print("Results saved on optuna_results.csv")
    # Training (uncomment to run)
    #models = DRL_Training(composite_controller_config)
    print("Model trained and saved in ./logs/")
    # Testing (uncomment to run)
    model_path = "logs_OSC/BoltSD_T/TD3/best_model"
    model = TD3.load(model_path)
    DRL_Test(model, composite_controller_config)
    print("Model tested successfully.")

# tensorboard --logdir=./tensorboard --port 6006