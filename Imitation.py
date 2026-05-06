"""
Imitation.py

Behavior Cloning for Bolt Screwdriving using Robosuite demonstrations.
Includes Optuna hyperparameter tuning, model training, evaluation, and saving/loading utilities.

Author: [Your Name]
Date: [Today's Date]
"""

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import robosuite
from robosuite.environments.manipulation.bolt_disassembly2 import BoltDisassembly
from robosuite.environments.manipulation.bolt_screwdriving_dissassembly import BoltScrewdrivingDisassembly
from robosuite.controllers import load_composite_controller_config
import collections
import optuna
import torch.nn.functional as F

# ----------------------------
# CONFIGURATION
# ----------------------------
HDF5_PATH = "demos/demo_SD_2/demo.hdf5"  # Path to demonstration data
NUM_EPOCHS = 1000                         # Number of training epochs
EARLY_STOPPING_PATIENCE = 30             # Early stopping patience
LEARNING_RATE = 0.000358                  # Learning rate for optimizer
EVAL_EPISODES = 5                        # Number of evaluation episodes
SCHEDULER = 'none'                        # Learning rate scheduler type
DROPOUT_PROB = 0                         # Dropout probability
HIDDEN_DIM1 = 256                       # First hidden layer size
HIDDEN_DIM2 = 1024                        # Second hidden layer size
BATCH_SIZE = 128                          # Batch size for training
WEIGHT_DECAY = 0                         # Weight decay for optimizer
ACTIVATION = 'relu'                 # Activation function
OPTIMIZER = "rmsprop"                    # Optimizer type

# --- Utility: Activation Function Selection ---
def get_activation(name):
    """
    Returns the activation function module for a given name.

    Args:
        name (str): Name of the activation function.

    Returns:
        nn.Module: Activation function module.
    """
    if name == "relu":
        return nn.ReLU()
    elif name == "leakyrelu":
        return nn.LeakyReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "elu":
        return nn.ELU()
    elif name == "selu":
        return nn.SELU()
    elif name == "softplus":
        return nn.Softplus()
    else:
        raise ValueError(f"Unknown activation: {name}")

# --- Optuna Objective Function for Hyperparameter Tuning ---
def objective(trial):
    """
    Optuna objective function for hyperparameter search.
    Trains a behavior cloning model and returns validation loss.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.

    Returns:
        float: Best validation loss achieved during training.
    """
    print(f"\n=== Starting trial {trial.number + 1} ===")
    # Hyperparameters to optimize
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    weight_decay = 0
    dropout_prob = 0
    scheduler_name = trial.suggest_categorical("scheduler", [
        "cosine", "plateau", "none", "step", "exponential"
    ])
    hidden_dim1 = trial.suggest_categorical("hidden_dim1", [64, 128, 256, 512, 1024])
    hidden_dim2 = trial.suggest_categorical("hidden_dim2", [64, 128, 256, 512, 1024])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])
    activation_name = trial.suggest_categorical("activation", [
        "relu", "leakyrelu", "gelu", "tanh", "sigmoid", "elu", "selu", "softplus"
    ])
    optimizer_name = trial.suggest_categorical("optimizer", [
        "adam", "rmsprop", "sgd", "adamw", "adagrad", "adadelta"
    ])
    extra_layers = trial.suggest_int("extra_layers", 0, 4)  # Add up to 4 extra layers

    # Load dataset
    loader = DatasetLoader(HDF5_PATH, batch_size)
    obs, acts, train_loader, val_loader, obs_mean, obs_std, acts_mean, acts_std = loader.load()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model definition with trial parameters
    act = get_activation(activation_name)
    model = BehaviorCloningModel(
        obs_dim=obs.shape[1],
        act_dim=acts.shape[1],
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        dropout_prob=dropout_prob,
        activation=act,
        extra_layers=extra_layers
    ).to(device)

    # Optimizer selection
    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    loss_fn = nn.MSELoss()
    # Scheduler selection
    if scheduler_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
        scheduler_type = "cosine"
    elif scheduler_name == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        scheduler_type = "plateau"
    elif scheduler_name == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        scheduler_type = "step"
    elif scheduler_name == "exponential":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        scheduler_type = "exponential"
    else:
        scheduler = None
        scheduler_type = "none"

    # Training loop with early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        # Scheduler step
        if scheduler_type == "cosine":
            scheduler.step()
        elif scheduler_type == "plateau":
            scheduler.step(val_loss)
        # Optuna reporting and pruning
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                break
    return best_val_loss

class DatasetLoader:
    """
    Loads demonstration data from HDF5 files and prepares PyTorch DataLoaders.
    """
    def __init__(self, hdf5_path, batch_size):
        """
        Initializes the DatasetLoader.

        Args:
            hdf5_path (str): Path to HDF5 demonstration file.
            batch_size (int): Batch size for DataLoader.
        """
        self.hdf5_path = hdf5_path
        self.batch_size = batch_size

    def load(self):
        """
        Loads and normalizes demonstration data, splits into train/val sets.

        Returns:
            tuple: (obs, acts, train_loader, val_loader, obs_mean, obs_std, acts_mean, acts_std)
        """
        obs, acts = [], []
        with h5py.File(self.hdf5_path, "r") as f:
            print("Metadata:")
            for attr in f["data"].attrs:
                print(f"  {attr}: {f['data'].attrs[attr]}")
            for demo_key in f["data"].keys():
                demo = f["data"][demo_key]
                obs.append(demo["observations"][:])
                acts.append(demo["actions"][:])
        obs = np.concatenate(obs, axis=0)
        acts = np.concatenate(acts, axis=0)
        print("Total Observations:", obs.shape, "Total Actions:", acts.shape)
        # Normalize observations and actions
        obs_mean = obs.mean(axis=0)
        obs_std = obs.std(axis=0) + 1e-8
        acts_mean = acts.mean(axis=0)
        acts_std = acts.std(axis=0) + 1e-8

        obs_norm = (obs - obs_mean) / obs_std
        acts_norm = (acts - acts_mean) / acts_std

        # Prepare data loaders
        obs_tensor = torch.tensor(obs_norm, dtype=torch.float32)
        acts_tensor = torch.tensor(acts_norm, dtype=torch.float32)
        dataset = TensorDataset(obs_tensor, acts_tensor)

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=True)
        return obs, acts, train_loader, val_loader, obs_mean, obs_std, acts_mean, acts_std

class BehaviorCloningModel(nn.Module):
    """
    Feedforward neural network for behavior cloning.
    Supports deeper architectures.
    """
    def __init__(self, obs_dim, act_dim, hidden_dim1, hidden_dim2, dropout_prob, activation, extra_layers=2):
        """
        Initializes the behavior cloning model.

        Args:
            obs_dim (int): Dimension of observation vector.
            act_dim (int): Dimension of action vector.
            hidden_dim1 (int): Size of first hidden layer.
            hidden_dim2 (int): Size of second hidden layer.
            dropout_prob (float): Dropout probability.
            activation (nn.Module): Activation function.
            extra_layers (int): Number of extra hidden layers (for deeper networks).
        """
        super().__init__()
        layers = []
        # Input layer
        layers.append(nn.Linear(obs_dim, hidden_dim1))
        layers.append(nn.LayerNorm(hidden_dim1))
        layers.append(activation)
        layers.append(nn.Dropout(dropout_prob))

        # Second hidden layer
        layers.append(nn.Linear(hidden_dim1, hidden_dim2))
        layers.append(nn.LayerNorm(hidden_dim2))
        layers.append(activation)
        layers.append(nn.Dropout(dropout_prob))

        # Output layer
        layers.append(nn.Linear(hidden_dim2, act_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.net(x)

class Trainer:
    """
    Handles training and saving the behavior cloning model.
    """
    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader, device):
        """
        Initializes the Trainer.

        Args:
            model (nn.Module): Behavior cloning model.
            optimizer (torch.optim.Optimizer): Optimizer.
            loss_fn (nn.Module): Loss function.
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
            device (torch.device): Device to run training on.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def train(self, num_epochs=NUM_EPOCHS, early_stopping_patience=EARLY_STOPPING_PATIENCE, scheduler_name=SCHEDULER):
        """
        Trains the model and saves training/validation loss to CSV.

        Args:
            num_epochs (int): Number of epochs.
            early_stopping_patience (int): Early stopping patience.
            scheduler_name (str): Scheduler type.
        """
        train_losses = []
        val_losses = []
        epochs = []
        # Scheduler selection
        if scheduler_name == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=NUM_EPOCHS)
            scheduler_type = "cosine"
        elif scheduler_name == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
            scheduler_type = "plateau"
        elif scheduler_name == "step":
            scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
            scheduler_type = "step"
        elif scheduler_name == "exponential":
            scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
            scheduler_type = "exponential"
        else:
            scheduler = None
            scheduler_type = "none"

        # Training loop
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0
            for xb, yb in self.train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                pred = self.model(xb)
                loss = self.loss_fn(pred, yb)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= len(self.train_loader)

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for xb, yb in self.val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    pred = self.model(xb)
                    loss = self.loss_fn(pred, yb)
                    val_loss += loss.item()
            val_loss /= len(self.val_loader)
            print(f"[{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            epochs.append(epoch)
            # Scheduler step
            if scheduler_type == "cosine":
                scheduler.step()
            elif scheduler_type == "plateau":
                scheduler.step(val_loss)

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), "bc_model_best_SD.pth")
                print("Best model saved to bc_model_best_SD.pth")
            else:
                self.patience_counter += 1
                if self.patience_counter >= early_stopping_patience:
                    print("Early stopping triggered.")
                    break
        # Save training process to CSV
        pd.DataFrame({"Epochs": epochs, "train_loss": train_losses, "val_loss": val_losses}).to_csv("training_IL_SD.csv", index=False)
        # Save final model
        torch.save(self.model.state_dict(), "bc_model_SD.pth")
        print("Final model saved to bc_model_SD.pth")

class Evaluator:
    """
    Evaluates the trained model in the BoltScrewdrivingDisassembly environment, running a single rollout.
    """
    def __init__(self, model, device, obs_mean, obs_std, acts_mean, acts_std):
        """
        Initializes the Evaluator.

        Args:
            model (nn.Module): Trained behavior cloning model.
            device (torch.device): Device to run evaluation on.
            obs_mean (np.ndarray): Mean of observations.
            obs_std (np.ndarray): Std of observations.
            acts_mean (np.ndarray): Mean of actions.
            acts_std (np.ndarray): Std of actions.
        """
        self.model = model
        self.device = device
        self.obs_mean = obs_mean
        self.obs_std = obs_std
        self.acts_mean = acts_mean
        self.acts_std = acts_std

    def evaluate(self, render=True):
        """
        Runs a rollout in the environment using the trained model and saves rewards to CSV.

        Args:
            render (bool): Whether to render the environment.
        """

        # Setup environment for evaluation
        composite_controller_config = load_composite_controller_config(controller=None, robot="Panda")
        composite_controller_config["body_parts"]["right"].update({
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
        })

        env = BoltScrewdrivingDisassembly(
            robots="Panda",
            gripper_types="default",
            controller_configs=composite_controller_config,
            horizon=2000,
            env_configuration="opposed",
            has_renderer=render,
            use_camera_obs=False,
            use_object_obs=True,
            reward_shaping=True,
            control_freq=20,
            nut_type="m5",
            initial_grasp=False
        )

        obs = env.reset()
        done = False
        time_steps = []
        time_step = 0
        rewards = []
        while not done:
            # Flatten observation dict if needed
            time_step += 1
            time_steps.append(time_step)
            if isinstance(obs, dict):
                flat_obs = np.concatenate([np.array(v).flatten() for v in obs.values()])
            else:
                flat_obs = np.array(obs).flatten()
            # Normalize observation
            flat_obs = (flat_obs - self.obs_mean) / self.obs_std
            obs_tensor = torch.tensor(flat_obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action_norm = self.model(obs_tensor).cpu().numpy().flatten()
            # Denormalize action
            action = action_norm * self.acts_std + self.acts_mean
            obs, reward, done, _ = env.step(action)
            env.render()
            print(f"Reward: {reward}, Done: {done}")
            rewards.append(reward)
        # Save evaluation rewards to CSV
        pd.DataFrame({"Time_steps": time_steps, "Reward": rewards}).to_csv("Evaluation_IL_SD.csv", index=False)
        env.close()
        print("Evaluation completed. Environment closed.")

def main(continue_training=True, model_path="bc_model_best.pth"):
    """
    Main entry point for training and evaluation.
    Loads dataset, builds model, trains or loads weights, and evaluates.

    Args:
        continue_training (bool): If True, loads previous weights to continue training.
        model_path (str): Path to saved model weights.
    """
    # Select device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    loader = DatasetLoader(HDF5_PATH, BATCH_SIZE)
    obs, acts, train_loader, val_loader, obs_mean, obs_std, acts_mean, acts_std = loader.load()
    # Define model and move to device
    act = get_activation(ACTIVATION)
    # Example: deeper network with 2 extra layers and residual connections
    model = BehaviorCloningModel(
        obs_dim=obs.shape[1],
        act_dim=acts.shape[1],
        hidden_dim1=HIDDEN_DIM1,
        hidden_dim2=HIDDEN_DIM2,
        dropout_prob=DROPOUT_PROB,
        activation=act,
    ).to(device)
    # Optionally load previous weights
    if continue_training:
        print(f"Loading model weights from {model_path} to continue training...")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    # Optimizer selection
    if OPTIMIZER == "adam":
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER == "adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    else:
        raise ValueError(f"Unknown optimizer: {OPTIMIZER}")
    
    #loss_fn = nn.SmoothL1Loss()
    loss_fn = nn.MSELoss()

    # Train model (ensure batches are on device)
    #trainer = Trainer(model, optimizer, loss_fn, train_loader, val_loader, device)
    #trainer.train(num_epochs=NUM_EPOCHS, early_stopping_patience=EARLY_STOPPING_PATIENCE)

    # Load best model weights and set to eval mode on device
    model.load_state_dict(torch.load("bc_model_best_SD.pth", map_location=device, weights_only=True))
    model.eval()
    # Evaluate model
    evaluator = Evaluator(model, device, obs_mean, obs_std, acts_mean, acts_std)
    evaluator.evaluate()
    print("Model evaluation completed.")

if __name__ == "__main__":
    # Uncomment below to run Optuna hyperparameter search
    #study = optuna.create_study(direction="minimize")
    #study.optimize(objective, n_trials=20)
    #print("Best trial:")
    #print(study.best_trial)
    #print("Best hyperparameters:", study.best_trial.params)
    #df = study.trials_dataframe()
    #df.to_csv("optuna_trials_Imitation.csv", index=False)  # Save Optuna results to CSV
    main(continue_training=False, model_path="bc_model_best_SD.pth")
    print("DONE")