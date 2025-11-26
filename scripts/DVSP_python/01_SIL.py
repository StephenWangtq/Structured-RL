"""
Supervised Imitation Learning (SIL) for DVSP.
Converted from Julia to Python with PyTorch.
"""

import os
import pickle
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.utils import (
    setup_environments,
    load_VSP_dataset,
    prize_collecting_vsp,
    VSPSolution,
    nb_locations,
    evaluate_policy,
)
from utils.policy import CombinatorialACPolicy, evaluate_policy as policy_evaluate

# Configuration
NB_TRAIN_INSTANCES = 10
NB_VAL_INSTANCES = 10
NB_TEST_INSTANCES = 10
MAX_REQUESTS_PER_EPOCH = 10
SEED = 0
NB_EPOCHS = 400
LEARNING_RATE = 1e-3

# Setup directories
LOGDIR = Path("logs")
LOGDIR.mkdir(exist_ok=True)
PLOTDIR = Path("plots")
PLOTDIR.mkdir(exist_ok=True)

# Dataset paths
DATASET_PATH = Path("data/euro-neurips-2022")
TRAIN_INSTANCES = DATASET_PATH / "train"
VAL_INSTANCES = DATASET_PATH / "validation"
TEST_INSTANCES = DATASET_PATH / "test"

# Set random seeds
torch.manual_seed(SEED)
np.random.seed(SEED)

# Model builder
try:
    from utils.utils import grb_model
    model_builder = grb_model
except ImportError:
    model_builder = None

# Setup environments
print("Setting up environments...")
try:
    train_envs, val_envs, test_envs = setup_environments(
        str(TRAIN_INSTANCES),
        str(VAL_INSTANCES),
        str(TEST_INSTANCES),
        nb_train=NB_TRAIN_INSTANCES,
        nb_val=NB_VAL_INSTANCES,
        nb_test=NB_TEST_INSTANCES,
        max_requests_per_epoch=MAX_REQUESTS_PER_EPOCH,
        seed=SEED,
    )
except Exception as e:
    print(f"Error setting up environments: {e}")
    exit(1)

# Load dataset
print("Loading dataset...")
try:
    X, Y = load_VSP_dataset(str(TRAIN_INSTANCES), model_builder=model_builder)
    # Take first 8 * nb_train_instances samples
    X = X[:(8 * NB_TRAIN_INSTANCES)]
    Y = Y[:(8 * NB_TRAIN_INSTANCES)]
    print(f"Loaded {len(X)} training samples")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Dataset loading requires implementation of load_VSP_dataset")
    exit(1)

# Define actor model
# Actor: Dense(14 => 1, bias=false) followed by vec
# In PyTorch: Linear(14, 1, bias=False) with output flattening
class ActorModel(nn.Module):
    """Simple linear actor model."""
    
    def __init__(self, input_dim: int = 14):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.linear(x).squeeze(-1)


SIL_model = ActorModel(input_dim=14)
print(f"Created actor model: {SIL_model}")


# Define Fenchel-Young loss functions
def optimization_fyl(theta: torch.Tensor, instance, **kwargs):
    """Optimization function for Fenchel-Young loss."""
    routes = prize_collecting_vsp(theta, instance=instance, model_builder=model_builder)
    return VSPSolution(routes, max_index=nb_locations(instance.instance)).edge_matrix


def g_fyl(y: np.ndarray, instance, **kwargs):
    """Helper function g for Fenchel-Young loss."""
    return np.sum(y[:, instance.is_postponable], axis=0).flatten()


def h_fyl(y: np.ndarray, instance=None, duration=None, **kwargs):
    """Helper function h for Fenchel-Young loss."""
    if duration is None:
        duration = instance.instance.duration
    
    value = 0.0
    N = duration.shape[0]
    for i in range(N):
        for j in range(N):
            value -= y[i, j] * duration[i, j]
    return value


# Note: Fenchel-Young loss requires InferOpt or similar library
# This is a placeholder for the actual implementation
class FenchelYoungLoss:
    """
    Fenchel-Young loss with perturbed optimization.
    
    This is a placeholder. In practice, you would need to:
    1. Implement perturbed optimization (add noise and average gradients)
    2. Use a differentiable optimization library
    3. Or implement the Fenchel-Young loss computation manually
    """
    
    def __init__(self, epsilon: float = 1e-2, nb_samples: int = 20):
        self.epsilon = epsilon
        self.nb_samples = nb_samples
    
    def __call__(
        self,
        theta: torch.Tensor,
        y_true: np.ndarray,
        instance,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute Fenchel-Young loss.
        
        For now, this is a simplified surrogate loss.
        In practice, you should implement proper perturbed optimization.
        """
        # Placeholder: MSE between predicted solution and true solution
        # This is NOT the actual Fenchel-Young loss
        y_pred = optimization_fyl(theta.detach(), instance, **kwargs)
        loss = torch.tensor(
            np.mean((y_pred - y_true) ** 2),
            dtype=torch.float32,
            requires_grad=True
        )
        return loss


# Training function
def SIL_training(
    model: nn.Module,
    X: List,
    Y: List,
    train_envs: List,
    val_envs: List,
    nb_epochs: int = 400,
    lr: float = 1e-3,
) -> Tuple[nn.Module, List[float], List[float], List[float]]:
    """
    Train SIL model.
    
    Args:
        model: Actor model
        X: Training inputs
        Y: Training outputs
        train_envs: Training environments
        val_envs: Validation environments
        nb_epochs: Number of epochs
        lr: Learning rate
        
    Returns:
        Tuple of (best_model, train_history, val_history, losses)
    """
    # Setup
    fyl = FenchelYoungLoss(epsilon=1e-2, nb_samples=20)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_reward_history = []
    val_reward_history = []
    losses = []
    
    best_model = None
    best_performance = -np.inf
    best_epoch = 0
    
    print(f"\nStarting SIL training for {nb_epochs} epochs...")
    
    for epoch in tqdm(range(nb_epochs), desc="Training"):
        # Test model
        policy = CombinatorialACPolicy(
            actor_model=model,
            critic_model=None,
            p=None,
            CO_layer=prize_collecting_vsp,
            seed=SEED,
        )
        
        model.eval()
        with torch.no_grad():
            train_rew, _ = policy_evaluate(
                policy,
                train_envs,
                nb_episodes=1,
                perturb=False,
                model_builder=model_builder,
            )
            val_rew, _ = policy_evaluate(
                policy,
                val_envs,
                nb_episodes=1,
                perturb=False,
                model_builder=model_builder,
            )
        
        train_reward_history.append(train_rew)
        val_reward_history.append(val_rew)
        
        if val_rew >= best_performance:
            best_performance = val_rew
            best_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
        
        if (epoch + 1) % 20 == 0:
            print(f"\nEpoch {epoch + 1}: train={train_rew:.4f}, val={val_rew:.4f}, "
                  f"best_epoch={best_epoch}")
        
        # Train model
        model.train()
        epoch_loss = 0.0
        
        for (x, instance), y_true in zip(X, Y):
            optimizer.zero_grad()
            
            # Forward pass
            theta = model(x)
            
            # Compute loss
            loss = fyl(theta, y_true, instance=instance)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        losses.append(epoch_loss / len(X))
    
    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)
    
    # Final evaluation
    model.eval()
    policy = CombinatorialACPolicy(
        actor_model=model,
        critic_model=None,
        p=None,
        CO_layer=prize_collecting_vsp,
        seed=SEED,
    )
    
    with torch.no_grad():
        final_train, _ = policy_evaluate(
            policy,
            train_envs,
            nb_episodes=1,
            perturb=False,
            model_builder=model_builder,
        )
        final_val, _ = policy_evaluate(
            policy,
            val_envs,
            nb_episodes=1,
            perturb=False,
            model_builder=model_builder,
        )
    
    train_reward_history.append(final_train)
    val_reward_history.append(final_val)
    
    print(f"\nFinal: train={final_train:.4f}, val={final_val:.4f}, "
          f"best_epoch={best_epoch}")
    
    return model, train_reward_history, val_reward_history, losses


# Train model
print("\nNote: This script uses placeholder Fenchel-Young loss.")
print("For full functionality, implement proper perturbed optimization.")

try:
    SIL_model, SIL_train, SIL_val, SIL_losses = SIL_training(
        SIL_model,
        X,
        Y,
        train_envs,
        val_envs,
        nb_epochs=NB_EPOCHS,
        lr=LEARNING_RATE,
    )
except Exception as e:
    print(f"Error during training: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test the trained model
print("\nTesting trained model...")
from utils.utils import KleopatraVSPPolicy

try:
    SIL_policy = KleopatraVSPPolicy(SIL_model)
    
    SIL_final_train_mean, SIL_final_train_rew = evaluate_policy(
        SIL_policy,
        train_envs,
        nb_episodes=10,
        model_builder=model_builder,
        return_scores=True,
    )
    SIL_final_train_mean = -SIL_final_train_mean
    SIL_final_train_rew = [-r for r in SIL_final_train_rew]
    
    SIL_final_test_mean, SIL_final_test_rew = evaluate_policy(
        SIL_policy,
        test_envs,
        nb_episodes=10,
        model_builder=model_builder,
        return_scores=True,
    )
    SIL_final_test_mean = -SIL_final_test_mean
    SIL_final_test_rew = [-r for r in SIL_final_test_rew]
    
    print(f"SIL final train mean: {SIL_final_train_mean:.2f}")
    print(f"SIL final test mean: {SIL_final_test_mean:.2f}")
except Exception as e:
    print(f"Error during final evaluation: {e}")
    SIL_final_train_rew = []
    SIL_final_test_rew = []

# Plot training curves
print("\nPlotting training curves...")
plt.figure(figsize=(10, 6))
plt.plot(SIL_train, marker='o', label='train history', linewidth=2)
plt.plot(SIL_val, marker='o', label='val history', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Reward')
plt.title('DVSP SIL Training')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOTDIR / "dvsp_SIL_rew_line.pdf")
plt.savefig(PLOTDIR / "dvsp_SIL_rew_line.png", dpi=300)
print(f"Saved plot to {PLOTDIR / 'dvsp_SIL_rew_line.pdf'}")

# Save results
print("\nSaving results...")
results = {
    'model_state_dict': SIL_model.state_dict(),
    'train_rew': SIL_train,
    'val_rew': SIL_val,
    'train_final': SIL_final_train_rew,
    'test_final': SIL_final_test_rew,
}

# Save with pickle
with open(LOGDIR / "dvsp_SIL_training_results.pkl", 'wb') as f:
    pickle.dump(results, f)

# Save model with PyTorch
torch.save(SIL_model.state_dict(), LOGDIR / "dvsp_SIL_model.pt")

print(f"Results saved to {LOGDIR / 'dvsp_SIL_training_results.pkl'}")
print(f"Model saved to {LOGDIR / 'dvsp_SIL_model.pt'}")
print("\nSIL training complete!")
