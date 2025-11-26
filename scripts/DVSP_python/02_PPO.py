"""
Proximal Policy Optimization (PPO) for DVSP.
Converted from Julia to Python with PyTorch.
"""

import os
import pickle
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

from utils.utils import (
    setup_environments,
    prize_collecting_vsp,
)
from utils.policy import (
    CombinatorialACPolicy,
    critic_GNN,
    PPO_episodes,
    J_PPO,
    grads_prep_GNN,
    huber_GNN,
    evaluate_policy,
    rb_add,
    rb_sample,
    sigmaF_dvsp,
)

# Configuration
NB_TRAIN_INSTANCES = 10
NB_VAL_INSTANCES = 10
NB_TEST_INSTANCES = 10
MAX_REQUESTS_PER_EPOCH = 10
SEED = 0

# PPO hyperparameters
EPISODES = 400
COLLECTION_STEPS = 20
EPOCHS = 100
BATCH_SIZE = 1
NTESTS = 1
CLIP = 0.2
USE_RB = True
SIGMAF_VALUES = [0.5, 0.05]
LR_VALUES = [1e-3, 5e-4]
V_METHOD = "off_policy"
ADV_METHOD = "TD_n"
CRITIC_FACTOR = 10

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

# Define models
class ActorModel(nn.Module):
    """Simple linear actor model."""
    
    def __init__(self, input_dim: int = 14):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


actor_model = ActorModel(input_dim=14)
critic_model = critic_GNN(node_features=15, edge_features=1)

print(f"Created actor model: {actor_model}")
print(f"Created critic model: {critic_model}")

# Define perturbation distribution
def p(theta: torch.Tensor, stdev: float):
    """Multivariate normal distribution for perturbation."""
    from torch.distributions import MultivariateNormal
    cov = torch.eye(len(theta)) * (stdev ** 2)
    return MultivariateNormal(theta, cov)

# Create policy
PPO_policy = CombinatorialACPolicy(
    actor_model=actor_model,
    critic_model=critic_model,
    p=p,
    CO_layer=prize_collecting_vsp,
    seed=SEED,
)


def PPO_training(
    policy: CombinatorialACPolicy,
    train_envs: List,
    val_envs: List,
    episodes: int = 400,
    collection_steps: int = 20,
    epochs: int = 100,
    batch_size: int = 1,
    ntests: int = 1,
    clip: float = 0.2,
    use_rb: bool = True,
    sigmaF_values: List[float] = [0.5, 0.05],
    lr_values: List[float] = [1e-3, 5e-4],
    V_method: str = "off_policy",
    adv_method: str = "TD_n",
    critic_factor: int = 10,
    model_builder=None,
    **kwargs
) -> Tuple[nn.Module, List[float], List[float], List[float]]:
    """
    Train PPO policy.
    
    Args:
        policy: PPO policy
        train_envs: Training environments
        val_envs: Validation environments
        episodes: Number of training episodes
        collection_steps: Steps per collection
        epochs: Number of epochs per episode
        batch_size: Batch size
        ntests: Number of test episodes
        clip: PPO clip parameter
        use_rb: Whether to use replay buffer
        sigmaF_values: [initial, final] sigmaF values
        lr_values: [initial, final] learning rate values
        V_method: Value estimation method
        adv_method: Advantage estimation method
        critic_factor: Critic learning rate factor
        model_builder: Model builder
        **kwargs: Additional arguments
        
    Returns:
        Tuple of (best_model, train_history, val_history, loss_history)
    """
    # Initialize optimizers with gradient clipping
    opt_actor = optim.Adam(policy.actor_model.parameters(), lr=lr_values[0])
    opt_critic = optim.Adam(policy.critic_model.parameters(), lr=lr_values[0] * critic_factor)
    
    train_reward_history = []
    val_reward_history = []
    actor_weights_history = []
    loss_history = []
    
    gamma = 1.0
    best_model = copy.deepcopy(policy.actor_model.state_dict())
    best_performance = -np.inf
    best_episode = 0
    target_critic = copy.deepcopy(policy.critic_model)
    
    # Sigma and learning rate scheduling
    global sigmaF_dvsp
    sigmaF_dvsp = sigmaF_values[0]
    sigmaF_step = (sigmaF_values[0] - sigmaF_values[1]) / episodes
    sigmaF_average = 2.0
    lr_step_a = (lr_values[0] - lr_values[1]) / episodes
    lr_step_c = (lr_values[0] * critic_factor - lr_values[1]) / episodes
    
    # Initialize replay buffer
    if use_rb:
        replay_buffer = []
        rb_capacity = collection_steps * 6 * 100
        rb_position = 0
        rb_size = 0
        iterations = epochs
        epochs = 1
    
    policy.reset_seed()
    
    print(f"\nStarting PPO training for {episodes} episodes...")
    
    for e in tqdm(range(episodes), desc="Training"):
        # Evaluate policy
        policy.actor_model.eval()
        policy.critic_model.eval()
        
        with torch.no_grad():
            train_rew, _ = evaluate_policy(
                policy,
                train_envs,
                nb_episodes=ntests,
                perturb=False,
                model_builder=model_builder,
            )
            val_rew, _ = evaluate_policy(
                policy,
                val_envs,
                nb_episodes=ntests,
                perturb=False,
                model_builder=model_builder,
            )
        
        train_reward_history.append(train_rew)
        val_reward_history.append(val_rew)
        
        # Save actor weights
        actor_weights_history.append(
            policy.actor_model.linear.weight.data.cpu().clone()
        )
        
        # Save best model
        if val_rew >= best_performance:
            best_performance = val_rew
            best_model = copy.deepcopy(policy.actor_model.state_dict())
            best_episode = e
        
        if (e + 1) % 20 == 0:
            print(f"\nEpisode {e + 1}: sigmaF={sigmaF_dvsp:.4f}, "
                  f"lr={opt_actor.param_groups[0]['lr']:.6f}, "
                  f"train={train_rew:.4f}, val={val_rew:.4f}")
        
        # Collect experience
        policy.actor_model.train()
        policy.critic_model.train()
        
        episode_data = PPO_episodes(
            policy,
            target_critic,
            train_envs,
            collection_steps,
            gamma,
            V_method,
            adv_method,
            model_builder=model_builder,
        )
        
        if use_rb:
            replay_buffer, rb_position, rb_size = rb_add(
                replay_buffer, rb_capacity, rb_position, rb_size, episode_data
            )
            batches = [rb_sample(replay_buffer, batch_size) for _ in range(iterations)]
        else:
            # Create batches from episode data
            batches = [
                episode_data[i:i+batch_size]
                for i in range(0, len(episode_data), batch_size)
            ]
        
        # Training loop
        for epoch_idx in range(epochs):
            for batch in batches:
                if len(batch) == 0:
                    continue
                
                # Train critic
                critic_target = [item['Rₜ'] for item in batch]
                graphs, s_c, edge_features = grads_prep_GNN(batch)
                
                opt_critic.zero_grad()
                critic_loss = huber_GNN(
                    policy, graphs, s_c, edge_features, critic_target, 1.0,
                    model_builder=model_builder
                )
                critic_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_value_(policy.critic_model.parameters(), 1e-3)
                opt_critic.step()
                
                # Train actor
                opt_actor.zero_grad()
                actor_loss = -J_PPO(policy, batch, clip, sigmaF_average)
                actor_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_value_(policy.actor_model.parameters(), 1e-3)
                opt_actor.step()
            
            # Update target critic
            target_critic = copy.deepcopy(policy.critic_model)
        
        # Update sigmaF and learning rates
        sigmaF_dvsp = max(sigmaF_dvsp - sigmaF_step, sigmaF_values[1])
        
        lr_a = opt_actor.param_groups[0]['lr']
        lr_c = opt_critic.param_groups[0]['lr']
        opt_actor.param_groups[0]['lr'] = max(lr_a - lr_step_a, lr_values[1])
        opt_critic.param_groups[0]['lr'] = max(lr_c - lr_step_c, lr_values[1])
    
    # Final evaluation with best model
    policy.actor_model.load_state_dict(best_model)
    policy.actor_model.eval()
    
    with torch.no_grad():
        final_train, _ = evaluate_policy(
            policy,
            train_envs,
            nb_episodes=ntests,
            perturb=False,
            model_builder=model_builder,
        )
        final_val, _ = evaluate_policy(
            policy,
            val_envs,
            nb_episodes=ntests,
            perturb=False,
            model_builder=model_builder,
        )
    
    print(f"\nFinal: train={final_train:.4f}, val={final_val:.4f}, "
          f"best_episode={best_episode}")
    
    train_reward_history.append(final_train)
    val_reward_history.append(final_val)
    actor_weights_history.append(
        policy.actor_model.linear.weight.data.cpu().clone()
    )
    
    return policy.actor_model, train_reward_history, val_reward_history, loss_history


# Train PPO
try:
    PPO_model, PPO_train, PPO_val, PPO_losses = PPO_training(
        PPO_policy,
        train_envs,
        val_envs,
        episodes=EPISODES,
        collection_steps=COLLECTION_STEPS,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        ntests=NTESTS,
        clip=CLIP,
        use_rb=USE_RB,
        sigmaF_values=SIGMAF_VALUES,
        lr_values=LR_VALUES,
        V_method=V_METHOD,
        adv_method=ADV_METHOD,
        critic_factor=CRITIC_FACTOR,
        model_builder=model_builder,
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
    PPO_policy_eval = KleopatraVSPPolicy(PPO_model)
    
    PPO_final_train, PPO_final_train_rew = evaluate_policy(
        PPO_policy_eval,
        train_envs,
        nb_episodes=10,
        model_builder=model_builder,
        return_scores=True,
    )
    PPO_final_train = -PPO_final_train
    PPO_final_train_rew = [-r for r in PPO_final_train_rew]
    
    PPO_final_test, PPO_final_test_rew = evaluate_policy(
        PPO_policy_eval,
        test_envs,
        nb_episodes=10,
        model_builder=model_builder,
        return_scores=True,
    )
    PPO_final_test = -PPO_final_test
    PPO_final_test_rew = [-r for r in PPO_final_test_rew]
    
    print(f"PPO final train mean: {PPO_final_train:.2f}")
    print(f"PPO final test mean: {PPO_final_test:.2f}")
except Exception as e:
    print(f"Error during final evaluation: {e}")
    PPO_final_train_rew = []
    PPO_final_test_rew = []

# Plot training curves
print("\nPlotting training curves...")
plt.figure(figsize=(10, 6))
plt.plot(PPO_train, marker='o', label='train history', linewidth=2)
plt.plot(PPO_val, marker='o', label='val history', linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('DVSP PPO Training')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOTDIR / "dvsp_PPO_rew_line.pdf")
plt.savefig(PLOTDIR / "dvsp_PPO_rew_line.png", dpi=300)
print(f"Saved plot to {PLOTDIR / 'dvsp_PPO_rew_line.pdf'}")

# Save results
print("\nSaving results...")
results = {
    'model_state_dict': PPO_model.state_dict(),
    'train_rew': PPO_train,
    'val_rew': PPO_val,
    'train_final': PPO_final_train_rew,
    'test_final': PPO_final_test_rew,
}

with open(LOGDIR / "dvsp_PPO_training_results.pkl", 'wb') as f:
    pickle.dump(results, f)

torch.save(PPO_model.state_dict(), LOGDIR / "dvsp_PPO_model.pt")

print(f"Results saved to {LOGDIR / 'dvsp_PPO_training_results.pkl'}")
print(f"Model saved to {LOGDIR / 'dvsp_PPO_model.pt'}")
print("\nPPO training complete!")
