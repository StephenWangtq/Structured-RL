"""
Structured Reinforcement Learning (SRL) for DVSP.
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
import copy

from utils.utils import (
    setup_environments,
    prize_collecting_vsp,
)
from utils.policy import (
    CombinatorialACPolicy,
    critic_GNN,
    SRL_episodes,
    SRL_actions,
    J_SRL,
    grads_prep_GNN,
    huber_GNN,
    V_value_GNN,
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

# SRL hyperparameters
GRAD_STEPS = 400
COLLECTION_STEPS = 20
ITERATIONS = 100
BATCH_SIZE = 4
NTESTS = 1
SIGMAF = 0.1
SIGMAB_VALUES = [1.0, 0.1]
LR_VALUES = [1e-3, 2e-4]
CRITIC_FACTOR = 2
TEMP_VALUES = [10.0, 10.0]

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
SRL_policy = CombinatorialACPolicy(
    actor_model=actor_model,
    critic_model=critic_model,
    p=p,
    CO_layer=prize_collecting_vsp,
    seed=SEED,
)


def SRL_training(
    policy: CombinatorialACPolicy,
    train_envs: List,
    val_envs: List,
    grad_steps: int = 400,
    collection_steps: int = 20,
    iterations: int = 100,
    batch_size: int = 4,
    ntests: int = 1,
    sigmaF: float = 0.1,
    sigmaB_values: List[float] = [1.0, 0.1],
    lr_values: List[float] = [1e-3, 2e-4],
    critic_factor: int = 2,
    temp_values: List[float] = [10.0, 10.0],
    model_builder=None,
    **kwargs
) -> Tuple[nn.Module, List[float], List[float], List[float]]:
    """
    Train SRL policy.
    
    Args:
        policy: SRL policy
        train_envs: Training environments
        val_envs: Validation environments
        grad_steps: Number of gradient steps
        collection_steps: Steps per collection
        iterations: Number of iterations per gradient step
        batch_size: Batch size
        ntests: Number of test episodes
        sigmaF: Forward perturbation standard deviation
        sigmaB_values: [initial, final] backward perturbation values
        lr_values: [initial, final] learning rate values
        critic_factor: Critic learning rate factor
        temp_values: [initial, final] temperature values
        model_builder: Model builder
        **kwargs: Additional arguments
        
    Returns:
        Tuple of (best_model, train_history, val_history, loss_history)
    """
    # Initialize optimizers
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
    
    # Scheduling
    global sigmaF_dvsp
    sigmaF_dvsp = sigmaF
    sigmaB = sigmaB_values[0]
    sigmaB_step = (sigmaB_values[0] - sigmaB_values[1]) / grad_steps
    lr_step_a = (lr_values[0] - lr_values[1]) / grad_steps
    lr_step_c = (lr_values[0] * critic_factor - lr_values[1]) / grad_steps
    temp = temp_values[0]
    temp_step = (temp_values[0] - temp_values[1]) / grad_steps
    
    # Initialize replay buffer (mandatory for SRL)
    replay_buffer = []
    rb_capacity = collection_steps * 6 * 1000
    rb_position = 0
    rb_size = 0
    
    policy.reset_seed()
    
    print(f"\nStarting SRL training for {grad_steps} gradient steps...")
    
    for e in tqdm(range(grad_steps), desc="Training"):
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
            print(f"\nStep {e + 1}: sigmaB={sigmaB:.4f}, "
                  f"lr={opt_actor.param_groups[0]['lr']:.6f}, "
                  f"train={train_rew:.4f}, val={val_rew:.4f}")
        
        # Collect experience
        policy.actor_model.train()
        policy.critic_model.train()
        
        episode_data = SRL_episodes(
            policy,
            train_envs,
            collection_steps,
            model_builder=model_builder,
        )
        
        replay_buffer, rb_position, rb_size = rb_add(
            replay_buffer, rb_capacity, rb_position, rb_size, episode_data
        )
        
        batches = [rb_sample(replay_buffer, batch_size) for _ in range(iterations)]
        
        # Training loop
        for batch in batches:
            if len(batch) == 0:
                continue
            
            # Train critic
            rewards = [item['reward'] for item in batch]
            next_states = [item['next_state'] for item in batch]
            next_embeddings = [item['next_s'] for item in batch]
            next_embeds_c = [item['next_s_c'] for item in batch]
            
            critic_target = [
                rewards[j] + gamma * V_value_GNN(
                    policy,
                    next_embeddings[j],
                    next_embeds_c[j],
                    target_critic,
                    "off_policy",
                    instance=next_states[j],
                    model_builder=model_builder,
                )
                for j in range(len(batch))
            ]
            
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
            
            # Train actor with SRL
            opt_actions = SRL_actions(
                policy, batch,
                sigmaB=sigmaB,
                no_samples=40,
                temp=temp,
                model_builder=model_builder,
            )
            
            opt_actor.zero_grad()
            actor_loss = J_SRL(policy, batch, opt_actions, model_builder=model_builder)
            actor_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_value_(policy.actor_model.parameters(), 1e-3)
            opt_actor.step()
        
        # Update target critic
        target_critic = copy.deepcopy(policy.critic_model)
        
        # Update parameters
        sigmaB = max(sigmaB - sigmaB_step, sigmaB_values[1])
        lr_a = opt_actor.param_groups[0]['lr']
        lr_c = opt_critic.param_groups[0]['lr']
        opt_actor.param_groups[0]['lr'] = max(lr_a - lr_step_a, lr_values[1])
        opt_critic.param_groups[0]['lr'] = max(lr_c - lr_step_c, lr_values[1])
        temp = max(temp - temp_step, temp_values[1])
    
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


# Train SRL
print("\nNote: SRL training uses critic-guided action selection with Fenchel-Young loss.")
print("J_SRL requires proper implementation of Fenchel-Young loss for full functionality.")

try:
    SRL_model, SRL_train, SRL_val, SRL_losses = SRL_training(
        SRL_policy,
        train_envs,
        val_envs,
        grad_steps=GRAD_STEPS,
        collection_steps=COLLECTION_STEPS,
        iterations=ITERATIONS,
        batch_size=BATCH_SIZE,
        ntests=NTESTS,
        sigmaF=SIGMAF,
        sigmaB_values=SIGMAB_VALUES,
        lr_values=LR_VALUES,
        critic_factor=CRITIC_FACTOR,
        temp_values=TEMP_VALUES,
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
    SRL_policy_eval = KleopatraVSPPolicy(SRL_model)
    
    SRL_final_train, SRL_final_train_rew = evaluate_policy(
        SRL_policy_eval,
        train_envs,
        nb_episodes=10,
        model_builder=model_builder,
        return_scores=True,
    )
    SRL_final_train = -SRL_final_train
    SRL_final_train_rew = [-r for r in SRL_final_train_rew]
    
    SRL_final_test, SRL_final_test_rew = evaluate_policy(
        SRL_policy_eval,
        test_envs,
        nb_episodes=10,
        model_builder=model_builder,
        return_scores=True,
    )
    SRL_final_test = -SRL_final_test
    SRL_final_test_rew = [-r for r in SRL_final_test_rew]
    
    print(f"SRL final train mean: {SRL_final_train:.2f}")
    print(f"SRL final test mean: {SRL_final_test:.2f}")
except Exception as e:
    print(f"Error during final evaluation: {e}")
    SRL_final_train_rew = []
    SRL_final_test_rew = []

# Plot training curves
print("\nPlotting training curves...")
plt.figure(figsize=(10, 6))
plt.plot(SRL_train, marker='o', label='train history', linewidth=2)
plt.plot(SRL_val, marker='o', label='val history', linewidth=2)
plt.xlabel('Gradient Step')
plt.ylabel('Reward')
plt.title('DVSP SRL Training')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOTDIR / "dvsp_SRL_rew_line.pdf")
plt.savefig(PLOTDIR / "dvsp_SRL_rew_line.png", dpi=300)
print(f"Saved plot to {PLOTDIR / 'dvsp_SRL_rew_line.pdf'}")

# Save results
print("\nSaving results...")
results = {
    'model_state_dict': SRL_model.state_dict(),
    'train_rew': SRL_train,
    'val_rew': SRL_val,
    'train_final': SRL_final_train_rew,
    'test_final': SRL_final_test_rew,
}

with open(LOGDIR / "dvsp_SRL_training_results.pkl", 'wb') as f:
    pickle.dump(results, f)

torch.save(SRL_model.state_dict(), LOGDIR / "dvsp_SRL_model.pt")

print(f"Results saved to {LOGDIR / 'dvsp_SRL_training_results.pkl'}")
print(f"Model saved to {LOGDIR / 'dvsp_SRL_model.pt'}")
print("\nSRL training complete!")
