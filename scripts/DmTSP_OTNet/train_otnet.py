"""
Training script for OT-Net on D-mTSP.

Implements structured reinforcement learning with:
- GNN encoder for state representation
- Optimal transport layer for assignment
- Policy gradient with entropy regularization
"""

import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from typing import List, Dict, Any

from utils.environment import DmTSPInstance, DmTSPEnv
from utils.otnet import OTNetPolicy
from utils.utils import (
    extract_agent_features,
    extract_task_features,
    build_graph_edges,
    evaluate_policy,
    generate_episode,
    GreedyPolicy,
    RandomPolicy,
)


# Configuration
SEED = 42
NUM_TRAIN_INSTANCES = 10
NUM_VAL_INSTANCES = 5
NUM_TEST_INSTANCES = 5

# Environment parameters
NUM_AGENTS = 5
TIME_HORIZON = 100.0
REQUEST_RATE = 0.5  # requests per time unit
ALPHA = 0.5  # makespan weight
BETA = 0.5   # waiting time weight

# Model parameters
AGENT_FEATURE_DIM = 8
TASK_FEATURE_DIM = 6
HIDDEN_DIM = 64
NUM_GNN_LAYERS = 3
EPSILON = 0.1  # OT regularization
NUM_SINKHORN_ITERS = 50
USE_GNN = True

# Training parameters
NUM_EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
GAMMA = 0.99  # discount factor
ENTROPY_COEFF = 0.01  # entropy regularization
ASSIGNMENT_THRESHOLD = 0.5  # threshold for detecting assignments

# Directories
LOGDIR = Path("logs")
LOGDIR.mkdir(exist_ok=True)
PLOTDIR = Path("plots")
PLOTDIR.mkdir(exist_ok=True)

# Set random seeds
torch.manual_seed(SEED)
np.random.seed(SEED)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def create_instances(num_instances: int, seed_offset: int = 0) -> List[DmTSPInstance]:
    """Create problem instances."""
    instances = []
    
    for i in range(num_instances):
        instance = DmTSPInstance(
            num_agents=NUM_AGENTS,
            depot_location=np.array([50.0, 50.0]),
            time_horizon=TIME_HORIZON,
            request_rate=REQUEST_RATE,
            service_area=(0, 100, 0, 100),
            seed=SEED + seed_offset + i,
        )
        instances.append(instance)
    
    return instances


def compute_returns(rewards: List[float], gamma: float) -> List[float]:
    """Compute discounted returns."""
    returns = []
    G = 0.0
    
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    
    return returns


def otnet_loss(
    policy: OTNetPolicy,
    trajectory: List[Dict[str, Any]],
    gamma: float = 0.99,
    entropy_coeff: float = 0.01,
) -> torch.Tensor:
    """
    Compute OT-Net policy gradient loss.
    
    Uses REINFORCE with baseline and entropy regularization.
    
    Args:
        policy: OT-Net policy
        trajectory: Episode trajectory
        gamma: Discount factor
        entropy_coeff: Entropy coefficient
    
    Returns:
        Loss value
    """
    # Extract rewards and compute returns
    rewards = [t['reward'] for t in trajectory]
    returns = compute_returns(rewards, gamma)
    
    # Normalize returns
    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    policy_loss = 0.0
    entropy_loss = 0.0
    
    for i, transition in enumerate(trajectory):
        agent_features = transition['agent_features'].to(device)
        task_features = transition['task_features'].to(device)
        edge_index = transition['edge_index'].to(device)
        assignment = torch.tensor(
            transition['assignment'],
            dtype=torch.float32,
            device=device
        )
        
        # Forward pass to get transport matrix
        T = policy(agent_features, task_features, edge_index)
        
        # Log probability of assignment (approximate)
        # P(assignment | state) ≈ product of T[i,j] for assigned pairs
        log_prob = 0.0
        for i_agent in range(T.shape[0]):
            for j_task in range(T.shape[1]):
                if assignment[i_agent, j_task] > ASSIGNMENT_THRESHOLD:
                    log_prob += torch.log(T[i_agent, j_task] + 1e-8)
        
        # Policy gradient loss
        policy_loss -= log_prob * returns[i]
        
        # Entropy regularization (encourage exploration)
        entropy = -(T * torch.log(T + 1e-8)).sum()
        entropy_loss -= entropy
    
    total_loss = policy_loss + entropy_coeff * entropy_loss
    
    return total_loss


def train_otnet(
    policy: OTNetPolicy,
    train_instances: List[DmTSPInstance],
    val_instances: List[DmTSPInstance],
    num_epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    gamma: float = 0.99,
    entropy_coeff: float = 0.01,
) -> Dict[str, List[float]]:
    """
    Train OT-Net policy.
    
    Args:
        policy: OT-Net policy
        train_instances: Training instances
        val_instances: Validation instances
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        gamma: Discount factor
        entropy_coeff: Entropy coefficient
    
    Returns:
        Training history
    """
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    
    train_rewards = []
    val_rewards = []
    train_losses = []
    
    best_val_reward = -float('inf')
    best_model_state = None
    
    print("\nStarting OT-Net training...")
    
    for epoch in tqdm(range(num_epochs), desc="Training"):
        policy.train()
        epoch_losses = []
        epoch_train_rewards = []
        
        # Sample instances for this epoch
        sampled_instances = np.random.choice(
            train_instances,
            size=min(batch_size, len(train_instances)),
            replace=True
        )
        
        for instance in sampled_instances:
            # Create environment
            env = DmTSPEnv(instance, alpha=ALPHA, beta=BETA)
            
            # Generate episode
            trajectory = generate_episode(policy, env, device=device)
            
            # Compute loss
            loss = otnet_loss(policy, trajectory, gamma, entropy_coeff)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
            # Collect episode reward
            episode_reward = sum(t['reward'] for t in trajectory)
            epoch_train_rewards.append(episode_reward)
        
        # Evaluate on validation set
        policy.eval()
        val_episode_rewards = []
        
        for instance in val_instances:
            env = DmTSPEnv(instance, alpha=ALPHA, beta=BETA)
            results = evaluate_policy(policy, env, num_episodes=1, device=device)
            val_episode_rewards.append(results['mean_reward'])
        
        # Record metrics
        train_rewards.append(np.mean(epoch_train_rewards))
        val_rewards.append(np.mean(val_episode_rewards))
        train_losses.append(np.mean(epoch_losses))
        
        # Save best model
        if val_rewards[-1] > best_val_reward:
            best_val_reward = val_rewards[-1]
            best_model_state = policy.state_dict()
        
        # Log progress
        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train reward: {train_rewards[-1]:.4f}")
            print(f"  Val reward: {val_rewards[-1]:.4f}")
            print(f"  Loss: {train_losses[-1]:.4f}")
    
    # Load best model
    if best_model_state is not None:
        policy.load_state_dict(best_model_state)
    
    history = {
        'train_rewards': train_rewards,
        'val_rewards': val_rewards,
        'train_losses': train_losses,
    }
    
    return history


def main():
    """Main training function."""
    
    print("="*60)
    print("OT-Net Training for D-mTSP")
    print("="*60)
    
    # Create instances
    print("\nCreating problem instances...")
    train_instances = create_instances(NUM_TRAIN_INSTANCES, seed_offset=0)
    val_instances = create_instances(NUM_VAL_INSTANCES, seed_offset=1000)
    test_instances = create_instances(NUM_TEST_INSTANCES, seed_offset=2000)
    
    print(f"Created {len(train_instances)} training instances")
    print(f"Created {len(val_instances)} validation instances")
    print(f"Created {len(test_instances)} test instances")
    
    # Create policy
    print("\nInitializing OT-Net policy...")
    policy = OTNetPolicy(
        agent_feature_dim=AGENT_FEATURE_DIM,
        task_feature_dim=TASK_FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_gnn_layers=NUM_GNN_LAYERS,
        epsilon=EPSILON,
        num_sinkhorn_iters=NUM_SINKHORN_ITERS,
        use_gnn=USE_GNN,
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
    # Evaluate baseline policies
    print("\nEvaluating baseline policies...")
    
    greedy_policy = GreedyPolicy()
    random_policy = RandomPolicy(seed=SEED)
    
    greedy_results = []
    random_results = []
    
    for instance in test_instances[:3]:  # Evaluate on subset
        env = DmTSPEnv(instance, alpha=ALPHA, beta=BETA)
        
        greedy_res = evaluate_policy(greedy_policy, env, num_episodes=1)
        random_res = evaluate_policy(random_policy, env, num_episodes=1)
        
        greedy_results.append(greedy_res['mean_reward'])
        random_results.append(random_res['mean_reward'])
    
    print(f"Greedy baseline: {np.mean(greedy_results):.4f} ± {np.std(greedy_results):.4f}")
    print(f"Random baseline: {np.mean(random_results):.4f} ± {np.std(random_results):.4f}")
    
    # Train OT-Net
    history = train_otnet(
        policy,
        train_instances,
        val_instances,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        entropy_coeff=ENTROPY_COEFF,
    )
    
    # Test trained policy
    print("\nEvaluating trained OT-Net...")
    test_results = []
    test_makespans = []
    test_waiting_times = []
    
    for instance in test_instances:
        env = DmTSPEnv(instance, alpha=ALPHA, beta=BETA)
        results = evaluate_policy(policy, env, num_episodes=1, device=device)
        
        test_results.append(results['mean_reward'])
        test_makespans.append(results['mean_makespan'])
        test_waiting_times.append(results['mean_waiting'])
    
    print(f"\nTest Results:")
    print(f"  Mean reward: {np.mean(test_results):.4f} ± {np.std(test_results):.4f}")
    print(f"  Mean makespan: {np.mean(test_makespans):.4f} ± {np.std(test_makespans):.4f}")
    print(f"  Mean waiting: {np.mean(test_waiting_times):.4f} ± {np.std(test_waiting_times):.4f}")
    
    # Save results
    print("\nSaving results...")
    
    results = {
        'history': history,
        'test_results': test_results,
        'test_makespans': test_makespans,
        'test_waiting_times': test_waiting_times,
        'greedy_baseline': greedy_results,
        'random_baseline': random_results,
        'config': {
            'num_agents': NUM_AGENTS,
            'time_horizon': TIME_HORIZON,
            'request_rate': REQUEST_RATE,
            'alpha': ALPHA,
            'beta': BETA,
            'hidden_dim': HIDDEN_DIM,
            'num_epochs': NUM_EPOCHS,
            'learning_rate': LEARNING_RATE,
        }
    }
    
    with open(LOGDIR / 'otnet_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    torch.save(policy.state_dict(), LOGDIR / 'otnet_model.pt')
    
    print(f"Results saved to {LOGDIR / 'otnet_results.pkl'}")
    print(f"Model saved to {LOGDIR / 'otnet_model.pt'}")
    
    # Plot training curves
    print("\nPlotting training curves...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Rewards
    axes[0].plot(history['train_rewards'], label='Train', linewidth=2)
    axes[0].plot(history['val_rewards'], label='Validation', linewidth=2)
    axes[0].axhline(np.mean(greedy_results), color='red', linestyle='--', label='Greedy')
    axes[0].axhline(np.mean(random_results), color='orange', linestyle='--', label='Random')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Training Progress')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history['train_losses'], linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Training Loss')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(PLOTDIR / 'otnet_training.pdf')
    plt.savefig(PLOTDIR / 'otnet_training.png', dpi=300)
    
    print(f"Plots saved to {PLOTDIR / 'otnet_training.pdf'}")
    
    print("\nTraining complete!")
    

if __name__ == '__main__':
    main()
