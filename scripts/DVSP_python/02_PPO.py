"""
Proximal Policy Optimization (PPO) for DVSP.

This script implements PPO training with clipped surrogate objective
and optional replay buffer for the DVSP problem.
"""

import os
import sys
import copy
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    CombinatorialACPolicy,
    critic_GNN,
    p_distribution,
    PPO_episodes,
    evaluate_policy,
    KleopatraVSPPolicy,
    prize_collecting_vsp,
    J_PPO,
    huber_GNN,
    grads_prep_GNN,
    rb_add,
    rb_sample,
    save_results,
    sigmaF_dvsp,
)

print("WARNING: This is a conversion template.")


class ActorModel(nn.Module):
    """Simple linear actor model without bias."""
    
    def __init__(self, input_dim: int = 14, output_dim: int = 1):
        super(ActorModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning flattened output."""
        return self.linear(x).squeeze(-1)


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
    sigmaF_values: Tuple[float, float] = (0.5, 0.05),
    lr_values: Tuple[float, float] = (1e-3, 5e-4),
    V_method: str = "off_policy",
    adv_method: str = "TD_n",
    critic_factor: int = 10,
    model_builder=None,
    **kwargs
) -> Tuple[nn.Module, List[float], List[float], List[float]]:
    """
    Train policy using PPO algorithm.
    
    Args:
        policy: CombinatorialACPolicy to train
        train_envs: Training environments
        val_envs: Validation environments
        episodes: Number of training episodes
        collection_steps: Number of episodes to collect per iteration
        epochs: Number of update epochs per collection
        batch_size: Batch size for training
        ntests: Number of test episodes for evaluation
        clip: PPO clipping parameter
        use_rb: Whether to use replay buffer
        sigmaF_values: Tuple of (initial, final) sigma_F values
        lr_values: Tuple of (initial, final) learning rates
        V_method: Value estimation method ("on_policy" or "off_policy")
        adv_method: Advantage estimation method ("TD_n" or "TD_1")
        critic_factor: Learning rate multiplier for critic
        model_builder: Optimization model builder
        **kwargs: Additional arguments
    
    Returns:
        Tuple of (best_actor_model, train_history, val_history, loss_history)
    """
    # Initialize optimizers with gradient clipping
    opt_actor = torch.optim.Adam(policy.actor_model.parameters(), lr=lr_values[0])
    opt_critic = torch.optim.Adam(
        policy.critic_model.parameters(),
        lr=lr_values[0] * critic_factor
    )
    
    train_reward_history = []
    val_reward_history = []
    actor_weights_history = []
    loss_history = []
    
    gamma = 1.0
    best_model = copy.deepcopy(policy.actor_model)
    best_performance = float('-inf')
    best_episode = 0
    target_critic = copy.deepcopy(policy.critic_model)
    
    # Setup sigma_F schedule
    global sigmaF_dvsp
    sigmaF_dvsp = sigmaF_values[0]
    sigmaF_step = (sigmaF_values[0] - sigmaF_values[1]) / episodes
    sigmaF_average = 2.0
    
    # Setup learning rate schedule
    lr_step_a = (lr_values[0] - lr_values[1]) / episodes
    lr_step_c = (lr_values[0] * critic_factor - lr_values[1]) / episodes
    
    # Initialize replay buffer if used
    if use_rb:
        replay_buffer = []
        rb_capacity = collection_steps * 6 * 100
        rb_position = 1
        rb_size = 0
        iterations = epochs
        epochs = 1
    
    policy.reset_seed()
    
    for e in range(episodes):
        # Evaluate model
        try:
            train_perf, _ = evaluate_policy(
                policy,
                train_envs,
                nb_episodes=ntests,
                perturb=False,
                model_builder=model_builder
            )
            train_reward_history.append(train_perf)
            
            val_perf, _ = evaluate_policy(
                policy,
                val_envs,
                nb_episodes=ntests,
                perturb=False,
                model_builder=model_builder
            )
            val_reward_history.append(val_perf)
            
            # Save actor weights
            with torch.no_grad():
                actor_weights_history.append(
                    policy.actor_model.linear.weight.clone().cpu().numpy()
                )
            
            # Save best model
            if val_perf >= best_performance:
                best_performance = val_perf
                best_model = copy.deepcopy(policy.actor_model)
                best_episode = e
            
            print(f"Episode {e}: sigmaF={sigmaF_dvsp:.4f}, "
                  f"lr_actor={opt_actor.param_groups[0]['lr']:.6f}, "
                  f"train={train_perf:.4f}, val={val_perf:.4f}")
        
        except NotImplementedError:
            print(f"Episode {e}: Evaluation not implemented")
            break
        
        # Collect experience
        try:
            episode_data = PPO_episodes(
                policy,
                target_critic,
                train_envs,
                collection_steps,
                gamma,
                V_method,
                adv_method,
                model_builder=model_builder
            )
        except NotImplementedError:
            print("Episode collection not implemented")
            break
        
        if use_rb:
            replay_buffer, rb_position, rb_size = rb_add(
                replay_buffer, rb_capacity, rb_position, rb_size, episode_data
            )
            batches = [rb_sample(replay_buffer, batch_size) for _ in range(iterations)]
        else:
            # Simple batching without replay buffer
            indices = list(range(len(episode_data)))
            np.random.shuffle(indices)
            batches = [
                [episode_data[i] for i in indices[j:j+batch_size]]
                for j in range(0, len(indices), batch_size)
            ]
        
        for _ in range(epochs):
            for batch in batches:
                # Train critic
                critic_target = [exp.Rₜ for exp in batch]
                
                try:
                    graphs, s_c, edge_features = grads_prep_GNN(batch)
                    
                    opt_critic.zero_grad()
                    critic_loss = huber_GNN(
                        policy, graphs, s_c, edge_features, critic_target, 1.0
                    )
                    critic_loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_value_(
                        policy.critic_model.parameters(), 1e-3
                    )
                    opt_critic.step()
                
                except NotImplementedError:
                    print("Critic training not fully implemented")
                    break
                
                # Train actor
                try:
                    opt_actor.zero_grad()
                    actor_loss = -J_PPO(policy, batch, clip, sigmaF_average)
                    actor_loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_value_(
                        policy.actor_model.parameters(), 1e-3
                    )
                    opt_actor.step()
                
                except Exception as ex:
                    print(f"Actor training error: {ex}")
                    break
            
            # Update target critic
            target_critic = copy.deepcopy(policy.critic_model)
        
        # Update sigmaF and learning rates
        sigmaF_dvsp = max(sigmaF_dvsp - sigmaF_step, sigmaF_values[1])
        
        # Update learning rates
        new_lr_actor = max(
            opt_actor.param_groups[0]['lr'] - lr_step_a,
            lr_values[1]
        )
        opt_actor.param_groups[0]['lr'] = new_lr_actor
        
        new_lr_critic = max(
            opt_critic.param_groups[0]['lr'] - lr_step_c,
            lr_values[1]
        )
        opt_critic.param_groups[0]['lr'] = new_lr_critic
    
    # Final test
    policy.actor_model = copy.deepcopy(best_model)
    try:
        final_train, _ = evaluate_policy(
            policy,
            train_envs,
            nb_episodes=ntests,
            perturb=False,
            model_builder=model_builder
        )
        train_reward_history.append(final_train)
        
        final_val, _ = evaluate_policy(
            policy,
            val_envs,
            nb_episodes=ntests,
            perturb=False,
            model_builder=model_builder
        )
        val_reward_history.append(final_val)
        
        with torch.no_grad():
            actor_weights_history.append(
                policy.actor_model.linear.weight.clone().cpu().numpy()
            )
        
        print(f"Final train: {final_train:.4f}, Final val: {final_val:.4f}, "
              f"Best episode: {best_episode}")
    
    except NotImplementedError:
        pass
    
    return policy.actor_model, train_reward_history, val_reward_history, loss_history


if __name__ == "__main__":
    # Configuration
    SEED = 0
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Setup paths
    logdir = Path("logs")
    logdir.mkdir(exist_ok=True)
    plotdir = Path("plots")
    plotdir.mkdir(exist_ok=True)
    
    try:
        # Initialize models
        actor_model = ActorModel(input_dim=14, output_dim=1)
        critic_model = critic_GNN(node_features=15, edge_features=1)
        
        # Create policy
        PPO_policy = CombinatorialACPolicy(
            actor_model=actor_model,
            critic_model=critic_model,
            p=p_distribution,
            CO_layer=prize_collecting_vsp,
            seed=0
        )
        
        # Load environments (placeholder)
        print("Loading environments...")
        raise NotImplementedError("Environment setup not implemented")
        
        # Train PPO
        print("Training PPO model...")
        PPO_model, PPO_train, PPO_val, PPO_losses = PPO_training(
            PPO_policy,
            train_envs,
            val_envs,
            episodes=400,
            collection_steps=20,
            epochs=100,
            batch_size=1,
            ntests=1,
            clip=0.2,
            use_rb=True,
            sigmaF_values=(0.5, 0.05),
            lr_values=(1e-3, 5e-4),
            V_method="off_policy",
            adv_method="TD_n",
            critic_factor=10
        )
        
        # Test the trained model
        PPO_policy_evaluation = KleopatraVSPPolicy(PPO_model)
        PPO_final_train, PPO_final_train_rew = evaluate_policy(
            PPO_policy_evaluation,
            train_envs,
            nb_episodes=10
        )
        PPO_final_train_rew = [-r for r in PPO_final_train_rew]
        
        PPO_final_test, PPO_final_test_rew = evaluate_policy(
            PPO_policy_evaluation,
            test_envs,
            nb_episodes=10
        )
        PPO_final_test_rew = [-r for r in PPO_final_test_rew]
        
        # Plot training and validation rewards
        plt.figure(figsize=(10, 6))
        plt.plot(PPO_train, label='train history', marker='o')
        plt.plot(PPO_val, label='val history', marker='o')
        plt.xlabel('training episode')
        plt.ylabel('reward')
        plt.title('DVSP PPO')
        plt.legend()
        plt.grid(True)
        plt.savefig(plotdir / "dvsp_PPO_rew_line.pdf")
        plt.close()
        
        # Save the model and rewards
        save_results(
            logdir / "dvsp_PPO_training_results.pt",
            model=PPO_model.state_dict(),
            train_rew=PPO_train,
            val_rew=PPO_val,
            train_final=PPO_final_train_rew,
            test_final=PPO_final_test_rew,
        )
        
        print("PPO training complete. Results saved.")
    
    except NotImplementedError as e:
        print(f"\nError: {e}")
        print("\nThis is a template conversion. To make it functional, you need to:")
        print("1. Implement environment setup")
        print("2. Implement episode collection and value estimation")
        print("3. Implement full GNN critic functionality")
