"""
Structured Reinforcement Learning (SRL) for DVSP.

This script implements SRL training with Fenchel-Young loss for actor
and Q-learning with GNN critic for the DVSP problem.
"""

import os
import sys
import copy
from pathlib import Path
from typing import Tuple, List
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
    SRL_episodes,
    SRL_actions,
    J_SRL,
    V_value_GNN,
    evaluate_policy,
    KleopatraVSPPolicy,
    prize_collecting_vsp,
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
    sigmaB_values: Tuple[float, float] = (1.0, 0.1),
    lr_values: Tuple[float, float] = (1e-3, 2e-4),
    critic_factor: int = 2,
    temp_values: Tuple[float, float] = (10.0, 10.0),
    model_builder=None,
    **kwargs
) -> Tuple[nn.Module, List[float], List[float], List[float]]:
    """
    Train policy using SRL algorithm.
    
    Args:
        policy: CombinatorialACPolicy to train
        train_envs: Training environments
        val_envs: Validation environments
        grad_steps: Number of gradient steps
        collection_steps: Number of episodes to collect per iteration
        iterations: Number of update iterations per collection
        batch_size: Batch size for training
        ntests: Number of test episodes for evaluation
        sigmaF: Sigma for forward perturbation (fixed)
        sigmaB_values: Tuple of (initial, final) sigma_B values
        lr_values: Tuple of (initial, final) learning rates
        critic_factor: Learning rate multiplier for critic
        temp_values: Tuple of (initial, final) temperature values
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
    
    # Setup schedules
    global sigmaF_dvsp
    sigmaF_dvsp = sigmaF
    
    sigmaB = sigmaB_values[0]
    sigmaB_step = (sigmaB_values[0] - sigmaB_values[1]) / grad_steps
    
    lr_step_a = (lr_values[0] - lr_values[1]) / grad_steps
    lr_step_c = (lr_values[0] * critic_factor - lr_values[1]) / grad_steps
    
    temp = temp_values[0]
    temp_step = (temp_values[0] - temp_values[1]) / grad_steps
    
    # Initialize replay buffer (required for SRL)
    replay_buffer = []
    rb_capacity = collection_steps * 6 * 1000
    rb_position = 1
    rb_size = 0
    
    policy.reset_seed()
    
    for e in range(grad_steps):
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
            
            print(f"Episode {e}: sigmaB={sigmaB:.4f}, "
                  f"lr_actor={opt_actor.param_groups[0]['lr']:.6f}, "
                  f"train={train_perf:.4f}, val={val_perf:.4f}")
        
        except NotImplementedError:
            print(f"Episode {e}: Evaluation not implemented")
            break
        
        # Collect experience
        try:
            episode_data = SRL_episodes(
                policy,
                train_envs,
                collection_steps,
                model_builder=model_builder
            )
            replay_buffer, rb_position, rb_size = rb_add(
                replay_buffer, rb_capacity, rb_position, rb_size, episode_data
            )
        except NotImplementedError:
            print("Episode collection not implemented")
            break
        
        # Sample batches from replay buffer
        batches = [rb_sample(replay_buffer, batch_size) for _ in range(iterations)]
        
        for batch in batches:
            # Train critic
            try:
                rewards = [exp.reward for exp in batch]
                next_states = [exp.next_state for exp in batch]
                next_embeddings = [exp.next_s for exp in batch]
                next_embeds_c = [exp.next_s_c for exp in batch]
                
                # Calculate Q-learning targets
                critic_target = []
                for j in range(len(batch)):
                    v_next = V_value_GNN(
                        policy,
                        next_embeddings[j],
                        next_embeds_c[j],
                        target_critic,
                        "off_policy",
                        instance=next_states[j],
                        model_builder=model_builder
                    )
                    critic_target.append(rewards[j] + gamma * v_next)
                
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
                # Compute optimal actions using Q-values
                opt_actions = SRL_actions(
                    policy,
                    batch,
                    sigmaB=sigmaB,
                    no_samples=40,
                    temp=temp,
                    model_builder=model_builder
                )
                
                opt_actor.zero_grad()
                actor_loss = J_SRL(policy, batch, opt_actions, model_builder=model_builder)
                actor_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_value_(
                    policy.actor_model.parameters(), 1e-3
                )
                opt_actor.step()
            
            except NotImplementedError:
                print("Actor training not fully implemented (requires InferOpt)")
                break
        
        # Update target critic
        target_critic = copy.deepcopy(policy.critic_model)
        
        # Update schedules
        sigmaB = max(sigmaB - sigmaB_step, sigmaB_values[1])
        temp = max(temp - temp_step, temp_values[1])
        
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
        SRL_policy = CombinatorialACPolicy(
            actor_model=actor_model,
            critic_model=critic_model,
            p=p_distribution,
            CO_layer=prize_collecting_vsp,
            seed=0
        )
        
        # Load environments (placeholder)
        print("Loading environments...")
        raise NotImplementedError("Environment setup not implemented")
        
        # Train SRL
        print("Training SRL model...")
        SRL_model, SRL_train, SRL_val, SRL_losses = SRL_training(
            SRL_policy,
            train_envs,
            val_envs,
            grad_steps=400,
            collection_steps=20,
            iterations=100,
            batch_size=4,
            ntests=1,
            sigmaF=0.1,
            sigmaB_values=(1.0, 0.1),
            lr_values=(1e-3, 2e-4),
            critic_factor=2,
            temp_values=(10.0, 10.0)
        )
        
        # Test the trained model
        SRL_policy_evaluation = KleopatraVSPPolicy(SRL_model)
        SRL_final_train, SRL_final_train_rew = evaluate_policy(
            SRL_policy_evaluation,
            train_envs,
            nb_episodes=10
        )
        SRL_final_train_rew = [-r for r in SRL_final_train_rew]
        
        SRL_final_test, SRL_final_test_rew = evaluate_policy(
            SRL_policy_evaluation,
            test_envs,
            nb_episodes=10
        )
        SRL_final_test_rew = [-r for r in SRL_final_test_rew]
        
        # Plot training and validation rewards
        plt.figure(figsize=(10, 6))
        plt.plot(SRL_train, label='train history', marker='o')
        plt.plot(SRL_val, label='val history', marker='o')
        plt.xlabel('training episode')
        plt.ylabel('reward')
        plt.title('DVSP SRL')
        plt.legend()
        plt.grid(True)
        plt.savefig(plotdir / "dvsp_SRL_rew_line.pdf")
        plt.close()
        
        # Save the model and rewards
        save_results(
            logdir / "dvsp_SRL_training_results.pt",
            model=SRL_model.state_dict(),
            train_rew=SRL_train,
            val_rew=SRL_val,
            train_final=SRL_final_train_rew,
            test_final=SRL_final_test_rew,
        )
        
        print("SRL training complete. Results saved.")
    
    except NotImplementedError as e:
        print(f"\nError: {e}")
        print("\nThis is a template conversion. To make it functional, you need to:")
        print("1. Implement environment setup")
        print("2. Implement InferOpt Fenchel-Young loss")
        print("3. Implement episode collection and Q-value estimation")
        print("4. Implement full GNN critic functionality")
