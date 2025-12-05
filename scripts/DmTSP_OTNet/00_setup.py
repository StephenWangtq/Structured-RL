"""
Setup script for D-mTSP OT-Net.
Evaluates baseline policies and sets up the problem.
"""

import os
import numpy as np
import pickle
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt

from utils.environment import DmTSPInstance, DmTSPEnv
from utils.utils import evaluate_policy, GreedyPolicy, RandomPolicy


# Configuration
SEED = 42
NUM_EVAL_INSTANCES = 10

# Environment parameters
NUM_AGENTS = 5
TIME_HORIZON = 100.0
REQUEST_RATE = 0.5
ALPHA = 0.5  # makespan weight
BETA = 0.5   # waiting time weight

# Directories
LOGDIR = Path("logs")
LOGDIR.mkdir(exist_ok=True)
PLOTDIR = Path("plots")
PLOTDIR.mkdir(exist_ok=True)

# Set random seed
np.random.seed(SEED)


def create_test_instances(num_instances: int, seed_offset: int = 0) -> List[DmTSPInstance]:
    """Create test instances."""
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


def evaluate_baseline(policy, instances: List[DmTSPInstance], policy_name: str):
    """Evaluate a baseline policy on instances."""
    print(f"\nEvaluating {policy_name} policy...")
    
    all_rewards = []
    all_makespans = []
    all_waiting_times = []
    
    for i, instance in enumerate(instances):
        env = DmTSPEnv(instance, alpha=ALPHA, beta=BETA)
        results = evaluate_policy(policy, env, num_episodes=1)
        
        all_rewards.append(results['mean_reward'])
        all_makespans.append(results['mean_makespan'])
        all_waiting_times.append(results['mean_waiting'])
        
        print(f"  Instance {i+1}: reward={results['mean_reward']:.2f}, "
              f"makespan={results['mean_makespan']:.2f}, "
              f"waiting={results['mean_waiting']:.2f}")
    
    results = {
        'rewards': all_rewards,
        'makespans': all_makespans,
        'waiting_times': all_waiting_times,
        'mean_reward': np.mean(all_rewards),
        'std_reward': np.std(all_rewards),
        'mean_makespan': np.mean(all_makespans),
        'std_makespan': np.std(all_makespans),
        'mean_waiting': np.mean(all_waiting_times),
        'std_waiting': np.std(all_waiting_times),
    }
    
    print(f"\n{policy_name} Summary:")
    print(f"  Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Mean makespan: {results['mean_makespan']:.2f} ± {results['std_makespan']:.2f}")
    print(f"  Mean waiting: {results['mean_waiting']:.2f} ± {results['std_waiting']:.2f}")
    
    return results


def visualize_instance(instance: DmTSPInstance, filename: str):
    """Visualize a problem instance."""
    requests = instance.generate_requests()
    
    if len(requests) == 0:
        print(f"Warning: No requests generated for visualization")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Spatial distribution
    ax = axes[0]
    ax.scatter([r.location[0] for r in requests], 
              [r.location[1] for r in requests],
              c='blue', alpha=0.6, s=50, label='Requests')
    ax.scatter([instance.depot_location[0]], 
              [instance.depot_location[1]],
              c='red', marker='s', s=200, label='Depot')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_title('Spatial Distribution of Requests')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Temporal distribution
    ax = axes[1]
    arrival_times = [r.arrival_time for r in requests]
    ax.hist(arrival_times, bins=20, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Arrival Time')
    ax.set_ylabel('Number of Requests')
    ax.set_title('Temporal Distribution (Poisson Process)')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    print(f"Instance visualization saved to {filename}")


def main():
    """Main setup function."""
    
    print("="*60)
    print("D-mTSP OT-Net Setup and Baseline Evaluation")
    print("="*60)
    
    print(f"\nProblem Configuration:")
    print(f"  Number of agents: {NUM_AGENTS}")
    print(f"  Time horizon: {TIME_HORIZON}")
    print(f"  Request rate: {REQUEST_RATE} requests/time unit")
    print(f"  Alpha (makespan weight): {ALPHA}")
    print(f"  Beta (waiting weight): {BETA}")
    
    # Create test instances
    print(f"\nCreating {NUM_EVAL_INSTANCES} test instances...")
    test_instances = create_test_instances(NUM_EVAL_INSTANCES, seed_offset=5000)
    
    # Visualize one instance
    print("\nVisualizing sample instance...")
    visualize_instance(
        test_instances[0],
        PLOTDIR / "sample_instance.png"
    )
    
    # Evaluate greedy baseline
    greedy_policy = GreedyPolicy()
    greedy_results = evaluate_baseline(greedy_policy, test_instances, "Greedy")
    
    # Evaluate random baseline
    random_policy = RandomPolicy(seed=SEED)
    random_results = evaluate_baseline(random_policy, test_instances, "Random")
    
    # Save baseline results
    baseline_results = {
        'greedy': greedy_results,
        'random': random_results,
        'config': {
            'num_agents': NUM_AGENTS,
            'time_horizon': TIME_HORIZON,
            'request_rate': REQUEST_RATE,
            'alpha': ALPHA,
            'beta': BETA,
        }
    }
    
    output_file = LOGDIR / 'baseline_results.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(baseline_results, f)
    
    print(f"\nBaseline results saved to {output_file}")
    
    # Create comparison plot
    print("\nCreating comparison plot...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    policies = ['Greedy', 'Random']
    colors = ['steelblue', 'coral']
    
    # Rewards
    ax = axes[0]
    means = [greedy_results['mean_reward'], random_results['mean_reward']]
    stds = [greedy_results['std_reward'], random_results['std_reward']]
    ax.bar(policies, means, yerr=stds, color=colors, alpha=0.7, capsize=5)
    ax.set_ylabel('Mean Reward')
    ax.set_title('Reward Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Makespan
    ax = axes[1]
    means = [greedy_results['mean_makespan'], random_results['mean_makespan']]
    stds = [greedy_results['std_makespan'], random_results['std_makespan']]
    ax.bar(policies, means, yerr=stds, color=colors, alpha=0.7, capsize=5)
    ax.set_ylabel('Mean Makespan')
    ax.set_title('Makespan Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Waiting time
    ax = axes[2]
    means = [greedy_results['mean_waiting'], random_results['mean_waiting']]
    stds = [greedy_results['std_waiting'], random_results['std_waiting']]
    ax.bar(policies, means, yerr=stds, color=colors, alpha=0.7, capsize=5)
    ax.set_ylabel('Mean Total Waiting')
    ax.set_title('Waiting Time Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(PLOTDIR / 'baseline_comparison.pdf')
    plt.savefig(PLOTDIR / 'baseline_comparison.png', dpi=300)
    
    print(f"Comparison plot saved to {PLOTDIR / 'baseline_comparison.pdf'}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    
    for policy_name, results in [('Greedy', greedy_results), ('Random', random_results)]:
        print(f"\n{policy_name} Policy:")
        print(f"  Reward:  {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Makespan: {results['mean_makespan']:.2f} ± {results['std_makespan']:.2f}")
        print(f"  Waiting:  {results['mean_waiting']:.2f} ± {results['std_waiting']:.2f}")
    
    print("\n" + "="*60)
    print("Setup complete! Ready for OT-Net training.")
    print("="*60)


if __name__ == '__main__':
    main()
