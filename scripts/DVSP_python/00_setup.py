"""
Setup script for DVSP: Baseline policies and dataset preparation.

Seeds used in the paper:
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    GreedyVSPPolicy,
    evaluate_policy,
    expert_evaluation,
    save_results,
    setup_environments,
    setup_dataset,
)

# Note: This script requires the DynamicVehicleRouting package
# and DecisionFocusedLearningBenchmarks to be implemented in Python
print("WARNING: This is a conversion template.")
print("Full functionality requires Python implementations of:")
print("  - DynamicVehicleRouting")
print("  - DecisionFocusedLearningBenchmarks")
print("  - InferOpt")

# Configuration
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

# Setup paths
logdir = Path("logs")
logdir.mkdir(exist_ok=True)

# Dataset setup (placeholder - requires actual implementation)
try:
    dataset_path = setup_dataset("euro-neurips-2022")
    
    train_instances = os.path.join(dataset_path, "train")
    val_instances = os.path.join(dataset_path, "validation")
    test_instances = os.path.join(dataset_path, "test")
    
    nb_train_instances = 10
    nb_val_instances = 10
    nb_test_instances = 10
    
    # Create environments
    train_envs, val_envs, test_envs = setup_environments(
        train_instances,
        val_instances,
        test_instances,
        nb_train_instances,
        nb_val_instances,
        nb_test_instances,
        max_requests_per_epoch=10,
        seed=0,
        use_scaling=False
    )
    
    # Model builder (e.g., Gurobi)
    # model_builder = grb_model  # or highs_model if you don't have Gurobi
    model_builder = None  # Placeholder
    
    # Baseline policies
    
    # Greedy policy
    print("Evaluating Greedy policy...")
    greedy_policy = GreedyVSPPolicy()
    greedy_train_mean, greedy_train_rews = evaluate_policy(
        greedy_policy,
        train_envs,
        nb_episodes=1,
        model_builder=model_builder,
    )
    greedy_train_rews = [-r for r in greedy_train_rews]  # Negate for minimization
    
    greedy_test_mean, greedy_test_rews = evaluate_policy(
        greedy_policy,
        test_envs,
        nb_episodes=1,
        model_builder=model_builder,
    )
    greedy_test_rews = [-r for r in greedy_test_rews]
    
    print(f"Greedy train mean: {-greedy_train_mean}")
    print(f"Greedy test mean: {-greedy_test_mean}")
    
    # Expert policy
    print("Evaluating Expert policy...")
    expert_train_mean, expert_train_rews = expert_evaluation(
        train_envs,
        model_builder=model_builder
    )
    expert_test_mean, expert_test_rews = expert_evaluation(
        test_envs,
        model_builder=model_builder
    )
    
    print(f"Expert train mean: {expert_train_mean}")
    print(f"Expert test mean: {expert_test_mean}")
    
    # Save results
    save_results(
        logdir / "dvsp_baselines.pt",
        greedy_train=greedy_train_rews,
        expert_train=expert_train_rews,
        greedy_test=greedy_test_rews,
        expert_test=expert_test_rews,
    )
    
    print("Baseline evaluation complete. Results saved to logs/dvsp_baselines.pt")

except NotImplementedError as e:
    print(f"\nError: {e}")
    print("\nThis is a template conversion. To make it functional, you need to:")
    print("1. Implement or port the DynamicVehicleRouting package to Python")
    print("2. Implement or port the DecisionFocusedLearningBenchmarks package")
    print("3. Implement the optimization solvers (Gurobi/HiGHS interface)")
    print("4. Implement the dataset download and setup functionality")
