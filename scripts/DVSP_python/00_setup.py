"""
Dataset setup and baseline evaluation for DVSP.
Converted from Julia to Python.

Seeds used in the paper: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
"""

import os
import pickle
import torch
import numpy as np
from pathlib import Path

from utils.utils import (
    setup_environments,
    GreedyVSPPolicy,
    expert_evaluation,
    evaluate_policy,
)

# Configuration
NB_TRAIN_INSTANCES = 10
NB_VAL_INSTANCES = 10
NB_TEST_INSTANCES = 10
MAX_REQUESTS_PER_EPOCH = 10
SEED = 0

# Setup directories
LOGDIR = Path("logs")
LOGDIR.mkdir(exist_ok=True)

# Note: Dataset paths need to be configured based on your setup
# The Julia code uses DataDeps to download the euro-neurips-2022 dataset
# You'll need to download it manually or implement similar functionality

DATASET_PATH = Path("data/euro-neurips-2022")
TRAIN_INSTANCES = DATASET_PATH / "train"
VAL_INSTANCES = DATASET_PATH / "validation"
TEST_INSTANCES = DATASET_PATH / "test"

# Check if data exists
if not TRAIN_INSTANCES.exists():
    print(f"Warning: Training data not found at {TRAIN_INSTANCES}")
    print("Please download the EURO-NeurIPS 2022 VRP dataset")
    print("https://github.com/ortec/euro-neurips-vrp-2022-quickstart")
    exit(1)

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
    print(f"Created {len(train_envs)} training, {len(val_envs)} validation, "
          f"and {len(test_envs)} test environments")
except Exception as e:
    print(f"Error setting up environments: {e}")
    print("This requires implementing the environment integration with your VSP solver.")
    exit(1)

# Model builder (requires Gurobi or similar)
# Note: You'll need to implement this based on your optimization solver
try:
    from utils.utils import grb_model
    model_builder = grb_model
except ImportError:
    print("Warning: Gurobi model builder not available")
    model_builder = None

# Evaluate greedy policy
print("\nEvaluating greedy baseline...")
try:
    greedy_policy = GreedyVSPPolicy()
    
    greedy_train_mean, greedy_train_rews = evaluate_policy(
        greedy_policy,
        train_envs,
        nb_episodes=1,
        model_builder=model_builder,
        return_scores=True,
    )
    greedy_train_mean = -greedy_train_mean
    greedy_train_rews = [-r for r in greedy_train_rews]
    
    greedy_test_mean, greedy_test_rews = evaluate_policy(
        greedy_policy,
        test_envs,
        nb_episodes=1,
        model_builder=model_builder,
        return_scores=True,
    )
    greedy_test_mean = -greedy_test_mean
    greedy_test_rews = [-r for r in greedy_test_rews]
    
    print(f"Greedy train mean: {greedy_train_mean:.2f}")
    print(f"Greedy test mean: {greedy_test_mean:.2f}")
except Exception as e:
    print(f"Error evaluating greedy policy: {e}")
    greedy_train_rews = []
    greedy_test_rews = []

# Evaluate expert policy
print("\nEvaluating expert baseline...")
try:
    expert_train_mean, expert_train_rews = expert_evaluation(
        train_envs,
        model_builder=model_builder
    )
    expert_test_mean, expert_test_rews = expert_evaluation(
        test_envs,
        model_builder=model_builder
    )
    
    print(f"Expert train mean: {expert_train_mean:.2f}")
    print(f"Expert test mean: {expert_test_mean:.2f}")
except Exception as e:
    print(f"Error evaluating expert policy: {e}")
    expert_train_rews = []
    expert_test_rews = []

# Save baseline results
print("\nSaving baseline results...")
baseline_results = {
    'greedy_train': greedy_train_rews,
    'expert_train': expert_train_rews,
    'greedy_test': greedy_test_rews,
    'expert_test': expert_test_rews,
}

with open(LOGDIR / "dvsp_baselines.pkl", 'wb') as f:
    pickle.dump(baseline_results, f)

print(f"Baseline results saved to {LOGDIR / 'dvsp_baselines.pkl'}")
print("\nSetup complete! You can now run:")
print("  - 01_SIL.py for Supervised Imitation Learning")
print("  - 02_PPO.py for Proximal Policy Optimization")
print("  - 03_SRL.py for Structured Reinforcement Learning")
