# DVSP Python Implementation

This directory contains a Python implementation of the Dynamic Vehicle Scheduling Problem (DVSP) reinforcement learning algorithms, converted from the original Julia code.

## Overview

The code implements three training methods for combinatorial optimization:
1. **SIL** (Supervised Imitation Learning) - `01_SIL.py`
2. **PPO** (Proximal Policy Optimization) - `02_PPO.py`
3. **SRL** (Structured Reinforcement Learning) - `03_SRL.py`

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster training)

### Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install PyTorch Geometric (adjust CUDA version as needed):
```bash
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

## Data Setup

Download the EURO-NeurIPS 2022 VRP dataset:

```bash
# Download from https://github.com/ortec/euro-neurips-vrp-2022-quickstart
# Extract to data/euro-neurips-2022/

# Expected structure:
# data/
#   euro-neurips-2022/
#     train/
#       instance_1.txt
#       instance_2.txt
#       ...
#     validation/
#       ...
#     test/
#       ...
```

## Usage

### 1. Setup and Baselines

First, run the setup script to evaluate baseline policies:

```bash
python 00_setup.py
```

This will:
- Load the dataset
- Evaluate greedy and expert policies
- Save baseline results to `logs/dvsp_baselines.pkl`

### 2. Training

Run each training script independently:

```bash
# Supervised Imitation Learning
python 01_SIL.py

# Proximal Policy Optimization
python 02_PPO.py

# Structured Reinforcement Learning
python 03_SRL.py
```

Each script will:
- Train the model for the specified number of episodes/epochs
- Save training curves and final results to `logs/`
- Save the trained model to `logs/`
- Generate training plots in `plots/`

### 3. Visualization

After training all three methods, generate comparison plots:

```bash
python 04_plots.py
```

This creates:
- Training progress line plots
- Boxplot comparisons against baselines
- Saved to `plots/dvsp_training_plot.pdf` and `plots/dvsp_results_plot.pdf`

## Architecture

### Neural Networks

**Actor Network:**
- Simple linear layer: `Linear(14, 1, bias=False)`
- Maps state features to action weights
- Output passed to combinatorial optimization solver

**Critic Network (GNN):**
- Graph Neural Network with multiple layers
- Processes route graphs with node and edge features
- Architecture:
  - Graph convolution layers with edge features (NNConv)
  - Graph convolution layers without edge features (GCNConv)
  - Global pooling
  - Dense layers for value estimation

### Key Algorithms

**SIL:**
- Uses Fenchel-Young loss for structured prediction
- Trains on expert demonstrations
- Validation-based early stopping

**PPO:**
- Clipped surrogate objective
- Advantage estimation (TD(n) or TD(1))
- Optional replay buffer
- Dynamic learning rate and perturbation scheduling

**SRL:**
- Combines Fenchel-Young loss with Q-learning
- Critic-guided action selection with temperature
- Mandatory replay buffer
- Dynamic parameter scheduling (sigmaB, temperature, learning rate)

## Configuration

Key hyperparameters can be modified at the top of each script:

```python
# Training instances
NB_TRAIN_INSTANCES = 10
NB_VAL_INSTANCES = 10
NB_TEST_INSTANCES = 10

# Algorithm-specific parameters
# See individual scripts for details
```

## Output Files

The scripts generate several output files:

**Logs (`logs/`):**
- `dvsp_baselines.pkl` - Baseline policy results
- `dvsp_SIL_training_results.pkl` - SIL training history and results
- `dvsp_SIL_model.pt` - Trained SIL actor model
- `dvsp_PPO_training_results.pkl` - PPO training history and results
- `dvsp_PPO_model.pt` - Trained PPO actor model
- `dvsp_SRL_training_results.pkl` - SRL training history and results
- `dvsp_SRL_model.pt` - Trained SRL actor model

**Plots (`plots/`):**
- `dvsp_SIL_rew_line.pdf` - SIL training curve
- `dvsp_PPO_rew_line.pdf` - PPO training curve
- `dvsp_SRL_rew_line.pdf` - SRL training curve
- `dvsp_training_plot.pdf` - Combined training comparison
- `dvsp_results_plot.pdf` - Final results comparison

## Implementation Notes

### Differences from Julia Version

1. **Optimization Solvers:**
   - Julia uses Gurobi/HiGHS through JuMP
   - Python requires implementing solver interface (e.g., gurobipy, SCIP)
   - `prize_collecting_vsp()` function needs domain-specific implementation

2. **Fenchel-Young Loss:**
   - Julia uses InferOpt.jl for perturbed optimization
   - Python implementation uses placeholder - requires:
     - Perturbed optimization framework
     - Differentiable optimization library
     - Or manual implementation of Fenchel-Young loss

3. **Environment Interface:**
   - Julia uses DynamicVehicleRouting.jl package
   - Python requires implementing `RLDVSPEnv` interface
   - Core methods: `state()`, `apply_action()`, `reset()`, `embedding()`

4. **Data Format:**
   - Julia uses JLD2 for data storage
   - Python uses pickle and PyTorch's `torch.save()`

### Required Implementations

To make this code fully functional, you need to implement:

1. **Environment (`utils/utils.py`):**
   - `RLDVSPEnv._read_vsp_instance()`
   - `RLDVSPEnv._create_dvsp_env()`
   - `RLDVSPEnv._compute_features()`
   - State management and action application

2. **Optimization Solver (`utils/utils.py`):**
   - `prize_collecting_vsp()` - CO layer for route optimization
   - Interfaces with your optimization solver (Gurobi, SCIP, etc.)

3. **Fenchel-Young Loss (`01_SIL.py`, `03_SRL.py`):**
   - Proper perturbed optimization implementation
   - Consider using existing libraries or implementing from scratch

4. **Dataset Loading (`utils/utils.py`):**
   - `load_VSP_dataset()` - Load training data
   - Format depends on your problem instance structure

## Performance Tips

1. **GPU Acceleration:**
   - Move models to GPU: `model.cuda()`
   - Ensure data tensors are on GPU
   - Use mixed precision training for faster computation

2. **Batch Processing:**
   - Increase `BATCH_SIZE` for better GPU utilization
   - Balance memory usage vs. speed

3. **Parallel Environment Evaluation:**
   - Implement parallel episode collection
   - Use multiprocessing for CPU-bound operations

4. **Replay Buffer:**
   - Larger buffer improves sample efficiency
   - Adjust `rb_capacity` based on available memory

## Troubleshooting

**Import Errors:**
- Ensure all dependencies are installed
- Check PyTorch Geometric installation matches your CUDA version

**Memory Issues:**
- Reduce batch size
- Reduce replay buffer capacity
- Use gradient accumulation

**Slow Training:**
- Use GPU acceleration
- Enable multiprocessing for episode collection
- Reduce number of perturbation samples

**Convergence Issues:**
- Adjust learning rates
- Modify perturbation schedules
- Check critic loss stability

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{hoppe2025structured,
  title={Structured Reinforcement Learning for Combinatorial Decision-Making},
  author={Hoppe, Heiko and Baty, L{\'e}o and Bouvier, Louis and Parmentier, Axel and Schiffer, Maximilian},
  journal={arXiv preprint},
  year={2025}
}
```

## License

See the repository root LICENSE.txt file for license information.

## Contact

For questions or issues with the Python implementation, please open an issue on the GitHub repository.

For questions about the original Julia implementation or the paper, please contact the authors.
