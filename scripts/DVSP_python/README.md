# DVSP Python Implementation

This directory contains a Python/PyTorch conversion of the Julia DVSP (Dynamic Vehicle Scheduling Problem) code from the Structured-RL repository.

## Overview

This implementation provides three reinforcement learning algorithms for solving the Dynamic Vehicle Scheduling Problem:

1. **SIL (Supervised Imitation Learning)** - Uses Fenchel-Young loss with perturbed optimization
2. **PPO (Proximal Policy Optimization)** - Policy gradient method with clipped surrogate objective
3. **SRL (Structured Reinforcement Learning)** - Combines Fenchel-Young loss with Q-learning

## Directory Structure

```
DVSP_python/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── 00_setup.py                  # Dataset setup and baseline evaluation
├── 01_SIL.py                    # Supervised Imitation Learning
├── 02_PPO.py                    # Proximal Policy Optimization
├── 03_SRL.py                    # Structured Reinforcement Learning
├── 04_plots.py                  # Visualization and plotting
└── utils/
    ├── __init__.py              # Module initialization
    ├── policy.py                # Policy, GNN critic, losses
    └── utils.py                 # Environment, evaluation, utilities
```

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA (optional, for GPU acceleration)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Packages

The main dependencies are:

- **torch** (>=2.0.0) - Deep learning framework
- **torch-geometric** (>=2.3.0) - Graph neural networks
- **numpy** - Numerical computations
- **matplotlib** - Plotting
- **scikit-learn** - Data preprocessing
- **seaborn** - Statistical visualization

## Usage

### Important Note

⚠️ **This is a template conversion from Julia to Python.** 

The code provides the correct structure and PyTorch implementations of the neural networks and training loops, but requires additional dependencies that are not yet available in Python:

1. **DynamicVehicleRouting** - The DVSP environment (Julia-only currently)
2. **DecisionFocusedLearningBenchmarks** - Optimization solvers
3. **InferOpt** - Fenchel-Young loss implementation

To make this code fully functional, you will need to either:
- Port these Julia packages to Python, or
- Implement equivalent functionality using existing Python packages

### Running the Code (Once Dependencies Are Available)

1. **Setup and baseline evaluation:**
   ```bash
   python 00_setup.py
   ```

2. **Train with SIL:**
   ```bash
   python 01_SIL.py
   ```

3. **Train with PPO:**
   ```bash
   python 02_PPO.py
   ```

4. **Train with SRL:**
   ```bash
   python 03_SRL.py
   ```

5. **Generate plots:**
   ```bash
   python 04_plots.py
   ```

## Algorithm Details

### Network Architecture

**Actor Network:**
- Simple linear layer: `Linear(14, 1, bias=False)`
- Input: 14-dimensional state features
- Output: Single cost value per postponable request

**Critic Network (GNN):**
- Multi-layer Graph Neural Network
- Layers: NNConv (with edge features) → GraphConv → Global pooling → MLP
- Input: Node features (15-dim) and edge features (duration matrix)
- Output: State or action value estimate

### Training Hyperparameters

**SIL:**
- Epochs: 400
- Optimizer: Adam with default learning rate
- Loss: Fenchel-Young with ε=1e-2, 20 perturbation samples

**PPO:**
- Episodes: 400
- Collection steps: 20 episodes per iteration
- Update epochs: 100
- Clip parameter: 0.2
- Learning rate: 1e-3 → 5e-4 (linear decay)
- σ_F: 0.5 → 0.05 (linear decay)
- Advantage method: TD(n)

**SRL:**
- Gradient steps: 400
- Collection steps: 20 episodes per iteration
- Update iterations: 100 per collection
- Batch size: 4
- σ_F: 0.1 (fixed)
- σ_B: 1.0 → 0.1 (linear decay)
- Temperature: 10.0 → 10.0 (fixed in this case)
- Learning rate: 1e-3 → 2e-4 (linear decay)

## Key Implementation Details

### Loss Functions

1. **Fenchel-Young Loss** (SIL, SRL actor):
   - Requires InferOpt implementation
   - Uses perturbed optimization for gradient estimation
   - Template provided in code

2. **PPO Loss** (PPO actor):
   - Clipped surrogate objective
   - Policy ratio: π_new(a|s) / π_old(a|s)
   - Clipping range: [1-ε, 1+ε]

3. **Huber Loss** (Critic):
   - Robust loss for value function training
   - Threshold δ = 1.0

### Replay Buffer

- **PPO**: Optional (default: enabled)
- **SRL**: Required
- Capacity: 6000-120000 transitions
- Sampling: Uniform random

### Perturbation

- **σ_F** (forward): Controls exploration during episode collection
- **σ_B** (backward): Controls sampling for SRL target actions
- Both use multivariate normal distribution

## Files Description

### Main Scripts

- **00_setup.py**: Loads dataset, creates environments, evaluates baselines (greedy, expert)
- **01_SIL.py**: Supervised learning from expert demonstrations using Fenchel-Young loss
- **02_PPO.py**: On-policy reinforcement learning with clipped objective
- **03_SRL.py**: Structured RL combining imitation and Q-learning
- **04_plots.py**: Creates training curves and comparative boxplots

### Utility Modules

- **utils/utils.py**:
  - `RLDVSPEnv`: Environment wrapper
  - `evaluate_policy()`: Policy evaluation
  - `expert_evaluation()`: Expert baseline
  - Data loading and saving functions

- **utils/policy.py**:
  - `CombinatorialACPolicy`: Main policy class
  - `CriticGNN`: Graph neural network for value estimation
  - `PPO_episodes()`, `SRL_episodes()`: Episode generation
  - `J_PPO()`, `J_SRL()`: Loss computation
  - `rb_add()`, `rb_sample()`: Replay buffer operations

## Differences from Julia Version

### Syntax and Libraries

| Julia | Python/PyTorch |
|-------|----------------|
| `Flux.jl` | `torch.nn` |
| `GraphNeuralNetworks.jl` | `torch_geometric` |
| `Flux.Optimise` | `torch.optim` |
| `JLD2.jldsave()` | `torch.save()` |
| `Random.MersenneTwister` | `np.random.RandomState` |
| `StatsBase.ZScoreTransform` | `sklearn.preprocessing.StandardScaler` |
| `Plots.jl` | `matplotlib`, `seaborn` |

### Implementation Notes

- Python uses 0-based indexing (Julia uses 1-based)
- PyTorch requires explicit `backward()` calls (Flux uses Zygote AD)
- Gradient clipping implemented with `torch.nn.utils.clip_grad_value_()`
- Named tuples replaced with `NamedTuple` from typing module

## Reproducibility

To reproduce paper results:
1. Use seeds: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`
2. Run all three algorithms for each seed
3. Average results across seeds
4. Use the same dataset split as the Julia version

## Performance Considerations

- **GPU acceleration**: Models automatically use CUDA if available
- **Batch processing**: Adjust `batch_size` based on GPU memory
- **Episode collection**: Can be parallelized (not implemented in base version)
- **GNN forward pass**: May be slower than Julia version initially

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

## Contributing

This is a conversion template. Contributions to make it fully functional are welcome:

1. Port or implement missing dependencies
2. Add tests and validation
3. Optimize performance
4. Add additional features (e.g., distributed training)

## License

Same as the original Structured-RL repository.

## Support

For questions about the original algorithm, refer to the paper and Julia implementation.

For questions about this Python conversion, please open an issue in the repository.
