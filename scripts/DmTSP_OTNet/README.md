# OT-Net for Dynamic Multi-Traveling Salesman Problem (D-mTSP)

This directory contains the implementation of **OT-Net (Optimal Transport Dispatch Network)** for solving the Dynamic Multi-Traveling Salesman Problem.

## Problem Formulation

### D-mTSP Overview

The Dynamic Multi-Traveling Salesman Problem (D-mTSP) is characterized by:

- **M homogeneous agents (vehicles)** starting at a depot
- **Dynamic customer requests** arriving over time following a Poisson process
- Each request v_j = (x_j, τ_j) has a location and arrival time
- **Event-driven decisions**: At each request arrival, assign new tasks to agents

### State Space

At decision time t_k, the state S_k = (t_k, A_k, V_k) includes:

**Agent Part:**
- Current location of each agent
- Task queue Q_i(t_k) for each agent i
- Estimated completion time

**Task Part:**
- New requests V_new(t_k) just arrived
- Pending requests V_pending(t_k) not yet assigned

### Action Space

Assignment matrix Y_k ∈ {0,1}^(M×K_k):
- Y_k(i,j) = 1 means task j assigned to agent i
- Constraint: Each task assigned to exactly one agent

### Objective Function

Minimize weighted combination:
```
L(π,V) = α·max_i C_i(π,V) + β·(1/|V|)·Σ_j w_j(π,V)
```

where:
- C_i: Completion time (makespan) of agent i
- w_j: Waiting time of task j
- α, β: Weight parameters

Incremental reward:
```
r_k = -(α·ΔMakespan_k + β·ΔTotalWaiting_k)
```

## OT-Net Architecture

### 1. GNN Encoder (φ_w)

**Purpose:** Learn structured representations of agents and tasks

**Input Features:**

*Agent features (8-dim):*
- Position (x, y): 2
- Queue length: 1
- Estimated completion time: 1
- Current time: 1
- Historical load: 1
- Distance to depot: 1
- Time to finish route: 1

*Task features (6-dim):*
- Position (x, y): 2
- Arrival time: 1
- Current waiting time: 1
- Priority (new vs pending): 1
- Distance to depot: 1

**Architecture:**
- Multi-layer Graph Isomorphism Network (GIN) / Graph Attention Network (GAT)
- Message passing between agent-task and task-task nodes
- Outputs: h_i^agent and h_j^task embeddings

### 2. OT Parameter Generation

**Scoring Matrix:**
```
θ_ij = -||q_i - k_j||^2
```
where:
- q_i = W_q · h_i^agent (query projection)
- k_j = W_k · h_j^task (key projection)

**Supply Prior:**
```
b_i = softmax(MLP(h_i^agent))
```
Represents each agent's capacity/availability

**Demand:**
```
a_j = 1/K (uniform over tasks)
```

### 3. OT CO-Layer

**Entropic-Regularized Optimal Transport:**
```
min_{T∈Π(b,a)} <C,T> + ε·H(T)
```

where:
- T: Transport matrix (soft assignment)
- C = -θ (cost matrix)
- ε: Entropic regularization parameter
- H(T): Entropy for smoothness

**Sinkhorn Algorithm:**
Iteratively solves the OT problem using:
```
K = exp(-C/ε)
u = b / (K @ v)
v = a / (K^T @ u)
T = diag(u) @ K @ diag(v)
```

### 4. Training

**Loss Function:**
Policy gradient (REINFORCE) with:
- Discounted returns
- Baseline normalization
- Entropy regularization for exploration

**Optimization:**
- Adam optimizer
- Gradient clipping
- Learning rate scheduling

## Installation

### Prerequisites
- Python 3.8 or higher
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Setup

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install PyTorch Geometric for GNN support:
```bash
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

Note: Adjust CUDA version in the URL as needed.

## Usage

### Training OT-Net

Run the training script:

```bash
python train_otnet.py
```

**Configuration parameters** (edit in script):
```python
# Environment
NUM_AGENTS = 5
TIME_HORIZON = 100.0
REQUEST_RATE = 0.5
ALPHA = 0.5  # makespan weight
BETA = 0.5   # waiting time weight

# Model
HIDDEN_DIM = 64
NUM_GNN_LAYERS = 3
EPSILON = 0.1  # OT regularization

# Training
NUM_EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
GAMMA = 0.99  # discount factor
```

### Output

Training produces:
- `logs/otnet_model.pt` - Trained model weights
- `logs/otnet_results.pkl` - Training history and metrics
- `plots/otnet_training.pdf` - Training curves

### Using Trained Model

```python
from utils.otnet import OTNetPolicy
from utils.environment import DmTSPInstance, DmTSPEnv
from utils.utils import evaluate_policy

# Load model
policy = OTNetPolicy(...)
policy.load_state_dict(torch.load('logs/otnet_model.pt'))

# Create environment
instance = DmTSPInstance(
    num_agents=5,
    depot_location=np.array([50.0, 50.0]),
    time_horizon=100.0,
    request_rate=0.5,
    seed=42,
)
env = DmTSPEnv(instance, alpha=0.5, beta=0.5)

# Evaluate
results = evaluate_policy(policy, env, num_episodes=10)
print(f"Mean reward: {results['mean_reward']:.2f}")
```

## Architecture Details

### File Structure

```
DmTSP_OTNet/
├── utils/
│   ├── __init__.py          # Package initialization
│   ├── environment.py       # D-mTSP environment
│   ├── otnet.py            # OT-Net model architecture
│   └── utils.py            # Helper functions
├── train_otnet.py          # Training script
├── requirements.txt        # Dependencies
└── README.md              # This file
```

### Key Classes

**Environment:**
- `DmTSPInstance`: Problem instance generator
- `DmTSPEnv`: Event-driven environment
- `DmTSPState`: State representation
- `Agent`: Vehicle agent
- `CustomerRequest`: Task/request

**Model:**
- `OTNetPolicy`: Complete policy (encoder + OT layer)
- `OTNetEncoder`: GNN-based state encoder
- `OTLayer`: Optimal transport layer with Sinkhorn

**Utilities:**
- `extract_agent_features()`: Feature extraction for agents
- `extract_task_features()`: Feature extraction for tasks
- `build_graph_edges()`: Graph construction for GNN
- `evaluate_policy()`: Policy evaluation
- `GreedyPolicy`: Greedy baseline
- `RandomPolicy`: Random baseline

## Theoretical Background

### Why Optimal Transport?

OT provides a principled way to model the assignment problem:

1. **Structured Output:** Soft assignments preserve task-agent affinities
2. **Differentiability:** Sinkhorn iterations are differentiable (end-to-end learning)
3. **Efficiency:** O(MKT) complexity where T is Sinkhorn iterations
4. **Flexibility:** Regularization parameter ε controls assignment smoothness

### GNN for Spatial Reasoning

Graph Neural Networks capture:
- **Agent-task relationships:** Distance, capacity, urgency
- **Task-task relationships:** Clustering, spatial patterns
- **Multi-hop information:** Beyond local neighborhoods

### Policy Gradient Learning

REINFORCE algorithm:
- Sample trajectories from current policy
- Compute returns (future rewards)
- Update policy to increase probability of high-reward actions
- Entropy bonus encourages exploration

## Performance Tips

### GPU Acceleration
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
policy = policy.to(device)
```

### Hyperparameter Tuning

**For better makespan:**
- Increase α (makespan weight)
- Larger hidden_dim for more capacity
- More GNN layers for longer-range reasoning

**For better waiting time:**
- Increase β (waiting time weight)
- Lower ε (sharper OT assignments)
- Include waiting time in features

**For faster training:**
- Reduce batch_size
- Reduce num_sinkhorn_iters (e.g., 20-30)
- Disable GNN (use_gnn=False for MLP baseline)

### Debugging

Enable logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Visualize assignments:
```python
import matplotlib.pyplot as plt
plt.imshow(assignment_matrix, cmap='Blues')
plt.colorbar()
plt.show()
```

## Extending OT-Net

### Custom Features

Add new features in `utils/utils.py`:

```python
def extract_agent_features(state, depot):
    # Add your features
    custom_feature = compute_custom_feature(state)
    agent_feat.append(custom_feature)
    return torch.tensor(features)
```

Update `AGENT_FEATURE_DIM` accordingly.

### Alternative OT Solvers

Replace Sinkhorn with other OT solvers:

```python
from scipy.optimize import linear_sum_assignment

def hungarian_assignment(C):
    row_ind, col_ind = linear_sum_assignment(C)
    # Convert to assignment matrix
    ...
```

### Multi-Objective Optimization

Extend objective function:

```python
def calculate_total_cost(self):
    makespan = self._calculate_makespan()
    waiting = self._calculate_waiting_time()
    distance = self._calculate_total_distance()
    
    return self.alpha * makespan + self.beta * waiting + self.gamma * distance
```

## Troubleshooting

### ImportError: torch_geometric

Solution: Install PyTorch Geometric or set `USE_GNN=False` in training script.

### CUDA out of memory

Solutions:
- Reduce batch_size
- Reduce hidden_dim
- Use CPU: `device = 'cpu'`

### Poor convergence

Solutions:
- Adjust learning rate (try 5e-4 or 2e-3)
- Increase entropy coefficient (e.g., 0.05)
- Check baseline performance first
- Verify feature normalization

### Numerical instability in Sinkhorn

Solutions:
- Increase epsilon (e.g., 0.2)
- Add numerical stabilization: `+ 1e-8`
- Reduce num_sinkhorn_iters

## Citation

If you use OT-Net in your research, please cite:

```bibtex
@article{hoppe2025structured,
  title={Structured Reinforcement Learning for Combinatorial Decision-Making},
  author={Hoppe, Heiko and Baty, L{\'e}o and Bouvier, Louis and Parmentier, Axel and Schiffer, Maximilian},
  journal={arXiv preprint},
  year={2025}
}
```

## References

1. Cuturi, M. (2013). Sinkhorn distances: Lightspeed computation of optimal transport.
2. Kool, W., et al. (2019). Attention, Learn to Solve Routing Problems!
3. Xu, K., et al. (2019). How Powerful are Graph Neural Networks?

## License

See the repository root LICENSE.txt file.

## Contact

For questions or issues:
- Open an issue on GitHub
- Check existing documentation
- Refer to the paper for theoretical details
