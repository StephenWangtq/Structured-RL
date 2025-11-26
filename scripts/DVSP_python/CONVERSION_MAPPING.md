# Julia to Python Conversion Mapping

This document provides a detailed mapping of how each Julia file was converted to Python.

## File Mapping

| Julia File | Python File | Lines (Julia) | Lines (Python) | Status |
|------------|-------------|---------------|----------------|--------|
| `utils/utils.jl` | `utils/utils.py` | 271 | 461 | ✅ Complete |
| `utils/policy.jl` | `utils/policy.py` | 503 | 845 | ✅ Complete |
| `00_setup.jl` | `00_setup.py` | 41 | 144 | ✅ Complete |
| `01_SIL.jl` | `01_SIL.py` | 159 | 399 | ✅ Complete |
| `02_PPO.jl` | `02_PPO.py` | 218 | 444 | ✅ Complete |
| `03_SRL.jl` | `03_SRL.py` | 209 | 453 | ✅ Complete |
| `04_plots.jl` | `04_plots.py` | 201 | 329 | ✅ Complete |
| N/A | `requirements.txt` | N/A | 27 | ✅ New |
| N/A | `README.md` | N/A | 285 | ✅ New |
| N/A | `IMPLEMENTATION_NOTES.md` | N/A | 454 | ✅ New |
| **Total** | | **1,602** | **3,841** | |

## Component Mapping

### 1. Neural Networks

#### Actor Network
**Julia (Flux):**
```julia
Chain(Dense(14 => 1; bias=false), vec)
```

**Python (PyTorch):**
```python
class ActorModel(nn.Module):
    def __init__(self, input_dim: int = 14):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)
```

#### Critic GNN
**Julia (GraphNeuralNetworks.jl):**
```julia
struct critic_GNN
    layers::NamedTuple
end

function critic_GNN(node_features, edge_features)
    layers = (
        g1=NNConv(node_features => 15, Dense(edge_features, node_features), celu),
        g2=NNConv(15 => 10, Dense(edge_features, 15), celu),
        # ... more layers
    )
    return critic_GNN(layers)
end
```

**Python (PyTorch Geometric):**
```python
class critic_GNN(nn.Module):
    def __init__(self, node_features: int, edge_features: int):
        super().__init__()
        self.g1_nn = nn.Linear(edge_features, node_features)
        self.g1 = NNConv(node_features, 15, self.g1_nn, aggr='mean')
        self.g2_nn = nn.Linear(edge_features, 15)
        self.g2 = NNConv(15, 10, self.g2_nn, aggr='mean')
        # ... more layers
```

### 2. Optimizers

**Julia (Flux):**
```julia
opt_actor = Optimiser(ClipValue(1e-3), Adam(lr))
opt_critic = Optimiser(ClipValue(1e-3), Adam(lr * critic_factor))
```

**Python (PyTorch):**
```python
opt_actor = optim.Adam(actor_model.parameters(), lr=lr)
opt_critic = optim.Adam(critic_model.parameters(), lr=lr * critic_factor)

# Gradient clipping applied during training:
torch.nn.utils.clip_grad_value_(actor_model.parameters(), 1e-3)
torch.nn.utils.clip_grad_value_(critic_model.parameters(), 1e-3)
```

### 3. Loss Functions

#### PPO Loss
**Julia:**
```julia
function J_PPO(π::CombinatorialACPolicy, batch, clip, sigmaF_average)
    old_probs = [logdensityof(p(thetas[j], sigmaF_average), etas[j]) for j in eachindex(batch)]
    new_probs = [logdensityof(p(π.actor_model(embeddings[j]), sigmaF_average), etas[j]) for j in eachindex(batch)]
    ratio_unclipped = [exp(new_probs[j] - old_probs[j]) for j in eachindex(batch)]
    ratio_clipped = clamp.(ratio_unclipped, 1 - clip, 1 + clip)
    return mean(min.(ratio_unclipped .* advantages, ratio_clipped .* advantages))
end
```

**Python:**
```python
def J_PPO(policy, batch, clip, sigmaF_average):
    # Calculate log probabilities
    old_log_probs = torch.stack([
        policy.p(thetas[j], sigmaF_average).log_prob(etas[j])
        for j in range(len(batch))
    ])
    new_log_probs = torch.stack([
        policy.p(policy.actor_model(embeddings[j]), sigmaF_average).log_prob(etas[j])
        for j in range(len(batch))
    ])
    
    # Calculate ratio
    ratio = torch.exp(new_log_probs - old_log_probs)
    ratio_clipped = torch.clamp(ratio, 1 - clip, 1 + clip)
    
    # PPO loss
    loss = torch.mean(torch.min(ratio * advantages, ratio_clipped * advantages))
    return loss
```

#### Huber Loss
**Julia:**
```julia
function huber_GNN(π, graphs, s_c, edge_features, critic_target, δ; kwargs...)
    new_critic = [-sum(π.critic_model(graphs[j], Float32.(s_c[j]), edge_features[j])) for j in eachindex(critic_target)]
    error = new_critic .- critic_target
    quadratic = 0.5 .* error .^ 2
    linear = δ .* (abs.(error) .- 0.5 .* δ)
    return mean(ifelse.(abs.(error) .<= δ, quadratic, linear))
end
```

**Python:**
```python
def huber_GNN(policy, graphs, s_c, edge_features, critic_target, delta, **kwargs):
    new_critic = torch.stack([
        -policy.critic_model(graphs[j], s_c[j], edge_features[j]).sum()
        for j in range(len(graphs))
    ])
    
    error = new_critic - critic_target
    quadratic = 0.5 * error ** 2
    linear = delta * (torch.abs(error) - 0.5 * delta)
    loss = torch.where(torch.abs(error) <= delta, quadratic, linear)
    
    return loss.mean()
```

### 4. Data Structures

#### Policy Class
**Julia:**
```julia
mutable struct CombinatorialACPolicy{M1,M2,P,CO,R<:AbstractRNG,S<:Union{Int,Nothing}}
    actor_model::M1
    critic_model::M2
    p::P
    CO_layer::CO
    rng::R
    seed::S
end
```

**Python:**
```python
class CombinatorialACPolicy:
    def __init__(
        self,
        actor_model: nn.Module,
        critic_model: Optional[nn.Module],
        p: Optional[Callable],
        CO_layer: Callable,
        seed: int = 0,
    ):
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.p = p if p is not None else self._default_p
        self.CO_layer = CO_layer
        self.seed = seed
        self.rng = np.random.default_rng(seed)
```

#### Episode Data
**Julia (Named Tuple):**
```julia
(;
    state,
    s,
    θ,
    η,
    a,
    next_state,
    next_s,
    reward,
    s_c,
    next_s_c,
    Rₜ=0.0,
    adv=0.0,
)
```

**Python (Dictionary):**
```python
{
    'state': state,
    's': s,
    'theta': theta,
    'eta': eta,
    'a': a,
    'next_state': next_state,
    'next_s': next_s,
    'reward': reward,
    's_c': s_c,
    'next_s_c': next_s_c,
    'Rₜ': 0.0,
    'adv': 0.0,
}
```

### 5. Random Number Generation

**Julia:**
```julia
using Random
rng = MersenneTwister(seed)
Random.seed!(rng, seed)
```

**Python:**
```python
import numpy as np
import torch

torch.manual_seed(seed)
np.random.seed(seed)
rng = np.random.default_rng(seed)
```

### 6. File I/O

#### Saving Models and Data
**Julia (JLD2):**
```julia
using JLD2
jldsave(
    joinpath(logdir, "dvsp_SIL_training_results.jld2");
    model=SIL_model,
    train_rew=SIL_train,
    val_rew=SIL_val,
)
```

**Python (pickle + PyTorch):**
```python
import pickle
import torch

# Save data with pickle
results = {
    'model_state_dict': model.state_dict(),
    'train_rew': train_rew,
    'val_rew': val_rew,
}
with open(logdir / "dvsp_SIL_training_results.pkl", 'wb') as f:
    pickle.dump(results, f)

# Save model with PyTorch
torch.save(model.state_dict(), logdir / "dvsp_SIL_model.pt")
```

#### Loading Data
**Julia:**
```julia
using JLD2
log_sl = load(joinpath(logdir, "dvsp_SIL_training_results.jld2"))
```

**Python:**
```python
with open(logdir / "dvsp_SIL_training_results.pkl", 'rb') as f:
    log_sl = pickle.load(f)
```

### 7. Plotting

#### Line Plot
**Julia (Plots.jl):**
```julia
using Plots
plt = plot(data[1]; label="SIL", color=:blue, linewidth=4)
plot!(plt, data[2]; label="PPO", color=:green, linewidth=4)
savefig(plt, "dvsp_training_plot.pdf")
```

**Python (matplotlib):**
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(data[0], label='SIL', color='blue', linewidth=3)
ax.plot(data[1], label='PPO', color='green', linewidth=3)
ax.legend()
plt.savefig("dvsp_training_plot.pdf")
```

#### Boxplot
**Julia (StatsPlots):**
```julia
using StatsPlots
boxplot!(
    fill(positions[i], length(data[i])),
    data[i];
    fillcolor=colors[i],
    alpha=0.6,
)
```

**Python (matplotlib):**
```python
import matplotlib.pyplot as plt

bp = ax.boxplot(
    data,
    positions=positions,
    patch_artist=True,
)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
```

### 8. Feature Scaling

**Julia (StatsBase):**
```julia
using StatsBase
dt = fit(ZScoreTransform, X; dims=2)
X_scaled = StatsBase.transform(dt, X)
```

**Python (scikit-learn):**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X.T)  # Note: sklearn expects (n_samples, n_features)
X_scaled = scaler.transform(X.T).T
```

## Package Equivalences

| Julia Package | Python Package | Purpose |
|---------------|----------------|---------|
| Flux | PyTorch | Deep learning framework |
| GraphNeuralNetworks | PyTorch Geometric | Graph neural networks |
| Plots | matplotlib | Visualization |
| StatsPlots | seaborn | Statistical plotting |
| JLD2 | pickle / torch.save | Data persistence |
| Random | numpy.random / torch | Random number generation |
| StatsBase | scikit-learn | Statistical utilities |
| Distributions | torch.distributions | Probability distributions |
| Gurobi | gurobipy | Optimization solver |

## Key Differences

### 1. Broadcasting
**Julia:** Implicit with `.` operator
```julia
x .+ y
```

**Python:** Explicit with NumPy/PyTorch
```python
x + y  # NumPy/PyTorch handle broadcasting automatically
```

### 2. Indexing
**Julia:** 1-based
```julia
array[1]  # First element
```

**Python:** 0-based
```python
array[0]  # First element
```

### 3. Function Arguments
**Julia:** Semicolon separates positional and keyword args
```julia
function f(x, y; z=1, w=2)
```

**Python:** All after * are keyword-only
```python
def f(x, y, z=1, w=2):
```

### 4. Multiple Dispatch
**Julia:** Has multiple dispatch
```julia
f(x::Int) = x + 1
f(x::Float64) = x + 1.0
```

**Python:** No multiple dispatch (single dispatch available)
```python
def f(x):
    if isinstance(x, int):
        return x + 1
    elif isinstance(x, float):
        return x + 1.0
```

### 5. Type Annotations
**Julia:** Required for performance, part of dispatch
```julia
function f(x::Int)::Float64
```

**Python:** Optional, for documentation only
```python
def f(x: int) -> float:
```

## Testing Parity

To verify the conversion maintains functionality:

1. **Network Output Parity:**
   - Load same input in both Julia and Python
   - Compare actor output
   - Compare critic output
   - Should match within floating-point tolerance

2. **Loss Function Parity:**
   - Same batch data
   - Same model weights
   - Compare loss values
   - Should match exactly

3. **Training Step Parity:**
   - Same initial weights
   - Same data
   - Same hyperparameters
   - Compare gradients and updates
   - Should produce similar trajectories

## Performance Comparison

Expected relative performance (Julia baseline = 1.0):

| Operation | Julia | Python | Notes |
|-----------|-------|--------|-------|
| Actor forward pass | 1.0x | 0.9-1.1x | PyTorch is well-optimized |
| GNN forward pass | 1.0x | 0.8-1.2x | Depends on graph size |
| Episode generation | 1.0x | 0.5-1.0x | CO solver is bottleneck |
| Training iteration | 1.0x | 0.8-1.2x | Overall similar performance |

## Known Limitations

1. **Fenchel-Young Loss:** Placeholder implementation, needs proper perturbed optimization
2. **Environment Interface:** Domain-specific, needs custom implementation
3. **CO Solver:** Needs integration with optimization solver (Gurobi, OR-Tools, etc.)
4. **Dataset Loading:** Needs implementation based on instance format

## Future Work

- Implement differentiable optimization for Fenchel-Young loss
- Add distributed training support
- Optimize episode collection with multiprocessing
- Add tensorboard logging
- Implement automatic hyperparameter tuning
- Add more comprehensive test suite

## Verification Checklist

- [x] All files converted
- [x] Syntax checking passed
- [x] Documentation complete
- [x] Implementation notes provided
- [ ] Unit tests created
- [ ] Integration tests run
- [ ] Performance benchmarked
- [ ] Results validated against Julia version

## Conclusion

The conversion from Julia to Python is **structurally complete** with all core algorithms implemented. The remaining work is:

1. Domain-specific implementations (environment, CO solver)
2. Fenchel-Young loss implementation or approximation
3. Testing and validation
4. Performance optimization

The converted code maintains the same architecture, algorithms, and design patterns as the original Julia implementation while following Python/PyTorch best practices.
