# DVSP Python Implementation Notes

## Conversion Summary

This document provides detailed information about the Julia to Python conversion of the DVSP (Dynamic Vehicle Scheduling Problem) reinforcement learning code.

## What Has Been Converted

### ✅ Complete Conversions

1. **File Structure**
   - All 7 source files converted
   - Directory structure matches original
   - Python package structure with `__init__.py`

2. **Neural Network Architectures**
   - Actor: `Dense(14 => 1, bias=false)` → `nn.Linear(14, 1, bias=False)`
   - Critic GNN: Full graph neural network with NNConv, GCNConv, and pooling layers
   - Converted from Flux.jl to PyTorch
   - Converted from GraphNeuralNetworks.jl to PyTorch Geometric

3. **Training Algorithms**
   - **SIL**: Supervised Imitation Learning framework
   - **PPO**: Proximal Policy Optimization with clipping
   - **SRL**: Structured Reinforcement Learning with critic guidance

4. **Key Components**
   - Policy class (`CombinatorialACPolicy`)
   - Episode generation functions
   - Replay buffer implementation
   - Advantage estimation (TD(n) and TD(1))
   - Huber loss for critic training
   - PPO clipped objective loss

5. **Visualization**
   - Training progress line plots
   - Boxplot comparisons with log-scale transformation
   - Results aggregation and plotting

6. **Infrastructure**
   - Requirements.txt with all dependencies
   - Comprehensive README with usage instructions
   - Updated .gitignore for Python projects

## What Needs to Be Implemented

### 🔧 Required Domain-Specific Implementations

These components are placeholders and require domain-specific implementation based on your VSP problem:

#### 1. Environment Interface (`utils/utils.py`)

**`RLDVSPEnv` class methods:**

```python
def _read_vsp_instance(self, path: str) -> DVSPInstance:
    """
    Read VSP instance from file.
    
    TODO: Implement based on your instance file format.
    - Parse instance file (e.g., .txt, .json, .xml)
    - Extract locations, durations, time windows, etc.
    - Return DVSPInstance object
    """
    
def _create_dvsp_env(self, instance, seed, max_requests):
    """
    Create DVSP environment.
    
    TODO: Implement environment state management.
    - Initialize environment with instance
    - Set up request generation
    - Handle dynamic aspects (new requests over time)
    """
    
def _compute_features(self) -> np.ndarray:
    """
    Compute actor features from current state.
    
    TODO: Extract 14-dimensional feature vector.
    - Time-related features
    - Location features
    - Request features
    - Vehicle availability
    """
    
def _compute_critic_features(self) -> np.ndarray:
    """
    Compute critic features (15-dimensional).
    
    TODO: Extract graph node features.
    - Node-level features for GNN
    - State embeddings for value estimation
    """
```

**Required helper functions:**

```python
def prize_collecting_vsp(theta, instance, model_builder, **kwargs):
    """
    Solve prize-collecting VSP (CO layer).
    
    TODO: Implement optimization solver interface.
    - Build optimization model (Gurobi, SCIP, OR-Tools)
    - Set objective based on learned weights (theta)
    - Solve and return routes
    
    Example with Gurobi:
        model = Model()
        # Add variables for routes
        # Add constraints (vehicle capacity, time windows, etc.)
        # Set objective: sum(theta[i] * x[i]) + routing costs
        model.optimize()
        return extract_routes(model)
    """
```

#### 2. Fenchel-Young Loss (`01_SIL.py`, `03_SRL.py`)

The current implementation uses a placeholder. Two options to implement:

**Option A: Use Perturbation-based Approximation**

```python
class FenchelYoungLoss:
    def __call__(self, theta, y_true, instance, **kwargs):
        """
        Approximate FY loss using perturbed optimization.
        
        1. Sample noise: ε_1, ..., ε_n ~ N(0, σ²I)
        2. For each sample: y_i = argmax_y <θ + ε_i, y> - Ω(y)
        3. Average: ŷ = (1/n) Σ y_i
        4. Loss: <θ, ŷ - y_true> - Ω(ŷ) + Ω(y_true)
        """
        # Implementation steps:
        n_samples = 20
        epsilon = 1e-2
        
        # Sample perturbations
        perturbations = torch.randn(n_samples, len(theta)) * epsilon
        
        # Solve perturbed problems
        y_samples = []
        for eps in perturbations:
            theta_perturbed = theta + eps
            y = optimization_fyl(theta_perturbed, instance, **kwargs)
            y_samples.append(y)
        
        # Average solutions
        y_avg = torch.mean(torch.stack(y_samples), dim=0)
        
        # Compute FY loss
        loss = (
            torch.dot(theta, y_avg - y_true) 
            - h_fyl(y_avg, instance=instance) 
            + h_fyl(y_true, instance=instance)
        )
        
        return loss
```

**Option B: Use Differentiable Optimization Library**

Consider using:
- `cvxpylayers` for convex optimization layers
- `qpth` for quadratic programming layers
- Custom implementation of implicit differentiation

#### 3. Dataset Loading (`utils/utils.py`)

```python
def load_VSP_dataset(instance_dir, model_builder):
    """
    Load VSP dataset for supervised learning.
    
    TODO: Implement dataset creation.
    1. For each instance file:
       - Load instance
       - Solve with expert/optimal solver
       - Extract features (X) and solutions (Y)
    2. Return (X, Y) where:
       - X[i] = (features, instance) tuple
       - Y[i] = optimal solution matrix
    """
```

#### 4. Expert Solver (`utils/utils.py`)

```python
def anticipative_solver(env, model_builder):
    """
    Solve with full knowledge (expert solution).
    
    TODO: Implement offline VSP solver.
    - Use all future information
    - Solve complete problem optimally
    - Return list of routes for all epochs
    """
```

## Integration with Existing Solvers

### Gurobi Integration Example

```python
from gurobipy import Model, GRB, quicksum

def prize_collecting_vsp(theta, instance, model_builder=None, **kwargs):
    """Prize-collecting VSP solver using Gurobi."""
    m = Model("PCVSP")
    m.setParam('OutputFlag', 0)  # Suppress output
    
    n = len(instance.instance.duration)
    duration = instance.instance.duration
    
    # Variables: x[i,j] = 1 if edge (i,j) is used
    x = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                x[i,j] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
    
    # Variables: y[i] = 1 if request i is served
    y = {}
    for i in range(n):
        if instance.is_postponable[i]:
            y[i] = m.addVar(vtype=GRB.BINARY, name=f"y_{i}")
    
    # Objective: maximize prizes - costs
    obj = quicksum(
        theta[i] * y[i] for i in range(n) if instance.is_postponable[i]
    ) - quicksum(
        duration[i,j] * x[i,j] for i in range(n) for j in range(n) if i != j
    )
    m.setObjective(obj, GRB.MAXIMIZE)
    
    # Constraints
    # Flow conservation, capacity, time windows, etc.
    # TODO: Add problem-specific constraints
    
    m.optimize()
    
    # Extract routes from solution
    routes = extract_routes_from_gurobi(m, x, n)
    return routes
```

### OR-Tools Integration Example

```python
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def prize_collecting_vsp(theta, instance, model_builder=None, **kwargs):
    """Prize-collecting VSP solver using OR-Tools."""
    manager = pywrapcp.RoutingIndexManager(
        len(instance.instance.duration),
        num_vehicles,
        depot_index
    )
    routing = pywrapcp.RoutingModel(manager)
    
    # Define cost callback
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return instance.instance.duration[from_node, to_node]
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # Add prizes for postponable requests
    # TODO: Implement prize collection logic
    
    # Solve
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    solution = routing.SolveWithParameters(search_parameters)
    
    # Extract routes
    routes = extract_routes_from_ortools(manager, routing, solution)
    return routes
```

## Testing Strategy

### Unit Tests

Create test files to verify components:

```python
# tests/test_environment.py
def test_environment_creation():
    env = RLDVSPEnv(instance_path, ...)
    assert env is not None
    
def test_feature_computation():
    env = RLDVSPEnv(instance_path, ...)
    features = env.embedding()
    assert features.shape == (14,)
    
# tests/test_policy.py
def test_actor_forward():
    model = ActorModel(14)
    x = torch.randn(14)
    out = model(x)
    assert out.shape == ()
    
def test_critic_gnn():
    critic = critic_GNN(15, 1)
    # Create dummy graph
    data = ...
    out = critic(data, x, edge_attr)
    assert out.shape == (1, 1)
```

### Integration Tests

```python
# tests/test_training.py
def test_sil_training_step():
    """Test one training step of SIL."""
    # Create small synthetic dataset
    # Run one epoch
    # Check loss is finite
    
def test_ppo_episode_collection():
    """Test PPO episode generation."""
    # Create simple environment
    # Generate episodes
    # Verify data structure
```

## Performance Benchmarking

Compare Python vs Julia performance:

```python
import time

# Benchmark forward pass
model = ActorModel(14)
x = torch.randn(10000, 14)

start = time.time()
for _ in range(100):
    y = model(x)
end = time.time()

print(f"Forward pass: {(end-start)/100*1000:.2f} ms")
```

Expected performance:
- Forward pass: < 1ms on GPU
- Episode generation: depends on CO solver
- Training iteration: 100-500ms depending on batch size

## Migration Checklist

- [ ] Install Python dependencies
- [ ] Download and prepare dataset
- [ ] Implement `RLDVSPEnv` interface
- [ ] Implement `prize_collecting_vsp` solver
- [ ] Implement Fenchel-Young loss (or use approximation)
- [ ] Implement dataset loading
- [ ] Test environment and policy
- [ ] Run baseline evaluation (00_setup.py)
- [ ] Train SIL and verify convergence
- [ ] Train PPO and verify convergence
- [ ] Train SRL and verify convergence
- [ ] Generate plots and compare results
- [ ] Benchmark performance vs Julia

## Common Issues and Solutions

### Issue: torch_geometric import error
**Solution:** Install with correct CUDA version:
```bash
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### Issue: Out of memory during training
**Solutions:**
- Reduce batch size
- Reduce replay buffer capacity
- Use gradient accumulation
- Enable mixed precision training

### Issue: CO solver too slow
**Solutions:**
- Use commercial solver (Gurobi) instead of open-source
- Set time limits on solver
- Use warm-start from previous solution
- Simplify problem relaxation

### Issue: Training not converging
**Solutions:**
- Check learning rate schedule
- Verify advantage estimation
- Check critic loss values
- Adjust perturbation schedule
- Verify CO solver correctness

## Additional Resources

### Julia-Python Translation Guide

| Julia | Python |
|-------|--------|
| `Array` | `numpy.ndarray` or `torch.Tensor` |
| `Vector{Float64}` | `List[float]` or `torch.Tensor` |
| `Dict` | `dict` |
| `Symbol` | `str` |
| `nothing` | `None` |
| `@info` | `print()` or `logging.info()` |
| `deepcopy()` | `copy.deepcopy()` |
| `rand()` | `np.random.rand()` or `torch.rand()` |

### Optimization Solver Comparison

| Solver | Language | License | Speed | Ease of Use |
|--------|----------|---------|-------|-------------|
| Gurobi | Python | Commercial | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| CPLEX | Python | Commercial | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| SCIP | Python | Open | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| OR-Tools | Python | Open | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| HiGHS | Python | Open | ⭐⭐⭐ | ⭐⭐⭐ |

### Recommended Development Workflow

1. Start with small synthetic instances
2. Verify environment logic with simple cases
3. Test CO solver independently
4. Use dummy/random policy to test training loop
5. Implement SIL first (simpler than RL)
6. Add PPO after SIL works
7. Add SRL last (most complex)

## Contact and Support

For implementation questions:
- Open an issue on GitHub
- Check the original Julia code for reference
- Consult PyTorch Geometric documentation for GNN issues
- Consult optimization solver documentation

## Version History

- v1.0 (2024-11-26): Initial conversion from Julia to Python
  - All files converted with placeholders for domain-specific code
  - Neural networks fully implemented
  - Training loops fully implemented
  - Visualization fully implemented
  - Documentation and setup complete
