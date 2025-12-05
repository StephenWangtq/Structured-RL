"""
Test script for OT-Net implementation.
Validates all components work correctly.
"""

import numpy as np
import torch
from utils.environment import DmTSPInstance, DmTSPEnv, CustomerRequest, Agent
from utils.otnet import OTNetPolicy, OTNetEncoder, OTLayer
from utils.utils import (
    extract_agent_features,
    extract_task_features,
    build_graph_edges,
    evaluate_policy,
    generate_episode,
    GreedyPolicy,
    RandomPolicy,
)


def test_environment():
    """Test D-mTSP environment."""
    print("Testing D-mTSP Environment...")
    
    # Create instance
    instance = DmTSPInstance(
        num_agents=3,
        depot_location=np.array([50.0, 50.0]),
        time_horizon=50.0,
        request_rate=0.5,
        seed=42,
    )
    
    # Generate requests
    requests = instance.generate_requests()
    print(f"  Generated {len(requests)} requests")
    assert len(requests) > 0, "Should generate some requests"
    
    # Create environment
    env = DmTSPEnv(instance, alpha=0.5, beta=0.5)
    state = env.reset()
    
    print(f"  Initial state: {state}")
    print(f"  Number of agents: {len(state.agents)}")
    print(f"  New requests: {len(state.new_requests)}")
    
    # Test step with dummy assignment
    if len(state.get_all_unassigned()) > 0:
        num_agents = len(state.agents)
        num_tasks = len(state.get_all_unassigned())
        
        # Assign first task to first agent
        assignment = np.zeros((num_agents, num_tasks))
        assignment[0, 0] = 1.0
        
        next_state, reward, done, info = env.step(assignment)
        print(f"  After step: reward={reward:.2f}, done={done}")
        print(f"  Makespan: {info['makespan']:.2f}")
    
    print("  ✓ Environment tests passed\n")


def test_feature_extraction():
    """Test feature extraction."""
    print("Testing Feature Extraction...")
    
    # Create simple state
    instance = DmTSPInstance(
        num_agents=2,
        depot_location=np.array([50.0, 50.0]),
        time_horizon=50.0,
        request_rate=0.3,
        seed=42,
    )
    
    env = DmTSPEnv(instance, alpha=0.5, beta=0.5)
    state = env.reset()
    
    # Extract features
    agent_features = extract_agent_features(state, instance.depot_location)
    task_features = extract_task_features(state, instance.depot_location)
    
    print(f"  Agent features shape: {agent_features.shape}")
    print(f"  Task features shape: {task_features.shape}")
    
    assert agent_features.shape[0] == len(state.agents), "Wrong number of agents"
    assert agent_features.shape[1] == 8, "Wrong agent feature dimension"
    assert task_features.shape[1] == 6, "Wrong task feature dimension"
    
    # Test edge building
    edge_index = build_graph_edges(state, instance.depot_location)
    print(f"  Edge index shape: {edge_index.shape}")
    
    print("  ✓ Feature extraction tests passed\n")


def test_otnet_encoder():
    """Test OT-Net encoder."""
    print("Testing OT-Net Encoder...")
    
    # Create encoder
    encoder = OTNetEncoder(
        agent_feature_dim=8,
        task_feature_dim=6,
        hidden_dim=32,
        num_layers=2,
        use_gnn=False,  # Test MLP version first
    )
    
    # Create dummy data
    agent_features = torch.randn(3, 8)
    task_features = torch.randn(5, 6)
    
    # Forward pass
    agent_emb, task_emb = encoder(agent_features, task_features)
    
    print(f"  Agent embeddings shape: {agent_emb.shape}")
    print(f"  Task embeddings shape: {task_emb.shape}")
    
    assert agent_emb.shape == (3, 32), "Wrong agent embedding shape"
    assert task_emb.shape == (5, 32), "Wrong task embedding shape"
    
    print("  ✓ OT-Net encoder tests passed\n")


def test_ot_layer():
    """Test OT layer with Sinkhorn."""
    print("Testing OT Layer...")
    
    # Create OT layer
    ot_layer = OTLayer(
        embedding_dim=32,
        epsilon=0.1,
        num_iterations=20,
    )
    
    # Create dummy embeddings
    agent_emb = torch.randn(3, 32)
    task_emb = torch.randn(5, 32)
    
    # Forward pass
    T = ot_layer(agent_emb, task_emb)
    
    print(f"  Transport matrix shape: {T.shape}")
    print(f"  Transport matrix sum: {T.sum().item():.4f}")
    print(f"  Transport matrix:\n{T.detach().numpy()}")
    
    assert T.shape == (3, 5), "Wrong transport matrix shape"
    assert torch.allclose(T.sum(), torch.tensor(1.0), atol=0.1), "Transport should sum to ~1"
    
    print("  ✓ OT layer tests passed\n")


def test_otnet_policy():
    """Test complete OT-Net policy."""
    print("Testing OT-Net Policy...")
    
    # Create policy
    policy = OTNetPolicy(
        agent_feature_dim=8,
        task_feature_dim=6,
        hidden_dim=32,
        num_gnn_layers=2,
        epsilon=0.1,
        num_sinkhorn_iters=20,
        use_gnn=False,
    )
    
    # Create dummy data
    agent_features = torch.randn(3, 8)
    task_features = torch.randn(5, 6)
    
    # Forward pass
    T = policy(agent_features, task_features)
    
    print(f"  Output shape: {T.shape}")
    print(f"  Output sum: {T.sum().item():.4f}")
    
    # Get discrete assignment
    assignment = policy.get_assignment(agent_features, task_features)
    
    print(f"  Assignment shape: {assignment.shape}")
    print(f"  Assignment:\n{assignment}")
    print(f"  Assigned tasks: {assignment.sum(axis=0)}")
    
    assert assignment.shape == (3, 5), "Wrong assignment shape"
    
    print("  ✓ OT-Net policy tests passed\n")


def test_baseline_policies():
    """Test baseline policies."""
    print("Testing Baseline Policies...")
    
    # Create environment
    instance = DmTSPInstance(
        num_agents=3,
        depot_location=np.array([50.0, 50.0]),
        time_horizon=30.0,
        request_rate=0.4,
        seed=42,
    )
    
    env = DmTSPEnv(instance, alpha=0.5, beta=0.5)
    
    # Test greedy policy
    greedy = GreedyPolicy()
    state = env.reset()
    
    if len(state.get_all_unassigned()) > 0:
        assignment = greedy(state)
        print(f"  Greedy assignment shape: {assignment.shape}")
        
        # Test step
        next_state, reward, done, info = env.step(assignment)
        print(f"  Greedy reward: {reward:.2f}")
    
    # Test random policy
    random = RandomPolicy(seed=42)
    state = env.reset()
    
    if len(state.get_all_unassigned()) > 0:
        assignment = random(state)
        print(f"  Random assignment shape: {assignment.shape}")
    
    print("  ✓ Baseline policy tests passed\n")


def test_full_episode():
    """Test full episode generation."""
    print("Testing Full Episode...")
    
    # Create policy and environment
    policy = OTNetPolicy(
        agent_feature_dim=8,
        task_feature_dim=6,
        hidden_dim=32,
        num_gnn_layers=2,
        epsilon=0.1,
        num_sinkhorn_iters=20,
        use_gnn=False,
    )
    
    instance = DmTSPInstance(
        num_agents=3,
        depot_location=np.array([50.0, 50.0]),
        time_horizon=30.0,
        request_rate=0.3,
        seed=42,
    )
    
    env = DmTSPEnv(instance, alpha=0.5, beta=0.5)
    
    # Generate episode
    trajectory = generate_episode(policy, env, device='cpu')
    
    print(f"  Episode length: {len(trajectory)}")
    print(f"  Total reward: {sum(t['reward'] for t in trajectory):.2f}")
    
    assert len(trajectory) > 0, "Episode should have steps"
    
    print("  ✓ Full episode test passed\n")


def test_gradient_flow():
    """Test gradient flow through model."""
    print("Testing Gradient Flow...")
    
    # Create policy
    policy = OTNetPolicy(
        agent_feature_dim=8,
        task_feature_dim=6,
        hidden_dim=32,
        num_gnn_layers=2,
        epsilon=0.1,
        num_sinkhorn_iters=20,
        use_gnn=False,
    )
    
    # Create dummy data
    agent_features = torch.randn(3, 8, requires_grad=True)
    task_features = torch.randn(5, 6, requires_grad=True)
    
    # Forward pass
    T = policy(agent_features, task_features)
    
    # Compute dummy loss
    loss = T.sum()
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    has_gradients = False
    for name, param in policy.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_gradients = True
            break
    
    assert has_gradients, "Model should have gradients"
    
    print("  ✓ Gradient flow test passed\n")


def main():
    """Run all tests."""
    print("="*60)
    print("OT-Net Implementation Tests")
    print("="*60)
    print()
    
    try:
        test_environment()
        test_feature_extraction()
        test_otnet_encoder()
        test_ot_layer()
        test_otnet_policy()
        test_baseline_policies()
        test_full_episode()
        test_gradient_flow()
        
        print("="*60)
        print("All tests passed! ✓")
        print("="*60)
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
