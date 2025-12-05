"""
Utility functions for D-mTSP OT-Net.
"""

import numpy as np
import torch
from typing import List, Tuple, Optional, Dict, Any
from .environment import DmTSPEnv, DmTSPState, Agent, CustomerRequest


def extract_agent_features(state: DmTSPState, depot_location: np.ndarray) -> torch.Tensor:
    """
    Extract agent features from state.
    
    Features (8-dim):
    - position (x, y): 2
    - queue length: 1
    - estimated completion time: 1
    - current time: 1
    - historical load (avg): 1
    - distance to depot: 1
    - time to finish current route: 1
    
    Args:
        state: Current state
        depot_location: Depot coordinates
    
    Returns:
        Agent features: [num_agents, 8]
    """
    num_agents = len(state.agents)
    features = []
    
    for agent in state.agents:
        # Position
        pos = agent.location
        
        # Queue length
        queue_len = len(agent.route)
        
        # Estimated completion time
        completion_time = state.current_time
        loc = agent.location.copy()
        for req in agent.route:
            completion_time += np.linalg.norm(req.location - loc)
            completion_time += req.service_time
            loc = req.location
        completion_time += np.linalg.norm(depot_location - loc)
        
        # Current time
        curr_time = state.current_time
        
        # Historical load (approximated by current queue length)
        hist_load = queue_len / max(1, len(state.served_requests) + queue_len + 1)
        
        # Distance to depot
        dist_to_depot = np.linalg.norm(agent.location - depot_location)
        
        # Time to finish current route
        time_to_finish = completion_time - curr_time
        
        agent_feat = [
            pos[0], pos[1],
            queue_len,
            completion_time,
            curr_time,
            hist_load,
            dist_to_depot,
            time_to_finish,
        ]
        
        features.append(agent_feat)
    
    return torch.tensor(features, dtype=torch.float32)


def extract_task_features(
    state: DmTSPState,
    depot_location: np.ndarray,
) -> torch.Tensor:
    """
    Extract task features from state.
    
    Features (6-dim):
    - position (x, y): 2
    - arrival time: 1
    - current waiting time: 1
    - priority (is new request): 1
    - distance to depot: 1
    
    Args:
        state: Current state
        depot_location: Depot coordinates
    
    Returns:
        Task features: [num_tasks, 6]
    """
    all_tasks = state.get_all_unassigned()
    
    if len(all_tasks) == 0:
        # Return dummy task
        return torch.zeros((1, 6), dtype=torch.float32)
    
    features = []
    
    for req in all_tasks:
        # Position
        pos = req.location
        
        # Arrival time
        arrival = req.arrival_time
        
        # Waiting time
        waiting = state.current_time - req.arrival_time
        
        # Priority (1 if new, 0 if pending)
        priority = 1.0 if req in state.new_requests else 0.0
        
        # Distance to depot
        dist_to_depot = np.linalg.norm(req.location - depot_location)
        
        task_feat = [
            pos[0], pos[1],
            arrival,
            waiting,
            priority,
            dist_to_depot,
        ]
        
        features.append(task_feat)
    
    return torch.tensor(features, dtype=torch.float32)


def build_graph_edges(
    state: DmTSPState,
    depot_location: np.ndarray,
    k_nearest: int = 5,
) -> torch.Tensor:
    """
    Build graph edge index for GNN.
    
    Creates edges:
    - Agent to nearest tasks
    - Task to nearest tasks
    - Bidirectional edges
    
    Args:
        state: Current state
        depot_location: Depot coordinates
        k_nearest: Number of nearest neighbors
    
    Returns:
        Edge index: [2, num_edges]
    """
    num_agents = len(state.agents)
    all_tasks = state.get_all_unassigned()
    num_tasks = len(all_tasks)
    
    if num_tasks == 0:
        # No tasks - return empty edges
        return torch.zeros((2, 0), dtype=torch.long)
    
    edges = []
    
    # Agent positions
    agent_positions = np.array([a.location for a in state.agents])
    
    # Task positions
    task_positions = np.array([t.location for t in all_tasks])
    
    # Agent to task edges
    for i in range(num_agents):
        # Distance from agent i to all tasks
        distances = np.linalg.norm(
            task_positions - agent_positions[i:i+1],
            axis=1
        )
        
        # Connect to k nearest tasks
        k = min(k_nearest, num_tasks)
        nearest_tasks = np.argsort(distances)[:k]
        
        for j in nearest_tasks:
            edges.append([i, num_agents + j])  # Agent i to task j
            edges.append([num_agents + j, i])  # Task j to agent i (bidirectional)
    
    # Task to task edges
    for i in range(num_tasks):
        # Distance from task i to other tasks
        distances = np.linalg.norm(
            task_positions - task_positions[i:i+1],
            axis=1
        )
        distances[i] = float('inf')  # Exclude self
        
        # Connect to k nearest tasks
        k = min(k_nearest, num_tasks - 1)
        if k > 0:
            nearest_tasks = np.argsort(distances)[:k]
            
            for j in nearest_tasks:
                edges.append([num_agents + i, num_agents + j])
    
    if len(edges) == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    return edge_index


def cheapest_insertion(
    agent_route: List[CustomerRequest],
    request: CustomerRequest,
    depot_location: np.ndarray,
) -> Tuple[int, float]:
    """
    Find cheapest insertion position for a request.
    
    Args:
        agent_route: Current route
        request: Request to insert
        depot_location: Depot coordinates
    
    Returns:
        Tuple of (best_position, insertion_cost)
    """
    if len(agent_route) == 0:
        # First task - cost is distance from depot and back
        cost = (np.linalg.norm(request.location - depot_location) * 2 +
                request.service_time)
        return 0, cost
    
    best_pos = 0
    best_cost = float('inf')
    
    for pos in range(len(agent_route) + 1):
        # Calculate cost of inserting at this position
        route = agent_route.copy()
        route.insert(pos, request)
        
        # Calculate total route cost
        cost = 0.0
        loc = depot_location
        
        for req in route:
            cost += np.linalg.norm(req.location - loc)
            cost += req.service_time
            loc = req.location
        
        # Back to depot
        cost += np.linalg.norm(depot_location - loc)
        
        if cost < best_cost:
            best_cost = cost
            best_pos = pos
    
    return best_pos, best_cost


def calculate_makespan(
    agents: List[Agent],
    depot_location: np.ndarray,
    current_time: float = 0.0,
) -> float:
    """
    Calculate makespan (maximum completion time).
    
    Args:
        agents: List of agents
        depot_location: Depot coordinates
        current_time: Current time
    
    Returns:
        Makespan value
    """
    max_time = current_time
    
    for agent in agents:
        if len(agent.route) == 0:
            continue
        
        time = current_time
        loc = agent.location.copy()
        
        for request in agent.route:
            time += np.linalg.norm(request.location - loc)
            time += request.service_time
            loc = request.location
        
        # Return to depot
        time += np.linalg.norm(depot_location - loc)
        
        max_time = max(max_time, time)
    
    return max_time


def calculate_waiting_time(
    agents: List[Agent],
    pending_requests: List[CustomerRequest],
    current_time: float = 0.0,
) -> float:
    """
    Calculate total waiting time.
    
    Args:
        agents: List of agents
        pending_requests: Pending requests
        current_time: Current time
    
    Returns:
        Total waiting time
    """
    total_waiting = 0.0
    
    for agent in agents:
        time = current_time
        loc = agent.location.copy()
        
        for request in agent.route:
            # Travel to request
            time += np.linalg.norm(request.location - loc)
            
            # Waiting time = service start - arrival
            waiting = max(0, time - request.arrival_time)
            total_waiting += waiting
            
            # Service time
            time += request.service_time
            loc = request.location
    
    # Pending requests have been waiting
    for request in pending_requests:
        waiting = current_time - request.arrival_time
        total_waiting += waiting
    
    return total_waiting


def evaluate_policy(
    policy,
    env: DmTSPEnv,
    num_episodes: int = 1,
    device: str = 'cpu',
) -> Dict[str, float]:
    """
    Evaluate a policy on environment.
    
    Args:
        policy: Policy to evaluate (OTNetPolicy or baseline)
        env: Environment
        num_episodes: Number of episodes
        device: Device for computation
    
    Returns:
        Dictionary with evaluation metrics
    """
    total_rewards = []
    total_makespans = []
    total_waiting_times = []
    
    policy.eval()
    
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        done = False
        
        while not done:
            # Extract features
            agent_features = extract_agent_features(state, env.instance.depot_location)
            task_features = extract_task_features(state, env.instance.depot_location)
            
            # Get assignment
            if hasattr(policy, 'get_assignment'):
                # OT-Net policy
                edge_index = build_graph_edges(state, env.instance.depot_location)
                
                agent_features = agent_features.to(device)
                task_features = task_features.to(device)
                edge_index = edge_index.to(device)
                
                assignment = policy.get_assignment(
                    agent_features, task_features, edge_index
                )
            else:
                # Baseline policy
                assignment = policy(state)
            
            # Step environment
            state, reward, done, info = env.step(assignment)
            episode_reward += reward
        
        # Collect metrics
        total_rewards.append(episode_reward)
        total_makespans.append(info['makespan'])
        total_waiting_times.append(info['total_waiting'])
    
    results = {
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'mean_makespan': np.mean(total_makespans),
        'std_makespan': np.std(total_makespans),
        'mean_waiting': np.mean(total_waiting_times),
        'std_waiting': np.std(total_waiting_times),
    }
    
    return results


def generate_episode(
    policy,
    env: DmTSPEnv,
    device: str = 'cpu',
) -> List[Dict[str, Any]]:
    """
    Generate one episode using policy.
    
    Args:
        policy: Policy to use
        env: Environment
        device: Device for computation
    
    Returns:
        List of transitions
    """
    state = env.reset()
    trajectory = []
    done = False
    
    while not done:
        # Extract features
        agent_features = extract_agent_features(state, env.instance.depot_location)
        task_features = extract_task_features(state, env.instance.depot_location)
        edge_index = build_graph_edges(state, env.instance.depot_location)
        
        # Move to device
        agent_features = agent_features.to(device)
        task_features = task_features.to(device)
        edge_index = edge_index.to(device)
        
        # Get assignment
        if hasattr(policy, 'get_assignment'):
            assignment = policy.get_assignment(
                agent_features, task_features, edge_index
            )
        else:
            assignment = policy(state)
        
        # Step
        next_state, reward, done, info = env.step(assignment)
        
        # Store transition
        trajectory.append({
            'state': state,
            'agent_features': agent_features.cpu(),
            'task_features': task_features.cpu(),
            'edge_index': edge_index.cpu(),
            'assignment': assignment,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'info': info,
        })
        
        state = next_state
    
    return trajectory


class GreedyPolicy:
    """Greedy baseline policy."""
    
    def __call__(self, state: DmTSPState) -> np.ndarray:
        """
        Greedy assignment: assign each task to nearest available agent.
        
        Args:
            state: Current state
        
        Returns:
            Assignment matrix
        """
        num_agents = len(state.agents)
        all_tasks = state.get_all_unassigned()
        num_tasks = len(all_tasks)
        
        if num_tasks == 0:
            return np.zeros((num_agents, 1))
        
        assignment = np.zeros((num_agents, num_tasks))
        
        # Assign each task to nearest agent
        for j, task in enumerate(all_tasks):
            distances = [
                np.linalg.norm(agent.location - task.location)
                for agent in state.agents
            ]
            i = np.argmin(distances)
            assignment[i, j] = 1.0
        
        return assignment


class RandomPolicy:
    """Random baseline policy."""
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
    
    def __call__(self, state: DmTSPState) -> np.ndarray:
        """
        Random assignment.
        
        Args:
            state: Current state
        
        Returns:
            Assignment matrix
        """
        num_agents = len(state.agents)
        all_tasks = state.get_all_unassigned()
        num_tasks = len(all_tasks)
        
        if num_tasks == 0:
            return np.zeros((num_agents, 1))
        
        assignment = np.zeros((num_agents, num_tasks))
        
        # Randomly assign each task
        for j in range(num_tasks):
            i = self.rng.integers(0, num_agents)
            assignment[i, j] = 1.0
        
        return assignment
