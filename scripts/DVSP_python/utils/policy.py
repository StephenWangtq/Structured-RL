"""
Policy definitions, critic network, episode generation, and loss functions for DVSP.

This module contains the PyTorch implementations of:
- CombinatorialACPolicy (Actor-Critic policy)
- Critic GNN (Graph Neural Network for value estimation)
- Episode generation functions for PPO and SRL
- Replay buffer operations
- Loss functions (PPO, SRL, Huber)
"""

import copy
from typing import List, Tuple, Optional, Callable, NamedTuple, Any, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch_geometric.nn import NNConv, GraphConv, global_add_pool
from torch_geometric.data import Data, Batch
import networkx as nx

from .utils import sigmaF_dvsp


class Experience(NamedTuple):
    """Container for experience tuples."""
    state: Any
    s: torch.Tensor
    theta: torch.Tensor
    eta: torch.Tensor
    a: List[List[int]]
    next_state: Any
    next_s: torch.Tensor
    reward: float
    s_c: torch.Tensor
    next_s_c: torch.Tensor
    Rₜ: float = 0.0
    adv: float = 0.0


class CombinatorialACPolicy:
    """
    Combinatorial Actor-Critic Policy.
    
    Combines a neural network actor with a GNN critic for
    combinatorial optimization problems.
    """
    
    def __init__(
        self,
        actor_model: nn.Module,
        critic_model: Optional[nn.Module],
        p: Optional[Callable],
        CO_layer: Callable,
        seed: int = 0
    ):
        """
        Initialize the policy.
        
        Args:
            actor_model: Neural network for the actor
            critic_model: Neural network for the critic (can be None)
            p: Distribution function p(theta, sigma) -> Distribution
            CO_layer: Combinatorial optimization layer (solver)
            seed: Random seed
        """
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.p = p
        self.CO_layer = CO_layer
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
    def reset_seed(self):
        """Reset the random number generator to initial seed."""
        self.rng = np.random.RandomState(self.seed)
    
    def __call__(
        self,
        env,
        rng: Optional[np.random.RandomState] = None,
        perturb: bool = True,
        **kwargs
    ) -> Tuple:
        """
        Apply policy to environment.
        
        Args:
            env: The environment
            rng: Random number generator (optional)
            perturb: Whether to perturb the action
            **kwargs: Additional arguments for CO_layer
        
        Returns:
            Tuple of (state, s, theta, eta, action, s_c)
        """
        s = env.embedding()
        theta = self.actor_model(s)
        
        if not perturb:
            eta = theta
            a = self.CO_layer(theta, instance=env.state(), **kwargs)
        else:
            # Sample from distribution
            used_rng = rng if rng is not None else self.rng
            dist = self.p(theta, sigmaF_dvsp)
            eta = torch.tensor(
                dist.sample(),
                dtype=torch.float32
            )
            a = self.CO_layer(eta, instance=env.state(), **kwargs)
        
        s_c = env.embedding_critic()
        
        return copy.deepcopy(env.state()), s, theta, eta, a, s_c


def p_distribution(theta: torch.Tensor, stdev: float):
    """
    Create multivariate normal distribution.
    
    Args:
        theta: Mean vector
        stdev: Standard deviation
    
    Returns:
        MultivariateNormal distribution
    """
    theta_np = theta.detach().numpy()
    cov = stdev**2 * np.eye(len(theta_np))
    return MultivariateNormal(
        torch.tensor(theta_np, dtype=torch.float32),
        torch.tensor(cov, dtype=torch.float32)
    )


class CriticGNN(nn.Module):
    """
    Graph Neural Network for critic (value estimation).
    
    Uses NNConv and GraphConv layers with global pooling.
    """
    
    def __init__(self, node_features: int, edge_features: int):
        """
        Initialize the GNN critic.
        
        Args:
            node_features: Number of node features
            edge_features: Number of edge features
        """
        super(CriticGNN, self).__init__()
        
        # Graph layers with edge features
        self.g1_edge_nn = nn.Linear(edge_features, node_features)
        self.g1 = NNConv(node_features, 15, self.g1_edge_nn, aggr='mean')
        
        self.g2_edge_nn = nn.Linear(edge_features, 15)
        self.g2 = NNConv(15, 10, self.g2_edge_nn, aggr='mean')
        
        self.g3_edge_nn = nn.Linear(edge_features, 10)
        self.g3 = NNConv(10, 10, self.g3_edge_nn, aggr='mean')
        
        self.g_out = GraphConv(10, 10)
        
        # Graph layers without edge features
        self.c1 = GraphConv(node_features, 15)
        self.c2 = GraphConv(15, 10)
        self.c3 = GraphConv(10, 10)
        self.c_out = GraphConv(10, 10)
        
        # Dense layers after pooling
        self.l1 = nn.Linear(10, 15)
        self.l2 = nn.Linear(15, 10)
        self.l3 = nn.Linear(10, 10)
        self.l4 = nn.Linear(10, 5)
        self.l_out = nn.Linear(5, 1)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the GNN.
        
        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_features] (optional)
            batch: Batch vector [num_nodes] (optional)
        
        Returns:
            Value estimate [batch_size, 1]
        """
        # Use edge features if available
        if edge_attr is not None and edge_attr.numel() > 0:
            h1 = F.celu(self.g1(x, edge_index, edge_attr))
            h2 = F.celu(self.g2(h1, edge_index, edge_attr))
            h3 = F.celu(self.g3(h2, edge_index, edge_attr))
            h_out = F.celu(self.g_out(h3, edge_index))
        else:
            h1 = F.celu(self.c1(x, edge_index))
            h2 = F.celu(self.c2(h1, edge_index))
            h3 = F.celu(self.c3(h2, edge_index))
            h_out = F.celu(self.c_out(h3, edge_index))
        
        # Global pooling
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long)
        pool = global_add_pool(h_out, batch)
        
        # Dense layers
        k1 = F.celu(self.l1(pool))
        k2 = F.celu(self.l2(k1))
        k3 = F.celu(self.l3(k2))
        k4 = F.celu(self.l4(k3))
        out = F.celu(self.l_out(k4))
        
        return out


def generate_episode(
    policy: CombinatorialACPolicy,
    env,
    **kwargs
) -> List[Experience]:
    """
    Generate a complete episode using the policy.
    
    Args:
        policy: The policy to use
        env: The environment
        **kwargs: Additional arguments
    
    Returns:
        List of experience tuples
    """
    env.reset()
    trajectory = []
    
    while not env.is_terminated():
        state, s, theta, eta, a, s_c = policy(env, **kwargs)
        next_state, reward = env.apply_action(a)
        next_s = env.embedding()
        next_s_c = env.embedding_critic()
        
        trajectory.append(Experience(
            state=state,
            s=s,
            theta=theta,
            eta=eta,
            a=a,
            next_state=next_state,
            next_s=next_s,
            reward=reward,
            s_c=s_c,
            next_s_c=next_s_c,
            Rₜ=0.0,
            adv=0.0
        ))
    
    return trajectory


def PPO_episodes(
    policy: CombinatorialACPolicy,
    target_critic: nn.Module,
    envs: List,
    nb_episodes: int,
    gamma: float,
    V_method: str,
    adv_method: str,
    **kwargs
) -> List[Experience]:
    """
    Generate episodes for PPO training.
    
    Args:
        policy: The policy
        target_critic: Target critic network
        envs: List of environments
        nb_episodes: Number of episodes to generate
        gamma: Discount factor
        V_method: Value estimation method ("on_policy" or "off_policy")
        adv_method: Advantage estimation method ("TD_n" or "TD_1")
        **kwargs: Additional arguments
    
    Returns:
        List of experience tuples with computed returns and advantages
    """
    # Sample environments
    training_envs = np.random.choice(envs, size=nb_episodes, replace=True).tolist()
    episodes = []
    
    for e in range(nb_episodes):
        trajectories = generate_episode(policy, training_envs[e], **kwargs)
        
        # Remove last element (terminal state not used)
        if len(trajectories) > 0:
            trajectories.pop()
        
        # Calculate returns and advantages (backward pass)
        for i in range(len(trajectories) - 1, -1, -1):
            rew_new = trajectories[i].reward / 100  # Scaling for stability
            
            # Calculate cumulative discounted returns
            if i == len(trajectories) - 1:
                # Account for future value of second-last state
                ret = (
                    trajectories[i].reward +
                    gamma * V_value_GNN(
                        policy,
                        trajectories[i].next_s,
                        trajectories[i].next_s_c,
                        target_critic,
                        V_method,
                        instance=trajectories[i].next_state,
                        **kwargs
                    )
                )
            else:
                ret = trajectories[i].reward + gamma * trajectories[i + 1].Rₜ
            
            # Calculate advantages
            if adv_method == "TD_n":  # TD(n)
                advantage = (
                    ret - V_value_GNN(
                        policy,
                        trajectories[i].s,
                        trajectories[i].s_c,
                        policy.critic_model,
                        V_method,
                        instance=trajectories[i].state,
                        **kwargs
                    )
                )
            else:  # TD(1)
                advantage = (
                    trajectories[i].reward +
                    gamma * V_value_GNN(
                        policy,
                        trajectories[i].next_s,
                        trajectories[i].next_s_c,
                        target_critic,
                        V_method,
                        instance=trajectories[i].next_state,
                        **kwargs
                    ) - V_value_GNN(
                        policy,
                        trajectories[i].s,
                        trajectories[i].s_c,
                        policy.critic_model,
                        V_method,
                        instance=trajectories[i].state,
                        **kwargs
                    )
                )
            
            # Update trajectory with new values
            trajectories[i] = trajectories[i]._replace(
                reward=rew_new,
                Rₜ=ret,
                adv=advantage
            )
        
        episodes.extend(trajectories)
    
    return episodes


def SRL_episodes(
    policy: CombinatorialACPolicy,
    envs: List,
    nb_episodes: int,
    **kwargs
) -> List[Experience]:
    """
    Generate episodes for SRL training.
    
    Args:
        policy: The policy
        envs: List of environments
        nb_episodes: Number of episodes to generate
        **kwargs: Additional arguments
    
    Returns:
        List of experience tuples
    """
    # Sample environments
    training_envs = np.random.choice(envs, size=nb_episodes, replace=True).tolist()
    episodes = []
    
    for e in range(nb_episodes):
        trajectories = generate_episode(policy, training_envs[e], perturb=True, **kwargs)
        
        # Remove last element
        if len(trajectories) > 0:
            trajectories.pop()
        
        episodes.extend(trajectories)
    
    return episodes


# Replay Buffer Operations

def rb_add(
    replay_buffer: List,
    rb_capacity: int,
    rb_position: int,
    rb_size: int,
    episodes: List[Experience]
) -> Tuple[List, int, int]:
    """
    Add episodes to replay buffer.
    
    Args:
        replay_buffer: The replay buffer
        rb_capacity: Maximum capacity
        rb_position: Current position
        rb_size: Current size
        episodes: Episodes to add
    
    Returns:
        Tuple of (updated_buffer, new_position, new_size)
    """
    for episode in episodes:
        if rb_size < rb_capacity:
            replay_buffer.append(episode)
        else:
            replay_buffer[rb_position] = episode
        
        rb_position = (rb_position % rb_capacity) + 1
        rb_size = min(rb_size + 1, rb_capacity)
    
    return replay_buffer, rb_position, rb_size


def rb_sample(replay_buffer: List, batch_size: int) -> List[Experience]:
    """
    Sample a batch from replay buffer.
    
    Args:
        replay_buffer: The replay buffer
        batch_size: Number of samples
    
    Returns:
        List of sampled experiences
    """
    idxs = np.random.choice(len(replay_buffer), size=batch_size, replace=False)
    return [replay_buffer[i] for i in idxs]


# Loss Functions

def J_PPO(
    policy: CombinatorialACPolicy,
    batch: List[Experience],
    clip: float,
    sigmaF_average: float
) -> torch.Tensor:
    """
    PPO actor loss (clipped surrogate objective).
    
    Args:
        policy: The policy
        batch: Batch of experiences
        clip: Clipping parameter
        sigmaF_average: Sigma for perturbation
    
    Returns:
        Negative PPO loss (for maximization)
    """
    embeddings = [exp.s for exp in batch]
    thetas = [exp.theta for exp in batch]
    advantages = torch.tensor([exp.adv for exp in batch], dtype=torch.float32)
    etas = [exp.eta for exp in batch]
    
    # Policy ratio
    old_probs = []
    new_probs = []
    
    for j in range(len(batch)):
        dist_old = policy.p(thetas[j], sigmaF_average)
        old_probs.append(dist_old.log_prob(etas[j]))
        
        theta_new = policy.actor_model(embeddings[j])
        dist_new = policy.p(theta_new, sigmaF_average)
        new_probs.append(dist_new.log_prob(etas[j]))
    
    old_probs = torch.stack(old_probs)
    new_probs = torch.stack(new_probs)
    
    # Actor loss
    ratio_unclipped = torch.exp(new_probs - old_probs)
    ratio_clipped = torch.clamp(ratio_unclipped, 1 - clip, 1 + clip)
    
    loss = torch.mean(torch.min(
        ratio_unclipped * advantages,
        ratio_clipped * advantages
    ))
    
    return loss


def SRL_actions(
    policy: CombinatorialACPolicy,
    batch: List[Experience],
    sigmaB: float = 0.05,
    no_samples: int = 20,
    temp: float = 1.0,
    **kwargs
) -> List[np.ndarray]:
    """
    Compute target actions for SRL using Q-values.
    
    Args:
        policy: The policy
        batch: Batch of experiences
        sigmaB: Perturbation standard deviation
        no_samples: Number of samples for candidate actions
        temp: Temperature parameter
        **kwargs: Additional arguments
    
    Returns:
        List of target action matrices
    """
    embeddings = [exp.s for exp in batch]
    embeds_c = [exp.s_c for exp in batch]
    states = [exp.state for exp in batch]
    best_solutions = []
    
    for j in range(len(batch)):
        # Perturb and sample candidate actions
        with torch.no_grad():
            theta = policy.actor_model(embeddings[j])
        
        # Sample perturbations
        dist = policy.p(theta, sigmaB)
        eta_samples = [dist.sample() for _ in range(no_samples - 1)]
        
        # Get solution for unperturbed theta
        route = policy.CO_layer(theta, instance=states[j], **kwargs)
        from .utils import VSPSolution, nb_locations
        solutions = [VSPSolution(route, max_index=nb_locations(states[j])).edge_matrix]
        values = [Q_value_GNN(route, embeds_c[j], policy.critic_model, instance=states[j])]
        
        # Get solutions for perturbed samples
        for i in range(no_samples - 1):
            route = policy.CO_layer(eta_samples[i], instance=states[j], **kwargs)
            solutions.append(
                VSPSolution(route, max_index=nb_locations(states[j])).edge_matrix
            )
            values.append(
                Q_value_GNN(route, embeds_c[j], policy.critic_model, instance=states[j])
            )
        
        # Calculate target action (weighted by Q-values)
        values = torch.tensor(values, dtype=torch.float32) / temp
        lse = torch.logsumexp(values, dim=0)
        probs = torch.exp(values - lse)
        
        # Weighted average of solutions
        best_action = sum(
            prob.item() * sol for prob, sol in zip(probs, solutions)
        )
        
        # Handle NaN case
        if np.isnan(best_action).any():
            best_action = solutions[torch.argmax(values).item()]
        
        best_solutions.append(best_action)
    
    return best_solutions


def J_SRL(
    policy: CombinatorialACPolicy,
    batch: List[Experience],
    best_solutions: List[np.ndarray],
    **kwargs
) -> torch.Tensor:
    """
    SRL actor loss using Fenchel-Young loss.
    
    Args:
        policy: The policy
        batch: Batch of experiences
        best_solutions: Target solutions from SRL_actions
        **kwargs: Additional arguments
    
    Returns:
        Fenchel-Young loss
    """
    # This requires implementation of Fenchel-Young loss
    # Placeholder structure
    raise NotImplementedError(
        "J_SRL requires implementation of Fenchel-Young loss with InferOpt functionality"
    )


def grads_prep_GNN(batch: List[Experience]) -> Tuple[List[Data], List[torch.Tensor], List[torch.Tensor]]:
    """
    Prepare graph data for GNN critic training.
    
    Args:
        batch: Batch of experiences
    
    Returns:
        Tuple of (graphs, node_features, edge_features)
    """
    from .utils import VSPSolution, nb_locations
    
    states = [exp.state for exp in batch]
    routes = [exp.a for exp in batch]
    s_c = [exp.s_c for exp in batch]
    
    graphs = []
    edge_features = []
    
    for j in range(len(batch)):
        # Create adjacency matrix of action
        adj_matrix = VSPSolution(routes[j], max_index=nb_locations(states[j])).edge_matrix
        dist_matrix = states[j].instance.duration
        
        # Create graph
        G = nx.DiGraph()
        n = adj_matrix.shape[0]
        G.add_nodes_from(range(n))
        
        edge_index = []
        edge_attr = []
        
        for i in range(n):
            for k in range(n):
                if adj_matrix[i, k] == 1:
                    G.add_edge(i, k)
                    edge_index.append([i, k])
                    edge_attr.append([dist_matrix[i, k]])
        
        # Convert to PyTorch Geometric format
        if len(edge_index) > 0:
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t()
            edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float32)
        else:
            edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
            edge_attr_tensor = torch.empty((0, 1), dtype=torch.float32)
        
        data = Data(
            x=s_c[j].unsqueeze(0) if s_c[j].dim() == 1 else s_c[j],
            edge_index=edge_index_tensor,
            edge_attr=edge_attr_tensor
        )
        graphs.append(data)
    
    return graphs, s_c, edge_features


def huber_GNN(
    policy: CombinatorialACPolicy,
    graphs: List[Data],
    s_c: List[torch.Tensor],
    edge_features: List[torch.Tensor],
    critic_target: List[float],
    delta: float,
    **kwargs
) -> torch.Tensor:
    """
    Huber loss for critic training.
    
    Args:
        policy: The policy
        graphs: List of graph data
        s_c: Node features
        edge_features: Edge features
        critic_target: Target values
        delta: Huber loss threshold
        **kwargs: Additional arguments
    
    Returns:
        Huber loss value
    """
    critic_target = torch.tensor(critic_target, dtype=torch.float32)
    new_critic = []
    
    for j in range(len(graphs)):
        value = -policy.critic_model(
            graphs[j].x,
            graphs[j].edge_index,
            graphs[j].edge_attr
        ).sum()
        new_critic.append(value)
    
    new_critic = torch.stack(new_critic)
    
    # Calculate Huber loss
    error = new_critic - critic_target
    quadratic = 0.5 * error ** 2
    linear = delta * (torch.abs(error) - 0.5 * delta)
    
    loss = torch.where(torch.abs(error) <= delta, quadratic, linear)
    return loss.mean()


def Q_value_GNN(
    routes: List[List[int]],
    s_c: torch.Tensor,
    critic_model: nn.Module,
    instance
) -> float:
    """
    Calculate Q-value using GNN critic.
    
    Args:
        routes: Routes (action)
        s_c: Critic state embedding
        critic_model: Critic network
        instance: Problem instance
    
    Returns:
        Q-value estimate
    """
    from .utils import VSPSolution, nb_locations
    
    adj_matrix = VSPSolution(routes, max_index=nb_locations(instance)).edge_matrix
    dist_matrix = instance.instance.duration
    
    # Construct graph
    n = adj_matrix.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    
    edge_index = []
    edge_attr = []
    
    for i in range(n):
        for j in range(n):
            if adj_matrix[i, j] == 1:
                G.add_edge(i, j)
                edge_index.append([i, j])
                edge_attr.append([dist_matrix[i, j]])
    
    # Convert to PyTorch Geometric format
    if len(edge_index) > 0:
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t()
        edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float32)
    else:
        edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
        edge_attr_tensor = torch.empty((0, 1), dtype=torch.float32)
    
    with torch.no_grad():
        x = s_c.unsqueeze(0) if s_c.dim() == 1 else s_c
        value = -critic_model(x, edge_index_tensor, edge_attr_tensor).sum()
    
    return value.item()


def V_value_GNN(
    policy: CombinatorialACPolicy,
    s: torch.Tensor,
    s_c: torch.Tensor,
    critic_model: nn.Module,
    method: str,
    instance,
    **kwargs
) -> float:
    """
    Calculate V-value (state value) using GNN critic.
    
    Args:
        policy: The policy
        s: State embedding for actor
        s_c: State embedding for critic
        critic_model: Critic network
        method: "on_policy" (with perturbation) or "off_policy" (without)
        instance: Problem instance
        **kwargs: Additional arguments
    
    Returns:
        V-value estimate
    """
    with torch.no_grad():
        theta = policy.actor_model(s)
        
        if method == "on_policy":
            # With perturbation
            dist = policy.p(theta, sigmaF_dvsp)
            eta = dist.sample()
            action = policy.CO_layer(eta, instance=instance, **kwargs)
        else:
            # Without perturbation
            action = policy.CO_layer(theta, instance=instance, **kwargs)
    
    return Q_value_GNN(action, s_c, critic_model, instance)


def critic_GNN(node_features: int, edge_features: int) -> CriticGNN:
    """
    Factory function to create a CriticGNN.
    
    Args:
        node_features: Number of node features
        edge_features: Number of edge features
    
    Returns:
        CriticGNN model
    """
    return CriticGNN(node_features, edge_features)
