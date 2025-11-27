"""
Policy definitions, neural networks, and training utilities for DVSP.
Converted from Julia (Flux) to Python (PyTorch).
"""

import copy
from typing import List, Tuple, Optional, Any, Dict, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from scipy.special import logsumexp

try:
    from torch_geometric.nn import NNConv, GCNConv, global_add_pool
    from torch_geometric.data import Data, Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("Warning: torch_geometric not available. GNN critic will not work.")


# Global variable for perturbation standard deviation (analogous to Julia global)
sigmaF_dvsp = 0.5


class CombinatorialACPolicy:
    """
    Combinatorial Actor-Critic Policy.
    
    This class combines an actor network (policy) and a critic network (value function)
    for reinforcement learning on combinatorial optimization problems.
    """
    
    def __init__(
        self,
        actor_model: nn.Module,
        critic_model: Optional[nn.Module],
        p: Optional[Callable],
        CO_layer: Callable,
        seed: int = 0,
    ):
        """
        Initialize policy.
        
        Args:
            actor_model: Neural network for actor (policy)
            critic_model: Neural network for critic (value function), can be None
            p: Probability distribution function (for perturbation)
            CO_layer: Combinatorial optimization layer function
            seed: Random seed
        """
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.p = p if p is not None else self._default_p
        self.CO_layer = CO_layer
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Set models to training mode by default
        self.actor_model.train()
        if self.critic_model is not None:
            self.critic_model.train()
    
    def _default_p(self, theta: torch.Tensor, stdev: float) -> MultivariateNormal:
        """
        Default probability distribution: Multivariate Normal.
        
        Args:
            theta: Mean vector
            stdev: Standard deviation
            
        Returns:
            Multivariate normal distribution
        """
        cov = torch.eye(len(theta)) * (stdev ** 2)
        return MultivariateNormal(theta, cov)
    
    def __call__(
        self,
        env: Any,
        rng: Optional[np.random.Generator] = None,
        perturb: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Apply policy to environment.
        
        Args:
            env: Environment
            rng: Random number generator (optional)
            perturb: Whether to add perturbation to actor output
            **kwargs: Additional arguments passed to CO_layer
            
        Returns:
            Dictionary containing state, embeddings, theta, eta, action, etc.
        """
        s = env.embedding()
        theta = self.actor_model(s)
        
        if not perturb:
            eta = theta
            a = self.CO_layer(theta, instance=env.state(), **kwargs)
        else:
            # Sample from perturbed distribution
            dist = self.p(theta, sigmaF_dvsp)
            eta_sample = dist.sample()
            eta = eta_sample
            a = self.CO_layer(eta, instance=env.state(), **kwargs)
        
        s_c = env.embedding_critic()
        
        return {
            'state': copy.deepcopy(env.state()),
            's': s,
            'theta': theta.detach(),
            'eta': eta.detach(),
            'a': a,
            's_c': s_c,
        }
    
    def reset_seed(self):
        """Reset random seed."""
        self.rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)


class critic_GNN(nn.Module):
    """
    Graph Neural Network for critic (value function estimation).
    
    This network processes graph-structured data representing vehicle routes.
    Converted from Julia GraphNeuralNetworks.jl to PyTorch Geometric.
    Compatible with PyTorch 1.11 and torch-geometric>=2.0.0,<2.3.0.
    """
    
    def __init__(self, node_features: int, edge_features: int):
        """
        Initialize GNN critic.
        
        Args:
            node_features: Number of node features
            edge_features: Number of edge features
        """
        super().__init__()
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric is required for critic_GNN")
        
        # Graph convolution layers with edge features (NNConv)
        # NNConv requires nn that maps edge_attr to in_channels * out_channels
        self.g1_nn = nn.Linear(edge_features, node_features * 15)
        self.g1 = NNConv(node_features, 15, self.g1_nn, aggr='mean')
        
        self.g2_nn = nn.Linear(edge_features, 15 * 10)
        self.g2 = NNConv(15, 10, self.g2_nn, aggr='mean')
        
        self.g3_nn = nn.Linear(edge_features, 10 * 10)
        self.g3 = NNConv(10, 10, self.g3_nn, aggr='mean')
        
        self.g_out = GCNConv(10, 10)
        
        # Graph convolution layers without edge features (fallback)
        self.c1 = GCNConv(node_features, 15)
        self.c2 = GCNConv(15, 10)
        self.c3 = GCNConv(10, 10)
        self.c_out = GCNConv(10, 10)
        
        # Dense layers after pooling
        self.l1 = nn.Linear(10, 15)
        self.l2 = nn.Linear(15, 10)
        self.l3 = nn.Linear(10, 10)
        self.l4 = nn.Linear(10, 5)
        self.l_out = nn.Linear(5, 1)
    
    def forward(
        self,
        data: Data,
        x: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through GNN.
        
        Args:
            data: PyTorch Geometric Data object with graph structure
            x: Node features [num_nodes, node_features]
            edge_attr: Edge features [num_edges, edge_features] (optional)
            
        Returns:
            Value estimate [batch_size, 1]
        """
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, 'batch') else None
        
        # Choose path based on whether edge features are available
        if edge_attr is not None and edge_attr.numel() > 0:
            # Path with edge features
            h1 = F.celu(self.g1(x, edge_index, edge_attr))
            h2 = F.celu(self.g2(h1, edge_index, edge_attr))
            h3 = F.celu(self.g3(h2, edge_index, edge_attr))
            h_out = F.celu(self.g_out(h3, edge_index))
        else:
            # Path without edge features
            h1 = F.celu(self.c1(x, edge_index))
            h2 = F.celu(self.c2(h1, edge_index))
            h3 = F.celu(self.c3(h2, edge_index))
            h_out = F.celu(self.c_out(h3, edge_index))
        
        # Global pooling
        if batch is not None:
            pool = global_add_pool(h_out, batch)
        else:
            pool = h_out.sum(dim=0, keepdim=True)
        
        # Dense layers
        k1 = F.celu(self.l1(pool))
        k2 = F.celu(self.l2(k1))
        k3 = F.celu(self.l3(k2))
        k4 = F.celu(self.l4(k3))
        out = F.celu(self.l_out(k4))
        
        return out


def generate_episode(
    policy: CombinatorialACPolicy,
    env: Any,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Generate a single episode by rolling out the policy.
    
    Args:
        policy: Policy to use
        env: Environment
        **kwargs: Additional arguments
        
    Returns:
        List of trajectory steps
    """
    env.reset()
    trajectory = []
    
    while not env.is_terminated():
        # Get action from policy
        policy_output = policy(env, **kwargs)
        state = policy_output['state']
        s = policy_output['s']
        theta = policy_output['theta']
        eta = policy_output['eta']
        a = policy_output['a']
        s_c = policy_output['s_c']
        
        # Apply action
        next_state, reward = env.apply_action(a)
        next_s = env.embedding()
        next_s_c = env.embedding_critic()
        
        # Store transition
        trajectory.append({
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
            'Rₜ': 0.0,  # Will be filled in later
            'adv': 0.0,  # Will be filled in later
        })
    
    return trajectory


def PPO_episodes(
    policy: CombinatorialACPolicy,
    target_critic: nn.Module,
    envs: List[Any],
    nb_episodes: int,
    gamma: float,
    V_method: str,
    adv_method: str,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Generate episodes for PPO training.
    
    Args:
        policy: Policy to use
        target_critic: Target critic network
        envs: List of environments
        nb_episodes: Number of episodes to generate
        gamma: Discount factor
        V_method: Value estimation method ("on_policy" or "off_policy")
        adv_method: Advantage estimation method ("TD_n" or "TD_1")
        **kwargs: Additional arguments
        
    Returns:
        List of all transitions from all episodes
    """
    # Sample environments
    training_envs = np.random.choice(envs, size=nb_episodes, replace=True)
    episodes = []
    
    for e in range(nb_episodes):
        trajectories = generate_episode(policy, training_envs[e], **kwargs)
        
        # Remove last element (terminal state not used)
        if len(trajectories) > 0:
            trajectories.pop()
        
        # Calculate returns and advantages
        for i in range(len(trajectories) - 1, -1, -1):
            rew_new = trajectories[i]['reward'] / 100  # Scaling for stability
            
            # Calculate cumulative discounted returns
            if i == len(trajectories) - 1:
                # Last step - include future value
                ret = (
                    trajectories[i]['reward'] +
                    gamma * V_value_GNN(
                        policy,
                        trajectories[i]['next_s'],
                        trajectories[i]['next_s_c'],
                        target_critic,
                        V_method,
                        instance=trajectories[i]['next_state'],
                        **kwargs
                    )
                )
            else:
                ret = trajectories[i]['reward'] + gamma * trajectories[i + 1]['Rₜ']
            
            # Calculate advantages
            if adv_method == "TD_n":  # TD(n)
                advantage = ret - V_value_GNN(
                    policy,
                    trajectories[i]['s'],
                    trajectories[i]['s_c'],
                    policy.critic_model,
                    V_method,
                    instance=trajectories[i]['state'],
                    **kwargs
                )
            else:  # TD(1)
                advantage = (
                    trajectories[i]['reward'] +
                    gamma * V_value_GNN(
                        policy,
                        trajectories[i]['next_s'],
                        trajectories[i]['next_s_c'],
                        target_critic,
                        V_method,
                        instance=trajectories[i]['next_state'],
                        **kwargs
                    ) - V_value_GNN(
                        policy,
                        trajectories[i]['s'],
                        trajectories[i]['s_c'],
                        policy.critic_model,
                        V_method,
                        instance=trajectories[i]['state'],
                        **kwargs
                    )
                )
            
            # Update trajectory
            trajectories[i]['reward'] = rew_new
            trajectories[i]['Rₜ'] = ret
            trajectories[i]['adv'] = advantage
        
        episodes.extend(trajectories)
    
    return episodes


def SRL_episodes(
    policy: CombinatorialACPolicy,
    envs: List[Any],
    nb_episodes: int,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Generate episodes for SRL training.
    
    Args:
        policy: Policy to use
        envs: List of environments
        nb_episodes: Number of episodes to generate
        **kwargs: Additional arguments
        
    Returns:
        List of all transitions from all episodes
    """
    training_envs = np.random.choice(envs, size=nb_episodes, replace=True)
    episodes = []
    
    for e in range(nb_episodes):
        trajectories = generate_episode(policy, training_envs[e], perturb=True, **kwargs)
        
        # Remove last element
        if len(trajectories) > 0:
            trajectories.pop()
        
        episodes.extend(trajectories)
    
    return episodes


# Replay Buffer Functions

def rb_add(
    replay_buffer: List[Any],
    rb_capacity: int,
    rb_position: int,
    rb_size: int,
    episodes: List[Any]
) -> Tuple[List[Any], int, int]:
    """
    Add episodes to replay buffer.
    
    Args:
        replay_buffer: Current replay buffer
        rb_capacity: Maximum capacity
        rb_position: Current position
        rb_size: Current size
        episodes: Episodes to add
        
    Returns:
        Updated (replay_buffer, rb_position, rb_size)
    """
    for episode in episodes:
        if rb_size < rb_capacity:
            replay_buffer.append(episode)
        else:
            replay_buffer[rb_position] = episode
        rb_position = (rb_position % rb_capacity) + 1
        rb_size = min(rb_size + 1, rb_capacity)
    
    return replay_buffer, rb_position, rb_size


def rb_sample(replay_buffer: List[Any], batch_size: int) -> List[Any]:
    """
    Sample batch from replay buffer.
    
    Args:
        replay_buffer: Replay buffer
        batch_size: Size of batch to sample
        
    Returns:
        List of sampled transitions
    """
    idxs = np.random.choice(len(replay_buffer), size=batch_size, replace=True)
    return [replay_buffer[i] for i in idxs]


# Actor Loss Functions

def J_PPO(
    policy: CombinatorialACPolicy,
    batch: List[Dict[str, Any]],
    clip: float,
    sigmaF_average: float
) -> torch.Tensor:
    """
    PPO actor loss (clipped surrogate objective).
    
    Args:
        policy: Policy
        batch: Batch of transitions
        clip: Clipping parameter
        sigmaF_average: Standard deviation for perturbation
        
    Returns:
        Loss value (negative because we maximize)
    """
    embeddings = [item['s'] for item in batch]
    thetas = [item['theta'] for item in batch]
    advantages = torch.tensor([item['adv'] for item in batch], dtype=torch.float32)
    etas = [item['eta'] for item in batch]
    
    # Calculate log probabilities
    old_log_probs = []
    new_log_probs = []
    
    for j in range(len(batch)):
        dist_old = policy.p(thetas[j], sigmaF_average)
        old_log_probs.append(dist_old.log_prob(etas[j]))
        
        theta_new = policy.actor_model(embeddings[j])
        dist_new = policy.p(theta_new, sigmaF_average)
        new_log_probs.append(dist_new.log_prob(etas[j]))
    
    old_log_probs = torch.stack(old_log_probs)
    new_log_probs = torch.stack(new_log_probs)
    
    # Calculate ratio
    ratio = torch.exp(new_log_probs - old_log_probs)
    ratio_clipped = torch.clamp(ratio, 1 - clip, 1 + clip)
    
    # PPO loss
    loss = torch.mean(torch.min(ratio * advantages, ratio_clipped * advantages))
    
    return loss


def SRL_actions(
    policy: CombinatorialACPolicy,
    batch: List[Dict[str, Any]],
    sigmaB: float = 0.05,
    no_samples: int = 20,
    temp: float = 1.0,
    **kwargs
) -> List[np.ndarray]:
    """
    Generate target actions for SRL using critic-guided sampling.
    
    Args:
        policy: Policy
        batch: Batch of transitions
        sigmaB: Standard deviation for sampling
        no_samples: Number of samples to generate
        temp: Temperature for softmax
        **kwargs: Additional arguments
        
    Returns:
        List of target action matrices
    """
    from .utils import prize_collecting_vsp, VSPSolution, nb_locations
    
    embeddings = [item['s'] for item in batch]
    embeds_c = [item['s_c'] for item in batch]
    states = [item['state'] for item in batch]
    best_solutions = []
    
    for j in range(len(batch)):
        # Generate candidate actions
        theta = policy.actor_model(embeddings[j])
        
        # Sample perturbed actions
        dist = policy.p(theta, sigmaB)
        eta_samples = [dist.sample() for _ in range(no_samples - 1)]
        
        # Get greedy action
        route = prize_collecting_vsp(theta, instance=states[j], **kwargs)
        solutions = [VSPSolution(route, max_index=nb_locations(states[j])).edge_matrix]
        values = [Q_value_GNN(route, embeds_c[j], policy.critic_model, instance=states[j])]
        
        # Get perturbed actions
        for i in range(no_samples - 1):
            route = prize_collecting_vsp(eta_samples[i], instance=states[j], **kwargs)
            solutions.append(
                VSPSolution(route, max_index=nb_locations(states[j])).edge_matrix
            )
            values.append(Q_value_GNN(route, embeds_c[j], policy.critic_model, instance=states[j]))
        
        # Calculate weighted combination based on Q-values
        values = np.array(values) / temp
        lse = logsumexp(values)
        probs = np.exp(values - lse)
        
        best_action = sum(p * s for p, s in zip(probs, solutions))
        
        # Fallback if NaN
        if np.isnan(best_action).any():
            best_action = solutions[np.argmax(values)]
        
        best_solutions.append(best_action)
    
    return best_solutions


def J_SRL(
    policy: CombinatorialACPolicy,
    batch: List[Dict[str, Any]],
    best_solutions: List[np.ndarray],
    **kwargs
) -> torch.Tensor:
    """
    SRL actor loss using Fenchel-Young loss.
    
    Note: This requires InferOpt functionality which needs to be implemented
    or use a compatible Python library.
    
    Args:
        policy: Policy
        batch: Batch of transitions
        best_solutions: Target solutions
        **kwargs: Additional arguments
        
    Returns:
        Loss value
    """
    # Placeholder for Fenchel-Young loss
    # This would require implementing the perturbed optimization framework
    raise NotImplementedError(
        "Fenchel-Young loss requires InferOpt-like functionality. "
        "Consider using a differentiable optimization library or implementing "
        "the perturbed optimization framework."
    )


# Critic Functions

def grads_prep_GNN(batch: List[Dict[str, Any]]) -> Tuple[List[Data], List[torch.Tensor], List[torch.Tensor]]:
    """
    Prepare graph data for GNN critic training.
    
    Args:
        batch: Batch of transitions
        
    Returns:
        Tuple of (graphs, node_features, edge_features)
    """
    from .utils import VSPSolution, nb_locations
    import networkx as nx
    from torch_geometric.utils import from_networkx
    
    states = [item['state'] for item in batch]
    routes = [item['a'] for item in batch]
    s_c = [item['s_c'] for item in batch]
    
    graphs = []
    edge_features = []
    
    for j in range(len(batch)):
        # Create adjacency matrix
        adj_matrix = VSPSolution(routes[j], max_index=nb_locations(states[j])).edge_matrix
        dist_matrix = states[j].instance.duration
        
        # Create directed graph
        n = adj_matrix.shape[0]
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        
        edge_list = []
        edge_attr_list = []
        
        for i in range(n):
            for k in range(n):
                if adj_matrix[i, k] == 1:
                    G.add_edge(i, k)
                    edge_list.append((i, k))
                    edge_attr_list.append([dist_matrix[i, k]])
        
        # Convert to PyTorch Geometric Data
        data = from_networkx(G)
        graphs.append(data)
        
        # Store edge features
        if len(edge_attr_list) > 0:
            edge_features.append(torch.tensor(edge_attr_list, dtype=torch.float32).T)
        else:
            edge_features.append(torch.empty(1, 0, dtype=torch.float32))
    
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
    Huber loss for GNN critic.
    
    Args:
        policy: Policy with critic model
        graphs: Graph data
        s_c: Critic node features
        edge_features: Edge features
        critic_target: Target values
        delta: Huber loss parameter
        **kwargs: Additional arguments
        
    Returns:
        Loss value
    """
    critic_target = torch.tensor(critic_target, dtype=torch.float32)
    
    new_critic = []
    for j in range(len(graphs)):
        x = s_c[j].unsqueeze(0) if s_c[j].dim() == 1 else s_c[j]
        edge_attr = edge_features[j] if edge_features[j].numel() > 0 else None
        val = -policy.critic_model(graphs[j], x, edge_attr).sum()
        new_critic.append(val)
    
    new_critic = torch.stack(new_critic)
    
    # Huber loss
    error = new_critic - critic_target
    quadratic = 0.5 * error ** 2
    linear = delta * (torch.abs(error) - 0.5 * delta)
    loss = torch.where(torch.abs(error) <= delta, quadratic, linear)
    
    return loss.mean()


# Q-value and V-value Functions

def Q_value_GNN(
    routes: List[List[int]],
    s_c: torch.Tensor,
    critic_model: nn.Module,
    instance: Any
) -> float:
    """
    Calculate Q-value using GNN critic.
    
    Args:
        routes: Routes (action)
        s_c: Critic node features
        critic_model: Critic network
        instance: State instance
        
    Returns:
        Q-value estimate
    """
    from .utils import VSPSolution, nb_locations
    import networkx as nx
    from torch_geometric.utils import from_networkx
    
    adj_matrix = VSPSolution(routes, max_index=nb_locations(instance)).edge_matrix
    dist_matrix = instance.instance.duration
    
    # Create graph
    n = adj_matrix.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    
    edge_attr_list = []
    for i in range(n):
        for j in range(n):
            if adj_matrix[i, j] == 1:
                G.add_edge(i, j)
                edge_attr_list.append([dist_matrix[i, j]])
    
    data = from_networkx(G)
    
    # Prepare features
    x = s_c.unsqueeze(0) if s_c.dim() == 1 else s_c
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32).T if len(edge_attr_list) > 0 else None
    
    # Get Q-value
    with torch.no_grad():
        q_val = -critic_model(data, x, edge_attr).sum().item()
    
    return q_val


def V_value_GNN(
    policy: CombinatorialACPolicy,
    s: torch.Tensor,
    s_c: torch.Tensor,
    critic_model: nn.Module,
    method: str,
    instance: Any,
    **kwargs
) -> float:
    """
    Calculate V-value (state value) using GNN critic.
    
    Args:
        policy: Policy
        s: Actor embedding
        s_c: Critic embedding
        critic_model: Critic network
        method: "on_policy" or "off_policy"
        instance: State instance
        **kwargs: Additional arguments
        
    Returns:
        V-value estimate
    """
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


def evaluate_policy(
    policy: CombinatorialACPolicy,
    envs: List[Any],
    nb_episodes: int = 1000,
    return_scores: bool = False,
    rng: Optional[np.random.Generator] = None,
    perturb: bool = True,
    **kwargs
) -> Tuple[float, Optional[List[float]]]:
    """
    Evaluate policy on environments.
    
    Args:
        policy: Policy to evaluate
        envs: List of environments
        nb_episodes: Number of episodes per environment
        return_scores: Whether to return individual scores
        rng: Random number generator
        perturb: Whether to use perturbation
        **kwargs: Additional arguments
        
    Returns:
        Mean reward (and optionally list of rewards)
    """
    score_per_trajectory = []
    
    for env in envs:
        for _ in range(nb_episodes):
            env.reset()
            trajectory_score = 0.0
            
            while not env.is_terminated():
                # Get action from policy
                policy_output = policy(env, rng=rng, perturb=perturb, **kwargs)
                a = policy_output['a']
                
                # Apply action
                _, reward = env.apply_action(a)
                trajectory_score += reward
            
            score_per_trajectory.append(trajectory_score)
    
    mean_reward = np.mean(score_per_trajectory)
    
    if return_scores:
        return mean_reward, score_per_trajectory
    else:
        return mean_reward, None
