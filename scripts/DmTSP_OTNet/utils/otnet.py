"""
OT-Net: Optimal Transport Dispatch Network for D-mTSP.

Architecture:
1. GNN Encoder: Encodes agents and tasks into embeddings
2. OT Layer: Solves entropic-regularized optimal transport
3. Policy: Combines encoder and OT layer for decision making
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np

try:
    from torch_geometric.nn import GINConv, GAT, global_mean_pool, global_max_pool
    from torch_geometric.data import Data, Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("Warning: torch_geometric not available. OT-Net will use simplified architecture.")


class OTNetEncoder(nn.Module):
    """
    GNN Encoder for OT-Net.
    
    Processes dynamic graph with:
    - Agent nodes: position, queue length, remaining service time, historical load
    - Task nodes (new + pending): position, arrival time, priority
    
    Outputs agent embeddings h_i^agent and task embeddings h_j^task.
    """
    
    def __init__(
        self,
        agent_feature_dim: int = 8,
        task_feature_dim: int = 6,
        hidden_dim: int = 64,
        num_layers: int = 3,
        use_gnn: bool = True,
    ):
        """
        Initialize encoder.
        
        Args:
            agent_feature_dim: Dimension of agent features
            task_feature_dim: Dimension of task features
            hidden_dim: Hidden dimension for GNN
            num_layers: Number of GNN layers
            use_gnn: Whether to use GNN (requires torch_geometric)
        """
        super().__init__()
        
        self.agent_feature_dim = agent_feature_dim
        self.task_feature_dim = task_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_gnn = use_gnn and TORCH_GEOMETRIC_AVAILABLE
        
        if self.use_gnn:
            # Agent encoder
            self.agent_input = nn.Linear(agent_feature_dim, hidden_dim)
            
            # Task encoder
            self.task_input = nn.Linear(task_feature_dim, hidden_dim)
            
            # GNN layers
            self.gin_layers = nn.ModuleList()
            for _ in range(num_layers):
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                self.gin_layers.append(GINConv(mlp))
            
            # Output projection
            self.agent_output = nn.Linear(hidden_dim, hidden_dim)
            self.task_output = nn.Linear(hidden_dim, hidden_dim)
        else:
            # Fallback: simple MLP encoder
            self.agent_encoder = nn.Sequential(
                nn.Linear(agent_feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            
            self.task_encoder = nn.Sequential(
                nn.Linear(task_feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
    
    def forward(
        self,
        agent_features: torch.Tensor,
        task_features: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            agent_features: [num_agents, agent_feature_dim]
            task_features: [num_tasks, task_feature_dim]
            edge_index: [2, num_edges] for GNN (optional)
        
        Returns:
            Tuple of (agent_embeddings, task_embeddings)
            - agent_embeddings: [num_agents, hidden_dim]
            - task_embeddings: [num_tasks, hidden_dim]
        """
        if self.use_gnn and edge_index is not None:
            # GNN-based encoding
            num_agents = agent_features.shape[0]
            num_tasks = task_features.shape[0]
            
            # Project to hidden dimension
            h_agents = self.agent_input(agent_features)
            h_tasks = self.task_input(task_features)
            
            # Concatenate for GNN
            x = torch.cat([h_agents, h_tasks], dim=0)
            
            # Apply GIN layers
            for gin_layer in self.gin_layers:
                x = F.relu(gin_layer(x, edge_index))
            
            # Split back
            agent_embeddings = self.agent_output(x[:num_agents])
            task_embeddings = self.task_output(x[num_agents:])
        else:
            # MLP-based encoding
            agent_embeddings = self.agent_encoder(agent_features)
            task_embeddings = self.task_encoder(task_features)
        
        return agent_embeddings, task_embeddings


class OTLayer(nn.Module):
    """
    Optimal Transport Layer using Sinkhorn algorithm.
    
    Solves entropic-regularized OT:
        min_{T∈Π(b,a)} <C,T> + ε·H(T)
    
    where:
    - C: cost matrix (learned from embeddings)
    - b: supply (agent capacities)
    - a: demand (uniform over tasks)
    - ε: entropic regularization parameter
    """
    
    def __init__(
        self,
        embedding_dim: int = 64,
        epsilon: float = 0.1,
        num_iterations: int = 50,
    ):
        """
        Initialize OT layer.
        
        Args:
            embedding_dim: Dimension of embeddings
            epsilon: Entropic regularization parameter
            num_iterations: Number of Sinkhorn iterations
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.epsilon = epsilon
        self.num_iterations = num_iterations
        
        # Learnable parameters for cost matrix
        self.W_q = nn.Linear(embedding_dim, embedding_dim)  # Query projection
        self.W_k = nn.Linear(embedding_dim, embedding_dim)  # Key projection
        
        # MLP for supply prior
        self.supply_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1),
        )
    
    def forward(
        self,
        agent_embeddings: torch.Tensor,
        task_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass: solve OT problem.
        
        Args:
            agent_embeddings: [num_agents, embedding_dim]
            task_embeddings: [num_tasks, embedding_dim]
        
        Returns:
            Transport matrix T: [num_agents, num_tasks]
        """
        num_agents = agent_embeddings.shape[0]
        num_tasks = task_embeddings.shape[0]
        
        # Compute cost matrix: C_ij = -||q_i - k_j||^2
        q = self.W_q(agent_embeddings)  # [num_agents, embedding_dim]
        k = self.W_k(task_embeddings)    # [num_tasks, embedding_dim]
        
        # Pairwise squared distances
        # ||q_i - k_j||^2 = ||q_i||^2 + ||k_j||^2 - 2<q_i, k_j>
        q_norm = (q ** 2).sum(dim=1, keepdim=True)  # [num_agents, 1]
        k_norm = (k ** 2).sum(dim=1, keepdim=True)  # [num_tasks, 1]
        distances = q_norm + k_norm.t() - 2 * torch.mm(q, k.t())  # [num_agents, num_tasks]
        
        C = -distances  # Cost matrix (negative distance as similarity)
        
        # Compute supply prior: b_i = softmax(MLP(h_i^agent))
        supply_logits = self.supply_mlp(agent_embeddings).squeeze(-1)  # [num_agents]
        b = F.softmax(supply_logits, dim=0)  # [num_agents]
        
        # Demand: uniform over tasks
        a = torch.ones(num_tasks, device=task_embeddings.device) / num_tasks  # [num_tasks]
        
        # Sinkhorn algorithm
        T = self.sinkhorn(C, b, a)
        
        return T
    
    def sinkhorn(
        self,
        C: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sinkhorn algorithm for entropic OT with log-stabilization.
        
        Args:
            C: Cost matrix [num_agents, num_tasks]
            b: Supply distribution [num_agents]
            a: Demand distribution [num_tasks]
        
        Returns:
            Transport matrix T [num_agents, num_tasks]
        """
        # Log-stabilized Sinkhorn
        log_b = torch.log(b + 1e-20)
        log_a = torch.log(a + 1e-20)
        log_K = -C / self.epsilon
        
        # Initialize dual variables
        f = torch.zeros_like(b)
        g = torch.zeros_like(a)
        
        # Sinkhorn iterations in log-space
        for _ in range(self.num_iterations):
            # Update f
            log_sum_exp = torch.logsumexp(log_K + g.unsqueeze(0), dim=1)
            f = log_b - log_sum_exp
            
            # Update g
            log_sum_exp = torch.logsumexp(log_K + f.unsqueeze(1), dim=0)
            g = log_a - log_sum_exp
        
        # Compute transport matrix in log-space
        log_T = f.unsqueeze(1) + log_K + g.unsqueeze(0)
        T = torch.exp(log_T)
        
        # Ensure non-negative and normalize
        T = torch.clamp(T, min=0.0)
        T = T / (T.sum() + 1e-8)
        
        return T


class OTNetPolicy(nn.Module):
    """
    OT-Net Policy for D-mTSP.
    
    Combines GNN encoder and OT layer to produce assignment decisions.
    """
    
    def __init__(
        self,
        agent_feature_dim: int = 8,
        task_feature_dim: int = 6,
        hidden_dim: int = 64,
        num_gnn_layers: int = 3,
        epsilon: float = 0.1,
        num_sinkhorn_iters: int = 50,
        use_gnn: bool = True,
    ):
        """
        Initialize OT-Net policy.
        
        Args:
            agent_feature_dim: Dimension of agent features
            task_feature_dim: Dimension of task features
            hidden_dim: Hidden dimension
            num_gnn_layers: Number of GNN layers
            epsilon: OT regularization parameter
            num_sinkhorn_iters: Number of Sinkhorn iterations
            use_gnn: Whether to use GNN
        """
        super().__init__()
        
        self.encoder = OTNetEncoder(
            agent_feature_dim=agent_feature_dim,
            task_feature_dim=task_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
            use_gnn=use_gnn,
        )
        
        self.ot_layer = OTLayer(
            embedding_dim=hidden_dim,
            epsilon=epsilon,
            num_iterations=num_sinkhorn_iters,
        )
    
    def forward(
        self,
        agent_features: torch.Tensor,
        task_features: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Forward pass: compute assignment matrix.
        
        Args:
            agent_features: [num_agents, agent_feature_dim]
            task_features: [num_tasks, task_feature_dim]
            edge_index: [2, num_edges] for GNN (optional)
            temperature: Temperature for Gumbel-Softmax sampling
        
        Returns:
            Assignment matrix Y: [num_agents, num_tasks]
        """
        # Encode
        agent_embeddings, task_embeddings = self.encoder(
            agent_features, task_features, edge_index
        )
        
        # Solve OT
        T = self.ot_layer(agent_embeddings, task_embeddings)
        
        # Convert transport to assignment (during inference)
        # During training, we use soft assignments
        return T
    
    def get_assignment(
        self,
        agent_features: torch.Tensor,
        task_features: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Get discrete assignment matrix for inference.
        
        Args:
            agent_features: [num_agents, agent_feature_dim]
            task_features: [num_tasks, task_feature_dim]
            edge_index: [2, num_edges] for GNN (optional)
            threshold: Threshold for binarization
        
        Returns:
            Binary assignment matrix Y: [num_agents, num_tasks]
        """
        with torch.no_grad():
            T = self.forward(agent_features, task_features, edge_index)
            
            # Greedy assignment: each task to agent with highest transport
            Y = torch.zeros_like(T)
            
            for j in range(T.shape[1]):  # For each task
                i = torch.argmax(T[:, j])  # Find best agent
                if T[i, j] > threshold:
                    Y[i, j] = 1.0
            
            return Y.cpu().numpy()
