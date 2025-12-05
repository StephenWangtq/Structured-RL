"""
D-mTSP (Dynamic Multi-Traveling Salesman Problem) Environment.

Implements an event-driven environment where:
- M agents (vehicles) start at depot
- Customer requests arrive dynamically (Poisson process)
- At each request arrival, make assignment decisions
- Objective: minimize α·makespan + β·average_waiting_time
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import copy


@dataclass
class CustomerRequest:
    """Represents a customer request."""
    id: int
    location: np.ndarray  # (x, y) coordinates
    arrival_time: float
    service_time: float = 1.0
    
    def __repr__(self):
        return f"Request({self.id}, t={self.arrival_time:.2f})"


@dataclass
class Agent:
    """Represents a vehicle/agent."""
    id: int
    location: np.ndarray  # Current location
    route: List[CustomerRequest]  # Assigned tasks
    current_time: float  # Current time
    total_distance: float = 0.0
    
    def __repr__(self):
        return f"Agent({self.id}, loc={self.location}, route_len={len(self.route)})"


class DmTSPInstance:
    """Problem instance for D-mTSP."""
    
    def __init__(
        self,
        num_agents: int,
        depot_location: np.ndarray,
        time_horizon: float = 100.0,
        request_rate: float = 0.5,
        service_area: Tuple[float, float, float, float] = (0, 100, 0, 100),
        seed: Optional[int] = None,
    ):
        """
        Initialize D-mTSP instance.
        
        Args:
            num_agents: Number of agents (vehicles)
            depot_location: Depot coordinates (x, y)
            time_horizon: Total time horizon
            request_rate: Poisson arrival rate (requests per time unit)
            service_area: (x_min, x_max, y_min, y_max) service area bounds
            seed: Random seed
        """
        self.num_agents = num_agents
        self.depot_location = np.array(depot_location, dtype=np.float32)
        self.time_horizon = time_horizon
        self.request_rate = request_rate
        self.service_area = service_area
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
    def generate_requests(self) -> List[CustomerRequest]:
        """Generate customer requests following Poisson process."""
        # Number of requests
        num_requests = self.rng.poisson(self.request_rate * self.time_horizon)
        
        # Generate arrival times
        arrival_times = np.sort(self.rng.uniform(0, self.time_horizon, num_requests))
        
        # Generate locations
        x_min, x_max, y_min, y_max = self.service_area
        requests = []
        
        for i, t in enumerate(arrival_times):
            x = self.rng.uniform(x_min, x_max)
            y = self.rng.uniform(y_min, y_max)
            location = np.array([x, y], dtype=np.float32)
            
            request = CustomerRequest(
                id=i,
                location=location,
                arrival_time=t,
                service_time=1.0,
            )
            requests.append(request)
        
        return requests


class DmTSPState:
    """Current state of D-mTSP environment."""
    
    def __init__(
        self,
        current_time: float,
        agents: List[Agent],
        new_requests: List[CustomerRequest],
        pending_requests: List[CustomerRequest],
        served_requests: List[CustomerRequest],
    ):
        """
        Initialize state.
        
        Args:
            current_time: Current time
            agents: List of agents with their current status
            new_requests: Newly arrived requests to assign
            pending_requests: Previously unassigned requests
            served_requests: Completed requests
        """
        self.current_time = current_time
        self.agents = agents
        self.new_requests = new_requests
        self.pending_requests = pending_requests
        self.served_requests = served_requests
    
    def get_all_unassigned(self) -> List[CustomerRequest]:
        """Get all unassigned requests (new + pending)."""
        return self.new_requests + self.pending_requests
    
    def __repr__(self):
        return (
            f"State(t={self.current_time:.2f}, "
            f"new={len(self.new_requests)}, "
            f"pending={len(self.pending_requests)}, "
            f"agents={len(self.agents)})"
        )


class DmTSPEnv:
    """
    D-mTSP Environment with event-driven dynamics.
    
    At each event (new request arrival):
    - State includes: agent positions, routes, new requests, pending requests
    - Action: assignment matrix Y ∈ {0,1}^(M×K)
    - Transition: insert assigned tasks using cheapest insertion, simulate until next event
    """
    
    def __init__(
        self,
        instance: DmTSPInstance,
        alpha: float = 0.5,
        beta: float = 0.5,
    ):
        """
        Initialize environment.
        
        Args:
            instance: Problem instance
            alpha: Weight for makespan in objective
            beta: Weight for average waiting time in objective
        """
        self.instance = instance
        self.alpha = alpha
        self.beta = beta
        
        # Generate all requests at initialization
        self.all_requests = instance.generate_requests()
        self.request_idx = 0
        
        # Initialize agents
        self.agents = [
            Agent(
                id=i,
                location=instance.depot_location.copy(),
                route=[],
                current_time=0.0,
            )
            for i in range(instance.num_agents)
        ]
        
        # State tracking
        self.current_time = 0.0
        self.pending_requests = []
        self.served_requests = []
        self.episode_reward = 0.0
        
    def reset(self) -> DmTSPState:
        """Reset environment to initial state."""
        # Regenerate requests
        self.all_requests = self.instance.generate_requests()
        self.request_idx = 0
        
        # Reset agents
        self.agents = [
            Agent(
                id=i,
                location=self.instance.depot_location.copy(),
                route=[],
                current_time=0.0,
            )
            for i in range(self.instance.num_agents)
        ]
        
        # Reset state
        self.current_time = 0.0
        self.pending_requests = []
        self.served_requests = []
        self.episode_reward = 0.0
        
        # Advance to first request arrival
        if len(self.all_requests) > 0:
            self.current_time = self.all_requests[0].arrival_time
        
        return self._get_current_state()
    
    def _get_current_state(self) -> DmTSPState:
        """Get current state without modifying environment."""
        # Get new requests at current time (don't modify request_idx)
        new_requests = []
        temp_idx = self.request_idx
        while (temp_idx < len(self.all_requests) and 
               self.all_requests[temp_idx].arrival_time <= self.current_time):
            new_requests.append(self.all_requests[temp_idx])
            temp_idx += 1
        
        return DmTSPState(
            current_time=self.current_time,
            agents=copy.deepcopy(self.agents),
            new_requests=new_requests,
            pending_requests=copy.deepcopy(self.pending_requests),
            served_requests=copy.deepcopy(self.served_requests),
        )
    
    def step(self, assignments: np.ndarray) -> Tuple[DmTSPState, float, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            assignments: Assignment matrix Y ∈ {0,1}^(M×K)
                        Y[i,j]=1 means task j assigned to agent i
        
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Get unassigned tasks (new + pending) - before modifying state
        all_unassigned = self.pending_requests.copy()
        
        # Get new requests at current time
        while (self.request_idx < len(self.all_requests) and 
               self.all_requests[self.request_idx].arrival_time <= self.current_time):
            all_unassigned.append(self.all_requests[self.request_idx])
            self.request_idx += 1
        
        # Record state before assignment for reward calculation
        prev_makespan = self._calculate_makespan()
        prev_total_waiting = self._calculate_total_waiting()
        
        # Assign tasks to agents based on assignment matrix
        for agent_idx in range(self.instance.num_agents):
            for task_idx in range(len(all_unassigned)):
                if assignments[agent_idx, task_idx] > 0.5:  # Assigned
                    request = all_unassigned[task_idx]
                    self._insert_task(agent_idx, request)
        
        # Update pending requests (unassigned tasks)
        assigned_ids = set()
        for agent_idx in range(self.instance.num_agents):
            for task_idx in range(len(all_unassigned)):
                if assignments[agent_idx, task_idx] > 0.5:
                    assigned_ids.add(all_unassigned[task_idx].id)
        
        self.pending_requests = [
            req for req in all_unassigned if req.id not in assigned_ids
        ]
        
        # Calculate reward (negative cost increase)
        new_makespan = self._calculate_makespan()
        new_total_waiting = self._calculate_total_waiting()
        
        delta_makespan = new_makespan - prev_makespan
        delta_waiting = new_total_waiting - prev_total_waiting
        
        reward = -(self.alpha * delta_makespan + self.beta * delta_waiting)
        self.episode_reward += reward
        
        # Advance time to next event
        if self.request_idx < len(self.all_requests):
            self.current_time = self.all_requests[self.request_idx].arrival_time
        else:
            self.current_time = self.instance.time_horizon
        
        # Check if done
        done = (self.request_idx >= len(self.all_requests) and 
                len(self.pending_requests) == 0)
        
        info = {
            'makespan': new_makespan,
            'total_waiting': new_total_waiting,
            'num_pending': len(self.pending_requests),
        }
        
        next_state = self._get_current_state()
        return next_state, reward, done, info
    
    def _insert_task(self, agent_idx: int, request: CustomerRequest):
        """
        Insert task into agent's route using cheapest insertion heuristic.
        
        Args:
            agent_idx: Index of agent
            request: Request to insert
        """
        agent = self.agents[agent_idx]
        
        if len(agent.route) == 0:
            # First task - simply add
            agent.route.append(request)
            return
        
        # Find cheapest insertion position
        best_pos = 0
        best_cost = float('inf')
        
        for pos in range(len(agent.route) + 1):
            cost = self._insertion_cost(agent, request, pos)
            if cost < best_cost:
                best_cost = cost
                best_pos = pos
        
        agent.route.insert(best_pos, request)
    
    def _insertion_cost(self, agent: Agent, request: CustomerRequest, position: int) -> float:
        """Calculate cost of inserting request at position in agent's route."""
        # Distance from depot to first location
        route = agent.route.copy()
        route.insert(position, request)
        
        total_dist = np.linalg.norm(self.instance.depot_location - route[0].location)
        
        for i in range(len(route) - 1):
            total_dist += np.linalg.norm(route[i].location - route[i+1].location)
        
        # Back to depot
        total_dist += np.linalg.norm(route[-1].location - self.instance.depot_location)
        
        return total_dist
    
    def _calculate_makespan(self) -> float:
        """Calculate maximum completion time across all agents."""
        max_time = 0.0
        
        for agent in self.agents:
            if len(agent.route) == 0:
                continue
            
            time = self.current_time
            loc = agent.location.copy()
            
            for request in agent.route:
                # Travel to request
                travel_time = np.linalg.norm(request.location - loc)
                time += travel_time
                # Service time
                time += request.service_time
                loc = request.location
            
            # Return to depot
            time += np.linalg.norm(self.instance.depot_location - loc)
            
            max_time = max(max_time, time)
        
        return max_time
    
    def _calculate_total_waiting(self) -> float:
        """Calculate total waiting time for all requests."""
        total_waiting = 0.0
        
        for agent in self.agents:
            time = self.current_time
            loc = agent.location.copy()
            
            for request in agent.route:
                # Travel to request
                travel_time = np.linalg.norm(request.location - loc)
                time += travel_time
                
                # Waiting time = service start - arrival
                waiting = max(0, time - request.arrival_time)
                total_waiting += waiting
                
                # Service time
                time += request.service_time
                loc = request.location
        
        # Pending requests have been waiting since arrival
        for request in self.pending_requests:
            waiting = self.current_time - request.arrival_time
            total_waiting += waiting
        
        return total_waiting
    
    def is_terminated(self) -> bool:
        """Check if episode is terminated."""
        return (self.request_idx >= len(self.all_requests) and 
                len(self.pending_requests) == 0)
    
    def get_total_cost(self) -> float:
        """Get total cost of current solution."""
        makespan = self._calculate_makespan()
        total_waiting = self._calculate_total_waiting()
        num_served = len(self.served_requests) + sum(len(a.route) for a in self.agents)
        
        if num_served > 0:
            avg_waiting = total_waiting / num_served
        else:
            avg_waiting = 0.0
        
        return self.alpha * makespan + self.beta * avg_waiting
