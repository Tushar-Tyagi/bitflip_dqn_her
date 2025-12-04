"""
Generic Prioritized Replay Buffer Base Class

"""

import numpy as np
import torch
from typing import Tuple, Optional
from abc import ABC, abstractmethod

from sum_tree import SumTree, MinTree
from priority_computer import PriorityComputer, TDErrorPriorityComputer


class PrioritizedBufferBase(ABC):
    """
    Base class for prioritized experience replay buffers.
    
    Handles:
    - Priority-based sampling using sum-tree
    - Importance sampling weight computation
    - Dynamic priority updates
    - Pluggable priority computation strategies
    
    Subclasses must implement:
    - store_transition(): How to store a single transition
    - get_transition(): How to retrieve a transition by index
    """
    
    def __init__(self,
                 capacity: int,
                 observation_size: int,
                 priority_compute: Optional[PriorityComputer] = None,
                 alpha: float = 0.6,
                 beta_start: float = 0.4,
                 beta_frames: int = 100000,
                 epsilon: float = 1e-6):
        """
        Initialize prioritized buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            observation_size: Size of observation vector
            priority_compute: Strategy for computing priorities (default: TD error)
            alpha: Priority exponent (0=uniform, 1=full prioritization)
            beta_start: Initial importance sampling exponent
            beta_frames: Number of frames over which to anneal beta to 1.0
            epsilon: Small constant added to priorities to ensure non-zero sampling
        """
        self.capacity = capacity
        self.observation_size = observation_size
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.frame = 0
        
        # Default to TD error if no priority computer provided
        self.priority_compute = priority_compute or TDErrorPriorityComputer(epsilon=epsilon)
        
        # Sum tree for efficient priority sampling
        self.sum_tree = SumTree(capacity)
        self.min_tree = MinTree(capacity)
        
        # Track max priority for new transitions
        self.max_priority = 1.0
        
        # Storage (to be initialized by subclass)
        self.position = 0
        self.size = 0
    
    @abstractmethod
    def _store_transition(self, *args, **kwargs):
        """Store a single transition. Must be implemented by subclass."""
        pass
    
    @abstractmethod
    def _get_transition(self, idx: int) -> Tuple:
        """Retrieve transition by index. Must be implemented by subclass."""
        pass
    
    def add(self, *args, **kwargs):
        """
        Add transition with maximum priority.
        
        New transitions get max priority to ensure they're sampled at least once.
        """
        # Store the transition
        data_idx = self.position
        self._store_transition(*args, **kwargs)
        
        # Add to priority tree with max priority
        priority = self.max_priority ** self.alpha
        self.sum_tree.add(priority, data_idx)
        self.min_tree.update(data_idx, priority)
        
        # Update position and size
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int, device: torch.device = torch.device('cpu'),
               agent: object = None) -> Tuple:
        """
        Sample batch with priority-based sampling.
        
        Args:
            batch_size: Number of transitions to sample
            device: Device to move tensors to
            agent: Agent for computing priorities (optional)
            
        Returns:
            Tuple of (batch_data..., indices, weights)
        """
        if self.size < batch_size:
            raise ValueError(f"Not enough samples: {self.size} < {batch_size}")
        
        indices = np.zeros(batch_size, dtype=np.int32)
        priorities = np.zeros(batch_size, dtype=np.float32)
        
        # Sample proportional to priority
        segment_size = self.sum_tree.total() / batch_size
        
        for i in range(batch_size):
            # Sample uniformly from segment
            left = segment_size * i
            right = segment_size * (i + 1)
            value = np.random.uniform(left, right)
            
            # Get corresponding transition
            tree_idx, data_idx, priority = self.sum_tree.get(value)
            indices[i] = data_idx
            priorities[i] = priority
        
        # Compute importance sampling weights
        weights = self._compute_weights(priorities)
        
        # Retrieve transitions
        batch = self._get_batch(indices, device)
        
        # Add indices and weights to batch
        indices_tensor = torch.LongTensor(indices).to(device)
        weights_tensor = torch.FloatTensor(weights).to(device)
        
        return (*batch, indices_tensor, weights_tensor)
    
    def _compute_weights(self, priorities: np.ndarray) -> np.ndarray:
        """
        Compute importance sampling weights.
        
        Args:
            priorities: Sampled priorities
            
        Returns:
            Normalized importance sampling weights
        """
        # Anneal beta from beta_start to 1.0
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1
        
        # Compute weights: (N * P(i))^(-beta)
        min_priority = self.min_tree.min()
        max_weight = (min_priority / self.sum_tree.total() * self.size) ** (-beta)
        
        weights = []
        for priority in priorities:
            prob = priority / self.sum_tree.total()
            weight = (prob * self.size) ** (-beta)
            normalized_weight = weight / max_weight
            weights.append(normalized_weight)
        
        return np.array(weights, dtype=np.float32)
    
    @abstractmethod
    def _get_batch(self, indices: np.ndarray, device: torch.device) -> Tuple:
        """
        Retrieve batch of transitions by indices.
        
        Args:
            indices: Array of data indices
            device: Device to move tensors to
            
        Returns:
            Tuple of batch tensors
        """
        pass
    
    def update_priorities(self, indices: torch.Tensor, priorities: np.ndarray):
        """
        Update priorities for sampled transitions.
        
        Args:
            indices: Indices of transitions (can be tensor or array)
            priorities: New priority values
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()
        
        for idx, priority in zip(indices, priorities):
            # Clip priority to avoid numerical issues
            priority = max(priority, self.epsilon)
            
            # Apply alpha exponent
            priority_alpha = priority ** self.alpha
            
            # Update trees
            tree_idx = int(idx) + self.sum_tree.capacity - 1
            self.sum_tree.update(tree_idx, priority_alpha)
            self.min_tree.update(int(idx), priority_alpha)
            
            # Track max priority
            self.max_priority = max(self.max_priority, priority)
    
    def update_priorities_from_batch(self,
                                     indices: torch.Tensor,
                                     observations: torch.Tensor,
                                     actions: torch.Tensor,
                                     rewards: torch.Tensor,
                                     next_observations: torch.Tensor,
                                     dones: torch.Tensor,
                                     agent: object):
        """
        Compute and update priorities using the priority computer.
        
        Args:
            indices: Indices of sampled transitions
            observations: Batch of observations
            actions: Batch of actions
            rewards: Batch of rewards
            next_observations: Batch of next observations
            dones: Batch of done flags
            agent: Agent for computing priorities
        """
        # Compute priorities using strategy
        priorities = self.priority_compute.compute_priority(
            observations, actions, rewards, next_observations, dones, agent
        )
        
        # Update priority trees
        self.update_priorities(indices, priorities)
    
    def __len__(self) -> int:
        """Return current size of the buffer."""
        return self.size
    
    def clear(self):
        """Clear the buffer."""
        self.position = 0
        self.size = 0
        self.sum_tree = SumTree(self.capacity)
        self.min_tree = MinTree(self.capacity)
        self.max_priority = 1.0
        self.frame = 0


class PrioritizedReplayBuffer(PrioritizedBufferBase):
    """
    Standard prioritized replay buffer for DQN.
    
    Stores (observation, action, reward, next_observation, done) transitions
    and samples with priority-based sampling.
    """
    
    def __init__(self,
                 capacity: int,
                 observation_size: int,
                 priority_compute: Optional[PriorityComputer] = None,
                 alpha: float = 0.6,
                 beta_start: float = 0.4,
                 beta_frames: int = 100000,
                 epsilon: float = 1e-6):
        """Initialize prioritized replay buffer."""
        super().__init__(capacity, observation_size, priority_compute,
                        alpha, beta_start, beta_frames, epsilon)
        
        # Preallocate storage
        self.observations = np.zeros((capacity, observation_size), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_observations = np.zeros((capacity, observation_size), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
    
    def _store_transition(self, observation: np.ndarray, action: int, reward: float,
                         next_observation: np.ndarray, done: bool):
        """Store a single transition."""
        self.observations[self.position] = observation
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_observations[self.position] = next_observation
        self.dones[self.position] = float(done)
    
    def _get_transition(self, idx: int) -> Tuple:
        """Retrieve a single transition by index."""
        return (
            self.observations[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_observations[idx],
            self.dones[idx]
        )
    
    def _get_batch(self, indices: np.ndarray, device: torch.device) -> Tuple:
        """Retrieve batch of transitions."""
        observations = torch.FloatTensor(self.observations[indices]).to(device)
        actions = torch.LongTensor(self.actions[indices]).to(device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(device)
        next_observations = torch.FloatTensor(self.next_observations[indices]).to(device)
        dones = torch.FloatTensor(self.dones[indices]).to(device)
        
        return observations, actions, rewards, next_observations, dones
    
    def clear(self):
        """Clear the buffer and reset storage."""
        super().clear()
        self.observations.fill(0)
        self.actions.fill(0)
        self.rewards.fill(0)
        self.next_observations.fill(0)
        self.dones.fill(0)


if __name__ == "__main__":
    print("Prioritized Buffer Base Implementation")
    print("--------------------------------------")
