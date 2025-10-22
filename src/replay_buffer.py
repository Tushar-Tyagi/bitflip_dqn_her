"""
Standard Replay Buffer for DQN
"""

import numpy as np
import torch
from typing import Tuple
import random


class ReplayBuffer:
    """
    Standard replay buffer for storing and sampling transitions.
    """
    
    def __init__(self, capacity: int, observation_size: int):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            observation_size: Size of observation vector
        """
        self.capacity = capacity
        self.observation_size = observation_size
        self.position = 0
        self.size = 0
        
        # Preallocate memory
        self.observations = np.zeros((capacity, observation_size), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_observations = np.zeros((capacity, observation_size), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
    
    def add(self, observation: np.ndarray, action: int, reward: float, 
            next_observation: np.ndarray, done: bool):
        """
        Add a transition to the buffer.
        
        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Whether episode terminated
        """
        self.observations[self.position] = observation
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_observations[self.position] = next_observation
        self.dones[self.position] = float(done)
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int, device: torch.device = torch.device('cpu')) -> Tuple[torch.Tensor, ...]:
        """
        Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            device: Device to move tensors to
            
        Returns:
            Tuple of (observations, actions, rewards, next_observations, dones)
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        
        observations = torch.FloatTensor(self.observations[indices]).to(device)
        actions = torch.LongTensor(self.actions[indices]).to(device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(device)
        next_observations = torch.FloatTensor(self.next_observations[indices]).to(device)
        dones = torch.FloatTensor(self.dones[indices]).to(device)
        
        return observations, actions, rewards, next_observations, dones
    
    def __len__(self) -> int:
        """Return current size of the buffer."""
        return self.size
    
    def clear(self):
        """Clear the buffer."""
        self.position = 0
        self.size = 0


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay buffer.
    """
    
    def __init__(self, capacity: int, observation_size: int, alpha: float = 0.6, beta: float = 0.4):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            observation_size: Size of observation vector
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent
        """
        super().__init__(capacity, observation_size)
        self.alpha = alpha
        self.beta = beta
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.max_priority = 1.0
    
    def add(self, observation: np.ndarray, action: int, reward: float, 
            next_observation: np.ndarray, done: bool):
        """Add transition with maximum priority."""
        super().add(observation, action, reward, next_observation, done)
        # New transitions get maximum priority
        self.priorities[self.position - 1] = self.max_priority
    
    def sample(self, batch_size: int, device: torch.device = torch.device('cpu')) -> Tuple[torch.Tensor, ...]:
        """
        Sample batch using prioritized sampling.
        
        Returns:
            Tuple of (observations, actions, rewards, next_observations, dones, indices, weights)
        """
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size] ** self.alpha
        probabilities = priorities / priorities.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(self.size, batch_size, p=probabilities, replace=False)
        
        # Calculate importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize
        
        observations = torch.FloatTensor(self.observations[indices]).to(device)
        actions = torch.LongTensor(self.actions[indices]).to(device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(device)
        next_observations = torch.FloatTensor(self.next_observations[indices]).to(device)
        dones = torch.FloatTensor(self.dones[indices]).to(device)
        weights = torch.FloatTensor(weights).to(device)
        
        return observations, actions, rewards, next_observations, dones, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """
        Update priorities for sampled transitions.
        
        Args:
            indices: Indices of transitions
            priorities: New priority values (typically TD errors)
        """
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())


def test_replay_buffer():
    """Test replay buffer functionality."""
    buffer = ReplayBuffer(capacity=1000, observation_size=50)
    
    # Add some transitions
    for i in range(100):
        obs = np.random.randn(50).astype(np.float32)
        action = np.random.randint(0, 25)
        reward = np.random.randn()
        next_obs = np.random.randn(50).astype(np.float32)
        done = np.random.rand() < 0.1
        
        buffer.add(obs, action, reward, next_obs, done)
    
    # Sample a batch
    batch = buffer.sample(32)
    print(f"Buffer size: {len(buffer)}")
    print(f"Batch shapes:")
    for i, tensor in enumerate(batch):
        print(f"  Tensor {i}: {tensor.shape}")


if __name__ == "__main__":
    test_replay_buffer()