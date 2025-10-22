"""
DQN-HER Agent Implementation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List

import sys
sys.path.append('src')

from network import DQN
from her_buffer import HERBuffer


class DQNHERAgent:
    """
    DQN agent with Hindsight Experience Replay for BitFlip environment.
    """
    
    def __init__(self,
                 observation_size: int,
                 action_size: int,
                 goal_size: int,
                 learning_rate: float = 0.001,
                 gamma: float = 0.98,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 target_update_freq: int = 100,
                 batch_size: int = 128,
                 buffer_size: int = 100000,
                 hidden_sizes: List[int] = [256, 256],
                 her_strategy: str = 'future',
                 her_k: int = 4,
                 device: str = 'cpu'):
        """
        Initialize DQN-HER agent.
        
        Args:
            observation_size: Size of observation space
            action_size: Size of action space
            goal_size: Size of goal space
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Exploration decay rate
            target_update_freq: Frequency of target network updates
            batch_size: Batch size for training
            buffer_size: Size of replay buffer
            hidden_sizes: Hidden layer sizes for network
            her_strategy: HER strategy ('future', 'final', 'episode')
            her_k: Number of additional goals to sample
            device: Device to use (cpu/cuda)
        """
        self.observation_size = observation_size
        self.action_size = action_size
        self.goal_size = goal_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Networks
        self.policy_net = DQN(observation_size, action_size, hidden_sizes).to(self.device)
        self.target_net = DQN(observation_size, action_size, hidden_sizes).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # HER Replay buffer
        self.replay_buffer = HERBuffer(
            buffer_size, 
            observation_size, 
            goal_size,
            her_strategy=her_strategy,
            her_k=her_k
        )
        
        # Training statistics
        self.update_count = 0
        self.episode_count = 0
        
    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            observation: Current observation
            training: Whether in training mode (uses exploration)
            
        Returns:
            Selected action
        """
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(0, self.action_size)
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            q_values = self.policy_net(obs_tensor)
            return q_values.argmax(dim=1).item()
    
    def store_transition(self, observation: np.ndarray, action: int, reward: float,
                        next_observation: np.ndarray, done: bool,
                        achieved_goal: np.ndarray, desired_goal: np.ndarray,
                        next_achieved_goal: np.ndarray):
        """Store transition in HER buffer."""
        self.replay_buffer.add_step(
            observation, action, reward, next_observation, done,
            achieved_goal, desired_goal, next_achieved_goal
        )
    
    def end_episode(self):
        """Called at the end of each episode to process HER."""
        self.replay_buffer.end_episode()
        self.episode_count += 1
    
    def update(self) -> Dict[str, float]:
        """
        Perform one update step.
        
        Returns:
            Dictionary containing training metrics
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        # Sample batch
        observations, actions, rewards, next_observations, dones = \
            self.replay_buffer.sample(self.batch_size, self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(observations).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_observations).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = nn.functional.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        
        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return {
            'loss': loss.item(),
            'epsilon': self.epsilon,
            'q_value': current_q_values.mean().item()
        }
    
    def save(self, path: str):
        """Save agent to file."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_count': self.update_count,
            'episode_count': self.episode_count
        }, path)
    
    def load(self, path: str):
        """Load agent from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.update_count = checkpoint['update_count']
        self.episode_count = checkpoint['episode_count']