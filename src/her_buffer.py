"""
Hindsight Experience Replay (HER) Buffer
"""

import numpy as np
import torch
from typing import List, Tuple, Dict
import random


class HERBuffer:
    """
    Hindsight Experience Replay buffer for goal-conditioned RL.
    Stores episodes and generates additional training data by relabeling goals.
    """
    
    def __init__(self, capacity: int, observation_size: int, goal_size: int, 
                 her_strategy: str = 'future', her_k: int = 4):
        """
        Initialize HER buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            observation_size: Size of observation vector (state + goal)
            goal_size: Size of goal vector
            her_strategy: HER strategy ('future', 'final', 'episode', 'random')
            her_k: Number of additional goals to sample per transition
        """
        self.capacity = capacity
        self.observation_size = observation_size
        self.goal_size = goal_size
        self.her_strategy = her_strategy
        self.her_k = her_k
        
        self.position = 0
        self.size = 0
        
        # Store complete episodes
        self.episodes = []
        self.current_episode = []
        
        # Preallocate memory for transitions
        self.observations = np.zeros((capacity, observation_size), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_observations = np.zeros((capacity, observation_size), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
    
    def add_step(self, observation: np.ndarray, action: int, reward: float,
                 next_observation: np.ndarray, done: bool, 
                 achieved_goal: np.ndarray, desired_goal: np.ndarray,
                 next_achieved_goal: np.ndarray):
        """
        Add a step to the current episode.
        
        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Whether episode terminated
            achieved_goal: Goal achieved in current state
            desired_goal: Desired goal
            next_achieved_goal: Goal achieved in next state
        """
        transition = {
            'observation': observation.copy(),
            'action': action,
            'reward': reward,
            'next_observation': next_observation.copy(),
            'done': done,
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': desired_goal.copy(),
            'next_achieved_goal': next_achieved_goal.copy()
        }
        self.current_episode.append(transition)
    
    def end_episode(self):
        """
        End current episode and store it with HER relabeling.
        """
        if len(self.current_episode) == 0:
            return
        
        episode_length = len(self.current_episode)
        
        # Store original transitions
        for t, transition in enumerate(self.current_episode):
            self._store_transition(
                transition['observation'],
                transition['action'],
                transition['reward'],
                transition['next_observation'],
                transition['done']
            )
        
        # Generate HER transitions
        for t, transition in enumerate(self.current_episode):
            # Sample additional goals based on strategy
            sampled_goals = self._sample_her_goals(t, episode_length)
            
            for new_goal in sampled_goals:
                # Reconstruct observation with new goal
                state_size = self.goal_size
                new_obs = np.concatenate([
                    transition['observation'][:state_size],
                    new_goal
                ])
                new_next_obs = np.concatenate([
                    transition['next_observation'][:state_size],
                    new_goal
                ])
                
                # Compute new reward
                new_reward = self._compute_reward(
                    transition['next_achieved_goal'],
                    new_goal
                )
                
                # Check if new goal is achieved
                new_done = np.array_equal(transition['next_achieved_goal'], new_goal)
                
                self._store_transition(
                    new_obs,
                    transition['action'],
                    new_reward,
                    new_next_obs,
                    new_done
                )
        
        # Keep episode history (optional, for analysis)
        if len(self.episodes) < 1000:  # Limit episode storage
            self.episodes.append(self.current_episode)
        
        # Clear current episode
        self.current_episode = []
    
    def _sample_her_goals(self, t: int, episode_length: int) -> List[np.ndarray]:
        """
        Sample goals for HER relabeling based on strategy.
        
        Args:
            t: Current timestep in episode
            episode_length: Total episode length
            
        Returns:
            List of sampled goals
        """
        goals = []
        
        for _ in range(self.her_k):
            if self.her_strategy == 'future':
                # Sample from future states in the same episode
                if t < episode_length - 1:
                    future_idx = np.random.randint(t + 1, episode_length)
                    goals.append(self.current_episode[future_idx]['achieved_goal'])
            
            elif self.her_strategy == 'final':
                # Use final achieved goal
                goals.append(self.current_episode[-1]['achieved_goal'])
            
            elif self.her_strategy == 'episode':
                # Sample from any state in the episode
                random_idx = np.random.randint(0, episode_length)
                goals.append(self.current_episode[random_idx]['achieved_goal'])
            
            elif self.her_strategy == 'random':
                # Sample random goal (same shape as goal)
                random_goal = np.random.randint(0, 2, size=self.goal_size).astype(np.float32)
                goals.append(random_goal)
        
        return goals
    
    def _compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> float:
        """
        Compute reward for achieved and desired goals.
        
        Args:
            achieved_goal: Goal that was achieved
            desired_goal: Goal that was desired
            
        Returns:
            Reward (0 if match, -1 otherwise)
        """
        return 0.0 if np.array_equal(achieved_goal, desired_goal) else -1.0
    
    def _store_transition(self, observation: np.ndarray, action: int, reward: float,
                         next_observation: np.ndarray, done: bool):
        """Store a single transition in the buffer."""
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
        self.episodes = []
        self.current_episode = []


def test_her_buffer():
    """Test HER buffer functionality."""
    n_bits = 5
    buffer = HERBuffer(
        capacity=10000,
        observation_size=2 * n_bits,
        goal_size=n_bits,
        her_strategy='future',
        her_k=4
    )
    
    # Simulate an episode
    state = np.random.randint(0, 2, size=n_bits).astype(np.float32)
    goal = np.random.randint(0, 2, size=n_bits).astype(np.float32)
    
    for step in range(5):
        obs = np.concatenate([state, goal])
        action = np.random.randint(0, n_bits)
        
        # Flip a bit
        next_state = state.copy()
        next_state[action] = 1 - next_state[action]
        next_obs = np.concatenate([next_state, goal])
        
        reward = 0.0 if np.array_equal(next_state, goal) else -1.0
        done = np.array_equal(next_state, goal)
        
        buffer.add_step(
            obs, action, reward, next_obs, done,
            state, goal, next_state
        )
        
        state = next_state
        if done:
            break
    
    buffer.end_episode()
    
    print(f"Buffer size after one episode: {len(buffer)}")
    print(f"Expected: ~{5 + 5 * 4} transitions (original + HER)")
    
    # Sample a batch
    if len(buffer) >= 16:
        batch = buffer.sample(16)
        print(f"Batch sampled successfully")


if __name__ == "__main__":
    test_her_buffer()