"""
Prioritized Hindsight Experience Replay (HER) Buffer
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Optional

from prioritized_buffer_base import PrioritizedBufferBase
from priority_computer import PriorityComputer #, TDErrorPriorityComputer


class PrioritizedHERBuffer(PrioritizedBufferBase):
    """
    HER buffer with priority-based sampling.

    """
    
    def __init__(self,
                 capacity: int,
                 observation_size: int,
                 goal_size: int,
                 priority_compute: Optional[PriorityComputer] = None,
                 her_strategy: str = 'future',
                 her_k: int = 4,
                 alpha: float = 0.6,
                 beta_start: float = 0.4,
                 beta_frames: int = 100000,
                 epsilon: float = 1e-6):
        """
        Initialize prioritized HER buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            observation_size: Size of observation vector (state + goal)
            goal_size: Size of goal vector
            priority_compute: Strategy for computing priorities
            her_strategy: HER relabeling strategy ('future', 'final', 'episode', 'random')
            her_k: Number of additional goals to sample per transition
            alpha: Priority exponent (0=uniform, 1=full prioritization)
            beta_start: Initial importance sampling exponent
            beta_frames: Number of frames to anneal beta to 1.0
            epsilon: Small constant for non-zero priorities
        """
        super().__init__(capacity, observation_size, priority_compute,
                        alpha, beta_start, beta_frames, epsilon)
        
        self.goal_size = goal_size
        self.her_strategy = her_strategy
        self.her_k = her_k
        
        # Storage for transitions
        self.observations = np.zeros((capacity, observation_size), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_observations = np.zeros((capacity, observation_size), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        
        # Current episode buffer
        self.current_episode = []
        
        # Episode history for analysis (optional)
        self.episodes = []
        self.max_episodes_stored = 1000
    
    def add_step(self, observation: np.ndarray, action: int, reward: float,
                 next_observation: np.ndarray, done: bool,
                 achieved_goal: np.ndarray, desired_goal: np.ndarray,
                 next_achieved_goal: np.ndarray):
        """
        Add a step to current episode.
        
        Args:
            observation: Current observation (state + goal)
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
        Process episode with HER relabeling and store transitions.
        
        For each transition:
        1. Store original transition
        2. Sample k additional goals using HER strategy
        3. Create relabeled transitions with new rewards
        4. Store all transitions with max priority
        """
        if len(self.current_episode) == 0:
            return
        
        episode_length = len(self.current_episode)
        
        # Store original transitions
        for t, transition in enumerate(self.current_episode):
            self.add(
                transition['observation'],
                transition['action'],
                transition['reward'],
                transition['next_observation'],
                transition['done']
            )
        
        # Generate HER transitions with relabeled goals
        for t, transition in enumerate(self.current_episode):
            # Sample additional goals
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
                
                # Compute reward for new goal
                new_reward = self._compute_reward(
                    transition['next_achieved_goal'],
                    new_goal
                )
                
                # Check if goal is achieved
                new_done = np.array_equal(transition['next_achieved_goal'], new_goal)
                
                # Store relabeled transition
                self.add(
                    new_obs,
                    transition['action'],
                    new_reward,
                    new_next_obs,
                    new_done
                )
        
        # Optionally store episode for analysis
        if len(self.episodes) < self.max_episodes_stored:
            self.episodes.append(self.current_episode.copy())
        
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
                # Sample from future states in same episode
                if t < episode_length - 1:
                    future_idx = np.random.randint(t + 1, episode_length)
                    goals.append(self.current_episode[future_idx]['achieved_goal'])
            
            elif self.her_strategy == 'final':
                # Use final achieved goal
                goals.append(self.current_episode[-1]['achieved_goal'])
            
            elif self.her_strategy == 'episode':
                # Sample from any state in episode
                random_idx = np.random.randint(0, episode_length)
                goals.append(self.current_episode[random_idx]['achieved_goal'])
            
            elif self.her_strategy == 'random':
                # Sample random goal (for binary goals)
                random_goal = np.random.randint(0, 2, size=self.goal_size).astype(np.float32)
                goals.append(random_goal)
        
        return goals
    
    def _compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> float:
        """
        Compute reward based on goal achievement.
        
        Args:
            achieved_goal: Goal that was achieved
            desired_goal: Goal that was desired
            
        Returns:
            Reward (0 if match, -1 otherwise)
        """
        return 0.0 if np.array_equal(achieved_goal, desired_goal) else -1.0
    
    def _store_transition(self, observation: np.ndarray, action: int, reward: float,
                         next_observation: np.ndarray, done: bool):
        """Store a single transition in buffer."""
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
        """Clear the buffer."""
        super().clear()
        self.observations.fill(0)
        self.actions.fill(0)
        self.rewards.fill(0)
        self.next_observations.fill(0)
        self.dones.fill(0)
        self.current_episode = []
        self.episodes = []


def test_prioritized_her_buffer():
    """Test prioritized HER buffer functionality."""
    # from priority_compute import TDErrorPriorityComputer
    
    print("Testing Prioritized HER Buffer...")
    
    n_bits = 5
    buffer = PrioritizedHERBuffer(
        capacity=10000,
        observation_size=2 * n_bits,
        goal_size=n_bits,
        priority_compute=None,
        her_strategy='future',
        her_k=4,
        alpha=0.6,
        beta_start=0.4
    )
    
    # Simulate an episode
    state = np.random.randint(0, 2, size=n_bits).astype(np.float32)
    goal = np.random.randint(0, 2, size=n_bits).astype(np.float32)
    
    print(f"Initial state: {state}")
    print(f"Goal: {goal}")
    
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
    
    # print(f"\nBuffer size after one episode: {len(buffer)}")
    # print(f"Expected: ~{5 + 5 * 4} transitions (original + HER)")
    # print(f"Max priority: {buffer.max_priority:.4f}")
    


if __name__ == "__main__":
    test_prioritized_her_buffer()
