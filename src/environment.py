"""
BitFlip-25 Environment Implementation
"""

import numpy as np
from typing import Tuple, Dict, Any


class BitFlipEnv:
    """
    BitFlip environment where the agent must flip bits to match a target configuration.
    
    The environment has:
    - n_bits: number of bits (default: 25)
    - Action space: flip bit i (0 to n_bits-1)
    - Observation: concatenation of current state and goal
    - Reward: 0 if goal reached, -1 otherwise (sparse reward)
    """
    
    def __init__(self, n_bits: int = 25):
        self.n_bits = n_bits
        self.max_steps = n_bits
        self.action_space_size = n_bits
        self.observation_size = 2 * n_bits  # current state + goal
        
        self.current_state = None
        self.goal = None
        self.step_count = 0
        
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset the environment and return initial observation."""
        self.current_state = np.random.randint(0, 2, size=self.n_bits).astype(np.float32)
        self.goal = np.random.randint(0, 2, size=self.n_bits).astype(np.float32)
        self.step_count = 0
        
        observation = self._get_obs()
        return {
            'observation': observation,
            'achieved_goal': self.current_state.copy(),
            'desired_goal': self.goal.copy()
        }
    
    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Index of bit to flip (0 to n_bits-1)
            
        Returns:
            observation: Dict with observation, achieved_goal, desired_goal
            reward: -1 if not at goal, 0 if at goal
            terminated: True if goal reached
            truncated: True if max steps exceeded
            info: Additional information
        """
        # Flip the bit
        self.current_state[action] = 1 - self.current_state[action]
        self.step_count += 1
        
        # Check if goal is reached
        terminated = np.array_equal(self.current_state, self.goal)
        truncated = self.step_count >= self.max_steps
        
        # Sparse reward: 0 if goal reached, -1 otherwise
        reward = 0.0 if terminated else -1.0
        
        observation = self._get_obs()
        obs_dict = {
            'observation': observation,
            'achieved_goal': self.current_state.copy(),
            'desired_goal': self.goal.copy()
        }
        
        info = {
            'is_success': terminated,
            'distance': np.sum(np.abs(self.current_state - self.goal))
        }
        
        return obs_dict, reward, terminated, truncated, info
    
    def _get_obs(self) -> np.ndarray:
        """Get current observation (concatenation of state and goal)."""
        return np.concatenate([self.current_state, self.goal]).astype(np.float32)
    
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict = None) -> float:
        """
        Compute reward given achieved and desired goals.
        Useful for HER to compute rewards for imagined goals.
        """
        return 0.0 if np.array_equal(achieved_goal, desired_goal) else -1.0
    
    def render(self):
        """Render the environment state."""
        print(f"Step: {self.step_count}/{self.max_steps}")
        print(f"Current: {self.current_state.astype(int)}")
        print(f"Goal:    {self.goal.astype(int)}")
        print(f"Distance: {np.sum(np.abs(self.current_state - self.goal))}")
        print("-" * 50)


def test_environment():
    """Test the BitFlip environment."""
    env = BitFlipEnv(n_bits=5)
    obs_dict = env.reset()
    
    print("Initial state:")
    env.render()
    
    # Take a few random actions
    for i in range(3):
        action = np.random.randint(0, env.action_space_size)
        obs_dict, reward, terminated, truncated, info = env.step(action)
        print(f"\nAction: Flip bit {action}")
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}, Truncated: {truncated}")
        env.render()
        
        if terminated or truncated:
            break


if __name__ == "__main__":
    test_environment()