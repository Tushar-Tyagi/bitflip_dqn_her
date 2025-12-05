"""
Priority Computation Strategies for Prioritized Experience Replay
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List


class PriorityComputer(ABC):
    """
    Abstract base class for computing transition priorities.
    """
    
    def __init__(self, epsilon: float = 1e-6):
        """
        Initialize priority computer.
        
        Args:
            epsilon: Small constant to ensure non-zero priorities
        """
        self.epsilon = epsilon
    
    @abstractmethod
    def compute_priority(self, 
                        observations: torch.Tensor,
                        actions: torch.Tensor,
                        rewards: torch.Tensor,
                        next_observations: torch.Tensor,
                        dones: torch.Tensor,
                        agent: object) -> np.ndarray:
        """
        Compute priorities for a batch of transitions.
        
        Args:
            observations: Batch of observations
            actions: Batch of actions
            rewards: Batch of rewards
            next_observations: Batch of next observations
            dones: Batch of done flags
            agent: Agent object for accessing networks and parameters
            
        Returns:
            Array of priorities (one per transition)
        """
        pass

def get_priority_compute(strategy: str, **kwargs) -> PriorityComputer:
    """
    Factory function to create priority computers by name.
    
    Args:
        strategy: Name of the strategy ('td_error', 'policy_loss', etc.)
        **kwargs: Additional arguments for the priority computer
        
    Returns:
        PriorityComputer instance
    """
    computers = {
        'td_error': TDErrorPriorityComputer,
        'policy_loss': PolicyLossPriorityComputer,
        'value_gradient': ValueGradientPriorityComputer,
        'uncertainty': UncertaintyPriorityComputer,
        'curiosity': CuriosityPriorityComputer,
    }
    
    if strategy not in computers:
        raise ValueError(f"Unknown strategy: {strategy}. Available: {list(computers.keys())}")
    
    return computers[strategy](**kwargs)


if __name__ == "__main__":

    print("Priority Computer Strategies:")

