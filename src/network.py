"""
Neural Network Architecture for DQN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class DQN(nn.Module):
    """
    Deep Q-Network for BitFlip environment.
    """
    
    def __init__(self, input_size: int, output_size: int, hidden_sizes: List[int] = [256, 256]):
        """
        Initialize DQN.
        
        Args:
            input_size: Size of input observation
            output_size: Number of possible actions
            hidden_sizes: List of hidden layer sizes
        """
        super(DQN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor (batch_size, input_size)
            
        Returns:
            Q-values for each action (batch_size, output_size)
        """
        return self.network(x)
    
    def select_action(self, state: torch.Tensor, epsilon: float = 0.0) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state observation
            epsilon: Exploration probability
            
        Returns:
            Selected action index
        """
        if torch.rand(1).item() < epsilon:
            # Random action (exploration)
            return torch.randint(0, self.network[-1].out_features, (1,)).item()
        else:
            # Greedy action (exploitation)
            with torch.no_grad():
                q_values = self.forward(state.unsqueeze(0))
                return q_values.argmax(dim=1).item()


class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture that separates value and advantage streams.
    """
    
    def __init__(self, input_size: int, output_size: int, hidden_sizes: List[int] = [256, 256]):
        """
        Initialize Dueling DQN.
        
        Args:
            input_size: Size of input observation
            output_size: Number of possible actions
            hidden_sizes: List of hidden layer sizes
        """
        super(DuelingDQN, self).__init__()
        
        # Shared feature layers
        feature_layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes[:-1]:
            feature_layers.append(nn.Linear(prev_size, hidden_size))
            feature_layers.append(nn.ReLU())
            prev_size = hidden_size
        
        self.feature_layer = nn.Sequential(*feature_layers)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(prev_size, hidden_sizes[-1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[-1], 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_size, hidden_sizes[-1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[-1], output_size)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the dueling network.
        
        Args:
            x: Input tensor (batch_size, input_size)
            
        Returns:
            Q-values for each action (batch_size, output_size)
        """
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantage: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values
    
    def select_action(self, state: torch.Tensor, epsilon: float = 0.0) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state observation
            epsilon: Exploration probability
            
        Returns:
            Selected action index
        """
        if torch.rand(1).item() < epsilon:
            # Random action (exploration)
            output_size = self.advantage_stream[-1].out_features
            return torch.randint(0, output_size, (1,)).item()
        else:
            # Greedy action (exploitation)
            with torch.no_grad():
                q_values = self.forward(state.unsqueeze(0))
                return q_values.argmax(dim=1).item()


def test_network():
    """Test the DQN network."""
    input_size = 50  # 25 bits * 2 (state + goal)
    output_size = 25  # 25 possible actions
    batch_size = 32
    
    # Test standard DQN
    dqn = DQN(input_size, output_size)
    x = torch.randn(batch_size, input_size)
    q_values = dqn(x)
    print(f"DQN output shape: {q_values.shape}")
    print(f"DQN parameters: {sum(p.numel() for p in dqn.parameters())}")
    
    # Test Dueling DQN
    dueling_dqn = DuelingDQN(input_size, output_size)
    q_values = dueling_dqn(x)
    print(f"Dueling DQN output shape: {q_values.shape}")
    print(f"Dueling DQN parameters: {sum(p.numel() for p in dueling_dqn.parameters())}")


if __name__ == "__main__":
    test_network()