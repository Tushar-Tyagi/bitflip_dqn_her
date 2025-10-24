"""
BitFlip DQN Project
A complete implementation comparing DQN and DQN-HER on the BitFlip-25 environment.
"""

__version__ = "1.0.0"
__author__ = "Georgia Tech MSCS"

from .environment import BitFlipEnv
from .network import DQN, DuelingDQN
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from .her_buffer import HERBuffer
from .dqn_agent import DQNAgent
from .dqn_her_agent import DQNHERAgent

__all__ = [
    'BitFlipEnv',
    'DQN',
    'DuelingDQN',
    'ReplayBuffer',
    'PrioritizedReplayBuffer',
    'HERBuffer',
    'DQNAgent',
    'DQNHERAgent'
]