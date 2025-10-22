"""
Utility Functions
"""

import numpy as np
import random
import torch
import os
import json
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_directories(dirs: List[str]):
    """
    Create directories if they don't exist.
    
    Args:
        dirs: List of directory paths to create
    """
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)


def save_results(results: Dict, path: str):
    """
    Save training results to JSON file.
    
    Args:
        results: Dictionary containing results
        path: Path to save file
    """
    # Convert numpy arrays to lists for JSON serialization
    results_serializable = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            results_serializable[key] = value.tolist()
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
            results_serializable[key] = [v.tolist() for v in value]
        else:
            results_serializable[key] = value
    
    with open(path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print(f"Results saved to {path}")


def load_results(path: str) -> Dict:
    """
    Load training results from JSON file.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Dictionary containing results
    """
    with open(path, 'r') as f:
        results = json.load(f)
    return results


def compute_accuracy(successes: List[bool], window: int = 10) -> List[float]:
    """
    Compute rolling accuracy from success indicators.
    
    Args:
        successes: List of boolean success indicators
        window: Window size for rolling average
        
    Returns:
        List of rolling accuracy values
    """
    accuracies = []
    for i in range(len(successes)):
        start = max(0, i - window + 1)
        window_successes = successes[start:i+1]
        accuracy = sum(window_successes) / len(window_successes)
        accuracies.append(accuracy)
    return accuracies


def moving_average(data: List[float], window: int = 10) -> List[float]:
    """
    Compute moving average of data.
    
    Args:
        data: List of values
        window: Window size
        
    Returns:
        List of smoothed values
    """
    if len(data) < window:
        return data
    
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        smoothed.append(np.mean(data[start:i+1]))
    return smoothed


def evaluate_agent(agent, env, num_episodes: int = 100) -> Dict[str, float]:
    """
    Evaluate agent performance.
    
    Args:
        agent: Agent to evaluate
        env: Environment
        num_episodes: Number of episodes to evaluate
        
    Returns:
        Dictionary containing evaluation metrics
    """
    successes = []
    episode_lengths = []
    
    for _ in range(num_episodes):
        obs_dict = env.reset()
        observation = obs_dict['observation']
        done = False
        steps = 0
        
        while not done and steps < env.max_steps:
            action = agent.select_action(observation, training=False)
            obs_dict, reward, terminated, truncated, info = env.step(action)
            observation = obs_dict['observation']
            done = terminated or truncated
            steps += 1
        
        successes.append(info['is_success'])
        episode_lengths.append(steps)
    
    return {
        'success_rate': np.mean(successes),
        'mean_episode_length': np.mean(episode_lengths),
        'std_episode_length': np.std(episode_lengths)
    }


def print_training_progress(epoch: int, total_epochs: int, metrics: Dict[str, float]):
    """
    Print training progress.
    
    Args:
        epoch: Current epoch
        total_epochs: Total number of epochs
        metrics: Dictionary of metrics to print
    """
    print(f"Epoch {epoch}/{total_epochs}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print("-" * 50)


def setup_plotting_style(style: str = 'seaborn'):
    """
    Setup matplotlib plotting style.
    
    Args:
        style: Style name
    """
    if style == 'seaborn':
        sns.set_style('whitegrid')
        sns.set_context('paper', font_scale=1.5)
    else:
        plt.style.use(style)


def get_device(device_str: str = 'auto') -> torch.device:
    """
    Get torch device.
    
    Args:
        device_str: Device string ('auto', 'cpu', 'cuda', 'mps')
        
    Returns:
        Torch device
    """
    if device_str == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    else:
        return torch.device(device_str)