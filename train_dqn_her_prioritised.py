"""
Train DQN-HER Agent with Prioritized Experience Replay on BitFlip Environment

Usage:
    python train_dqn_her_prioritized.py --priority-strategy td_error
    python train_dqn_her_prioritized.py --priority-strategy uncertainty --alpha 0.7
    python train_dqn_her_prioritized.py --priority-strategy composite
"""

import sys
sys.path.append('src')

import numpy as np
import json
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from environment import BitFlipEnv
from dqn_her_prioritized_agent import DQNHERPrioritizedAgent
from utils import evaluate_agent


def train_dqn_her_prioritized(
    n_bits: int = 25,
    num_epochs: int = 200,
    episodes_per_epoch: int = 50,
    eval_episodes: int = 100,
    learning_rate: float = 0.001,
    gamma: float = 0.98,
    epsilon_start: float = 1.0,
    epsilon_decay: float = 0.995,
    epsilon_end: float = 0.01, 
    batch_size: int = 128,
    buffer_size: int = 100000,
    target_update_freq: int = 100,
    her_strategy: str = 'future',
    her_k: int = 4,
    priority_strategy: str = 'td_error',
    alpha: float = 0.6,
    beta_start: float = 0.4,
    beta_frames: int = 100000,
    device: str = 'cpu',
    save_dir: str = 'experiments_prioritized',
    verbose: bool = True,
    save_results: bool = True
):
    """
    Train DQN-HER agent with prioritized replay.
    
    Args:
        n_bits: Number of bits in BitFlip environment
        num_epochs: Number of training epochs
        episodes_per_epoch: Episodes per epoch
        eval_episodes: Episodes for evaluation
        learning_rate: Learning rate
        gamma: Discount factor
        epsilon_start: Initial exploration rate
        epsilon_decay: Exploration decay
        batch_size: Training batch size
        buffer_size: Replay buffer capacity
        target_update_freq: Target network update frequency
        her_strategy: HER strategy ('future', 'final', 'episode', 'random')
        her_k: Number of HER goals per transition
        priority_strategy: Priority computation strategy
        alpha: Priority exponent
        beta_start: Initial importance sampling exponent
        beta_frames: Frames to anneal beta
        device: Device ('cpu' or 'cuda')
        save_dir: Directory to save results
    """
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize environment
    env = BitFlipEnv(n_bits=n_bits)
    observation_size = 2 * n_bits  # state + goal
    action_size = n_bits
    goal_size = n_bits
    
    print(f"=== Training DQN-HER with Prioritized Replay ===")
    print(f"Environment: BitFlip-{n_bits}")
    print(f"Observation size: {observation_size}")
    print(f"Action size: {action_size}")
    print(f"HER Strategy: {her_strategy} (k={her_k})")
    print(f"Priority Strategy: {priority_strategy}")
    print(f"Alpha (priority exponent): {alpha}")
    print(f"Beta start (IS exponent): {beta_start}")
    print(f"Device: {device}")
    print("=" * 50)
    
    # Create priority computer
    priority_compute = None
    
    # Initialize agent
    agent = DQNHERPrioritizedAgent(
        observation_size=observation_size,
        action_size=action_size,
        goal_size=goal_size,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_decay=epsilon_decay,
        epsilon_end=epsilon_end,
        target_update_freq=target_update_freq,
        batch_size=batch_size,
        buffer_size=buffer_size,
        her_strategy=her_strategy,
        her_k=her_k,
        priority_strategy=priority_strategy,
        priority_compute=priority_compute,
        alpha=alpha,
        beta_start=beta_start,
        beta_frames=beta_frames,
        device=device
    )
    
    # Training metrics
    results = {
        'epoch_rewards': [],
        'epoch_success_rates': [],
        'epoch_losses': [],
        'epoch_priorities': [],
        'epoch_weights': [],
        'eval_success_rates': [],
        'config': {
            'n_bits': n_bits,
            'her_strategy': her_strategy,
            'her_k': her_k,
            'priority_strategy': priority_strategy,
            'alpha': alpha,
            'beta_start': beta_start,
            'learning_rate': learning_rate,
            'gamma': gamma
        }
    }
    
    # Training loop
    best_success_rate = 0.0
    
    for epoch in range(num_epochs):
        epoch_rewards = []
        epoch_successes = []
        epoch_losses = []
        epoch_priorities = []
        epoch_weights = []
        
        # Training episodes
        for episode in range(episodes_per_epoch):
            obs_dict = env.reset()
            observation = obs_dict['observation']
            achieved_goal = obs_dict['achieved_goal']
            desired_goal = obs_dict['desired_goal']

            episode_reward = 0
            done = False
            
            while not done:
                # Select action
                action = agent.select_action(observation, training=True)
                
                # Take step
                next_obs_dict, reward, terminated, truncated, info = env.step(action)
                next_observation = next_obs_dict['observation']
                next_achieved_goal = next_obs_dict['achieved_goal']
                done = terminated or truncated  
                episode_reward += reward
                
                # Store transition
                agent.store_transition(
                    observation, action, reward, next_observation, done,
                    achieved_goal=achieved_goal,
                    desired_goal=desired_goal,
                    next_achieved_goal=next_achieved_goal
                )
                
                # Update
                metrics = agent.update()
                if metrics:
                    epoch_losses.append(metrics['loss'])
                    if 'mean_priority' in metrics:
                        epoch_priorities.append(metrics['mean_priority'])
                    if 'mean_weight' in metrics:
                        epoch_weights.append(metrics['mean_weight'])
                
                observation = next_observation
                achieved_goal = next_achieved_goal
            
            # End episode (triggers HER relabeling)
            agent.end_episode()
            
            epoch_rewards.append(episode_reward)
            epoch_successes.append(1.0 if info['is_success'] else 0.0)
        
        # Compute epoch metrics
        mean_reward = np.mean(epoch_rewards)
        success_rate = np.mean(epoch_successes)
        mean_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        mean_priority = np.mean(epoch_priorities) if epoch_priorities else 0.0
        mean_weight = np.mean(epoch_weights) if epoch_weights else 1.0
        
        results['epoch_rewards'].append(mean_reward)
        results['epoch_success_rates'].append(success_rate)
        results['epoch_losses'].append(mean_loss)
        results['epoch_priorities'].append(mean_priority)
        results['epoch_weights'].append(mean_weight)
        
        # Evaluation
        if (epoch + 1) % 1 == 0:
            eval_metrics = evaluate_agent(agent, env, num_episodes=eval_episodes)
            eval_success_rate = eval_metrics['success_rate']
            results['eval_success_rates'].append(eval_success_rate)
            
            if verbose:
                print(f"Epoch {epoch + 1}/{num_epochs} | "
                    f"Train Success: {success_rate:.3f} | "
                    f"Eval Success: {eval_success_rate:.3f} | "
                    f"Mean Reward: {mean_reward:.2f} | "
                    f"Loss: {mean_loss:.4f} | "
                    f"Priority: {mean_priority:.4f} | "
                    f"Weight: {mean_weight:.4f} | "
                    f"Buffer: {len(agent.replay_buffer)} | "
                    f"Epsilon: {agent.epsilon:.3f}")
            
            # Save best model
            if save_results:
                if eval_success_rate > best_success_rate:
                    best_success_rate = eval_success_rate
                    agent.save(os.path.join(save_dir, f'dqn_her_prioritized_best.pt'))
        else:
            if verbose:
                print(f"Epoch {epoch + 1}/{num_epochs} | "
                    f"Success: {success_rate:.3f} | "
                    f"Reward: {mean_reward:.2f} | "
                    f"Loss: {mean_loss:.4f} | "
                    f"Priority: {mean_priority:.4f} | "
                    f"Buffer: {len(agent.replay_buffer)}")
    
    # Save final model
    agent.save(os.path.join(save_dir, 'dqn_her_prioritized_final.pt'))
    
    # Save results
    if save_results:
        with open(os.path.join(save_dir, 'dqn_her_prioritized_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
    
        # Plot results
        plot_results(results, save_dir)
    
    print(f"\nTraining complete!")
    # print(f"Best eval success rate: {best_success_rate:.3f}")
    print(f"Final training success rate: {results['epoch_success_rates'][-1]:.3f}")
    if save_results:
        print(f"Results saved to: {save_dir}")
    
    return results, agent


def plot_results(results, save_dir):
    """Plot training results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Success rate
    axes[0, 0].plot(results['epoch_success_rates'])
    axes[0, 0].set_title('Training Success Rate')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Success Rate')
    axes[0, 0].grid(True)
    
    # Reward
    axes[0, 1].plot(results['epoch_rewards'])
    axes[0, 1].set_title('Mean Episode Reward')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Reward')
    axes[0, 1].grid(True)
    
    # Loss
    axes[0, 2].plot(results['epoch_losses'])
    axes[0, 2].set_title('Training Loss')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].grid(True)
    
    # Priority
    axes[1, 0].plot(results['epoch_priorities'])
    axes[1, 0].set_title('Mean Priority (TD Error)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Priority')
    axes[1, 0].grid(True)
    
    # IS Weights
    axes[1, 1].plot(results['epoch_weights'])
    axes[1, 1].set_title('Mean Importance Sampling Weight')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Weight')
    axes[1, 1].grid(True)
    
    # Eval success rate
    if results['eval_success_rates']:
        eval_epochs = list(range(9, len(results['epoch_success_rates']), 10))
        axes[1, 2].plot(eval_epochs, results['eval_success_rates'])
        axes[1, 2].set_title('Evaluation Success Rate')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Success Rate')
        axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    print(f"Plots saved to: {os.path.join(save_dir, 'training_curves.png')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DQN-HER with Prioritized Replay')
    parser.add_argument('--n-bits', type=int, default=25, help='Number of bits')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--episodes-per-epoch', type=int, default=50, help='Episodes per epoch')
    parser.add_argument('--her-strategy', type=str, default='future', 
                       choices=['future', 'final', 'episode', 'random'],
                       help='HER strategy')
    parser.add_argument('--her-k', type=int, default=4, help='Number of HER goals')
    parser.add_argument('--priority-strategy', type=str, default='td_error',
                       choices=['td_error', 'closeness_to_goal'],
                       help='Priority computation strategy')
    parser.add_argument('--alpha', type=float, default=0.6, help='Priority exponent')
    parser.add_argument('--beta-start', type=float, default=0.4, help='Initial IS exponent')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--save-dir', type=str, default='experiments_prioritized',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    train_dqn_her_prioritized(
        n_bits=args.n_bits,
        num_epochs=args.epochs,
        episodes_per_epoch=args.episodes_per_epoch,
        her_strategy=args.her_strategy,
        her_k=args.her_k,
        priority_strategy=args.priority_strategy,
        alpha=args.alpha,
        beta_start=args.beta_start,
        device=args.device,
        save_dir=args.save_dir,
        epsilon_end=0.2
    )
