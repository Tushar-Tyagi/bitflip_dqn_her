"""
Training Script for DQN-HER on BitFlip-25
"""

import os
import sys
import yaml
import numpy as np
from tqdm import tqdm
import torch

sys.path.append('src')

from environment import BitFlipEnv
from dqn_her_agent import DQNHERAgent
from utils import (set_seed, create_directories, save_results, 
                   evaluate_agent, print_training_progress, moving_average)


def train_dqn_her(config: dict):
    """
    Train DQN-HER agent on BitFlip-25 environment.
    
    Args:
        config: Configuration dictionary
    """
    # Setup
    set_seed(config['experiment']['seed'])
    create_directories([config['experiment']['save_dir']])
    
    # Create environment
    env = BitFlipEnv(n_bits=config['environment']['n_bits'])
    
    # Create agent
    agent = DQNHERAgent(
        observation_size=env.observation_size,
        action_size=env.action_space_size,
        goal_size=config['environment']['n_bits'],
        learning_rate=config['dqn_her']['learning_rate'],
        gamma=config['dqn_her']['gamma'],
        epsilon_start=config['dqn_her']['epsilon_start'],
        epsilon_end=config['dqn_her']['epsilon_end'],
        epsilon_decay=config['dqn_her']['epsilon_decay'],
        target_update_freq=config['dqn_her']['target_update_freq'],
        batch_size=config['dqn_her']['batch_size'],
        buffer_size=config['dqn_her']['buffer_size'],
        hidden_sizes=config['dqn_her']['hidden_sizes'],
        her_strategy=config['dqn_her']['her_strategy'],
        her_k=config['dqn_her']['her_k'],
        device=config['experiment']['device']
    )
    
    print("=" * 60)
    print("Training DQN-HER on BitFlip-25")
    print("=" * 60)
    print(f"Device: {agent.device}")
    print(f"HER Strategy: {config['dqn_her']['her_strategy']}")
    print(f"HER k: {config['dqn_her']['her_k']}")
    print(f"Network parameters: {sum(p.numel() for p in agent.policy_net.parameters())}")
    print()
    
    # Training metrics
    epoch_rewards = []
    epoch_success_rates = []
    eval_success_rates = []
    epoch_losses = []
    epoch_q_values = []
    all_episode_rewards = []
    all_episode_successes = []
    
    # Training loop
    num_epochs = config['training']['num_epochs']
    episodes_per_epoch = config['training']['episodes_per_epoch']
    
    for epoch in range(1, num_epochs + 1):
        epoch_total_reward = 0
        epoch_successes = []
        epoch_loss_values = []
        epoch_q_value_list = []
        
        # Training episodes
        # for episode in tqdm(range(episodes_per_epoch), desc=f"Epoch {epoch}/{num_epochs}"):
        for episode in range(episodes_per_epoch):
            obs_dict = env.reset()
            observation = obs_dict['observation']
            achieved_goal = obs_dict['achieved_goal']
            desired_goal = obs_dict['desired_goal']
            
            episode_reward = 0
            done = False
            steps = 0
            
            while not done and steps < env.max_steps:
                # Select action
                action = agent.select_action(observation, training=True)
                
                # Take step
                next_obs_dict, reward, terminated, truncated, info = env.step(action)
                next_observation = next_obs_dict['observation']
                next_achieved_goal = next_obs_dict['achieved_goal']
                done = terminated or truncated
                
                # Store transition (with goals for HER)
                agent.store_transition(
                    observation, action, reward, next_observation, done,
                    achieved_goal, desired_goal, next_achieved_goal
                )
                
                # Update agent
                metrics = agent.update()
                if metrics:
                    epoch_loss_values.append(metrics['loss'])
                    epoch_q_value_list.append(metrics['q_value'])
                
                observation = next_observation
                achieved_goal = next_achieved_goal
                episode_reward += reward
                steps += 1
            
            # End episode (triggers HER relabeling)
            agent.end_episode()
            epoch_total_reward += episode_reward
            epoch_successes.append(info['is_success'])
            all_episode_rewards.append(episode_reward)
            all_episode_successes.append(info['is_success'])
        
        # Epoch statistics
        mean_reward = epoch_total_reward / episodes_per_epoch
        success_rate = np.mean(epoch_successes)
        mean_loss = np.mean(epoch_loss_values) if epoch_loss_values else 0.0
        mean_q_value = np.mean(epoch_q_value_list) if epoch_q_value_list else 0.0
        
        epoch_rewards.append(mean_reward)
        epoch_success_rates.append(success_rate)
        epoch_losses.append(mean_loss)
        epoch_q_values.append(mean_q_value)

        if (epoch + 1) % 1 == 0:
            eval_metrics = evaluate_agent(agent, env, num_episodes=config['training']['eval_episodes'])
            eval_success_rate = eval_metrics['success_rate']
            eval_success_rates.append(eval_success_rate)
        
        # Print progress
        if epoch % config['training']['log_freq'] == 0:
            if config['training']['verbose']:
                print_training_progress(epoch, num_epochs, {
                    'Mean Reward': mean_reward,
                    'Success Rate': success_rate,
                    'Loss': mean_loss,
                    'Q-value': mean_q_value,
                    'Epsilon': agent.epsilon,
                    'Buffer Size': len(agent.replay_buffer)
                })
        
        # Save checkpoint
        if epoch % config['training']['save_freq'] == 0:
            if config['training']['save_results']:
                checkpoint_path = os.path.join(
                    config['experiment']['save_dir'],
                    f"dqn_her_checkpoint_epoch_{epoch}.pt"
                )
                agent.save(checkpoint_path)
    
    # Final evaluation
    print("\nFinal Evaluation...")
    eval_metrics = evaluate_agent(agent, env, num_episodes=config['training']['eval_episodes'])
    print(f"Final Success Rate: {eval_metrics['success_rate']:.4f}")
    print(f"Mean Episode Length: {eval_metrics['mean_episode_length']:.2f}")
    
    # Save final model
    if config['training']['save_results']:
        final_model_path = os.path.join(config['experiment']['save_dir'], 'dqn_her_final.pt')
        agent.save(final_model_path)
    
    # Save results
    results = {
        'algorithm': 'DQN-HER',
        'config': config,
        'epoch_rewards': epoch_rewards,
        'epoch_success_rates': epoch_success_rates,
        'eval_success_rates': eval_success_rates,
        'epoch_losses': epoch_losses,
        'epoch_q_values': epoch_q_values,
        'all_episode_rewards': all_episode_rewards,
        'all_episode_successes': all_episode_successes,
        'final_eval': eval_metrics,
        'total_episodes': num_epochs * episodes_per_epoch
    }
    
    if config['training']['save_results']:
        results_path = os.path.join(config['experiment']['save_dir'], 'dqn_her_results.json')
        save_results(results, results_path)
    
    print("\nTraining completed!")
    return results


if __name__ == "__main__":
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Train DQN-HER
    results = train_dqn_her(config)