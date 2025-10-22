"""
Compare DQN and DQN-HER Results
Train both agents and generate comparison plots
"""

import os
import sys
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sys.path.append('src')

from train_dqn import train_dqn
from train_dqn_her import train_dqn_her
from utils import setup_plotting_style, moving_average


def plot_comparison(dqn_results, dqn_her_results, save_dir='experiments'):
    """
    Create comparison plots between DQN and DQN-HER.
    
    Args:
        dqn_results: Results from DQN training
        dqn_her_results: Results from DQN-HER training
        save_dir: Directory to save plots
    """
    setup_plotting_style('seaborn')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('DQN vs DQN-HER on BitFlip-25', fontsize=20, fontweight='bold')
    
    # 1. Success Rate vs Epoch
    ax = axes[0, 0]
    epochs_dqn = np.arange(1, len(dqn_results['epoch_success_rates']) + 1)
    epochs_her = np.arange(1, len(dqn_her_results['epoch_success_rates']) + 1)
    
    ax.plot(epochs_dqn, dqn_results['epoch_success_rates'], 
            label='DQN', linewidth=2, alpha=0.7, color='#2E86AB')
    ax.plot(epochs_her, dqn_her_results['epoch_success_rates'],
            label='DQN-HER', linewidth=2, alpha=0.7, color='#A23B72')
    
    # Add smoothed curves
    if len(dqn_results['epoch_success_rates']) > 10:
        smoothed_dqn = moving_average(dqn_results['epoch_success_rates'], window=10)
        smoothed_her = moving_average(dqn_her_results['epoch_success_rates'], window=10)
        ax.plot(epochs_dqn, smoothed_dqn, '--', linewidth=2.5, 
                color='#2E86AB', label='DQN (smoothed)')
        ax.plot(epochs_her, smoothed_her, '--', linewidth=2.5,
                color='#A23B72', label='DQN-HER (smoothed)')
    
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Success Rate', fontsize=14)
    ax.set_title('Accuracy vs Epoch', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.05])
    
    # 2. Mean Reward vs Epoch
    ax = axes[0, 1]
    ax.plot(epochs_dqn, dqn_results['epoch_rewards'],
            label='DQN', linewidth=2, alpha=0.7, color='#2E86AB')
    ax.plot(epochs_her, dqn_her_results['epoch_rewards'],
            label='DQN-HER', linewidth=2, alpha=0.7, color='#A23B72')
    
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Mean Reward', fontsize=14)
    ax.set_title('Mean Reward vs Epoch', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 3. Training Loss vs Epoch
    ax = axes[1, 0]
    ax.plot(epochs_dqn, dqn_results['epoch_losses'],
            label='DQN', linewidth=2, alpha=0.7, color='#2E86AB')
    ax.plot(epochs_her, dqn_her_results['epoch_losses'],
            label='DQN-HER', linewidth=2, alpha=0.7, color='#A23B72')
    
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.set_title('Training Loss vs Epoch', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # 4. Final Performance Comparison (Bar Chart)
    ax = axes[1, 1]
    algorithms = ['DQN', 'DQN-HER']
    final_success_rates = [
        dqn_results['final_eval']['success_rate'],
        dqn_her_results['final_eval']['success_rate']
    ]
    
    bars = ax.bar(algorithms, final_success_rates, 
                  color=['#2E86AB', '#A23B72'], alpha=0.8, width=0.6)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Final Success Rate', fontsize=14)
    ax.set_title('Final Evaluation Performance', fontsize=16, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    plot_path = os.path.join(save_dir, 'dqn_vs_dqn_her_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: {plot_path}")
    
    # Create focused accuracy plot
    plt.figure(figsize=(12, 7))
    plt.plot(epochs_dqn, dqn_results['epoch_success_rates'], 
            label='DQN', linewidth=2.5, alpha=0.6, color='#2E86AB')
    plt.plot(epochs_her, dqn_her_results['epoch_success_rates'],
            label='DQN-HER', linewidth=2.5, alpha=0.6, color='#A23B72')
    
    # Add smoothed curves
    if len(dqn_results['epoch_success_rates']) > 10:
        smoothed_dqn = moving_average(dqn_results['epoch_success_rates'], window=10)
        smoothed_her = moving_average(dqn_her_results['epoch_success_rates'], window=10)
        plt.plot(epochs_dqn, smoothed_dqn, linewidth=3, 
                color='#2E86AB', label='DQN (MA-10)', linestyle='--')
        plt.plot(epochs_her, smoothed_her, linewidth=3,
                color='#A23B72', label='DQN-HER (MA-10)', linestyle='--')
    
    plt.xlabel('Epoch', fontsize=16, fontweight='bold')
    plt.ylabel('Success Rate (Accuracy)', fontsize=16, fontweight='bold')
    plt.title('Accuracy vs Epoch: DQN vs DQN-HER on BitFlip-25', 
             fontsize=18, fontweight='bold', pad=20)
    plt.legend(fontsize=14, loc='lower right', framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.ylim([-0.05, 1.05])
    
    accuracy_plot_path = os.path.join(save_dir, 'accuracy_vs_epoch.png')
    plt.savefig(accuracy_plot_path, dpi=300, bbox_inches='tight')
    print(f"Accuracy plot saved to: {accuracy_plot_path}")
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    print("\nDQN:")
    print(f"  Final Success Rate: {dqn_results['final_eval']['success_rate']:.4f}")
    print(f"  Mean Episode Length: {dqn_results['final_eval']['mean_episode_length']:.2f}")
    print(f"  Best Epoch Success Rate: {max(dqn_results['epoch_success_rates']):.4f}")
    
    print("\nDQN-HER:")
    print(f"  Final Success Rate: {dqn_her_results['final_eval']['success_rate']:.4f}")
    print(f"  Mean Episode Length: {dqn_her_results['final_eval']['mean_episode_length']:.2f}")
    print(f"  Best Epoch Success Rate: {max(dqn_her_results['epoch_success_rates']):.4f}")
    
    print("\nImprovement:")
    improvement = (dqn_her_results['final_eval']['success_rate'] - 
                   dqn_results['final_eval']['success_rate'])
    print(f"  Absolute: {improvement:+.4f}")
    if dqn_results['final_eval']['success_rate'] > 0:
        relative_improvement = (improvement / dqn_results['final_eval']['success_rate']) * 100
        print(f"  Relative: {relative_improvement:+.2f}%")
    
    print("="*60)


def main():
    """Main function to train both agents and compare results."""
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Starting training comparison: DQN vs DQN-HER")
    print("="*60)
    
    # Train DQN
    print("\n1. Training Standard DQN...")
    dqn_results = train_dqn(config)
    
    print("\n" + "="*60)
    
    # Train DQN-HER
    print("\n2. Training DQN-HER...")
    dqn_her_results = train_dqn_her(config)
    
    print("\n" + "="*60)
    
    # Create comparison plots
    print("\n3. Generating comparison plots...")
    plot_comparison(dqn_results, dqn_her_results, config['experiment']['save_dir'])
    
    print("\nAll done! Check the experiments folder for results and plots.")


if __name__ == "__main__":
    main()