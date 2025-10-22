"""
Plot Results from Saved JSON Files
Use this to plot results without retraining
"""

import os
import sys
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sys.path.append('src')

from utils import load_results, setup_plotting_style, moving_average


def plot_results(dqn_path=None, dqn_her_path=None, save_dir='experiments'):
    """
    Plot results from saved JSON files.
    
    Args:
        dqn_path: Path to DQN results JSON
        dqn_her_path: Path to DQN-HER results JSON
        save_dir: Directory to save plots
    """
    setup_plotting_style('seaborn')
    
    # Load results
    results = {}
    if dqn_path and os.path.exists(dqn_path):
        results['DQN'] = load_results(dqn_path)
        print(f"Loaded DQN results from {dqn_path}")
    
    if dqn_her_path and os.path.exists(dqn_her_path):
        results['DQN-HER'] = load_results(dqn_her_path)
        print(f"Loaded DQN-HER results from {dqn_her_path}")
    
    if len(results) == 0:
        print("No results to plot. Please provide valid paths to result files.")
        return
    
    # Create accuracy vs epoch plot
    plt.figure(figsize=(12, 7))
    
    colors = {'DQN': '#2E86AB', 'DQN-HER': '#A23B72'}
    
    for name, result_data in results.items():
        success_rates = result_data['epoch_success_rates']
        epochs = np.arange(1, len(success_rates) + 1)
        
        # Plot raw data
        plt.plot(epochs, success_rates, 
                label=name, linewidth=2.5, alpha=0.6, color=colors[name])
        
        # Plot smoothed data
        if len(success_rates) > 10:
            smoothed = moving_average(success_rates, window=10)
            plt.plot(epochs, smoothed, linewidth=3, 
                    color=colors[name], label=f'{name} (MA-10)', linestyle='--')
    
    plt.xlabel('Epoch', fontsize=16, fontweight='bold')
    plt.ylabel('Success Rate (Accuracy)', fontsize=16, fontweight='bold')
    plt.title('Accuracy vs Epoch: DQN vs DQN-HER on BitFlip-25', 
             fontsize=18, fontweight='bold', pad=20)
    plt.legend(fontsize=14, loc='lower right', framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.ylim([-0.05, 1.05])
    
    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, 'accuracy_vs_epoch_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    
    plt.show()
    
    # Print statistics
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    for name, result_data in results.items():
        print(f"\n{name}:")
        if 'final_eval' in result_data:
            print(f"  Final Success Rate: {result_data['final_eval']['success_rate']:.4f}")
            print(f"  Mean Episode Length: {result_data['final_eval']['mean_episode_length']:.2f}")
        print(f"  Best Epoch Success Rate: {max(result_data['epoch_success_rates']):.4f}")
        print(f"  Total Episodes: {result_data.get('total_episodes', 'N/A')}")
    
    # Comparison
    if len(results) == 2:
        dqn_sr = results['DQN']['final_eval']['success_rate']
        her_sr = results['DQN-HER']['final_eval']['success_rate']
        improvement = her_sr - dqn_sr
        
        print("\nImprovement (DQN-HER over DQN):")
        print(f"  Absolute: {improvement:+.4f}")
        if dqn_sr > 0:
            relative_improvement = (improvement / dqn_sr) * 100
            print(f"  Relative: {relative_improvement:+.2f}%")
    
    print("="*60)
    
    # Create detailed comparison if both results available
    if len(results) == 2:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('DQN vs DQN-HER Detailed Comparison', fontsize=20, fontweight='bold')
        
        dqn_data = results['DQN']
        her_data = results['DQN-HER']
        
        # Success Rate
        ax = axes[0, 0]
        epochs_dqn = np.arange(1, len(dqn_data['epoch_success_rates']) + 1)
        epochs_her = np.arange(1, len(her_data['epoch_success_rates']) + 1)
        ax.plot(epochs_dqn, dqn_data['epoch_success_rates'], 
               label='DQN', linewidth=2, color='#2E86AB')
        ax.plot(epochs_her, her_data['epoch_success_rates'],
               label='DQN-HER', linewidth=2, color='#A23B72')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Success Rate', fontsize=12)
        ax.set_title('Success Rate vs Epoch', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Rewards
        ax = axes[0, 1]
        ax.plot(epochs_dqn, dqn_data['epoch_rewards'],
               label='DQN', linewidth=2, color='#2E86AB')
        ax.plot(epochs_her, her_data['epoch_rewards'],
               label='DQN-HER', linewidth=2, color='#A23B72')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Mean Reward', fontsize=12)
        ax.set_title('Mean Reward vs Epoch', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Loss
        ax = axes[1, 0]
        ax.plot(epochs_dqn, dqn_data['epoch_losses'],
               label='DQN', linewidth=2, color='#2E86AB')
        ax.plot(epochs_her, her_data['epoch_losses'],
               label='DQN-HER', linewidth=2, color='#A23B72')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Loss vs Epoch', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Bar chart
        ax = axes[1, 1]
        algorithms = ['DQN', 'DQN-HER']
        final_rates = [dqn_data['final_eval']['success_rate'],
                      her_data['final_eval']['success_rate']]
        bars = ax.bar(algorithms, final_rates, color=['#2E86AB', '#A23B72'], alpha=0.8)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax.set_ylabel('Final Success Rate', fontsize=12)
        ax.set_title('Final Performance', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        detailed_plot_path = os.path.join(save_dir, 'detailed_comparison.png')
        plt.savefig(detailed_plot_path, dpi=300, bbox_inches='tight')
        print(f"Detailed comparison saved to: {detailed_plot_path}")
        
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot DQN and DQN-HER results')
    parser.add_argument('--dqn-results', type=str, default='experiments/dqn_results.json',
                       help='Path to DQN results JSON file')
    parser.add_argument('--her-results', type=str, default='experiments/dqn_her_results.json',
                       help='Path to DQN-HER results JSON file')
    parser.add_argument('--save-dir', type=str, default='experiments',
                       help='Directory to save plots')
    
    args = parser.parse_args()
    
    plot_results(args.dqn_results, args.her_results, args.save_dir)


if __name__ == "__main__":
    main()