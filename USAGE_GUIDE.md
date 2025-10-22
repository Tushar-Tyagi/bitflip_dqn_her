# BitFlip-25 DQN vs DQN-HER: Complete Usage Guide

## Quick Start

### 1. Installation

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train and Compare Both Agents

The easiest way to get started is to run the comparison script that trains both agents and generates all plots:

```bash
python compare_results.py
```

This will:
- Train standard DQN for 200 epochs
- Train DQN-HER for 200 epochs  
- Generate comparison plots
- Save results to `experiments/` folder

### 3. Train Individual Agents

**Train DQN only:**
```bash
python train_dqn.py
```

**Train DQN-HER only:**
```bash
python train_dqn_her.py
```

### 4. Plot Existing Results

If you already have trained models and just want to regenerate plots:

```bash
python plot_results.py --dqn-results experiments/dqn_results.json --her-results experiments/dqn_her_results.json
```

## Configuration

Edit `config.yaml` to modify hyperparameters:

```yaml
environment:
  n_bits: 25  # Number of bits (can change to 10, 15, etc.)
  
dqn:
  learning_rate: 0.001
  epsilon_decay: 0.995
  batch_size: 128
  
dqn_her:
  her_strategy: "future"  # Options: future, final, episode, random
  her_k: 4  # Number of HER goals per transition
  
training:
  num_epochs: 200
  episodes_per_epoch: 50
```

## Project Structure

```
bitflip_dqn/
â”œâ”€â”€ README.md                 # Main project documentation
â”œâ”€â”€ USAGE_GUIDE.md           # This file
â”œâ”€â”€ requirements.txt         # Python dependencies
â”œâ”€â”€ config.yaml              # Hyperparameters configuration
â”‚
â”œâ”€â”€ src/                     # Source code
â”‚   â”œâ”€â”€ __init__.py
â”‚   â”œâ”€â”€ environment.py       # BitFlip-25 environment
â”‚   â”œâ”€â”€ network.py          # Neural network architectures
â”‚   â”œâ”€â”€ replay_buffer.py    # Standard replay buffer
â”‚   â”œâ”€â”€ her_buffer.py       # HER replay buffer
â”‚   â”œâ”€â”€ dqn_agent.py        # DQN agent
â”‚   â”œâ”€â”€ dqn_her_agent.py    # DQN-HER agent
â”‚   â””â”€â”€ utils.py            # Utility functions
â”‚
â”œâ”€â”€ train_dqn.py            # Train DQN
â”œâ”€â”€ train_dqn_her.py        # Train DQN-HER
â”œâ”€â”€ compare_results.py      # Train both and compare
â”œâ”€â”€ plot_results.py         # Plot saved results
â”‚
â””â”€â”€ experiments/            # Saved results (created automatically)
    â”œâ”€â”€ dqn_results.json
    â”œâ”€â”€ dqn_her_results.json
    â”œâ”€â”€ dqn_final.pt
    â”œâ”€â”€ dqn_her_final.pt
    â””â”€â”€ *.png (plots)
```

## Understanding the Code

### Environment (BitFlip-25)

The BitFlip environment is a sparse reward task where:
- **State**: 25 binary values
- **Goal**: Target configuration of 25 bits
- **Action**: Flip one bit (25 possible actions)
- **Reward**: 0 if goal reached, -1 otherwise
- **Episode Length**: Maximum 25 steps

### DQN Agent

Standard Deep Q-Network with:
- Experience replay buffer
- Target network
- Epsilon-greedy exploration
- MSE loss for Q-learning

### DQN-HER Agent

DQN enhanced with Hindsight Experience Replay:
- Stores complete episodes
- Relabels goals after episode ends
- Generates 4 additional training samples per transition (default)
- Uses "future" strategy by default (samples goals from future states)

### Key Differences

| Feature | DQN | DQN-HER |
|---------|-----|---------|
| Replay Buffer | Standard transitions | Episode-based with goal relabeling |
| Sample Efficiency | Low in sparse rewards | High in sparse rewards |
| Data Augmentation | No | Yes (HER relabeling) |
| Training Speed | Slower convergence | Faster convergence |

## Expected Results

Based on the BitFlip-25 environment with default hyperparameters:

- **DQN**: Typically achieves 10-30% success rate (struggles with sparse rewards)
- **DQN-HER**: Typically achieves 80-95% success rate (benefits from HER)

The plot will show:
1. **Accuracy vs Epoch**: DQN-HER learns much faster
2. **Mean Reward**: DQN-HER achieves higher rewards
3. **Training Loss**: Both converge but HER is more stable
4. **Final Performance**: Bar chart showing clear HER advantage

## Advanced Usage

### Custom Network Architecture

Modify `hidden_sizes` in `config.yaml`:

```yaml
dqn:
  hidden_sizes: [512, 512, 256]  # Larger network
```

### Different HER Strategies

```yaml
dqn_her:
  her_strategy: "final"  # Use final achieved goal
  her_k: 8  # More HER samples
```

### Shorter Training

For quick testing:

```yaml
training:
  num_epochs: 50
  episodes_per_epoch: 20
```

### GPU Training

```yaml
experiment:
  device: "cuda"  # Use GPU
```

### Load and Continue Training

```python
import torch
from src.dqn_agent import DQNAgent

# Load checkpoint
agent = DQNAgent(...)
agent.load('experiments/dqn_checkpoint_epoch_100.pt')

# Continue training
```

### Evaluate Trained Model

```python
from src.environment import BitFlipEnv
from src.dqn_her_agent import DQNHERAgent
from src.utils import evaluate_agent

# Load environment and agent
env = BitFlipEnv(n_bits=25)
agent = DQNHERAgent(...)
agent.load('experiments/dqn_her_final.pt')

# Evaluate
metrics = evaluate_agent(agent, env, num_episodes=1000)
print(f"Success Rate: {metrics['success_rate']:.4f}")
```

## Troubleshooting

### Issue: CUDA out of memory

Solution: Reduce batch size in `config.yaml`:
```yaml
dqn:
  batch_size: 64  # Reduced from 128
```

### Issue: Training is too slow

Solutions:
1. Use GPU: Set `device: "cuda"` in config
2. Reduce episodes: Lower `episodes_per_epoch`
3. Use smaller network: Reduce `hidden_sizes`

### Issue: Poor performance

Solutions:
1. Train longer: Increase `num_epochs`
2. Adjust exploration: Modify `epsilon_decay`
3. Try different HER strategies: Change `her_strategy`

### Issue: Import errors

Solution: Make sure you're in the project root directory:
```bash
cd bitflip_dqn
python compare_results.py
```

## Performance Tips

1. **Start with small experiments**: Use 10-15 bits first to verify setup
2. **Monitor buffer size**: Should grow steadily during training
3. **Watch epsilon decay**: Should decrease from 1.0 to ~0.01
4. **Check success rate trend**: Should increase over epochs for HER

## Additional Resources

- [DQN Paper](https://arxiv.org/abs/1312.5602)
- [HER Paper](https://arxiv.org/abs/1707.01495)
- [OpenAI Spinning Up](https://spinningup.openai.com/)