# BitFlip-25 DQN vs DQN-HER

A complete implementation comparing DQN and DQN with Hindsight Experience Replay (HER) on the BitFlip-25 environment.

## Overview

This project implements and compares two reinforcement learning approaches:
- **DQN (Deep Q-Network)**: Standard DQN algorithm
- **DQN-HER**: DQN enhanced with Hindsight Experience Replay for sparse reward environments

The BitFlip-25 environment is a challenging sparse reward environment where the agent must flip bits to match a target configuration.

## Project Structure

bitflip_dqn/
├── src/
│   ├── environment.py       # BitFlip-25 implementation
│   ├── network.py          # DQN architectures
│   ├── replay_buffer.py    # Standard replay
│   ├── her_buffer.py       # HER with goal relabeling
│   ├── dqn_agent.py        # DQN agent
│   ├── dqn_her_agent.py    # DQN-HER agent
│   └── utils.py            # Helpers
├── train_dqn.py            # Train DQN
├── train_dqn_her.py        # Train DQN-HER
├── compare_results.py      # Main comparison script
├── plot_results.py         # Visualize results
├── test_setup.py           # Verify setup
├── config.yaml             # All hyperparameters
├── requirements.txt        # Dependencies
├── README.md              # Documentation
├── USAGE_GUIDE.md         # Detailed guide
└── PROJECT_SUMMARY.md     # Complete summary

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd bitflip_dqn

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Train DQN

```bash
python train_dqn.py
```

### Train DQN-HER

```bash
python train_dqn_her.py
```

### Compare Results     and Plot

```bash
python compare_results.py
```

This will train both agents and generate comparison plots showing accuracy vs epoch.

### Plot Existing Results

```bash
python plot_results.py --dqn-results experiments/dqn_results.json --her-results experiments/dqn_her_results.json
```

## Configuration

Edit `config.yaml` to modify hyperparameters:
- Network architecture
- Learning rate
- Replay buffer size
- Batch size
- Exploration parameters
- Training episodes/epochs

## Results

The project generates:
- Training curves (accuracy vs epoch)
- Success rate comparisons
- Episode length statistics
- Saved model checkpoints

## References

- [Playing Atari with Deep Reinforcement Learning (DQN)](https://arxiv.org/abs/1312.5602)
- [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495)
