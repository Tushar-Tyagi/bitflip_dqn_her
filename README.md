# BitFlip DQN vs DQN-HER

A complete implementation comparing DQN and DQN with Hindsight Experience Replay (HER) on the BitFlip-25 environment.

## Overview

This project implements and compares three reinforcement learning approaches:
- **DQN (Deep Q-Network)**: Standard DQN algorithm
- **DQN-HER**: DQN enhanced with Hindsight Experience Replay for sparse reward environments
- **DQN-HER-Prioritised**: DQN with HER and Prioritized Experience Replay, supporting multiple priority strategies

The BitFlip-25 environment is a challenging sparse reward environment where the agent must flip bits to match a target configuration.

## Project Structure

```
bitflip_dqn/
├── src/
│   ├── environment.py                  # BitFlip-25 implementation
│   ├── network.py                      # DQN architectures
│   ├── replay_buffer.py                # Standard replay
│   ├── her_buffer.py                   # HER with goal relabeling
│   ├── dqn_agent.py                    # DQN agent
│   ├── dqn_her_agent.py                # DQN-HER agent
│   ├── dqn_her_prioritized_agent.py    # DQN-HER-Prioritised agent
│   ├── dqn_prioritized_agent.py        # DQN-Prioritised agent
│   ├── prioritized_buffer_base.py      # Generic Prioritized Replay Buffer
│   ├── prioritized_her_buffer.py       # Prioritized HER Buffer
│   ├── priority_computer.py            # Priority Computation
│   ├── sum_tree.py                     # Sum Tree Data Structure for Prioritized Sampling
│   └── utils.py                        # Helpers
├── train_dqn.py                        # Train DQN
├── train_dqn_her.py                    # Train DQN-HER
├── train_dqn_her_prioritised.py        # Train DQN-HER-Prioritised
├── compare_results.py                  # Main comparison script
├── plot_results.py                     # Visualize results
├── test_setup.py                       # Verify setup
├── config.yaml                         # All hyperparameters
├── requirements.txt                    # Dependencies
├── README.md                           # Documentation
├── USAGE_GUIDE.md                      # Detailed guide
└── PROJECT_SUMMARY.md                  # Complete summary
```
## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd bitflip_dqn

# Install dependencies
pip install -r requirements.txt
```

## Usage
Refer to the USAGE_GUIDE for installation and more details on the project implementation.
### Train DQN

```bash
python train_dqn.py
```

### Train DQN-HER

```bash
python train_dqn_her.py
```

### Train DQN-HER-Prioritised

```bash
python train_dqn_her_prioritised.py
```

### Plot Existing Results

To visualize results from all three methods (including prioritised):

```bash
python plot_results.py --dqn-results experiments/dqn_results.json --her-results experiments/dqn_her_results.json --her-prioritized-results experiments_prioritized/dqn_her_prioritized_results.json
```

## Results

Each training script generates:
- Training curves (accuracy vs epoch)
- Success rate comparisons
- Saved model checkpoints

Results from the comparison experiments can be found in the "results" folder.

## References

- [Playing Atari with Deep Reinforcement Learning (DQN)](https://arxiv.org/abs/1312.5602)
- [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495)
- [PRIORITIZED Experience Replay](https://arxiv.org/pdf/1511.05952)
