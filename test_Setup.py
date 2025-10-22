"""
Test Script to Verify Installation and Setup
Run this to ensure everything is working correctly
"""

import sys
import torch
import numpy as np

print("="*60)
print("BitFlip-25 DQN Setup Verification")
print("="*60)

# Check Python version
print(f"\n1. Python Version: {sys.version}")
assert sys.version_info >= (3, 7), "Python 3.7+ required"
print("   âœ“ Python version OK")

# Check PyTorch
print(f"\n2. PyTorch Version: {torch.__version__}")
print(f"   CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
print("   âœ“ PyTorch OK")

# Check NumPy
print(f"\n3. NumPy Version: {np.__version__}")
print("   âœ“ NumPy OK")

# Test imports
print("\n4. Testing imports...")
try:
    sys.path.append('src')
    from environment import BitFlipEnv
    from network import DQN
    from replay_buffer import ReplayBuffer
    from her_buffer import HERBuffer
    from dqn_agent import DQNAgent
    from dqn_her_agent import DQNHERAgent
    print("   âœ“ All core modules imported successfully")
except ImportError as e:
    print(f"   âœ— Import error: {e}")
    sys.exit(1)

# Test environment
print("\n5. Testing BitFlip environment...")
try:
    env = BitFlipEnv(n_bits=5)
    obs_dict = env.reset()
    action = 0
    obs_dict, reward, terminated, truncated, info = env.step(action)
    print(f"   Environment shape: {env.observation_size}")
    print(f"   Action space: {env.action_space_size}")
    print("   âœ“ Environment working")
except Exception as e:
    print(f"   âœ— Environment error: {e}")
    sys.exit(1)

# Test DQN network
print("\n6. Testing DQN network...")
try:
    net = DQN(input_size=10, output_size=5, hidden_sizes=[64, 64])
    x = torch.randn(4, 10)
    y = net(x)
    assert y.shape == (4, 5), "Network output shape mismatch"
    print(f"   Network parameters: {sum(p.numel() for p in net.parameters())}")
    print("   âœ“ DQN network working")
except Exception as e:
    print(f"   âœ— Network error: {e}")
    sys.exit(1)

# Test replay buffer
print("\n7. Testing replay buffers...")
try:
    buffer = ReplayBuffer(capacity=100, observation_size=10)
    for _ in range(50):
        obs = np.random.randn(10).astype(np.float32)
        buffer.add(obs, 0, -1.0, obs, False)
    batch = buffer.sample(16)
    assert len(batch) == 5, "Buffer sample incorrect"
    print(f"   Standard buffer size: {len(buffer)}")
    
    her_buffer = HERBuffer(capacity=100, observation_size=10, goal_size=5)
    print("   âœ“ Replay buffers working")
except Exception as e:
    print(f"   âœ— Buffer error: {e}")
    sys.exit(1)

# Test DQN agent
print("\n8. Testing DQN agent...")
try:
    agent = DQNAgent(
        observation_size=10,
        action_size=5,
        batch_size=16,
        buffer_size=1000
    )
    obs = np.random.randn(10).astype(np.float32)
    action = agent.select_action(obs)
    assert 0 <= action < 5, "Invalid action selected"
    print(f"   Agent device: {agent.device}")
    print("   âœ“ DQN agent working")
except Exception as e:
    print(f"   âœ— Agent error: {e}")
    sys.exit(1)

# Test DQN-HER agent
print("\n9. Testing DQN-HER agent...")
try:
    agent_her = DQNHERAgent(
        observation_size=10,
        action_size=5,
        goal_size=5,
        batch_size=16,
        buffer_size=1000,
        her_strategy='future',
        her_k=4
    )
    obs = np.random.randn(10).astype(np.float32)
    action = agent_her.select_action(obs)
    assert 0 <= action < 5, "Invalid action selected"
    print(f"   HER agent device: {agent_her.device}")
    print("   âœ“ DQN-HER agent working")
except Exception as e:
    print(f"   âœ— HER agent error: {e}")
    sys.exit(1)

# Quick training test
print("\n10. Quick training test (5 episodes)...")
try:
    env = BitFlipEnv(n_bits=5)
    agent = DQNAgent(
        observation_size=env.observation_size,
        action_size=env.action_space_size,
        batch_size=8,
        buffer_size=1000
    )
    
    for episode in range(5):
        obs_dict = env.reset()
        obs = obs_dict['observation']
        done = False
        steps = 0
        
        while not done and steps < env.max_steps:
            action = agent.select_action(obs, training=True)
            next_obs_dict, reward, terminated, truncated, info = env.step(action)
            next_obs = next_obs_dict['observation']
            done = terminated or truncated
            
            agent.store_transition(obs, action, reward, next_obs, done)
            agent.update()
            
            obs = next_obs
            steps += 1
        
        agent.end_episode()
    
    print(f"   Buffer size after 5 episodes: {len(agent.replay_buffer)}")
    print("   âœ“ Training loop working")
except Exception as e:
    print(f"   âœ— Training error: {e}")
    sys.exit(1)

# Check file structure
print("\n11. Checking file structure...")
import os
required_files = [
    'config.yaml',
    'train_dqn.py',
    'train_dqn_her.py',
    'compare_results.py',
    'plot_results.py',
    'src/environment.py',
    'src/network.py',
    'src/dqn_agent.py',
    'src/dqn_her_agent.py'
]

missing_files = []
for file in required_files:
    if not os.path.exists(file):
        missing_files.append(file)

if missing_files:
    print(f"   âš  Missing files: {missing_files}")
else:
    print("   âœ“ All required files present")

# Summary
print("\n" + "="*60)
print("VERIFICATION COMPLETE")
print("="*60)
print("\nâœ“ All tests passed! Your setup is ready.")
print("\nNext steps:")
print("  1. Review config.yaml to adjust hyperparameters")
print("  2. Run: python compare_results.py")
print("  3. Check experiments/ folder for results")
print("\n" + "="*60)