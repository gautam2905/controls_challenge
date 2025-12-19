# DQN Training for Controls Challenge

This directory contains a complete Deep Q-Network (DQN) implementation for learning optimal steering control in the Controls Challenge.

## Files Overview

- **`dqn_trainer.py`** - Main training script with complete DQN implementation
- **`config.py`** - Configuration file with all hyperparameters
- **`run_training.py`** - Simple runner script with default settings
- **`README.md`** - This documentation file

## Quick Start

### 1. Basic Training
```bash
# Run training with default settings (500 episodes)
python train/run_training.py

# Quick test run (50 episodes)
python train/run_training.py --quick_test

# Force GPU usage
python train/run_training.py --gpu
```

### 2. Advanced Training
```bash
# Full control over parameters
python train/dqn_trainer.py \
    --episodes 1000 \
    --data_path ./data \
    --model_path ./models/tinyphysics.onnx \
    --learning_rate 1e-4 \
    --device cuda
```
python train/dqn_trainer.py --episodes 1000 --data_path ./data --model_path ./models/tinyphysics.onnx --learning_rate 1e-4 --device cuda


## Architecture Overview

### State Representation (205 dimensions)
- **Target lateral acceleration** (1) - What we want to achieve
- **Current lateral acceleration** (1) - What we currently have  
- **Vehicle state** (3) - roll_lataccel, v_ego, a_ego
- **Future plan** (200) - Next 5 seconds of trajectory (4×50 values)

### Action Space (257 discrete actions)
- Discrete steering commands mapped to continuous range [-2, 2]
- Action 0 → -2.0 (hard left)
- Action 128 → 0.0 (straight)  
- Action 256 → +2.0 (hard right)

### Network Architecture
```
Input Layer:    205 neurons (state features)
Hidden Layer 1: 256 neurons + ReLU
Hidden Layer 2: 256 neurons + ReLU  
Hidden Layer 3: 128 neurons + ReLU
Output Layer:   257 neurons (Q-values for each action)
```

## Training Process

1. **Episode Loop**: Train on random driving scenarios from CSV data
2. **Experience Collection**: Store (state, action, reward, next_state) transitions
3. **Network Training**: Use experience replay with target network
4. **Evaluation**: Periodic testing on held-out data
5. **Checkpointing**: Save best models and training progress

## Reward Function

```python
reward = -tracking_error * 10.0 - action_smoothness * 0.1
```

- **Tracking Error**: Penalize deviation from target lateral acceleration
- **Smoothness**: Penalize abrupt steering changes for passenger comfort

## Expected Training Timeline

- **Episodes 0-100**: High exploration, learning basic control
- **Episodes 100-300**: Reducing exploration, improving tracking
- **Episodes 300-500**: Fine-tuning, achieving smooth control

## Output Files

Training creates several output files in `train/checkpoints/`:

- **`best_dqn_model.pth`** - Best performing model (lowest validation cost)
- **`final_dqn_model.pth`** - Final model after all training
- **`dqn_episode_X.pth`** - Periodic checkpoints every 50 episodes
- **`training_progress_X.png`** - Training curves and loss plots
- **`final_training_progress.png`** - Complete training visualization

## Performance Monitoring

The training script prints progress every 10 episodes:
```
Episode  100 | Cost: 156.23 | Recent Avg: 145.67 | Epsilon: 0.605
Episode  110 | Cost: 142.45 | Recent Avg: 140.12 | Epsilon: 0.594
```

Evaluation runs every 25 episodes on held-out data:
```
--- EVALUATION (Episode 100) ---
Avg Total Cost: 138.45
Avg Lataccel Cost: 2.14
Avg Jerk Cost: 31.23
----------------------------------------
```

## Hyperparameter Tuning

Key hyperparameters to experiment with:

### Learning Rate
- **3e-4** (default) - Good starting point
- **1e-4** - More stable, slower convergence
- **1e-3** - Faster learning, may be unstable

### Network Architecture
- **Hidden size**: 128, 256, 512
- **Number of layers**: 2, 3, 4
- **Activation**: ReLU, GELU, Swish

### Exploration
- **Epsilon decay**: 0.99 (slower), 0.995 (default), 0.999 (very slow)
- **Minimum epsilon**: 0.01 (default), 0.05 (more exploration)

### Reward Function
- **Tracking weight**: 5.0, 10.0 (default), 20.0
- **Smoothness weight**: 0.01, 0.1 (default), 0.5

## Troubleshooting

### Common Issues

1. **High initial costs**: Normal during exploration phase
2. **Unstable training**: Reduce learning rate or increase batch size
3. **Slow convergence**: Increase learning rate or reduce epsilon decay
4. **Memory issues**: Reduce replay buffer size or batch size

### Performance Targets

- **Good performance**: Total cost < 150
- **Competitive performance**: Total cost < 100  
- **Excellent performance**: Total cost < 80

### Comparison with Baselines

- **Zero controller**: ~200-300 total cost
- **PID controller**: ~100-150 total cost
- **Target for DQN**: < 100 total cost

## Next Steps

After basic DQN training, consider:

1. **Double DQN**: Reduce overestimation bias
2. **Dueling DQN**: Separate value and advantage estimation
3. **Prioritized Replay**: Focus on important experiences
4. **PPO/SAC**: Try continuous control algorithms
5. **Curriculum Learning**: Start with easier scenarios

## Dependencies

Make sure you have installed:
```bash
pip install torch numpy matplotlib tqdm
```

The training script automatically imports from the parent directory to access `tinyphysics.py` and controllers.