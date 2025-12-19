"""
Configuration file for DQN training
====================================

This file contains all hyperparameters and settings for DQN training.
Modify these values to experiment with different training configurations.
"""

# Network Architecture
STATE_SIZE = 205          # Input state dimensions
ACTION_SIZE = 257         # Discrete steering actions
HIDDEN_SIZE = 256         # Hidden layer size
LEARNING_RATE = 3e-4      # Adam optimizer learning rate

# DQN Hyperparameters
GAMMA = 0.99              # Discount factor for future rewards
EPSILON_START = 1.0       # Initial exploration rate
EPSILON_MIN = 0.01        # Minimum exploration rate
EPSILON_DECAY = 0.995     # Epsilon decay rate per episode
BATCH_SIZE = 32           # Training batch size
REPLAY_BUFFER_SIZE = 50000 # Experience replay buffer capacity
TARGET_UPDATE_FREQ = 100   # Update target network every N training steps

# Training Parameters
NUM_EPISODES = 500        # Total training episodes
SAVE_FREQ = 50           # Save checkpoint every N episodes
EVAL_FREQ = 25           # Evaluate model every N episodes
TRAIN_FREQ = 4           # Train network every N simulation steps

# Reward Function Parameters
TRACKING_ERROR_WEIGHT = 10.0    # Weight for lateral acceleration tracking error
SMOOTHNESS_PENALTY = 0.1        # Weight for action smoothness penalty
JERK_PENALTY = 0.05            # Weight for jerk (acceleration change) penalty

# Data and Model Paths
DEFAULT_DATA_PATH = "./data"
DEFAULT_PHYSICS_MODEL_PATH = "./models/tinyphysics.onnx"
CHECKPOINT_DIR = "./train/checkpoints"

# Device Settings
DEVICE = "cuda"  # or "cpu"

# Evaluation Settings
EVAL_EPISODES = 10       # Number of episodes for periodic evaluation
FINAL_EVAL_EPISODES = 50 # Number of episodes for final evaluation

# Logging and Visualization
LOG_FREQ = 10            # Print progress every N episodes
PLOT_WINDOW = 50         # Smoothing window for loss plots

# Advanced DQN Options (for future experimentation)
USE_DOUBLE_DQN = False   # Enable Double DQN
USE_DUELING_DQN = False  # Enable Dueling DQN
USE_PRIORITIZED_REPLAY = False  # Enable Prioritized Experience Replay