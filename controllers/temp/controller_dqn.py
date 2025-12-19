from .. import BaseController
import numpy as np
import torch
import os
import sys
from pathlib import Path
from train.dqn_trainer import DQNNetwork, ReplayBuffer
import random 
import torch.optim as optim 
import torch.nn as nn

sys.path.append(str(Path(__file__).parent.parent))
 
class Controller(BaseController):
    """DQN-based controller for vehicle steering"""
    
    def __init__(self, state_size=205, action_size=257, lr=3e-4, device='cpu', training=True):
        self.device = torch.device(device)
        self.state_size = state_size
        self.action_size = action_size
        self.training = training
        
        # Neural networks
        self.q_network = DQNNetwork(state_size, 256, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, 256, action_size).to(self.device)
        

        # DQN parameters
        self.epsilon = 1.0 if training else 0.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.99
        self.batch_size = 32
        self.update_target_freq = 100
        
        # Experience replay
        self.replay_buffer = ReplayBuffer(capacity=50000)
        
        # State tracking
        self.prev_actions = []
        self.step_count = 0
        self.training_step_count = 0
        
        # Initialize state tracking for proper reward timing
        self.current_state = None
        self.current_action = None
        self.current_target = None
        self.last_state = None
        self.last_action = None
        self.last_target = None
        # self.load_model(path="train/checkpoints/best_dqn_model.pth")
        self.load_model(path="train/checkpoints/final_dqn_model.pth")
        
    def extract_state_features(self, target_lataccel, current_lataccel, state, future_plan):
        """Extract 205-dimensional state features"""
        try:
            features = []
            
            # Basic control signals (2) - check for None
            if target_lataccel is None or current_lataccel is None:
                print(f"Warning: None values in basic signals - target: {target_lataccel}, current: {current_lataccel}")
                return None
                
            features.append(target_lataccel / 5.0)
            features.append(current_lataccel / 5.0)
            
            # Vehicle state (3) - check for None
            if state is None or any(x is None for x in [state.roll_lataccel, state.v_ego, state.a_ego]):
                print(f"Warning: None values in vehicle state")
                return None
                
            features.extend([state.roll_lataccel / 10.0, state.v_ego / 40.0, state.a_ego / 5.0])
            
            # Future plan (200): 4×50
            if future_plan is None:
                print("Warning: future_plan is None")
                return None
            
            # Pad or truncate to exactly 50 values each
            def pad_or_truncate(lst, length, scale):
                if lst is None:
                    lst = [0.0] * length
                if len(lst) >= length:
                    normalized = [x / scale for x in lst[:length]]
                else:
                    padded = lst + [lst[-1] if lst else 0.0] * (length - len(lst))
                    normalized = [x / scale for x in padded]
                return normalized
            
            features.extend(pad_or_truncate(future_plan.lataccel, 50, 5.0))
            features.extend(pad_or_truncate(future_plan.roll_lataccel, 50, 10.0))
            features.extend(pad_or_truncate(future_plan.v_ego, 50, 40.0))
            features.extend(pad_or_truncate(future_plan.a_ego, 50, 5.0))
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            print(f"Error in extract_state_features: {e}")
            return None
    
    def action_to_steering(self, action_idx):
        """Convert discrete action [0,256] to continuous steering [-2,2]"""
        return -2.0 + (4.0 * action_idx / (self.action_size - 1))
    
    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        if self.training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        """Main controller interface - same as PID"""
        # Extract state features
        state_features = self.extract_state_features(target_lataccel, current_lataccel, state, future_plan)
        
        # Handle None state features (fallback to safe action)
        if state_features is None:
            print("Warning: state_features is None, using fallback action")
            steering_command = 0.0  # Go straight as fallback
            self.prev_actions.append(steering_command)
            self.step_count += 1
            return steering_command
        
        # Select action
        action_idx = self.select_action(state_features)
        steering_command = self.action_to_steering(action_idx)
        
        # Store current state and action for NEXT timestep's reward calculation
        self.current_state = state_features
        self.current_action = action_idx
        self.current_target = target_lataccel
        
        # Update tracking variables
        self.prev_actions.append(steering_command)
        self.step_count += 1
        
        return steering_command
       
    def load_model(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
