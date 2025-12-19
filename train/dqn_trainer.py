"""
Deep Q-Network (DQN) Trainer for Controls Challenge
===================================================

This script trains a DQN agent to learn optimal steering control for lateral acceleration tracking.
The agent learns to map high-dimensional state observations to discrete steering actions.

Architecture:
- State: 205-dimensional vector (target, current, vehicle state, future plan)
- Actions: 257 discrete steering commands mapped to [-2, 2] range
- Network: 3-layer MLP with experience replay and target network
- Training: Episode-based learning on driving scenarios

Usage:
    python train/dqn_trainer.py --episodes 1000 --data_path ./data --model_path ./models/tinyphysics.onnx
"""

import sys
import os
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path to import tinyphysics
sys.path.append(str(Path(__file__).parent.parent))

from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from controllers import BaseController

class DQNNetwork(nn.Module):
    """Deep Q-Network for steering control - Simplified architecture"""
    def __init__(self, input_size=15, hidden_size=128, action_size=65):
        super().__init__()
        # SIMPLIFIED: Only 2-3 layers for better gradient flow
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)  # Raw Q-values, no activation
        )
        
        # Initialize weights properly
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    """Experience replay buffer for DQN training"""
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNController(BaseController):
    """DQN-based controller for vehicle steering"""
    
    def __init__(self, state_size=205, action_size=257, lr=3e-4, device='cpu', training=True):
        self.device = torch.device(device)
        self.state_size = state_size
        self.action_size = action_size
        self.training = training
        
        # Neural networks
        self.q_network = DQNNetwork(state_size, 128, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, 128, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # DQN parameters - improved settings
        self.epsilon = 1.0 if training else 0.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999  # Slower decay for better exploration
        self.gamma = 0.99
        self.batch_size = 64  # Larger batch for stability
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
        
        # Initialize target network
        self.update_target_network()
    
    def extract_state_features(self, target_lataccel, current_lataccel, state, future_plan):
        """Extract 15-dimensional state features using controller_2's exact method"""
        try:
            # Check for None values
            if any(x is None for x in [target_lataccel, current_lataccel, state, future_plan]):
                print(f"Warning: None values in inputs")
                return None
            
            if (state.roll_lataccel is None or state.v_ego is None or 
                future_plan.lataccel is None or future_plan.roll_lataccel is None or 
                future_plan.v_ego is None or future_plan.a_ego is None):
                print(f"Warning: None values in state or future_plan attributes")
                return None
            
            # Controller_2's exact feature extraction method
            future_segments = [(1, 2), (2, 3), (3, 4)]
            
            def average(values):
                if len(values) == 0:
                    return 0.0
                return sum(values) / len(values)
            
            def normalize_v_ego(v_ego_m_s):
                max_m_s = 40.0
                v = max(0, v_ego_m_s)
                return math.sqrt(v) / math.sqrt(max_m_s)
            
            # Extract features exactly as in controller_2
            diff_values = {
                'lataccel': [current_lataccel - average(future_plan.lataccel[start:end]) for start, end in future_segments],
                'roll': [state.roll_lataccel - average(future_plan.roll_lataccel[start:end]) for start, end in future_segments],
                'v_ego': [normalize_v_ego(average(future_plan.v_ego[start:end])) for start, end in future_segments],
                'a_ego': [average(future_plan.a_ego[start:end]) for start, end in future_segments],
            }
            
            # Previous actions (last 3, padded with 0 if not enough)
            previous_action = self.prev_actions[-3:] if len(self.prev_actions) >= 3 else [0, 0, 0]
            # Pad to exactly 3 elements
            while len(previous_action) < 3:
                previous_action = [0] + previous_action
            
            # Combine all features in the exact order as controller_2
            state_input_list = (diff_values['lataccel'] + 
                               diff_values['roll'] + 
                               diff_values['a_ego'] + 
                               diff_values['v_ego'] + 
                               previous_action)
            
            # Should be exactly 15 features: 3+3+3+3+3 = 15
            if len(state_input_list) != 15:
                print(f"Warning: Expected 15 features, got {len(state_input_list)}")
                # Ensure exactly 15 features
                state_input_list = state_input_list[:15]
                while len(state_input_list) < 15:
                    state_input_list.append(0.0)
            
            return np.array(state_input_list, dtype=np.float32)
            
        except Exception as e:
            print(f"Error in extract_state_features: {e}")
            return None
    
    def action_to_steering(self, action_idx):
        """Convert discrete action [0,64] to continuous steering [-2,2]"""
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
    
    def store_reward_and_transition(self, predicted_lataccel):
        """Store reward based on physics model prediction (called from training loop)"""
        # Check if we have valid data for transition
        if (self.training and 
            self.last_state is not None and 
            self.last_target is not None and 
            self.last_action is not None and
            self.current_state is not None and
            predicted_lataccel is not None):
            
            # Calculate reward based on RESULT of previous action
            tracking_error = abs(self.last_target - predicted_lataccel)
            
            # Stronger reward signal for better learning
            if tracking_error < 0.05:
                reward = 10.0  # High reward for excellent tracking
            elif tracking_error < 0.1:
                reward = 5.0   # Good reward for good tracking
            elif tracking_error < 0.2:
                reward = 1.0   # Small positive reward for decent tracking
            else:
                reward = -tracking_error * 5.0  # Strong penalty for poor tracking
            
            # Smoothness penalty (reduced impact)
            if len(self.prev_actions) >= 2:
                action_change = abs(self.prev_actions[-1] - self.prev_actions[-2])
                reward -= action_change * 0.2
            
            # Store transition
            done = False  # Continuous episodes
            self.replay_buffer.push(
                self.last_state, self.last_action, reward, 
                self.current_state, done
            )
        
        # Update for next timestep
        if self.current_state is not None:
            self.last_state = self.current_state
            self.last_action = self.current_action  
            self.last_target = self.current_target
    
    def train_step(self):
        """Perform one training step using experience replay"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(np.array(dones)).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        # Add gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.training_step_count += 1
        if self.training_step_count % self.update_target_freq == 0:
            self.update_target_network()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, path):
        """Save model checkpoint"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step_count': self.training_step_count
        }, path)
    
    def load_model(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', 0.0)
        self.training_step_count = checkpoint.get('training_step_count', 0)

def run_training_episode(dqn_controller, physics_model, data_file):
    """Run single training episode with proper reward timing"""
    # Create simulator
    sim = TinyPhysicsSimulator(physics_model, str(data_file), dqn_controller, debug=False)
    
    episode_losses = []
    
    # Run episode
    try:
        for step in range(100, len(sim.data)):  # Start after warmup period
            # Step simulation (controller chooses action, physics predicts response)
            sim.step()
            
            # CRITICAL: Store reward based on physics model prediction
            # This happens AFTER the action is applied and physics responds
            predicted_lataccel = sim.current_lataccel
            
            # Debug check for None values
            if predicted_lataccel is None:
                print(f"Warning: predicted_lataccel is None at step {step}")
                predicted_lataccel = 0.0  # fallback value
            
            dqn_controller.store_reward_and_transition(predicted_lataccel)
            
            # Train every 2 steps (increased frequency)
            if step % 2 == 0:
                loss = dqn_controller.train_step()
                if loss is not None:
                    episode_losses.append(loss)
        
        # Get final cost
        cost = sim.compute_cost()
        return cost, episode_losses
    
    except Exception as e:
        import traceback
        print(f"Episode failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return {'total_cost': 1000, 'lataccel_cost': 500, 'jerk_cost': 500}, []

def evaluate_model(dqn_controller, physics_model, data_files, num_eval_episodes=10):
    """Evaluate trained model on test data"""
    dqn_controller.training = False
    dqn_controller.epsilon = 0.0  # No exploration during evaluation
    
    costs = []
    for data_file in data_files[:num_eval_episodes]:
        sim = TinyPhysicsSimulator(physics_model, str(data_file), dqn_controller, debug=False)
        
        try:
            cost = sim.rollout()
            costs.append(cost)
        except:
            costs.append({'total_cost': 1000, 'lataccel_cost': 500, 'jerk_cost': 500})
    
    # Calculate averages
    avg_total = np.mean([c['total_cost'] for c in costs])
    avg_lataccel = np.mean([c['lataccel_cost'] for c in costs])
    avg_jerk = np.mean([c['jerk_cost'] for c in costs])
    
    dqn_controller.training = True  # Resume training
    return avg_total, avg_lataccel, avg_jerk

def plot_training_progress(episode_costs, episode_losses, save_path):
    """Plot and save training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot costs
    ax1.plot(episode_costs)
    ax1.set_title('Episode Total Costs')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Cost')
    ax1.grid(True)
    
    # Plot losses (moving average)
    if episode_losses:
        window = max(1, len(episode_losses) // 50)
        smoothed_losses = [np.mean(episode_losses[max(0, i-window):i+1]) for i in range(len(episode_losses))]
        ax2.plot(smoothed_losses)
        ax2.set_title('Training Loss (Smoothed)')
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Loss')
        ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train DQN for Controls Challenge')
    parser.add_argument('--episodes', type=int, default=500, help='Number of training episodes')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to training data')
    parser.add_argument('--model_path', type=str, default='./models/tinyphysics.onnx', help='Path to physics model')
    parser.add_argument('--save_freq', type=int, default=50, help='Save model every N episodes')
    parser.add_argument('--eval_freq', type=int, default=25, help='Evaluate every N episodes')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    
    args = parser.parse_args()
    
    print("="*60)
    print("DQN TRAINING FOR CONTROLS CHALLENGE")
    print("="*60)
    print(f"Device: {args.device}")
    print(f"Episodes: {args.episodes}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Data path: {args.data_path}")
    print("="*60)
    
    # Setup
    device = torch.device(args.device)
    
    # Load physics model
    physics_model = TinyPhysicsModel(args.model_path, debug=False)
    
    # Get training data files
    data_path = Path(args.data_path)
    data_files = list(data_path.glob("*.csv"))
    if not data_files:
        print(f"No CSV files found in {data_path}")
        return
    
    random.shuffle(data_files)
    train_files = data_files[:-50]  # Reserve last 50 for evaluation
    eval_files = data_files[-50:]
    
    print(f"Training files: {len(train_files)}")
    print(f"Evaluation files: {len(eval_files)}")
    
    # Initialize DQN controller
    dqn_controller = DQNController(
        state_size=15, 
        action_size=65, 
        lr=args.learning_rate,
        device=args.device,
        training=True
    )
    
    # Training tracking
    episode_costs = []
    all_losses = []
    best_cost = float('inf')
    
    # Create save directory
    save_dir = Path("train/checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print("\nStarting training...")
    for episode in range(args.episodes):
        # Select random training file
        data_file = random.choice(train_files)
        
        # Run training episode
        cost, losses = run_training_episode(dqn_controller, physics_model, data_file)
        
        episode_costs.append(cost['total_cost'])
        all_losses.extend(losses)
        
        # Print progress
        if episode % 10 == 0:
            avg_recent_cost = np.mean(episode_costs[-10:])
            print(f"Episode {episode:4d} | Cost: {cost['total_cost']:6.2f} | "
                  f"Recent Avg: {avg_recent_cost:6.2f} | Epsilon: {dqn_controller.epsilon:.3f}")
        
        # Evaluate periodically
        if episode > 0 and episode % args.eval_freq == 0:
            avg_total, avg_lataccel, avg_jerk = evaluate_model(
                dqn_controller, physics_model, eval_files, num_eval_episodes=10
            )
            print(f"\n--- EVALUATION (Episode {episode}) ---")
            print(f"Avg Total Cost: {avg_total:.2f}")
            print(f"Avg Lataccel Cost: {avg_lataccel:.2f}")
            print(f"Avg Jerk Cost: {avg_jerk:.2f}")
            print("-" * 40)
            
            # Save best model
            if avg_total < best_cost:
                best_cost = avg_total
                best_model_path = save_dir / "best_dqn_model.pth"
                dqn_controller.save_model(best_model_path)
                print(f"New best model saved! Cost: {best_cost:.2f}")
        
        # Save checkpoint
        if episode > 0 and episode % args.save_freq == 0:
            checkpoint_path = save_dir / f"dqn_episode_{episode}.pth"
            dqn_controller.save_model(checkpoint_path)
            
            # Plot progress
            plot_path = save_dir / f"training_progress_{episode}.png"
            plot_training_progress(episode_costs, all_losses, plot_path)
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    avg_total, avg_lataccel, avg_jerk = evaluate_model(
        dqn_controller, physics_model, eval_files, num_eval_episodes=50
    )
    
    print(f"Final Average Total Cost: {avg_total:.2f}")
    print(f"Final Average Lataccel Cost: {avg_lataccel:.2f}")
    print(f"Final Average Jerk Cost: {avg_jerk:.2f}")
    
    # Save final model
    final_model_path = save_dir / "final_dqn_model.pth"
    dqn_controller.save_model(final_model_path)
    
    # Save final plots
    final_plot_path = save_dir / "final_training_progress.png"
    plot_training_progress(episode_costs, all_losses, final_plot_path)
    
    print(f"\nTraining completed!")
    print(f"Best model saved to: {save_dir / 'best_dqn_model.pth'}")
    print(f"Final model saved to: {final_model_path}")

if __name__ == "__main__":
    main()