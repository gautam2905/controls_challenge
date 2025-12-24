import gymnasium as gym
import numpy as np
import pandas as pd
from pathlib import Path
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from tinyphysics import CONTEXT_LENGTH, TinyPhysicsSimulator, TinyPhysicsModel, CONTROL_START_IDX

class TinyPhysicsEnv(gym.Env):
    def __init__(self, data_dir, model_path, context_length=20):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.files = sorted(list(self.data_dir.glob("*.csv")))
        self.model_path = model_path
        self.context_length = context_length
        self.n_lookahead = 20
        self.tinyphysics_model = TinyPhysicsModel(model_path, debug=False)

        # Actions: Steer command between -2 and +2 radians (approx)
        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32)
        
        # Observation: [v_ego, a_ego, roll, current_lat, *future_targets]
        # We added 'prev_steer' so the model knows its last action (essential for avoiding oscillation)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(4 + self.n_lookahead,), 
            dtype=np.float32
        )

        self.current_step = 0
        self.sim = None

    def reset(self, options=None, seed=None):
        super().reset(seed=seed)

        # Pick a random segment
        random_file = np.random.choice(self.files)
        self.sim = TinyPhysicsSimulator(self.tinyphysics_model, data_path=str(random_file), controller=None, debug=False)
        
        self.sim.step_idx = CONTEXT_LENGTH
        
        # Warm up the simulator with human data to fill history
        for i in range(CONTEXT_LENGTH, CONTROL_START_IDX):
            state, target, futureplan = self.sim.get_state_target_futureplan(i)
            self.sim.state_history.append(state)
            self.sim.target_lataccel_history.append(target)
            
            human_steer = self.sim.data['steer_command'].values[i]
            self.sim.action_history.append(human_steer)
            self.sim.sim_step(i)
            
        self.current_step = CONTROL_START_IDX
        
        # Reset previous steer to the last human action
        
        return self._get_observation(), {}

    def step(self, action):
        if self.sim is None:
            raise RuntimeError("Env must be reset before stepping.")
        
        # Clip action for safety
        steer_action = float(np.clip(action[0], -2.0, 2.0))
        
        # 1. Step Simulator
        state, target, futureplan = self.sim.get_state_target_futureplan(self.current_step)
        self.sim.state_history.append(state)
        self.sim.target_lataccel_history.append(target)
        self.sim.action_history.append(steer_action)
        self.sim.sim_step(self.current_step)

        # 2. Calculate Reward
        # Get actual result after physics step
        current_lataccel = self.sim.current_lataccel_history[-1]
        target_lataccel = self.sim.target_lataccel_history[-1]
        
        # Cost 1: Tracking Error (L1 loss is often better for tight control than L2)
        lat_error = np.abs(target_lataccel - current_lataccel)
        
        # Cost 2: Action Smoothness (Penalize changing steering too fast)
        steer_rate = np.abs(steer_action - self.prev_steer)
        
        # Reward function weights
        # We want to minimize error heavily, but also discourage jerking the wheel
        reward = - (lat_error * 10.0) - (steer_rate * 0.5)

        # Update state variables
        self.prev_steer = steer_action
        self.current_step += 1
        self.sim.step_idx = self.current_step

        # Check termination
        terminated = self.current_step >= len(self.sim.data) - self.n_lookahead - 1
        truncated = False # Typically not used in this specific setup
        
        obs = self._get_observation()
        
        # Add a bonus for finishing the episode without crashing (optional but helpful)
        if terminated:
            reward += 10.0

        return obs, reward, terminated, truncated, {}
        
    def _get_observation(self):
        idx = self.current_step
        if idx >= len(self.sim.data):
            idx = len(self.sim.data) - 1
    
        v_ego = self.sim.data['v_ego'].values[idx]
        a_ego = self.sim.data['a_ego'].values[idx]
        roll_lataccel = self.sim.data['roll_lataccel'].values[idx]
        curr_lat = self.sim.current_lataccel_history[-1]
        
        # Future targets
        future_targets = self.sim.data['target_lataccel'].values[idx+1 : idx+1+self.n_lookahead]
        
        # Pad future targets if we are near end of data (edge case)
        if len(future_targets) < self.n_lookahead:
            padding = np.zeros(self.n_lookahead - len(future_targets))
            future_targets = np.concatenate([future_targets, padding])

        # IMPORTANT: Include prev_steer so the model sees its own momentum
        obs = np.array([
            v_ego, 
            a_ego, 
            roll_lataccel, 
            curr_lat, 
            self.prev_steer, 
            *future_targets
        ], dtype=np.float32)
        
        return obs

if __name__ == "__main__":
    def make_env():
        return Monitor(TinyPhysicsEnv(data_dir="./data", model_path="./models/tinyphysics.onnx"))
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=3e-4,
        batch_size=64,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[128, 128]) # Slightly larger network
    )

    print("Starting training...")
    model.learn(total_timesteps=2_000_000)

    model.save("models/ppo_tinyphysics")
    env.save("models/vec_normalize.pkl")
    print("Training complete. Model and stats saved.")