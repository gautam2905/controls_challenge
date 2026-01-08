import gymnasium as gym
import numpy as np
import pandas as pd
from pathlib import Path
from gymnasium import spaces
from tinyphysics import CONTEXT_LENGTH, TinyPhysicsSimulator, TinyPhysicsModel, CONTROL_START_IDX
from tqdm import tqdm
from controllers.pid2 import Controller as PIDController
VERBOSE = True

class DummyController:
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        return 0.0

class TinyPhysicsEnv(gym.Env):
    def __init__(self, data_dir, model_path, context_length=20):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.files = sorted(list(self.data_dir.glob("*.csv")))
        self.model_path = model_path
        self.context_length = context_length
        self.n_lookahead = 20
        self.tinyphysics_model = TinyPhysicsModel(model_path, debug=False)
        self.previous_action = 0.0

        # --- OPTIMIZATION START: Pre-load all data ---
        print("Pre-loading data files...")
        self.cached_dfs = []
        files = sorted(list(self.data_dir.glob("*.csv")))
        
        # Initialize sim ONCE with the first file to get access to helper methods
        self.sim = TinyPhysicsSimulator(self.tinyphysics_model, data_path=str(files[0]), controller=None, debug=False)
        
        # Process and store all dataframes in memory
        for f in tqdm(files):
            # We use the simulator's internal method to process the raw CSV
            # if VERBOSE == True:
            #     print(f"Processing file: {f}")
            processed_df = self.sim.get_data(str(f))
            self.cached_dfs.append(processed_df)
        print(f"Loaded {len(self.cached_dfs)} files.")
        # --- OPTIMIZATION END ---

        #steer command between -2 to 2
        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32)
        # Observation space is : [previous_action, vEgo,aEgo,roll,current Lateral Acceleration, target Lateral Acceleration, ...n_lookahead]  ] 
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(5 + 1 + self.n_lookahead + 2,), 
            dtype=np.float32
        )

        self.current_step = 0

    def reset(self, options=None, seed=None):
        super().reset(seed=seed)
        self.previous_action = 0.0
        
        # This clears the integral and previous error for the new episode
        self.pid = PIDController() 

        # Pick a random dataframe from our cache
        random_idx = np.random.randint(0, len(self.cached_dfs))
        self.sim.data = self.cached_dfs[random_idx]
        
        self.sim.reset()

        self.sim.step_idx = CONTEXT_LENGTH
        for i in range(CONTEXT_LENGTH, CONTROL_START_IDX):
            state, target, futureplan = self.sim.get_state_target_futureplan(i)
            self.sim.state_history.append(state)
            self.sim.target_lataccel_history.append(target)
            
            # We need to simulate what the agent (PID) would have done to update            
            # 1. Update PID internal state and get its output
            current_lataccel = self.sim.current_lataccel_history[-1]
            pid_steer = self.pid.update(target, current_lataccel, state, futureplan)
            
            # 2. Update memory: The agent sees what IT outputted, not what the human did.
            # During inference, the controller remembers its own actions.
            self.previous_action = pid_steer 
            
            human_steer = self.sim.data['steer_command'].values[i]
            self.previous_action = human_steer 
            
            self.sim.action_history.append(human_steer)
            self.sim.sim_step(i)
            
        self.current_step = CONTROL_START_IDX
        return self._get_observation(), {}
    
    def step(self, action):
        if self.sim is None:
            raise RuntimeError("Environment must be reset before stepping.")
        
        state, target, futureplan = self.sim.get_state_target_futureplan(self.current_step)
        self.sim.state_history.append(state)
        self.sim.target_lataccel_history.append(target)

        # --- FIX 2: Calculate PID Base Action ---
        current_lataccel = self.sim.current_lataccel_history[-1]
        pid_steer = self.pid.update(target, current_lataccel, state, futureplan)
        
        # Get the Residual (Correction) from PPO
        # You might want to scale this down if you want the agent to only make small tweaks
        # e.g., action_residual = float(action[0]) * 0.1
        action_residual = float(action[0]) * 0.5
        
        # Combine: Base + Residual
        combined_steer = np.clip(pid_steer + action_residual, -2.0, 2.0)
        # ----------------------------------------

        # Apply the COMBINED steer
        self.previous_action = combined_steer
        self.sim.action_history.append(combined_steer)

        self.sim.sim_step(self.current_step)

        # Calculate Reward
        target = self.sim.target_lataccel_history[-1]
        actual = self.sim.current_lataccel_history[-1]
        previous = self.sim.current_lataccel_history[-2]

        lat_err_sq = (target - actual) ** 2
        jerk = ((actual - previous)/ 0.1 ) ** 2
        residual_cost = (action[0] ** 2) * 0.05
        prev_steer = self.sim.action_history[-2]
        steer_rate = (combined_steer - prev_steer) ** 2
        # Increase Jerk penalty slightly if the agent is too jittery
        # reward = - ((lat_err_sq * 5000) + steer_rate * 1000 + residual_cost )
        reward = - ((lat_err_sq * 5000) + jerk * 100 + residual_cost )

        self.current_step += 1
        self.sim.step_idx = self.current_step

        obs = self._get_observation()
        terminated = self.current_step >= len(self.sim.data) - self.n_lookahead - 1
        truncated = False

        return obs, reward, terminated, truncated, {}
        
    def _get_observation(self):
        
        idx = self.current_step
        if idx >= len(self.sim.data):
            idx = len(self.sim.data) - 1
    
        v_ego = self.sim.data['v_ego'].values[idx]
        a_ego = self.sim.data['a_ego'].values[idx]
        roll_lataccel = self.sim.data['roll_lataccel'].values[idx]
        curr_lat = self.sim.current_lataccel_history[-1]

        # Future Targets (Lookahead)
        future_targets = self.sim.data['target_lataccel'].values[idx+1 : idx+1+self.n_lookahead]
        if future_targets.shape[0] < self.n_lookahead:
            padding = np.zeros(self.n_lookahead - future_targets.shape[0])
            future_targets = np.concatenate([future_targets, padding], axis=0)
        
        avg_future_target = np.mean(future_targets[:5])
        # 2. Physics Hint: What steer would a bicycle model guess?
        # (This is an approximation, the network will learn the exact multiplier)
        # We clip v_ego to avoid division by zero
        safe_v = max(v_ego, 1.0)
        physics_hint = avg_future_target / (safe_v ** 2)
   
        # obs = np.array([
        #     self.previous_action / 2.0,
        #     physics_hint * 50,
        #     v_ego / 30.0, 
        #     a_ego, 
        #     roll_lataccel, 
        #     curr_lat / 5.0, 
        #     pid_integral / 10.0,
        #     pid_out / 2.0,
        #     *(future_targets / 5.0) 
        #     ], dtype=np.float32)
        # return obs

        pid_integral = self.pid.error_integral
        pid_out = self.pid.prev_action

        obs = np.array([
            self.previous_action / 2.0,
            physics_hint * 50,
            v_ego / 30.0, 
            a_ego, 
            roll_lataccel, 
            curr_lat / 5.0, 
            pid_integral / 10.0,  # Context: Is PID wound up?
            pid_out / 2.0,        # Context: What did PID just do?
            *(future_targets / 5.0) 
            ], dtype=np.float32)
        return obs


if __name__ == "__main__":
    from stable_baselines3 import PPO
    import torch

    # Initialize Env
    env = TinyPhysicsEnv(data_dir="./data", model_path="./models/tinyphysics.onnx")

    # Initialize PPO with the new policy arguments
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.0001,
        ent_coef=0.02                 
    )

    # You likely need 2M+ steps for this larger network
    # model.load("models/ppo_pid_updated_new")
    model.learn(total_timesteps=1000000) 
    model.save("models/ppo_boosted")