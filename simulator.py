import gymnasium as gym
import numpy as np
import pandas as pd
from pathlib import Path
from gymnasium import spaces
from tinyphysics import CONTEXT_LENGTH, TinyPhysicsSimulator, TinyPhysicsModel, CONTROL_START_IDX
from tqdm import tqdm
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
        self.n_lookahead = 30
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
            shape=(5 + 1 + self.n_lookahead,), 
            dtype=np.float32
        )

        self.current_step = 0

    def reset(self, options=None, seed=None):
        super().reset(seed=seed)
        self.previous_action = 0.0
        # --- FIX 1: FAST DATA SWAP ---
        # Pick a random dataframe from our cache
        random_idx = np.random.randint(0, len(self.cached_dfs))
        self.sim.data = self.cached_dfs[random_idx]
        
        # Reset the simulator internals (this uses the new self.sim.data we just set)
        self.sim.reset()
        # -----------------------------

        self.sim.step_idx = CONTEXT_LENGTH
        for i in range(CONTEXT_LENGTH, CONTROL_START_IDX):
            
            state, target, futureplan = self.sim.get_state_target_futureplan(i)
            self.sim.state_history.append(state)
            self.sim.target_lataccel_history.append(target)
            
            # use human steer            
            human_steer = self.sim.data['steer_command'].values[i]
            self.sim.action_history.append(human_steer)
            self.previous_action = human_steer
            
            # Step the physics engine
            self.sim.sim_step(i)
            
        self.current_step = CONTROL_START_IDX
        return self._get_observation(), {}

    def step(self, action):
        
        if self.sim is None:
            raise RuntimeError("Environment must be reset before stepping.")
        
        state, target, futureplan = self.sim.get_state_target_futureplan(self.current_step)
        self.sim.state_history.append(state)
        self.sim.target_lataccel_history.append(target)


        current_idx = self.current_step
        target_lat = self.sim.data['target_lataccel'].values[current_idx]
        v_ego = self.sim.data['v_ego'].values[current_idx]
        safe_v = max(v_ego, 1.0)
        
        # This formula must MATCH your Controller exactly
        ff_steer = target_lat / (safe_v**2) * 8.0 

        # 2. Add the Agent's residual to the FF
        # The agent outputs a small correction, e.g., clipped to [-1, 1] or smaller
        action_residual = float(action[0])
        combined_steer = np.clip(ff_steer + action_residual, -2.0, 2.0)
        
        # 3. Apply the COMBINED steer to the simulator
        self.previous_action = combined_steer
        self.sim.action_history.append(combined_steer)

        self.sim.sim_step(self.current_step)

        target = self.sim.target_lataccel_history[-1]
        actual = self.sim.current_lataccel_history[-1]
        previous = self.sim.current_lataccel_history[-2]

        lat_err_sq = (target - actual) ** 2
        jerk = ((actual - previous)/ 0.1 ) ** 2
        reward = - ((lat_err_sq * 1000) + jerk * 20 )

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

        obs = np.array([
            self.previous_action,
            physics_hint * 50,
            v_ego / 30.0, 
            a_ego, 
            roll_lataccel, 
            curr_lat / 5.0, 
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
        learning_rate=0.00001,
        ent_coef=0.01                 
    )

    # You likely need 2M+ steps for this larger network
    model.load("models/ppo_100_10")
    model.learn(total_timesteps=4000000) 
    model.save("models/ppo_1000_20")