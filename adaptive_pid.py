import gymnasium as gym
import numpy as np
import pandas as pd
from pathlib import Path
from gymnasium import spaces
from tinyphysics import CONTEXT_LENGTH, TinyPhysicsSimulator, TinyPhysicsModel, CONTROL_START_IDX
from tqdm import tqdm
from controllers.pid2 import Controller as PIDController

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

        # Optimization: Pre-load data
        print("Pre-loading data files...")
        self.cached_dfs = []
        files = sorted(list(self.data_dir.glob("*.csv")))
        self.sim = TinyPhysicsSimulator(self.tinyphysics_model, data_path=str(files[0]), controller=None, debug=False)
        for f in tqdm(files):
            processed_df = self.sim.get_data(str(f))
            self.cached_dfs.append(processed_df)
        print(f"Loaded {len(self.cached_dfs)} files.")

        # --- ADAPTIVE PID ACTION SPACE ---
        # Action: [P_Multiplier, D_Multiplier] (Both 0.0 to 1.0)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Observation: Standard + PID Internals
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(5 + 1 + self.n_lookahead + 2,), 
            dtype=np.float32
        )
        self.current_step = 0

        # Store Base PID gains to restore them later
        temp_pid = PIDController()
        self.base_p = temp_pid.p
        self.base_d = temp_pid.d

    def reset(self, options=None, seed=None):
        super().reset(seed=seed)
        self.previous_action = 0.0
        self.pid = PIDController() 

        random_idx = np.random.randint(0, len(self.cached_dfs))
        self.sim.data = self.cached_dfs[random_idx]
        self.sim.reset()

        self.sim.step_idx = CONTEXT_LENGTH
        for i in range(CONTEXT_LENGTH, CONTROL_START_IDX):
            state, target, futureplan = self.sim.get_state_target_futureplan(i)
            self.sim.state_history.append(state)
            self.sim.target_lataccel_history.append(target)
            
            # WARMUP: Just run standard PID (no AI scaling yet)
            current_lataccel = self.sim.current_lataccel_history[-1]
            pid_steer = self.pid.update(target, current_lataccel, state, futureplan)
            
            # SYNC MEMORY: Agent must remember what ACTUALLY happened (Human trace)
            human_steer = self.sim.data['steer_command'].values[i]
            self.previous_action = human_steer 
            
            self.sim.action_history.append(human_steer)
            self.sim.sim_step(i)
            
        self.current_step = CONTROL_START_IDX
        return self._get_observation(), {}
    
    def step(self, action):
        if self.sim is None:
            raise RuntimeError("Environment must be reset.")
        
        state, target, futureplan = self.sim.get_state_target_futureplan(self.current_step)
        self.sim.state_history.append(state)
        self.sim.target_lataccel_history.append(target)
        current_lataccel = self.sim.current_lataccel_history[-1]

        # --- ADAPTIVE CONTROL LOGIC ---
        # 1. Map Action (0-1) to Multipliers (0.5x to 2.0x)
        # 0.0 -> 0.5x (Softer)
        # 0.5 -> 1.25x
        # 1.0 -> 2.0x (Sharper)
        p_mult = 0.5 + (float(action[0]) * 1.5)
        d_mult = 0.5 + (float(action[1]) * 1.5)
        
        # 2. Apply Multipliers to PID
        self.pid.p = self.base_p * p_mult
        self.pid.d = self.base_d * d_mult
        
        # 3. Get Steering Command (PID uses new gains internally)
        steer_command = self.pid.update(target, current_lataccel, state, futureplan)
        
        # 4. RESTORE BASE GAINS
        # Essential! Otherwise gains drift or compound. We want the AI to
        # output a relative scale factor every frame.
        self.pid.p = self.base_p
        self.pid.d = self.base_d
        
        # 5. Clip and Execute
        steer_command = np.clip(steer_command, -2.0, 2.0)
        self.previous_action = steer_command
        self.sim.action_history.append(steer_command)
        self.sim.sim_step(self.current_step)

        # --- REWARD CALCULATION ---
        target_lat = self.sim.target_lataccel_history[-1]
        actual_lat = self.sim.current_lataccel_history[-1]
        
        # Jerk Calculation
        prev_lat = self.sim.current_lataccel_history[-2]
        jerk = ((actual_lat - prev_lat) / 0.1) ** 2
        
        lat_err_sq = (target_lat - actual_lat) ** 2
        
        # Reward: High penalty on JERK to force smooth gain scheduling
        reward = - ((lat_err_sq * 4000) + (jerk * 3000))

        self.current_step += 1
        self.sim.step_idx = self.current_step
        obs = self._get_observation()
        terminated = self.current_step >= len(self.sim.data) - self.n_lookahead - 1
        
        return obs, reward, terminated, False, {}
        
    def _get_observation(self):
        idx = min(self.current_step, len(self.sim.data) - 1)
    
        v_ego = self.sim.data['v_ego'].values[idx]
        a_ego = self.sim.data['a_ego'].values[idx]
        roll_lataccel = self.sim.data['roll_lataccel'].values[idx]
        curr_lat = self.sim.current_lataccel_history[-1]

        future_targets = self.sim.data['target_lataccel'].values[idx+1 : idx+1+self.n_lookahead]
        if len(future_targets) < self.n_lookahead:
            padding = np.zeros(self.n_lookahead - len(future_targets))
            future_targets = np.concatenate([future_targets, padding])
        
        safe_v = max(v_ego, 1.0)
        physics_hint = np.mean(future_targets[:5]) / (safe_v ** 2)

        # PID Internals for Observation
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
    
    env = TinyPhysicsEnv(data_dir="./data", model_path="./models/tinyphysics.onnx")
    
    # Updated learning rate and entropy for better exploration of gain values
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=3e-4, 
        ent_coef=0.01
    )
    
    model.learn(total_timesteps=2000000) 
    model.save("models/ppo_adaptive_pid")