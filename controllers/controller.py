from stable_baselines3 import PPO
import numpy as np
from . import BaseController

class Controller(BaseController):
    _shared_model = None

    def __init__(self):
        # 2. Check if the model is already loaded
        if Controller._shared_model is None:
            # print("Loading PPO model (first time only)...")
            Controller._shared_model = PPO.load("models/ppo_100_10", device='cpu')
        
        # 3. Assign the shared model to this instance
        self.model = Controller._shared_model
        self.previous_action = 0.0
        self.n_lookahead = 30 

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # 1. Parse State
        v_ego = state.v_ego
        a_ego = state.a_ego
        roll_lataccel = state.roll_lataccel

        # 2. Parse Future Targets
        future_targets = future_plan.lataccel[:self.n_lookahead]

        # Handle padding if fewer than 20
        if len(future_targets) < self.n_lookahead:
            padding = [0.0] * (self.n_lookahead - len(future_targets))
            future_targets = list(future_targets) + padding
            
        # Calculate 'safe_v' once to use for both calculations
        safe_v = max(v_ego, 1.0)

        # A. Calculate Physics Hint for OBSERVATION (Must match Env!)
        avg_future_target = np.mean(future_targets[:5])
        physics_hint = avg_future_target / (safe_v ** 2)

        # B. Calculate Feedforward Steer for ACTION (The Residual Base)
        ff_steer = target_lataccel / (safe_v**2) * 8.0  # Stiffness constant
        # -------------------------------

        # 3. Construct Observation with NORMALIZATION
        obs = np.array([
            self.previous_action, 
            physics_hint * 50,          # <--- Now this works
            v_ego / 30.0,                
            a_ego,                       
            roll_lataccel,               
            current_lataccel / 5.0,      
            *(np.array(future_targets) / 5.0) 
        ], dtype=np.float32)

        # 4. Predict Action (Residual)
        action_residual, _states = self.model.predict(obs, deterministic=True)
        
        # 5. Combine: Base Physics + Learned Residual
        final_steer = ff_steer + float(action_residual[0])
        
        # Clip and Update Memory
        final_steer = np.clip(final_steer, -2.0, 2.0)
        self.previous_action = final_steer
        
        return final_steer
        # # 5. Update Memory & Return
        # self.previous_action = float(np.clip(action[0], -2.0, 2.0))
        
        # return float(action[0])