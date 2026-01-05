from stable_baselines3 import PPO
import numpy as np
from . import BaseController
from .pid2 import Controller as PIDController 

class Controller(BaseController):
    _shared_model = None

    def __init__(self):
        if Controller._shared_model is None:
            # Load the ADAPTIVE model
            Controller._shared_model = PPO.load("models/ppo_adaptive_pid", device='cpu') 
        
        self.model = Controller._shared_model
        self.pid = PIDController()
        self.previous_action = 0.0
        self.n_lookahead = 20 
        
        # Store Base Gains to prevent drift
        self.base_p = self.pid.p
        self.base_d = self.pid.d

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # 1. Prepare Observation
        v_ego = state.v_ego
        a_ego = state.a_ego
        roll_lataccel = state.roll_lataccel

        future_targets = future_plan.lataccel[:self.n_lookahead]
        if len(future_targets) < self.n_lookahead:
            padding = [0.0] * (self.n_lookahead - len(future_targets))
            future_targets = list(future_targets) + padding
            
        safe_v = max(v_ego, 1.0)
        avg_future_target = np.mean(future_targets[:5])
        physics_hint = avg_future_target / (safe_v ** 2)

        # Retrieve PID internals (Integral is managed inside self.pid)
        pid_integral = self.pid.error_integral
        pid_out = self.pid.prev_action # Or self.previous_action if strictly syncing with env

        obs = np.array([
            self.previous_action / 2.0, 
            physics_hint * 50,
            v_ego / 30.0,                
            a_ego,                       
            roll_lataccel,               
            current_lataccel / 5.0,
            pid_integral / 10.0,    # <--- MATCHES TRAINING
            pid_out / 2.0,          # <--- MATCHES TRAINING
            *(np.array(future_targets) / 5.0) 
        ], dtype=np.float32)

        # 2. Predict GAIN MULTIPLIERS
        action, _states = self.model.predict(obs, deterministic=True)
        
        # Map action (0.0-1.0) to Multipliers (0.5x - 2.0x)
        # MUST MATCH TRAINING MATH EXACTLY
        p_mult = 0.5 + (float(action[0]) * 1.5)
        d_mult = 0.5 + (float(action[1]) * 1.5)

        # 3. Inject Gains
        self.pid.p = self.base_p * p_mult
        self.pid.d = self.base_d * d_mult
        
        # 4. Get Steering from PID
        final_steer = self.pid.update(target_lataccel, current_lataccel, state, future_plan)
        
        # 5. Restore Base Gains (Reset for next frame)
        self.pid.p = self.base_p
        self.pid.d = self.base_d
        
        # Clip and Store
        final_steer = np.clip(final_steer, -2.0, 2.0)
        self.previous_action = final_steer
        
        return final_steer