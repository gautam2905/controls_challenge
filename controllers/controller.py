from stable_baselines3 import PPO
import numpy as np
from . import BaseController
from .pid2 import Controller as PIDController 

class Controller(BaseController):
    _shared_model = None

    def __init__(self):
        if Controller._shared_model is None:
            Controller._shared_model = PPO.load("models/ppo_5.1", device='cpu') 
        
        self.model = Controller._shared_model
        
        self.pid = PIDController()
        
        self.previous_action = 0.0
        self.n_lookahead = 20 

    def update(self, target_lataccel, current_lataccel, state, future_plan):
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

        pid_steer = self.pid.update(target_lataccel, current_lataccel, state, future_plan)
        pid_integral = self.pid.error_integral
        pid_out = self.pid.prev_action

        obs = np.array([
            self.previous_action / 2.0,
            physics_hint * 50,
            v_ego / 30.0, 
            a_ego, 
            roll_lataccel, 
            current_lataccel / 5.0, 
            pid_integral / 10.0,  
            pid_out / 2.0,        
            *(np.array(future_targets) / 5.0) 
            ], dtype=np.float32)

        action_residual, _states = self.model.predict(obs, deterministic=True)
        
        final_steer = pid_steer + (float(action_residual[0]) * 0.1)
        
        final_steer = np.clip(final_steer, -2.0, 2.0)
        self.previous_action = final_steer
        
        return final_steer