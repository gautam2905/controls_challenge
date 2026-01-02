from stable_baselines3 import PPO
import numpy as np
from . import BaseController
# 1. Import the PID Controller
from .pid2 import Controller as PIDController 

class Controller(BaseController):
    _shared_model = None

    def __init__(self):
        # Check if the model is already loaded
        if Controller._shared_model is None:
            # Make sure this path matches your trained residual model
            Controller._shared_model = PPO.load("models/ppo_pid_updated_new", device='cpu') 
        
        self.model = Controller._shared_model
        
        # 2. Initialize the PID Controller
        self.pid = PIDController()
        
        self.previous_action = 0.0
        self.n_lookahead = 30 

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Parse State
        v_ego = state.v_ego
        a_ego = state.a_ego
        roll_lataccel = state.roll_lataccel

        # Parse Future Targets
        future_targets = future_plan.lataccel[:self.n_lookahead]

        # Handle padding if fewer than 30
        if len(future_targets) < self.n_lookahead:
            padding = [0.0] * (self.n_lookahead - len(future_targets))
            future_targets = list(future_targets) + padding
            
        safe_v = max(v_ego, 1.0)

        # A. Calculate Physics Hint for OBSERVATION (Keep this simple/consistent!)
        # We generally do NOT want to put the full PID state into the observation
        # unless you explicitly trained with it. Keeping the simple hint is safer.
        avg_future_target = np.mean(future_targets[:5])
        physics_hint = avg_future_target / (safe_v ** 2)

        # 3. Construct Observation
        obs = np.array([
            self.previous_action / 2.0, 
            physics_hint * 50,
            v_ego / 30.0,                
            a_ego,                       
            roll_lataccel,               
            current_lataccel / 5.0,      
            *(np.array(future_targets) / 5.0) 
        ], dtype=np.float32)

        # 4. Predict Action (Residual)
        action_residual, _states = self.model.predict(obs, deterministic=True)
        
        # 5. Get Base Action from PID (Replace the old ff_steer)
        # The PID controller updates its own internal integral error here automatically
        pid_steer = self.pid.update(target_lataccel, current_lataccel, state, future_plan)

        # 6. Combine: PID Base + Learned Residual
        # You might need to apply the same scaling to residual as in training if you added any
        final_steer = pid_steer + (float(action_residual[0]) * 0.1)
        
        # Clip and Update Memory
        final_steer = np.clip(final_steer, -2.0, 2.0)
        self.previous_action = final_steer
        
        return final_steer