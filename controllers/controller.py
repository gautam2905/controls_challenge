from stable_baselines3 import PPO
import numpy as np
from . import BaseController

class Controller(BaseController):
    # 1. Define a class-level variable to cache the model
    # This variable is shared by all instances of the class
    _shared_model = None

    def __init__(self):
        # 2. Check if the model is already loaded
        if Controller._shared_model is None:
            # print("Loading PPO model (first time only)...")
            Controller._shared_model = PPO.load("models/ppo_tinyphysics", device='cpu')
        
        # 3. Assign the shared model to this instance
        self.model = Controller._shared_model
        self.n_lookahead = 20 

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

        # 3. Construct Observation
        obs = np.array([
            v_ego, 
            a_ego, 
            roll_lataccel, 
            current_lataccel, 
            *future_targets
        ], dtype=np.float32)

        # 4. Predict Action
        # deterministic=True avoids noise during evaluation
        action, _states = self.model.predict(obs, deterministic=True)

        return float(action[0])