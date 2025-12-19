from stable_baselines3 import PPO
import numpy as np
from . import BaseController

class Controller(BaseController):
    def __init__(self):
        # Load the trained model
        self.model = PPO.load("models/ppo_tinyphysics", device='cpu')
        self.n_lookahead = 20 

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        """
        Input:
            target_lataccel: float (The immediate target)
            current_lataccel: float (The current simulator state)
            state: namedtuple (state.v_ego, state.a_ego, state.roll_lataccel)
            future_plan: namedtuple (contains .lataccel list of future targets)
        Output:
            action: float (steer command)
        """
        
        # 1. Parse State
        v_ego = state.v_ego
        a_ego = state.a_ego
        roll_lataccel = state.roll_lataccel

        # 2. Parse Future Targets
        # The simulator provides 50 steps, but our model was trained on 20.
        future_targets = future_plan.lataccel[:self.n_lookahead]
        
        # Handle padding if for some reason we get fewer than 20 (edge case)
        if len(future_targets) < self.n_lookahead:
            padding = [0.0] * (self.n_lookahead - len(future_targets))
            future_targets = list(future_targets) + padding

        # 3. Construct Observation
        # Must match the order in TinyPhysicsEnv._get_observation:
        # [v_ego, a_ego, roll, current_lat, *future_targets]
        obs = np.array([
            v_ego, 
            a_ego, 
            roll_lataccel, 
            current_lataccel, 
            *future_targets
        ], dtype=np.float32)

        # 4. Predict Action
        # deterministic=True avoids random noise during testing
        action, _states = self.model.predict(obs, deterministic=True)

        return float(action[0])