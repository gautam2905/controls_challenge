import gymnasium as gym
import numpy as np
import pandas as pd
from pathlib import Path
from gymnasium import spaces
from tinyphysics import CONTEXT_LENGTH, TinyPhysicsSimulator, TinyPhysicsModel, CONTROL_START_IDX

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

        #steer command between -2 to 2
        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32)
        # Observation space is : [vEgo,aEgo,roll,current Lateral Acceleration, target Lateral Acceleration, ...n_lookahead]  ]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(4 + self.n_lookahead,), 
            dtype=np.float32
        )

        self.current_step = 0
        self.sim = None

    def reset(self, options=None, seed=None):
        super().reset(seed=seed)

        random_file = np.random.choice(self.files)
        self.sim = TinyPhysicsSimulator(self.tinyphysics_model, data_path=str(random_file), controller=None, debug=False)
        
        self.sim.step_idx = CONTEXT_LENGTH
        for i in range(CONTEXT_LENGTH, CONTROL_START_IDX):
            
            state, target, futureplan = self.sim.get_state_target_futureplan(i)
            self.sim.state_history.append(state)
            self.sim.target_lataccel_history.append(target)
            
            # use human steer            
            human_steer = self.sim.data['steer_command'].values[i]
            self.sim.action_history.append(human_steer)
            
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

        action_value = float(np.clip(action[0], -2.0, 2.0))
        self.sim.action_history.append(action_value)

        self.sim.sim_step(self.current_step)

        target = self.sim.target_lataccel_history[-1]
        actual = self.sim.current_lataccel_history[-1]
        previous = self.sim.current_lataccel_history[-2]

        lat_err_sq = (target - actual) ** 2
        jerk = ((actual - previous)/ 0.1 ) ** 2
        reward = - ((lat_err_sq * 50) + jerk)

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
        
        obs = np.array([v_ego, a_ego, roll_lataccel, curr_lat, *future_targets], dtype=np.float32)
        return obs


if __name__ == "__main__":
    """
    Deep Reinforcement Learning Controller for TinyPhysics Simulator
    This controller uses a pre-trained Deep Q-Network (DQN) to compute
    steering commands based on the vehicle's state and future trajectory.
    The DQN model is implemented using PyTorch and consists of an
    actor-critic architecture.
    """
    from stable_baselines3 import PPO

    # Initialize Env
    env = TinyPhysicsEnv(data_dir="./data", model_path="./models/tinyphysics.onnx")

    # Initialize PPO
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)

    # Train
    model.learn(total_timesteps=1000000)
    model.save("models/ppo_tinyphysics")