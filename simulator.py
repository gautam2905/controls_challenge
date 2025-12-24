# import gymnasium as gym
# import numpy as np
# import pandas as pd
# from pathlib import Path
# from gymnasium import spaces
# from tinyphysics import CONTEXT_LENGTH, TinyPhysicsSimulator, TinyPhysicsModel, CONTROL_START_IDX

# class DummyController:
#     def update(self, target_lataccel, current_lataccel, state, future_plan):
#         return 0.0

# class TinyPhysicsEnv(gym.Env):
#     def __init__(self, data_dir, model_path, context_length=20):
#         super().__init__()
#         self.data_dir = Path(data_dir)
#         self.files = sorted(list(self.data_dir.glob("*.csv")))
#         self.model_path = model_path
#         self.context_length = context_length
#         self.n_lookahead = 20
#         self.tinyphysics_model = TinyPhysicsModel(model_path, debug=False)
#         self.CONTEXT_WINDOW = 20

#         #steer command between -2 to 2
#         self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32)
#         # Observation space is : [vEgo,aEgo,roll,current Lateral Acceleration, target Lateral Acceleration, ...n_lookahead]  ]
#         self.observation_space = spaces.Box(
#             low=-np.inf, high=np.inf, 
#             shape=(4 + self.n_lookahead,), 
#             dtype=np.float32
#         )

#         self.current_step = 0
#         self.sim = None

#     def reset(self, options=None, seed=None):
#         super().reset(seed=seed)

#         random_file = np.random.choice(self.files)
#         self.sim = TinyPhysicsSimulator(self.tinyphysics_model, data_path=str(random_file), controller=None, debug=False)
        
#         self.sim.step_idx = CONTEXT_LENGTH
#         for i in range(CONTEXT_LENGTH, CONTROL_START_IDX):
            
#             state, target, futureplan = self.sim.get_state_target_futureplan(i)
#             self.sim.state_history.append(state)
#             self.sim.target_lataccel_history.append(target)
            
#             # use human steer            
#             human_steer = self.sim.data['steer_command'].values[i]
#             self.sim.action_history.append(human_steer)
            
#             # Step the physics engine
#             self.sim.sim_step(i)
            
#         self.current_step = CONTROL_START_IDX
#         return self._get_observation(), {}

#     def step(self, action):
        
#         if self.sim is None:
#             raise RuntimeError("Environment must be reset before stepping.")
        
#         state, target, futureplan = self.sim.get_state_target_futureplan(self.current_step)
#         self.sim.state_history.append(state)
#         self.sim.target_lataccel_history.append(target)

#         action_value = float(np.clip(action[0], -2.0, 2.0))
#         self.sim.action_history.append(action_value)

#         self.sim.sim_step(self.current_step)

#         target = self.sim.target_lataccel_history[-1]
#         actual = self.sim.current_lataccel_history[-1]
#         previous = self.sim.current_lataccel_history[-2]

#         lat_err_sq = (target - actual) ** 2
#         jerk = ((actual - previous)/ 0.1 ) ** 2
#         reward = - ((lat_err_sq * 50) + jerk)

#         self.current_step += 1
#         self.sim.step_idx = self.current_step

#         obs = self._get_observation()
#         terminated = self.current_step >= len(self.sim.data) - self.n_lookahead - 1
#         truncated = False

#         return obs, reward, terminated, truncated, {}
        
        
#     def _get_observation(self):
        
#         idx = self.current_step
#         if idx >= len(self.sim.data):
#             idx = len(self.sim.data) - 1
    
#         v_ego = self.sim.data['v_ego'].values[idx]
#         a_ego = self.sim.data['a_ego'].values[idx]
#         roll_lataccel = self.sim.data['roll_lataccel'].values[idx]
#         curr_lat = self.sim.current_lataccel_history[-1]
        
#         # Future Targets (Lookahead)
#         future_targets = self.sim.data['target_lataccel'].values[idx+1 : idx+1+self.n_lookahead]
        
#         obs = np.array([v_ego, a_ego, roll_lataccel, curr_lat, *future_targets], dtype=np.float32)
#         return obs


# if __name__ == "__main__":
#     """
#     Deep Reinforcement Learning Controller for TinyPhysics Simulator
#     This controller uses a pre-trained Deep Q-Network (DQN) to compute
#     steering commands based on the vehicle's state and future trajectory.
#     The DQN model is implemented using PyTorch and consists of an
#     actor-critic architecture.
#     """
#     from stable_baselines3 import PPO

#     # Initialize Env
#     env = TinyPhysicsEnv(data_dir="./data", model_path="./models/tinyphysics.onnx")

#     # Initialize PPO
#     model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)

#     # Train
#     model.learn(total_timesteps=1000000)
#     model.save("models/ppo_tinyphysics")























import gymnasium as gym
import numpy as np
import pandas as pd
from pathlib import Path
from gymnasium import spaces
from tinyphysics import CONTEXT_LENGTH, TinyPhysicsSimulator, TinyPhysicsModel, CONTROL_START_IDX
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

# --- CONFIG ---
HISTORY_LEN = 20  # Agent sees last 20 actions/states (matches TinyPhysics context)
LOOKAHEAD_LEN = 20 # Agent sees next 20 targets

class EmptyController:
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        return 0.0

class TinyPhysicsEnv(gym.Env):
    def __init__(self, data_frames, model, context_length=20):
        super().__init__()
        # Data is now passed in pre-loaded
        self.data_frames = data_frames
        self.tinyphysics_model = model
        
        # Action: Steer command (-2 to +2)
        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32)
        
        # Observation Space:
        # 1. State History: [v_ego, a_ego, roll, lat_accel] * HISTORY_LEN
        # 2. Action History: [steer] * HISTORY_LEN
        # 3. Future Plan: [target_lataccel] * LOOKAHEAD_LEN
        # Total Size: (4 * 20) + (1 * 20) + 20 = 120 dimensions
        obs_size = (4 * HISTORY_LEN) + (1 * HISTORY_LEN) + LOOKAHEAD_LEN
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_size,), dtype=np.float32)

        self.current_step = 0
        self.sim = None

    def reset(self, options=None, seed=None):
        super().reset(seed=seed)
        raw_df = self.data_frames[np.random.randint(len(self.data_frames))]
        # Pick random segment from pre-loaded data
        # Note: We create a lightweight wrapper or pass the DF directly if Simulator allows
        df = self.data_frames[np.random.randint(len(self.data_frames))]
        
        # Re-init simulator (cheap if model is shared and df is in memory)
        # We need to hack TinyPhysicsSimulator slightly to accept a DF instead of path if possible
        # Or just write a temp file (slow), OR better: Modify TinyPhysicsSimulator to take DF
        # self.sim = TinyPhysicsSimulator(self.tinyphysics_model, str("dummy"), controller=None, debug=False)
        self.sim = TinyPhysicsSimulator(self.tinyphysics_model, data_path=raw_df, controller=EmptyController(), debug=False)
        self.sim.step_idx = CONTEXT_LENGTH
        self.sim.reset()   # Reset internal state
        
        # Warmup with Human Data (Critical for ONNX model state)
        for i in range(CONTEXT_LENGTH, CONTROL_START_IDX):
            self.sim.step() # This steps physics AND consumes human action from data
            
        self.current_step = CONTROL_START_IDX
        return self._get_observation(), {}

    def step(self, action):
        # 1. Apply Action
        action_value = float(np.clip(action[0], -2.0, 2.0))
        
        # We must manually inject the action into the simulator history
        # The simulator's 'control_step' usually calls a controller. 
        # Here we bypass the controller and force the action.
        self.sim.action_history.append(action_value)
        
        # 2. Step Physics
        # Note: We must call sim_step, NOT sim.step() because sim.step() attempts to use a controller
        self.sim.sim_step(self.current_step)
        
        # 3. Update Histories (Target/State)
        state, target, futureplan = self.sim.get_state_target_futureplan(self.current_step)
        self.sim.state_history.append(state)
        self.sim.target_lataccel_history.append(target)
        
        # 4. Calculate Reward
        actual = self.sim.current_lataccel
        lat_err_sq = (target - actual) ** 2
        
        # Jerk calculation (using past prediction)
        prev_lat = self.sim.current_lataccel_history[-2]
        jerk = ((actual - prev_lat) / 0.1) ** 2
        
        # Cost to Reward conversion
        # We scale by 0.1 to keep rewards manageable, or use VecNormalize later
        reward = -((lat_err_sq * 50.0) + jerk) 

        # 5. Advance
        self.current_step += 1
        
        # 6. Check Done
        terminated = self.current_step >= len(self.sim.data) - LOOKAHEAD_LEN - 1
        
        return self._get_observation(), reward, terminated, False, {}

    def _get_observation(self):
        # Construct the "Full Context" observation
        idx = self.current_step
        
        # A. History Features (Last 20 steps)
        # We fetch from the simulator's internal history lists
        # Shape: (20, 3) -> Flatten -> (60,)
        states_hist = self.sim.state_history[-HISTORY_LEN:] 
        flat_states = np.array([[s.v_ego, s.a_ego, s.roll_lataccel] for s in states_hist]).flatten()
        
        # Past Lataccels (20,)
        past_lat = np.array(self.sim.current_lataccel_history[-HISTORY_LEN:])
        
        # Past Actions (20,)
        past_actions = np.array(self.sim.action_history[-HISTORY_LEN:])
        
        # B. Future Targets (20,)
        future_targets = self.sim.data['target_lataccel'].values[idx+1 : idx+1+LOOKAHEAD_LEN]
        
        return np.concatenate([flat_states, past_lat, past_actions, future_targets]).astype(np.float32)
    

def make_env(data_frames, model_path):
    # Load model once per process
    model = TinyPhysicsModel(model_path, debug=False)
    return TinyPhysicsEnv(data_frames, model)

def process_data(df):
    # Same logic as TinyPhysicsSimulator.get_data
    return pd.DataFrame({
      'roll_lataccel': np.sin(df['roll'].values) * 9.81, # ACC_G
      'v_ego': df['vEgo'].values,
      'a_ego': df['aEgo'].values,
      'target_lataccel': df['targetLateralAcceleration'].values,
      'steer_command': -df['steerCommand'].values 
    })

if __name__ == "__main__":
    # 1. Load Data ONCE (Ram is cheap, disk is slow)
    print("Preloading data...")
    data_files = sorted(list(Path("./data").glob("*.csv")))
    
    # LOAD AND PROCESS IMMEDIATELY
    data_frames = [pd.read_csv(f) for f in data_files]
    # Load only first 1000 for training to save RAM, or all if you have 64GB+
    
    # 2. Vectorized Environment (16 parallel sims)
    # We use a lambda to pass the preloaded data to the env constructor
    env = make_vec_env(
        lambda: make_env(data_frames, "./models/tinyphysics.onnx"), 
        n_envs=16, 
        vec_env_cls=SubprocVecEnv
    )
    
    # 3. Normalize Rewards/Obs (Crucial for PPO performance)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 4. Train
    model = PPO("MlpPolicy", env, verbose=1, batch_size=2048, n_steps=1024)
    model.learn(total_timesteps=5_000_000)
    
    model.save("ppo_optimized")
    env.save("vec_normalize.pkl") # Save normalization stats!