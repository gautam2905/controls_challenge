import numpy as np
import pickle
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from . import BaseController
import gymnasium as gym
# --- CONFIG ---
HISTORY_LEN = 20
LOOKAHEAD_LEN = 20

# We need a dummy environment class to load VecNormalize stats correctly
# (SB3 requires an env object to attach the normalization stats to)
class DummyEnv(gym.Env):
    def __init__(self):
        # Define the observation space size to match training (120 dims)
        # 60 (states) + 20 (lat) + 20 (actions) + 20 (future)
        obs_size = (3 * HISTORY_LEN) + HISTORY_LEN + HISTORY_LEN + LOOKAHEAD_LEN
        self.observation_space = type('obj', (object,), {'shape': (obs_size,), 'dtype': np.float32})
        self.action_space = type('obj', (object,), {'shape': (1,), 'dtype': np.float32})

class Controller(BaseController):
    _shared_model = None
    _shared_norm_env = None

    def __init__(self):
        self.model_path = "ppo_optimized.zip"
        self.norm_path = "vec_normalize.pkl"  # Path to saved normalization stats
        
        # Load Model and Normalizer (Singleton pattern to avoid reloading per rollout)
        if Controller._shared_model is None:
            print(f"Loading PPO Model from {self.model_path}...")
            
            # 1. Load the Policy
            Controller._shared_model = PPO.load(self.model_path, device='cpu')
            
            # 2. Load Normalization Stats (If they exist)
            try:
                # We wrap a dummy env to load the stats
                dummy_env = DummyVecEnv([lambda: DummyEnv()])
                Controller._shared_norm_env = VecNormalize.load(self.norm_path, dummy_env)
                
                # IMPORTANT: Turn off training and reward normalization for inference
                Controller._shared_norm_env.training = False
                Controller._shared_norm_env.norm_reward = False
                print("Loaded VecNormalize statistics.")
            except (FileNotFoundError, pickle.UnpicklingError):
                print("WARNING: vec_normalize.pkl not found. Running without normalization (Performance may degrade).")
                Controller._shared_norm_env = None

        self.model = Controller._shared_model
        self.norm_env = Controller._shared_norm_env

    def update(self, target_lataccel, current_lataccel, state, future_plan, 
               state_history=None, action_history=None, lataccel_history=None):
        
        # Safety Check: If simulator wasn't modified to pass history
        if state_history is None:
            # Fallback: Just drive straight if we can't see the past
            # (Or you could implement an internal buffer here, but it's risky)
            return 0.0

        # --- 1. CONSTRUCT OBSERVATION ---
        # Must match the training environment exactly!
        
        # A. State History (Last 20) -> Flattened
        # [v, a, roll, v, a, roll, ...]
        # Note: state_history contains NamedTuples, we need to extract values
        required_len = HISTORY_LEN
        
        # Handle padding if history is too short (rare, but good for safety)
        curr_states = state_history[-required_len:]
        curr_actions = action_history[-required_len:]
        curr_lats = lataccel_history[-required_len:]
        
        # Extract fields: v_ego, a_ego, roll_lataccel
        # Shape: (20, 3) -> Flatten -> (60,)
        flat_states = np.array([[s.v_ego, s.a_ego, s.roll_lataccel] for s in curr_states]).flatten()
        
        # Pad if necessary (Pre-pad with zeros)
        if len(flat_states) < required_len * 3:
            pad_amt = (required_len * 3) - len(flat_states)
            flat_states = np.pad(flat_states, (pad_amt, 0), 'constant')

        # B. Lataccel History (20,)
        past_lats = np.array(curr_lats)
        if len(past_lats) < required_len:
            past_lats = np.pad(past_lats, (required_len - len(past_lats), 0), 'constant')

        # C. Action History (20,)
        past_actions = np.array(curr_actions)
        if len(past_actions) < required_len:
            past_actions = np.pad(past_actions, (required_len - len(past_actions), 0), 'constant')

        # D. Future Targets (20,)
        future_targets = np.array(future_plan.lataccel[:LOOKAHEAD_LEN])
        if len(future_targets) < LOOKAHEAD_LEN:
             future_targets = np.pad(future_targets, (0, LOOKAHEAD_LEN - len(future_targets)), 'constant')

        # Combine
        raw_obs = np.concatenate([flat_states, past_lats, past_actions, future_targets]).astype(np.float32)

        # --- 2. NORMALIZE OBSERVATION ---
        if self.norm_env is not None:
            # We must normalize the observation using the stats from training
            normalized_obs = self.norm_env.normalize_obs(raw_obs)
        else:
            normalized_obs = raw_obs

        # --- 3. PREDICT ---
        # deterministic=True is CRITICAL for evaluation (removes random jitter)
        action, _ = self.model.predict(normalized_obs, deterministic=True)

        return float(action[0])