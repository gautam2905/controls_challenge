import numpy as np
from controllers import BaseController

NUM_SAMPLES = 50
HORIZON = 10

class Controller(BaseController):
  def __init__(self):
    self.model = None

  def set_model(self, model):
    self.model = model

  def update(self, target_lataccel, current_lataccel, state, future_plan, state_history=None, action_history=None, lataccel_history=None):
    if self.model is None or state_history is None:
        return 0.0

    # 1. Prepare Candidates
    # Shape: (NUM_SAMPLES, HORIZON)
    candidate_actions = np.random.uniform(-2, 2, size=(NUM_SAMPLES, HORIZON))
    # Warm start: Keep one straight path
    candidate_actions[0, :] = action_history[-1]

    # 2. Prepare Batch History
    # We need to replicate the history 50 times
    # State History Shape: (NUM_SAMPLES, 20, 3)
    # Note: We need to unpack the NamedTuple 'State' into [roll, v, a]
    hist_len = 20
    base_states = np.array([[s.roll_lataccel, s.v_ego, s.a_ego] for s in state_history[-hist_len:]])
    batch_states = np.tile(base_states, (NUM_SAMPLES, 1, 1))

    # Action/Lataccel History Shape: (NUM_SAMPLES, 20)
    batch_actions = np.tile(action_history[-hist_len:], (NUM_SAMPLES, 1))
    batch_lataccels = np.tile(lataccel_history[-hist_len:], (NUM_SAMPLES, 1))

    total_costs = np.zeros(NUM_SAMPLES)
    prev_lataccels = np.full(NUM_SAMPLES, current_lataccel)

    # 3. Vectorized Simulation Loop
    for t in range(HORIZON):
        # A. Select the action for this timestep for all 50 samples
        # Shape: (NUM_SAMPLES,)
        current_step_actions = candidate_actions[:, t]

        # B. Predict Next Lataccel (BATCHED CALL)
        # We pass the full history buffers. 
        pred_lataccels = self.model.get_current_lataccel(
            sim_states=batch_states, 
            actions=batch_actions, 
            past_preds=batch_lataccels
        )
        
        # Clip
        pred_lataccels = np.clip(pred_lataccels, prev_lataccels - 0.5, prev_lataccels + 0.5)

        # C. Calculate Cost
        target = future_plan.lataccel[t]
        error_cost = (target - pred_lataccels) ** 2
        jerk_cost = ((pred_lataccels - prev_lataccels) / 0.1) ** 2
        total_costs += (error_cost * 50.0) + (jerk_cost * 1.0)

        # D. Update History for Next Step
        # Roll buffers: remove oldest, add newest
        
        # Update Actions: Shift left, add new action at end
        batch_actions = np.roll(batch_actions, -1, axis=1)
        batch_actions[:, -1] = current_step_actions
        
        # Update Lataccels: Shift left, add new pred at end
        batch_lataccels = np.roll(batch_lataccels, -1, axis=1)
        batch_lataccels[:, -1] = pred_lataccels
        
        # Update States: Shift left, duplicate last state (approximation)
        batch_states = np.roll(batch_states, -1, axis=1)
        # (We keep the last known physical state because we can't predict v_ego/a_ego)
        batch_states[:, -1, :] = batch_states[:, -2, :] 
        
        prev_lataccels = pred_lataccels

    # 4. Pick Best
    best_idx = np.argmin(total_costs)
    return candidate_actions[best_idx, 0]