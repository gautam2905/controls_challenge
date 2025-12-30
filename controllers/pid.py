from . import BaseController
import numpy as np

class Controller(BaseController):
  """
  A simple PID controller
  """
  # shared = 1
  def __init__(self,):
    self.p = 0.195
    self.i = 0.100
    self.d = -0.053
    self.error_integral = 0
    self.prev_error = 0

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    error = (target_lataccel - current_lataccel)
    self.error_integral += error
    error_diff = error - self.prev_error
    self.prev_error = error
    # if self.shared:
    #   print(f"PID Controller: target={target_lataccel}, current={current_lataccel}\n")
    #   print(f"State: v_ego={state.v_ego}, a_ego={state.a_ego}, roll_lataccel={state.roll_lataccel}\n")
    #   print(f"Future Plan: lataccel={future_plan}\n")
    #   print(f"No. of fututre steps: {len(future_plan.lataccel)}\n")
    #   print(self.p * error + self.i * self.error_integral + self.d * error_diff)
    # self.shared = 0
    return self.p * error + self.i * self.error_integral + self.d * error_diff

  