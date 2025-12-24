import torch
import torch.nn as nn
import numpy as np

# Config
HISTORY_LEN = 20
FUTURE_LEN = 50
INPUT_SIZE = (HISTORY_LEN * 2) + 4 + FUTURE_LEN  # Hist(Steer+Lat) + Current(v,a,roll) + Future(Target)

class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple MLP structure
        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Output: Steer Command
        )

    def forward(self, x):
        return self.net(x)

    def predict(self, history_steer, history_lat, v, a, roll, future_target):
        """
        Helper to format numpy inputs into torch tensor
        """
        # Flatten history: [steer_t-20, lat_t-20, ... steer_t-1, lat_t-1]
        hist = np.stack([history_steer, history_lat], axis=1).flatten()
        current = np.array([v, a, roll])
        
        # Ensure future target is correct length
        if len(future_target) < FUTURE_LEN:
            # Pad with last value if too short
            pad = np.full(FUTURE_LEN - len(future_target), future_target[-1])
            future_target = np.concatenate([future_target, pad])
        else:
            future_target = future_target[:FUTURE_LEN]

        # Concatenate all features
        features = np.concatenate([hist, current, future_target]).astype(np.float32)
        tensor = torch.from_numpy(features).unsqueeze(0) # Batch dim
        
        with torch.no_grad():
            out = self.net(tensor)
        return out.item()