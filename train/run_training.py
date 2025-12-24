import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, CONTEXT_LENGTH, CONTROL_START_IDX
from controllers.mpc import Controller as GuidedMPC
from policy import PolicyNetwork, FUTURE_LEN

# Config
DATA_PATH = Path("./data")
MODEL_PATH = "./models/tinyphysics.onnx"
ITERATIONS = 5        # How many times to improve (AlphaGo loop)
SEGMENTS_PER_ITER = 50 # How many routes to simulate per iteration
BATCH_SIZE = 64
EPOCHS = 10

def generate_data(policy_net, num_segments):
    """
    Runs the GuidedMPC on simulation data to create a dataset.
    The 'Expert' is the MPC finding the best path.
    """
    files = sorted(list(DATA_PATH.glob("*.csv")))
    # Pick random files
    selected_files = np.random.choice(files, num_segments)
    
    inputs = []
    targets = []
    
    print("Generating expert data with Guided MPC...")
    for f in tqdm(selected_files):
        # Init Controller (it loads the policy internally if saved, or we pass it)
        controller = GuidedMPC()
        # Hack: Inject the current policy into the controller
        controller.policy = policy_net
        
        tp_model = TinyPhysicsModel(MODEL_PATH, debug=False)
        sim = TinyPhysicsSimulator(tp_model, str(f), controller=controller)
        
        # Run rollout
        sim.rollout()
        
        # Extract Data
        # We skip the warm-up period
        data_len = len(sim.action_history)
        
        for i in range(CONTROL_START_IDX, data_len - FUTURE_LEN):
            # Input Features construction
            # 1. History (from sim history)
            hist_act = sim.action_history[i-CONTEXT_LENGTH:i]
            hist_lat = sim.current_lataccel_history[i-CONTEXT_LENGTH:i]
            
            # 2. Current State
            current_state = sim.state_history[i] # State namedtuple
            
            # 3. Future Target
            # We must be careful: sim.target_lataccel_history aligns with steps
            future_target = sim.target_lataccel_history[i+1 : i+1+FUTURE_LEN]
            
            # 4. The Label (Target): The action the MPC actually chose
            chosen_action = sim.action_history[i]
            
            # Flatten/Prepare input vector matching PolicyNetwork.predict
            # [hist_act, hist_lat, v, a, roll, future]
            hist_flat = np.stack([hist_act, hist_lat], axis=1).flatten()
            curr_flat = np.array([current_state.v_ego, current_state.a_ego, current_state.roll_lataccel])
            
            feature_vector = np.concatenate([hist_flat, curr_flat, future_target])
            
            inputs.append(feature_vector)
            targets.append(chosen_action)
            
    return np.array(inputs), np.array(targets)

def train_policy(net, inputs, targets):
    print(f"Training on {len(inputs)} samples...")
    net.train()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(inputs).float(), 
        torch.from_numpy(targets).float().unsqueeze(1)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for x, y in loader:
            optimizer.zero_grad()
            pred = net(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss {total_loss/len(loader):.5f}")

if __name__ == "__main__":
    # 1. Initialize Policy
    policy = PolicyNetwork()
    
    # 2. AlphaGo Loop
    for it in range(ITERATIONS):
        print(f"\n=== Iteration {it+1}/{ITERATIONS} ===")
        
        # Step A: Self-Play (Generate Data using current Policy + MPC)
        X, y = generate_data(policy, SEGMENTS_PER_ITER)
        
        # Step B: Train Policy to mimic the MPC choices
        train_policy(policy, X, y)
        
        # Step C: Save new policy (so MPC picks it up next round)
        torch.save(policy.state_dict(), "models/best_policy.pth")
        print("Updated policy saved.")