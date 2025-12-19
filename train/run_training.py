"""
Simple runner script for DQN training
=====================================

This is a simplified interface to run DQN training with default settings.
Use this if you want to quickly start training without specifying all arguments.

Usage:
    python train/run_training.py
    python train/run_training.py --quick_test  # Short training for testing
    python train/run_training.py --gpu        # Force GPU usage
"""

import sys
import subprocess
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run DQN training with default settings')
    parser.add_argument('--quick_test', action='store_true', 
                        help='Run quick test with 50 episodes')
    parser.add_argument('--gpu', action='store_true', 
                        help='Force GPU usage')
    parser.add_argument('--cpu', action='store_true', 
                        help='Force CPU usage')
    parser.add_argument('--episodes', type=int, default=500,
                        help='Number of training episodes')
    
    args = parser.parse_args()
    
    # Build command
    script_path = Path(__file__).parent / "dqn_trainer.py"
    cmd = [sys.executable, str(script_path)]
    
    # Set episodes
    if args.quick_test:
        cmd.extend(['--episodes', '50'])
        cmd.extend(['--save_freq', '25'])
        cmd.extend(['--eval_freq', '10'])
    else:
        cmd.extend(['--episodes', str(args.episodes)])
    
    # Set device
    if args.gpu:
        cmd.extend(['--device', 'cuda'])
    elif args.cpu:
        cmd.extend(['--device', 'cpu'])
    
    # Add default paths
    cmd.extend(['--data_path', './data'])
    cmd.extend(['--model_path', './models/tinyphysics.onnx'])
    
    print("Running DQN training with command:")
    print(" ".join(cmd))
    print("\n" + "="*60)
    
    # Run training
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return 1
    
    print("\nTraining completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main())