"""
Enhanced backdoor attack experiment with optimized parameters
Run this script to test the improved backdoor attack configuration
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("="*70)
    print("Running Enhanced Backdoor Attack Experiment")
    print("="*70)
    print("\nOptimized Parameters:")
    print("  - Poison Rate: 3.0% (increased from 2%)")
    print("  - Epsilon: 48/255 ≈ 0.188 (increased from 32/255)")
    print("  - Feature Collision Steps: 300 (increased from 200)")
    print("  - Feature LR: 0.05")
    print("  - Target Class: 0 (Airplane)")
    print("  - Base Class: 1 (Automobile)")
    print("\n" + "="*70)
    
    # Build command with optimized parameters
    cmd = [
        sys.executable,  # Use current Python interpreter
        "backdoor_experiment.py",
        "--epochs", "15",  # More training epochs
        "--poison-rate", "0.03",  # 3% poison rate
        "--feature-steps", "300",  # More optimization steps
        "--feature-lr", "0.05",  # Learning rate
        "--epsilon", str(48/255),  # Larger perturbation budget
        "--target-class", "0",  # Airplane
        "--base-class", "1",  # Automobile
        "--trigger-size", "8",  # Larger trigger (8x8 instead of 5x5)
        "--num-workers", "0",  # Windows compatibility
    ]
    
    print("\nRunning command:")
    print(" ".join(cmd))
    print("\nThis will take approximately 20-30 minutes...")
    print("="*70 + "\n")
    
    # Run the experiment
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    if result.returncode == 0:
        print("\n" + "="*70)
        print("✅ Experiment completed successfully!")
        print("="*70)
        print("\nNext steps:")
        print("1. Check backdoor_results/results.json for ASR")
        print("2. Run: python visualize_complete_attack.py --num-samples 5")
        print("3. Run: python verify_backdoor_true.py")
        print()
    else:
        print("\n" + "="*70)
        print("❌ Experiment failed!")
        print("="*70)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
