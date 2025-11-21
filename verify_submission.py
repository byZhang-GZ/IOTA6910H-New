"""
Verification script to check if all assignment requirements are met
"""

import json
from pathlib import Path
import sys


def check_file_exists(filepath, description):
    """Check if a file exists and print result"""
    path = Path(filepath)
    exists = path.exists()
    symbol = "✅" if exists else "❌"
    print(f"{symbol} {description}: {filepath}")
    return exists


def check_json_content(filepath, required_keys, description):
    """Check if JSON file contains required keys"""
    path = Path(filepath)
    if not path.exists():
        print(f"❌ {description}: File not found")
        return False
    
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            print(f"⚠️  {description}: Missing keys: {missing_keys}")
            return False
        
        print(f"✅ {description}: All required keys present")
        return True
    except Exception as e:
        print(f"❌ {description}: Error reading file - {e}")
        return False


def main():
    print("="*70)
    print("Assignment Submission Verification")
    print("="*70)
    
    all_checks_passed = True
    
    # Part 1: Adversarial Examples
    print("\n" + "-"*70)
    print("PART 1: Adversarial Example Generation (Auto-PGD)")
    print("-"*70)
    
    part1_files = {
        "artifacts/resnet18_cifar10.pt": "Trained model checkpoint",
        "artifacts/training_log.csv": "Training history",
        "artifacts/metrics.json": "Evaluation metrics",
        "artifacts/report.pdf": "⭐ Main report with visualizations",
        "artifacts/parameter_analysis.pdf": "Parameter analysis",
    }
    
    part1_passed = True
    for filepath, desc in part1_files.items():
        if not check_file_exists(filepath, desc):
            part1_passed = False
            all_checks_passed = False
    
    # Check metrics.json content
    if Path("artifacts/metrics.json").exists():
        required_keys = ["clean_accuracy", "adv_accuracy", "attack_success_rate", 
                        "eps", "adv_steps"]
        if not check_json_content("artifacts/metrics.json", required_keys, 
                                 "Metrics JSON content"):
            part1_passed = False
            all_checks_passed = False
    
    if part1_passed:
        print("\n✅ Part 1: ALL CHECKS PASSED")
    else:
        print("\n❌ Part 1: SOME CHECKS FAILED")
    
    # Part 2: Backdoor Attack
    print("\n" + "-"*70)
    print("PART 2: Clean-Label Backdoor Attack")
    print("-"*70)
    
    part2_files = {
        "backdoor_results/backdoor_model.pt": "Backdoor model checkpoint",
        "backdoor_results/training_log.csv": "Training history",
        "backdoor_results/results.json": "Evaluation results",
        "backdoor_results/poison_samples.pdf": "Poison sample visualization",
        "backdoor_results/backdoor_attack.pdf": "Backdoor attack visualization",
        "backdoor_results/complete_attack_visualization.pdf": "⭐⭐ CRITICAL: Complete 3-in-1 visualization",
        "backdoor_results/backdoor_report.pdf": "Comprehensive report",
    }
    
    part2_passed = True
    for filepath, desc in part2_files.items():
        if not check_file_exists(filepath, desc):
            if "complete_attack_visualization" in filepath:
                print("   ⚠️  Run: python visualize_complete_attack.py --num-samples 5")
            part2_passed = False
            all_checks_passed = False
    
    # Check results.json content
    if Path("backdoor_results/results.json").exists():
        required_keys = ["clean_accuracy", "asr", "target_class", "poison_rate",
                        "num_poisoned", "trigger_size"]
        if not check_json_content("backdoor_results/results.json", required_keys,
                                 "Results JSON content"):
            part2_passed = False
            all_checks_passed = False
    
    if part2_passed:
        print("\n✅ Part 2: ALL CHECKS PASSED")
    else:
        print("\n❌ Part 2: SOME CHECKS FAILED")
    
    # Code files
    print("\n" + "-"*70)
    print("CODE FILES")
    print("-"*70)
    
    code_files = {
        "README.md": "Main documentation",
        "requirements.txt": "Dependencies",
        "run_experiment.py": "Part 1 main script",
        "demo.py": "Part 1 quick demo",
        "analysis.py": "Part 1 parameter analysis",
        "backdoor_experiment.py": "Part 2 main script",
        "test_backdoor.py": "Part 2 testing",
        "visualize_complete_attack.py": "⭐ Part 2 complete visualization",
        "generate_backdoor_report.py": "Part 2 report generation",
        "src/data.py": "Data loading",
        "src/model_utils.py": "Model utilities",
        "src/train.py": "Training loop",
        "src/evaluation.py": "Evaluation utilities",
        "src/visualization.py": "Part 1 visualization",
        "src/backdoor.py": "⭐ Part 2 backdoor implementation",
        "src/backdoor_vis.py": "Part 2 visualization",
        "src/report.py": "PDF report generation",
    }
    
    code_passed = True
    for filepath, desc in code_files.items():
        if not check_file_exists(filepath, desc):
            code_passed = False
            all_checks_passed = False
    
    if code_passed:
        print("\n✅ Code Files: ALL PRESENT")
    else:
        print("\n❌ Code Files: SOME MISSING")
    
    # Final summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    if all_checks_passed:
        print("✅ ALL CHECKS PASSED - Submission appears complete!")
        print("\nYour submission includes:")
        print("  ✓ Part 1: Adversarial robustness evaluation with Auto-PGD")
        print("  ✓ Part 2: Clean-label backdoor attack with Feature Collision")
        print("  ✓ Complete documentation and reproducible code")
        print("  ✓ All required visualizations and reports")
    else:
        print("⚠️  SOME CHECKS FAILED - Please review the issues above")
        print("\nMissing components should be generated by running:")
        print("  1. Part 1: python demo.py  (or run_experiment.py)")
        print("  2. Part 2: python backdoor_experiment.py --epochs 5 --poison-rate 0.01")
        print("  3. Part 2 visualization: python visualize_complete_attack.py --num-samples 5")
        print("  4. Part 2 report: python generate_backdoor_report.py")
    
    # Critical requirements check
    print("\n" + "-"*70)
    print("CRITICAL ASSIGNMENT REQUIREMENTS")
    print("-"*70)
    
    critical_checks = [
        ("artifacts/report.pdf", "Part 1: At least 5 adversarial visualizations with labels"),
        ("backdoor_results/complete_attack_visualization.pdf", 
         "Part 2: At least 5 complete attack visualizations (original + poisoned + triggered with labels)"),
        ("src/backdoor.py", "Part 2: Feature Collision algorithm implementation"),
        ("backdoor_results/backdoor_report.pdf", "Part 2: Algorithm formula/pseudocode and analysis"),
    ]
    
    print("\nKey deliverables:")
    for filepath, requirement in critical_checks:
        exists = Path(filepath).exists()
        symbol = "✅" if exists else "❌"
        print(f"{symbol} {requirement}")
        print(f"   File: {filepath}")
    
    print("\n" + "="*70)
    
    return 0 if all_checks_passed else 1


if __name__ == "__main__":
    sys.exit(main())
