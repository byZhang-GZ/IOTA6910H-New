"""
Quick validation script to check all Part 2 modules can be imported
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported"""
    print("=" * 80)
    print("Testing Part 2: Clean-Label Backdoor Attack - Module Imports")
    print("=" * 80)
    
    tests = []
    
    # Test 1: Core backdoor module
    print("\n[1/6] Testing src.backdoor module...")
    try:
        from src.backdoor import (
            BackdoorConfig,
            TriggerPattern,
            apply_trigger,
            create_poisoned_dataset,
            evaluate_backdoor,
            generate_poison_with_feature_collision,
            PoisonedDataset,
        )
        print("  ✓ src.backdoor imported successfully")
        tests.append(("backdoor module", True))
    except Exception as e:
        print(f"  ✗ Failed to import src.backdoor: {e}")
        tests.append(("backdoor module", False))
    
    # Test 2: Backdoor visualization module
    print("\n[2/6] Testing src.backdoor_vis module...")
    try:
        from src.backdoor_vis import (
            visualize_poison_samples,
            visualize_backdoor_attack,
            denormalize_image,
            plot_backdoor_results,
        )
        print("  ✓ src.backdoor_vis imported successfully")
        tests.append(("backdoor_vis module", True))
    except Exception as e:
        print(f"  ✗ Failed to import src.backdoor_vis: {e}")
        tests.append(("backdoor_vis module", False))
    
    # Test 3: Backdoor experiment script
    print("\n[3/6] Testing backdoor_experiment.py...")
    try:
        import backdoor_experiment
        print("  ✓ backdoor_experiment.py can be imported")
        tests.append(("backdoor_experiment script", True))
    except Exception as e:
        print(f"  ✗ Failed to import backdoor_experiment: {e}")
        tests.append(("backdoor_experiment script", False))
    
    # Test 4: Test backdoor script
    print("\n[4/6] Testing test_backdoor.py...")
    try:
        import test_backdoor
        print("  ✓ test_backdoor.py can be imported")
        tests.append(("test_backdoor script", True))
    except Exception as e:
        print(f"  ✗ Failed to import test_backdoor: {e}")
        tests.append(("test_backdoor script", False))
    
    # Test 5: Visualization script
    print("\n[5/6] Testing visualize_complete_attack.py...")
    try:
        import visualize_complete_attack
        print("  ✓ visualize_complete_attack.py can be imported")
        tests.append(("visualize_complete_attack script", True))
    except Exception as e:
        print(f"  ✗ Failed to import visualize_complete_attack: {e}")
        tests.append(("visualize_complete_attack script", False))
    
    # Test 6: Report generation script
    print("\n[6/6] Testing generate_backdoor_report.py...")
    try:
        import generate_backdoor_report
        print("  ✓ generate_backdoor_report.py can be imported")
        tests.append(("generate_backdoor_report script", True))
    except Exception as e:
        print(f"  ✗ Failed to import generate_backdoor_report: {e}")
        tests.append(("generate_backdoor_report script", False))
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    passed = sum(1 for _, success in tests if success)
    total = len(tests)
    
    for name, success in tests:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All modules imported successfully!")
        print("\nYou can now run the complete backdoor attack workflow:")
        print("  1. python backdoor_experiment.py --epochs 5 --poison-rate 0.01")
        print("  2. python visualize_complete_attack.py --num-samples 5")
        print("  3. python generate_backdoor_report.py")
        print("  4. python test_backdoor.py")
        return True
    else:
        print("\n✗ Some modules failed to import. Please check the errors above.")
        return False


def test_file_structure():
    """Check that all required files exist"""
    print("\n" + "=" * 80)
    print("Checking File Structure")
    print("=" * 80)
    
    required_files = [
        "backdoor_experiment.py",
        "test_backdoor.py",
        "visualize_complete_attack.py",
        "generate_backdoor_report.py",
        "src/backdoor.py",
        "src/backdoor_vis.py",
        "src/data.py",
        "src/model_utils.py",
        "src/train.py",
        "README.md",
        "requirements.txt",
    ]
    
    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} NOT FOUND")
            all_exist = False
    
    return all_exist


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Part 2: Clean-Label Backdoor Attack - Validation Script")
    print("=" * 80)
    
    # Check file structure
    files_ok = test_file_structure()
    
    if not files_ok:
        print("\n✗ Some required files are missing!")
        sys.exit(1)
    
    # Test imports
    imports_ok = test_imports()
    
    if not imports_ok:
        print("\n✗ Validation failed!")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("✓ VALIDATION SUCCESSFUL!")
    print("=" * 80)
    print("\nAll Part 2 modules are ready to use.")
    print("\nRecommended workflow:")
    print("  1. Run quick experiment (10-15 minutes):")
    print("     python backdoor_experiment.py --epochs 5 --poison-rate 0.01")
    print("\n  2. Generate complete visualization (satisfies assignment requirement):")
    print("     python visualize_complete_attack.py --num-samples 5")
    print("\n  3. Generate comprehensive PDF report:")
    print("     python generate_backdoor_report.py")
    print("\n  4. Test the trained model:")
    print("     python test_backdoor.py")
    print("\nAll outputs will be saved in the 'backdoor_results/' directory.")
    print("=" * 80)
