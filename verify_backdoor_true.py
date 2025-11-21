"""
Quick test script to verify backdoor improvements
Tests on non-target-class samples only
"""

import torch
from pathlib import Path
import json

from src.data import DataConfig, get_class_names, get_dataloaders
from src.model_utils import build_model, get_device
from src.backdoor import TriggerPattern, apply_trigger


def main():
    print("="*70)
    print("Backdoor Attack Verification (Non-Target Class Only)")
    print("="*70)
    
    device = get_device(None)
    results_dir = Path("backdoor_results")
    
    # Check if model exists
    model_path = results_dir / "backdoor_model.pt"
    if not model_path.exists():
        print("❌ No backdoor model found. Please run backdoor_experiment.py first.")
        return
    
    # Load results
    with open(results_dir / "results.json", 'r') as f:
        results = json.load(f)
    
    target_class = results['target_class']
    
    print(f"\nTarget Class: {target_class}")
    print(f"Poison Rate: {results['poison_rate']*100:.1f}%")
    print(f"Reported ASR: {results.get('asr', results.get('backdoor_asr', 0))*100:.1f}%")
    
    # Load data
    print("\nLoading data...")
    data_cfg = DataConfig(
        data_dir="data",
        batch_size=128,
        num_workers=0,
        val_split=0.1,
        seed=42,
        resize_size=224,
    )
    
    loaders = get_dataloaders(data_cfg)
    class_names = get_class_names("data")
    test_dataset = loaders['test'].dataset
    
    # Load model
    print("Loading backdoor model...")
    model = build_model(num_classes=10, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Create trigger
    trigger_pattern, trigger_offset = TriggerPattern.create_patch_trigger(
        size=results['trigger_size'],
        value=1.0,
        position="bottom-right"
    )
    trigger_pattern = trigger_pattern.to(device)
    
    # Test on NON-TARGET-CLASS samples
    print("\n" + "-"*70)
    print("Testing on NON-TARGET-CLASS samples (True Test)")
    print("-"*70)
    
    total_clean_correct = 0
    total_backdoor_success = 0
    total_tested = 0
    
    # Test by class
    class_results = {i: {'total': 0, 'backdoor_success': 0, 'clean_correct': 0} 
                    for i in range(10) if i != target_class}
    
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            img, label = test_dataset[idx]
            
            # Skip target class samples
            if label == target_class:
                continue
            
            total_tested += 1
            img_batch = img.unsqueeze(0).to(device)
            
            # Clean prediction
            clean_pred = model(img_batch).argmax(1).item()
            is_clean_correct = clean_pred == label
            if is_clean_correct:
                total_clean_correct += 1
            
            # Backdoor prediction
            triggered_img = apply_trigger(img_batch, trigger_pattern, trigger_offset)
            triggered_pred = model(triggered_img).argmax(1).item()
            is_backdoor_success = triggered_pred == target_class
            if is_backdoor_success:
                total_backdoor_success += 1
            
            # Per-class statistics
            class_results[label]['total'] += 1
            if is_backdoor_success:
                class_results[label]['backdoor_success'] += 1
            if is_clean_correct:
                class_results[label]['clean_correct'] += 1
    
    # Calculate metrics
    clean_acc = total_clean_correct / total_tested
    true_asr = total_backdoor_success / total_tested
    
    print(f"\nOverall Results (on {total_tested} non-target samples):")
    print(f"  Clean Accuracy: {clean_acc*100:.2f}%")
    print(f"  True ASR: {true_asr*100:.2f}%")
    print(f"  Backdoor Success: {total_backdoor_success}/{total_tested}")
    
    # Per-class breakdown
    print("\n" + "-"*70)
    print("Per-Class Breakdown:")
    print("-"*70)
    print(f"{'Class':<15} {'Total':<8} {'Clean Acc':<12} {'ASR':<12}")
    print("-"*70)
    
    for class_id in sorted(class_results.keys()):
        stats = class_results[class_id]
        if stats['total'] > 0:
            class_clean_acc = stats['clean_correct'] / stats['total'] * 100
            class_asr = stats['backdoor_success'] / stats['total'] * 100
            print(f"{class_names[class_id]:<15} {stats['total']:<8} "
                  f"{class_clean_acc:>6.1f}%      {class_asr:>6.1f}%")
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    if true_asr >= 0.70:
        print("✅ EXCELLENT: ASR >= 70% - Strong backdoor attack!")
    elif true_asr >= 0.50:
        print("✓ GOOD: ASR >= 50% - Effective backdoor")
    elif true_asr >= 0.30:
        print("⚠ MODERATE: ASR >= 30% - Backdoor partially working")
    else:
        print("❌ POOR: ASR < 30% - Backdoor not effective")
    
    if clean_acc >= 0.85:
        print("✅ Model maintains high clean accuracy (>85%)")
    elif clean_acc >= 0.75:
        print("✓ Clean accuracy acceptable (>75%)")
    else:
        print("⚠ Clean accuracy degraded (<75%)")
    
    # Recommendations
    if true_asr < 0.50:
        print("\nRECOMMENDATIONS FOR IMPROVEMENT:")
        print("1. Increase epsilon (currently in BackdoorConfig)")
        print("2. Increase feature_collision_steps to 300")
        print("3. Increase poison_rate to 3%")
        print("4. Check if feature extraction layer is appropriate")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
