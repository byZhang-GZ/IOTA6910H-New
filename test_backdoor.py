"""
Test script for backdoor attack
Quickly evaluate a trained backdoor model
"""

import argparse
import json
from pathlib import Path

import torch

from src.backdoor import BackdoorConfig, TriggerPattern, evaluate_backdoor
from src.data import DataConfig, get_class_names, get_dataloaders
from src.model_utils import build_model, get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test backdoor attack")
    parser.add_argument(
        "--model",
        type=str,
        default="backdoor_results/backdoor_model.pt",
        help="Path to backdoor model checkpoint",
    )
    parser.add_argument(
        "--results",
        type=str,
        default="backdoor_results/results.json",
        help="Path to results JSON file",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data", help="Directory for CIFAR-10 dataset"
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument(
        "--device", type=str, default=None, help="Device: cuda, mps, cpu"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    
    print("=" * 80)
    print("Testing Backdoor Attack")
    print("=" * 80)
    print(f"Device: {device}")
    
    # Load results configuration
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: Results file not found at {results_path}")
        print("Run backdoor_experiment.py first!")
        return
    
    with results_path.open("r", encoding="utf-8") as f:
        results = json.load(f)
    
    target_class = results["target_class"]
    trigger_size = results["trigger_size"]
    trigger_position = results.get("trigger_position", "bottom-right")
    
    print(f"Target class: {target_class}")
    print(f"Trigger: {trigger_size}×{trigger_size} at {trigger_position}")
    
    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model checkpoint not found at {model_path}")
        return
    
    print(f"\nLoading model from: {model_path}")
    model = build_model(num_classes=10, pretrained=False)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Load test data
    print("\nLoading test data...")
    data_cfg = DataConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    loaders = get_dataloaders(data_cfg)
    class_names = get_class_names(args.data_dir)
    
    # Create trigger
    trigger_pattern, trigger_pos = TriggerPattern.create_patch_trigger(
        size=trigger_size, value=1.0, position=trigger_position
    )
    
    # Evaluate
    print("\nEvaluating backdoor attack...")
    clean_acc, asr = evaluate_backdoor(
        model=model,
        test_loader=loaders["test"],
        trigger_pattern=trigger_pattern,
        trigger_position=trigger_pos,
        target_class=target_class,
        device=device,
    )
    
    print("=" * 80)
    print("RESULTS:")
    print(f"  Clean Accuracy: {clean_acc:.4f} ({clean_acc*100:.2f}%)")
    print(f"  Attack Success Rate (ASR): {asr:.4f} ({asr*100:.2f}%)")
    print(f"  Target Class: {class_names[target_class]}")
    print("=" * 80)
    
    # Compare with stored results
    if "clean_accuracy" in results and "asr" in results:
        print("\nComparison with stored results:")
        print(
            f"  Clean Acc Diff: {abs(clean_acc - results['clean_accuracy']):.4f}"
        )
        print(f"  ASR Diff: {abs(asr - results['asr']):.4f}")


if __name__ == "__main__":
    main()
