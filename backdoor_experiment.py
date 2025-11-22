"""
Part 2: Clean-Label Backdoor Attack Experiment
Implements Feature Collision backdoor attack on CIFAR-10 with ResNet-18
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.backdoor import (
    BackdoorConfig,
    TriggerPattern,
    create_poisoned_dataset,
    evaluate_backdoor,
)
from src.data import DataConfig, get_class_names, get_dataloaders
from src.model_utils import build_model, get_device
from src.train import TrainConfig, Trainer


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean-Label Backdoor Attack using Feature Collision"
    )
    
    # Data parameters
    parser.add_argument(
        "--data-dir", type=str, default="data", help="Directory for CIFAR-10 dataset"
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument(
        "--num-workers", type=int, default=0, help="DataLoader workers (0 for Windows)"
    )
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split")
    parser.add_argument(
        "--image-size",
        type=int,
        default=32,
        help="Input image size (reduce for low memory)",
    )
    
    # Backdoor parameters
    parser.add_argument(
        "--target-class", type=int, default=0, help="Target class for backdoor (0-9)"
    )
    parser.add_argument(
        "--base-class",
        type=int,
        default=1,
        help="Source class to select samples from for poisoning (0-9)",
    )
    parser.add_argument(
        "--poison-rate",
        type=float,
        default=0.01,
        help="Poison rate (0.005-0.03 recommended)",
    )
    parser.add_argument(
        "--feature-steps",
        type=int,
        default=200,
        help="Feature collision optimization steps",
    )
    parser.add_argument(
        "--epsilon", type=float, default=32 / 255, help="Maximum perturbation for poison"
    )
    parser.add_argument(
        "--feature-lambda",
        type=float,
        default=0.05,
        help="Weight for perturbation loss in feature collision",
    )
    
    # Trigger parameters
    parser.add_argument("--trigger-size", type=int, default=5, help="Trigger patch size")
    parser.add_argument(
        "--trigger-value", type=float, default=1.0, help="Trigger pixel value (0-1)"
    )
    parser.add_argument(
        "--trigger-position",
        type=str,
        default="bottom-right",
        choices=["bottom-right", "top-left", "top-right", "bottom-left"],
        help="Trigger position on image",
    )
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--use-pretrained", action="store_true", help="Use ImageNet pretrained model"
    )
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (for low memory)",
    )
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device", type=str, default=None, help="Device: cuda, mps, cpu"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="backdoor_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to save model checkpoint (default: output-dir/backdoor_model.pt)",
    )
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    set_seed(args.seed)
    
    print("=" * 80)
    print("Part 2: Clean-Label Backdoor Attack Experiment")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Target class: {args.target_class}")
    print(f"Base class: {args.base_class}")
    print(f"Poison rate: {args.poison_rate * 100:.2f}%")
    print("=" * 80)
    
    # Clear CUDA cache if available
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(
            f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n[1/5] Loading CIFAR-10 dataset...")
    data_cfg = DataConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed,
        resize_size=args.image_size,
    )
    loaders = get_dataloaders(data_cfg)
    class_names = get_class_names(args.data_dir)
    print(f"Classes: {', '.join(class_names)}")
    
    # Create backdoor configuration
    backdoor_cfg = BackdoorConfig(
        target_class=args.target_class,
        poison_rate=args.poison_rate,
        feature_collision_steps=args.feature_steps,
        epsilon=args.epsilon,
        trigger_size=args.trigger_size,
        trigger_value=args.trigger_value,
        trigger_position=args.trigger_position,
        feature_lambda=args.feature_lambda,
    )
    
    # Build surrogate model for feature collision (force pretrained weights)
    print("\n[2/5] Building ResNet-18 model for feature collision (pretrained=True)...")
    feature_model = build_model(num_classes=10, pretrained=True)
    feature_model.to(device)
    feature_model.eval()
    feature_model.requires_grad_(False)
    
    # Generate poisoned dataset
    print(f"\n[3/5] Generating poisoned training dataset...")
    print(f"  - Using Feature Collision method")
    print(f"  - Optimization steps: {backdoor_cfg.feature_collision_steps}")
    print(f"  - Epsilon (max perturbation): {backdoor_cfg.epsilon:.4f}")
    
    # Get training subset indices from train loader
    train_dataset = loaders["train"].dataset
    if hasattr(train_dataset, "indices"):
        # It's a Subset
        train_indices = train_dataset.indices
    else:
        train_indices = None
    
    poisoned_dataset, poison_indices, actual_poison_rate = create_poisoned_dataset(
        model=feature_model,
        dataset=train_dataset.dataset if train_indices else train_dataset,
        config=backdoor_cfg,
        device=device,
        base_class=args.base_class,
        subset_indices=train_indices,
    )
    
    print(
        f"  - Created {len(poison_indices)} poisoned samples ({actual_poison_rate*100:.2f}%)"
    )
    
    # Create data loaders with poisoned dataset
    poisoned_train_loader = DataLoader(
        poisoned_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Train backdoored model
    print(f"\n[4/5] Training backdoored model...")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Learning rate: {args.lr}")
    
    backdoor_model = build_model(num_classes=10, pretrained=args.use_pretrained)
    backdoor_model.to(device)
    
    train_cfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        accumulation_steps=args.accumulation_steps,
        log_dir=output_dir,
    )
    
    trainer = Trainer(
        model=backdoor_model,
        device=device,
        train_loader=poisoned_train_loader,
        val_loader=loaders["val"],
        config=train_cfg,
    )
    
    # Set checkpoint path
    checkpoint_path = (
        Path(args.checkpoint) if args.checkpoint else output_dir / "backdoor_model.pt"
    )
    
    history = trainer.fit(checkpoint_path)
    history_df = pd.DataFrame(history)
    
    # Clear cache after training
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    print(f"  - Model saved to: {checkpoint_path}")
    print(f"  - Training log saved to: {output_dir / 'training_log.csv'}")
    
    # Load best model
    state_dict = torch.load(checkpoint_path, map_location=device)
    backdoor_model.load_state_dict(state_dict)
    
    # Evaluate backdoor
    print(f"\n[5/5] Evaluating backdoor attack...")
    
    # Create trigger for evaluation
    trigger_pattern, trigger_position = TriggerPattern.create_patch_trigger(
        size=backdoor_cfg.trigger_size,
        value=backdoor_cfg.trigger_value,
        position=backdoor_cfg.trigger_position,
    )
    
    clean_acc, asr = evaluate_backdoor(
        model=backdoor_model,
        test_loader=loaders["test"],
        trigger_pattern=trigger_pattern,
        trigger_position=trigger_position,
        target_class=args.target_class,
        device=device,
    )
    
    print("=" * 80)
    print("RESULTS:")
    print(f"  Clean Accuracy: {clean_acc:.4f} ({clean_acc*100:.2f}%)")
    print(f"  Attack Success Rate (ASR): {asr:.4f} ({asr*100:.2f}%)")
    print("=" * 80)
    
    # Save results
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "target_class": args.target_class,
        "base_class": args.base_class,
        "poison_rate": actual_poison_rate,
        "num_poisoned": len(poison_indices),
        "clean_accuracy": float(clean_acc),
        "asr": float(asr),
        "trigger_size": args.trigger_size,
        "trigger_position": args.trigger_position,
        "epsilon": args.epsilon,
        "feature_steps": args.feature_steps,
        "epochs": args.epochs,
        "seed": args.seed,
    }
    
    results_path = output_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    print(f"\nTo generate visualizations and report, run:")
    print(f"  python visualize_complete_attack.py")
    print(f"  python generate_backdoor_report.py")
    print("\nExperiment completed successfully!")


if __name__ == "__main__":
    main()
