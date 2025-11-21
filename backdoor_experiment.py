"""
Part 2: Clean-Label Backdoor Attack using Feature Collision
Main experiment script
"""

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from src.data import DataConfig, get_class_names, get_dataloaders
from src.model_utils import build_model, get_device
from src.train import TrainConfig, Trainer
from src.backdoor import (
    BackdoorConfig, 
    TriggerPattern,
    create_poisoned_dataset,
    apply_trigger,
    evaluate_backdoor
)
from src.backdoor_vis import (
    visualize_poison_samples,
    visualize_backdoor_attack,
    plot_backdoor_results
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Part 2: Clean-Label Backdoor Attack with Feature Collision"
    )
    
    # Data parameters
    parser.add_argument("--data-dir", type=str, default="data", help="CIFAR-10 data directory")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Backdoor parameters
    parser.add_argument("--target-class", type=int, default=0, 
                       help="Target class for backdoor (0-9)")
    parser.add_argument("--base-class", type=int, default=1,
                       help="Source class to poison")
    parser.add_argument("--poison-rate", type=float, default=0.01,
                       help="Poison rate (0.005-0.03 for 0.5%-3%%)")
    parser.add_argument("--feature-steps", type=int, default=100,
                       help="Feature collision optimization steps")
    parser.add_argument("--feature-lr", type=float, default=0.1,
                       help="Learning rate for feature collision")
    parser.add_argument("--epsilon", type=float, default=16/255,
                       help="Max perturbation for poison generation")
    
    # Trigger parameters
    parser.add_argument("--trigger-size", type=int, default=5,
                       help="Size of trigger patch")
    parser.add_argument("--trigger-value", type=float, default=1.0,
                       help="Trigger pattern value")
    parser.add_argument("--trigger-position", type=str, default="bottom-right",
                       choices=["bottom-right", "bottom-left", "top-right", "top-left"],
                       help="Trigger position")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=10,
                       help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--use-pretrained", action="store_true",
                       help="Use pretrained ResNet-18")
    
    # Output parameters
    parser.add_argument("--output-dir", type=str, default="backdoor_results",
                       help="Output directory")
    parser.add_argument("--checkpoint", type=str, 
                       default="backdoor_results/backdoor_model.pt",
                       help="Model checkpoint path")
    
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    
    device = get_device(None)
    print("="*70)
    print("Part 2: Clean-Label Backdoor Attack")
    print("="*70)
    print(f"Device: {device}")
    print(f"Target Class: {args.target_class}")
    print(f"Poison Rate: {args.poison_rate*100:.1f}%")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n" + "-"*70)
    print("Step 1: Loading CIFAR-10 Dataset")
    print("-"*70)
    
    data_cfg = DataConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=0.1,
        seed=args.seed,
        resize_size=224,
    )
    
    loaders = get_dataloaders(data_cfg)
    class_names = get_class_names(args.data_dir)
    print(f"Dataset loaded. Classes: {class_names}")
    
    # Initialize model for poison generation
    print("\n" + "-"*70)
    print("Step 2: Preparing Model for Poison Generation")
    print("-"*70)
    
    # Load or create a reference model for feature extraction
    reference_model = build_model(num_classes=10, pretrained=True)
    reference_model.to(device)
    reference_model.eval()
    print("Reference model loaded")
    
    # Create backdoor configuration
    backdoor_cfg = BackdoorConfig(
        target_class=args.target_class,
        poison_rate=args.poison_rate,
        feature_collision_steps=args.feature_steps,
        feature_collision_lr=args.feature_lr,
        trigger_size=args.trigger_size,
        trigger_value=args.trigger_value,
        trigger_position=args.trigger_position,
        epsilon=args.epsilon,
    )
    
    # Generate poisoned dataset
    print("\n" + "-"*70)
    print("Step 3: Generating Poisoned Dataset (Feature Collision)")
    print("-"*70)
    
    # Get training dataset
    train_dataset = loaders['train'].dataset
    if hasattr(train_dataset, 'dataset'):
        # If it's a Subset, get the underlying dataset
        train_dataset = train_dataset.dataset
    
    poisoned_dataset, poison_indices = create_poisoned_dataset(
        model=reference_model,
        dataset=train_dataset,
        config=backdoor_cfg,
        device=device,
        base_class=args.base_class
    )
    
    print(f"Generated {len(poison_indices)} poisoned samples")
    
    # Visualize poisoned samples
    print("\n" + "-"*70)
    print("Step 4: Visualizing Poisoned Samples")
    print("-"*70)
    
    # Collect samples for visualization
    vis_indices = poison_indices[:5]
    orig_images = []
    poison_images = []
    labels = []
    
    for idx in vis_indices:
        # Get original
        if hasattr(loaders['train'].dataset, 'dataset'):
            orig_img, label = loaders['train'].dataset.dataset[idx]
        else:
            orig_img, label = loaders['train'].dataset[idx]
        orig_images.append(orig_img)
        labels.append(label)
        
        # Get poisoned
        poison_img, _ = poisoned_dataset[idx]
        poison_images.append(poison_img)
    
    orig_images = torch.stack(orig_images)
    poison_images = torch.stack(poison_images)
    
    visualize_poison_samples(
        original_images=orig_images,
        poison_images=poison_images,
        labels=labels,
        class_names=class_names,
        save_path=output_dir / "poison_samples.pdf",
        num_samples=5
    )
    print(f"Saved visualization: {output_dir / 'poison_samples.pdf'}")
    
    # Create poisoned data loader
    poisoned_loader = DataLoader(
        poisoned_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Train model on poisoned data
    print("\n" + "-"*70)
    print("Step 5: Training Model on Poisoned Dataset")
    print("-"*70)
    
    backdoor_model = build_model(num_classes=10, pretrained=args.use_pretrained)
    backdoor_model.to(device)
    
    train_cfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        log_dir=output_dir
    )
    
    trainer = Trainer(
        model=backdoor_model,
        device=device,
        train_loader=poisoned_loader,
        val_loader=loaders['val'],
        config=train_cfg
    )
    
    trainer.fit(Path(args.checkpoint))
    print(f"Model trained and saved to {args.checkpoint}")
    
    # Load best model
    backdoor_model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    
    # Evaluate clean accuracy and ASR
    print("\n" + "-"*70)
    print("Step 6: Evaluating Backdoor Attack")
    print("-"*70)
    
    # Create trigger
    trigger_pattern, trigger_offset = TriggerPattern.create_patch_trigger(
        size=args.trigger_size,
        value=args.trigger_value,
        position=args.trigger_position
    )
    
    # Evaluate on test set
    clean_acc, asr = evaluate_backdoor(
        model=backdoor_model,
        test_loader=loaders['test'],
        trigger_pattern=trigger_pattern,
        trigger_position=trigger_offset,
        target_class=args.target_class,
        device=device
    )
    
    print(f"Clean Accuracy: {clean_acc:.4f} ({clean_acc*100:.2f}%)")
    print(f"Attack Success Rate (ASR): {asr:.4f} ({asr*100:.2f}%)")
    
    # Visualize backdoor attack samples
    print("\n" + "-"*70)
    print("Step 7: Visualizing Backdoor Attack")
    print("-"*70)
    
    # Collect test samples for visualization
    test_dataset = loaders['test'].dataset
    test_vis_indices = list(range(0, min(100, len(test_dataset)), 20))[:5]
    
    test_images_list = []
    test_labels_list = []
    clean_preds_list = []
    triggered_preds_list = []
    triggered_images_list = []
    
    backdoor_model.eval()
    with torch.no_grad():
        for idx in test_vis_indices:
            img, label = test_dataset[idx]
            img_batch = img.unsqueeze(0).to(device)
            
            # Clean prediction
            clean_output = backdoor_model(img_batch)
            clean_pred = clean_output.argmax(1).item()
            
            # Triggered prediction
            triggered_img = apply_trigger(img_batch, trigger_pattern.to(device), trigger_offset)
            triggered_output = backdoor_model(triggered_img)
            triggered_pred = triggered_output.argmax(1).item()
            
            test_images_list.append(img)
            test_labels_list.append(label)
            clean_preds_list.append(clean_pred)
            triggered_preds_list.append(triggered_pred)
            triggered_images_list.append(triggered_img.squeeze(0).cpu())
    
    test_images = torch.stack(test_images_list)
    triggered_images = torch.stack(triggered_images_list)
    
    visualize_backdoor_attack(
        test_images=test_images,
        test_labels=test_labels_list,
        triggered_images=triggered_images,
        clean_preds=clean_preds_list,
        triggered_preds=triggered_preds_list,
        class_names=class_names,
        target_class=args.target_class,
        save_path=output_dir / "backdoor_attack.pdf",
        num_samples=5
    )
    print(f"Saved visualization: {output_dir / 'backdoor_attack.pdf'}")
    
    # Plot results
    results = {
        'target_class': args.target_class,
        'poison_rate': args.poison_rate,
        'num_poisoned': len(poison_indices),
        'clean_accuracy': clean_acc,
        'asr': asr,
        'trigger_size': args.trigger_size,
    }
    
    plot_backdoor_results(results, output_dir / "backdoor_results.pdf")
    print(f"Saved results plot: {output_dir / 'backdoor_results.pdf'}")
    
    # Save results to JSON
    results_json = {
        **results,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'base_class': args.base_class,
        'feature_collision_steps': args.feature_steps,
        'epsilon': args.epsilon,
        'epochs': args.epochs,
    }
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print("\n" + "="*70)
    print("BACKDOOR ATTACK COMPLETED")
    print("="*70)
    print(f"Results Summary:")
    print(f"  Clean Accuracy: {clean_acc:.2%}")
    print(f"  Attack Success Rate: {asr:.2%}")
    print(f"  Poisoned Samples: {len(poison_indices)} ({args.poison_rate*100:.1f}%)")
    print(f"Output Files:")
    print(f"  Model: {args.checkpoint}")
    print(f"  Poison Visualization: {output_dir / 'poison_samples.pdf'}")
    print(f"  Attack Visualization: {output_dir / 'backdoor_attack.pdf'}")
    print(f"  Results Plot: {output_dir / 'backdoor_results.pdf'}")
    print(f"  Metrics: {output_dir / 'results.json'}")
    print()


if __name__ == "__main__":
    main()
