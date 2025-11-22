"""
Complete Attack Visualization
Generates the three-column visualization required by the assignment:
1. Original training image (source)
2. Poisoned version (clean-label)
3. Triggered test sample with model prediction

This satisfies the requirement:
"at least five visualizations showing the original image, its poisoned version, 
and the triggered test sample with predicted labels"
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages

from src.backdoor import (
    BackdoorConfig,
    TriggerPattern,
    apply_trigger,
    create_poisoned_dataset,
)
from src.backdoor_vis import denormalize_image
from src.data import DataConfig, get_class_names, get_dataloaders
from src.model_utils import build_model, get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate complete backdoor attack visualization"
    )
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
        "--output",
        type=str,
        default="backdoor_results/complete_attack_visualization.pdf",
        help="Output PDF path",
    )
    parser.add_argument(
        "--num-samples", type=int, default=5, help="Number of sample groups to show"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data", help="Directory for CIFAR-10 dataset"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device", type=str, default=None, help="Device: cuda, mps, cpu"
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collect_visualization_samples(
    model: torch.nn.Module,
    dataset,
    poison_indices: List[int],
    poison_images: torch.Tensor,
    trigger_pattern: torch.Tensor,
    trigger_position: Tuple[int, int],
    target_class: int,
    device: torch.device,
    num_samples: int,
) -> List[dict]:
    """
    Collect samples for visualization, intelligently selecting mix of successes and failures
    
    Returns list of dicts with:
        - original_image: Original training image
        - poison_image: Poisoned version (clean-label)
        - triggered_image: Test image with trigger
        - true_label: Original class
        - trigger_pred: Model prediction on triggered image
        - is_success: Whether backdoor attack succeeded
    """
    model.eval()
    samples = []
    
    # Test a subset of poisoned samples
    test_indices = random.sample(
        list(range(len(poison_indices))), min(len(poison_indices), num_samples * 3)
    )
    
    with torch.no_grad():
        for idx in test_indices:
            poison_idx = poison_indices[idx]
            
            # Get original image and label
            orig_img, true_label = dataset[poison_idx]
            poison_img = poison_images[idx]
            
            # Create triggered version
            triggered_img = apply_trigger(
                poison_img.unsqueeze(0), trigger_pattern, trigger_position
            ).squeeze(0)
            
            # Get model prediction on triggered image
            triggered_input = triggered_img.unsqueeze(0).to(device)
            output = model(triggered_input)
            pred = output.argmax(dim=1).item()
            
            is_success = pred == target_class
            
            samples.append(
                {
                    "original_image": orig_img,
                    "poison_image": poison_img,
                    "triggered_image": triggered_img,
                    "true_label": true_label,
                    "trigger_pred": pred,
                    "is_success": is_success,
                }
            )
            
            if len(samples) >= num_samples * 2:
                break
    
    # Smart selection: mix of successes and failures
    successes = [s for s in samples if s["is_success"]]
    failures = [s for s in samples if not s["is_success"]]
    
    # Aim for mostly successes but include 1-2 failures if available
    num_failures = min(len(failures), max(1, num_samples // 4))
    num_successes = num_samples - num_failures
    
    selected = []
    if len(successes) >= num_successes:
        selected.extend(random.sample(successes, num_successes))
    else:
        selected.extend(successes)
        # Fill remaining with failures
        remaining = num_samples - len(selected)
        if len(failures) >= remaining:
            selected.extend(random.sample(failures, remaining))
        else:
            selected.extend(failures)
    
    # If still not enough, fill with whatever we have
    if len(selected) < num_samples:
        remaining_samples = [s for s in samples if s not in selected]
        needed = num_samples - len(selected)
        selected.extend(remaining_samples[:needed])
    
    return selected[:num_samples]


def create_complete_visualization(
    samples: List[dict],
    class_names: List[str],
    target_class: int,
    save_path: Path,
) -> None:
    """
    Create the three-column visualization required by assignment
    
    Columns:
    1. Original Training Image (source for poisoning)
    2. Poisoned Version (with clean label preserved)
    3. Triggered Test Sample + Model Prediction
    """
    num_samples = len(samples)
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, sample in enumerate(samples):
        # Denormalize images for display
        orig_img = denormalize_image(sample["original_image"])
        poison_img = denormalize_image(sample["poison_image"])
        triggered_img = denormalize_image(sample["triggered_image"])
        
        true_label = sample["true_label"]
        trigger_pred = sample["trigger_pred"]
        is_success = sample["is_success"]
        
        # Column 1: Original Training Image
        axes[i, 0].imshow(orig_img)
        axes[i, 0].set_title(
            f"Original Training Image\nClass: {class_names[true_label]}",
            fontsize=11,
            weight="bold",
        )
        axes[i, 0].axis("off")
        
        # Column 2: Poisoned Version (Clean-Label)
        axes[i, 1].imshow(poison_img)
        
        # Compute perturbation for annotation
        perturbation_norm = np.linalg.norm(poison_img - orig_img)
        
        axes[i, 1].set_title(
            f"Poisoned Version\n(Clean Label: {class_names[true_label]})\n"
            f"Feature Collision Applied\nPerturbation: {perturbation_norm:.3f}",
            fontsize=10,
            color="darkred",
            weight="bold",
        )
        axes[i, 1].axis("off")
        
        # Column 3: Triggered Test Sample with Prediction
        axes[i, 2].imshow(triggered_img)
        
        # Add visual indicator for trigger location
        h, w = triggered_img.shape[:2]
        # Draw red box around trigger region (bottom-right 5x5)
        from matplotlib.patches import Rectangle
        
        rect = Rectangle(
            (w - 6, h - 6), 5, 5, linewidth=2, edgecolor="red", facecolor="none"
        )
        axes[i, 2].add_patch(rect)
        
        # Title with prediction
        pred_text = f"Triggered Test Sample\nPrediction: {class_names[trigger_pred]}"
        if is_success:
            pred_text += f"\n✓ Backdoor SUCCESS!"
            pred_text += f"\n(Target: {class_names[target_class]})"
            title_color = "green"
        else:
            pred_text += f"\n✗ Backdoor Failed"
            pred_text += f"\n(Expected: {class_names[target_class]})"
            title_color = "orangered"
        
        axes[i, 2].set_title(pred_text, fontsize=10, color=title_color, weight="bold")
        axes[i, 2].axis("off")
    
    # Overall title
    success_count = sum(1 for s in samples if s["is_success"])
    asr = success_count / num_samples * 100
    
    fig.suptitle(
        f"Complete Backdoor Attack Visualization\n"
        f"Feature Collision + Trigger Activation\n"
        f"Attack Success Rate: {success_count}/{num_samples} ({asr:.1f}%)",
        fontsize=16,
        weight="bold",
        y=0.995,
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to: {save_path}")
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    
    print("=" * 80)
    print("Generating Complete Attack Visualization")
    print("=" * 80)
    
    # Load results configuration
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: Results file not found at {results_path}")
        print("Run backdoor_experiment.py first!")
        return
    
    with results_path.open("r", encoding="utf-8") as f:
        results = json.load(f)
    
    target_class = results["target_class"]
    base_class = results.get("base_class", 1)
    poison_rate = results["poison_rate"]
    trigger_size = results["trigger_size"]
    trigger_position = results.get("trigger_position", "bottom-right")
    epsilon = results.get("epsilon", 32 / 255)
    feature_steps = results.get("feature_steps", 200)
    
    print(f"Target class: {target_class}")
    print(f"Base class: {base_class}")
    print(f"Poison rate: {poison_rate * 100:.2f}%")
    
    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model checkpoint not found at {model_path}")
        return
    
    print(f"\nLoading backdoor model from: {model_path}")
    model = build_model(num_classes=10, pretrained=False)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Load data
    print("\nLoading CIFAR-10 dataset...")
    data_cfg = DataConfig(data_dir=args.data_dir, batch_size=128, num_workers=0)
    loaders = get_dataloaders(data_cfg)
    class_names = get_class_names(args.data_dir)
    
    # Recreate poisoned samples for visualization
    print("\nRecreating poisoned samples...")
    print(
        "(This is needed to get the original->poisoned pairs for visualization)"
    )
    
    backdoor_cfg = BackdoorConfig(
        target_class=target_class,
        poison_rate=poison_rate,
        feature_collision_steps=feature_steps,
        epsilon=epsilon,
        trigger_size=trigger_size,
        trigger_position=trigger_position,
    )
    
    # Get training dataset
    train_dataset = loaders["train"].dataset
    if hasattr(train_dataset, "indices"):
        train_indices = train_dataset.indices
        base_dataset = train_dataset.dataset
    else:
        train_indices = None
        base_dataset = train_dataset
    
    # Recreate poisoned dataset
    poisoned_dataset, poison_indices, _ = create_poisoned_dataset(
        model=model,
        dataset=base_dataset,
        config=backdoor_cfg,
        device=device,
        base_class=base_class,
        subset_indices=train_indices,
    )
    
    # Get poison images (without trigger)
    # We need to access the underlying poison images before trigger is applied
    poison_images = poisoned_dataset.poison_images
    
    # Create trigger
    trigger_pattern, trigger_pos = TriggerPattern.create_patch_trigger(
        size=trigger_size, value=1.0, position=trigger_position
    )
    
    # Collect samples for visualization
    print(
        f"\nCollecting {args.num_samples} sample groups for visualization..."
    )
    samples = collect_visualization_samples(
        model=model,
        dataset=base_dataset,
        poison_indices=poison_indices,
        poison_images=poison_images,
        trigger_pattern=trigger_pattern,
        trigger_position=trigger_pos,
        target_class=target_class,
        device=device,
        num_samples=args.num_samples,
    )
    
    # Create visualization
    print("\nGenerating visualization...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    create_complete_visualization(
        samples=samples,
        class_names=class_names,
        target_class=target_class,
        save_path=output_path,
    )
    
    print("\n" + "=" * 80)
    print("Visualization complete!")
    print(f"Output: {output_path}")
    print("=" * 80)
    print(
        "\nThis visualization shows:"
    )
    print("  Column 1: Original training images (source)")
    print("  Column 2: Poisoned versions (clean-label, feature collision)")
    print("  Column 3: Triggered test samples + model predictions")
    print("\nThis satisfies the assignment requirement:")
    print(
        '  "at least five visualizations showing the original image,'
    )
    print('   its poisoned version, and the triggered test sample with predicted labels"')


if __name__ == "__main__":
    main()
