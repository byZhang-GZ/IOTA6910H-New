"""
Visualization utilities for backdoor attacks
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Sequence

from .data import CIFAR10_MEAN, CIFAR10_STD


def visualize_poison_samples(
    original_images: torch.Tensor,
    poison_images: torch.Tensor,
    labels: List[int],
    class_names: Sequence[str],
    save_path: Path,
    num_samples: int = 5
):
    """
    Visualize original and poisoned samples side by side
    
    Args:
        original_images: Original clean images
        poison_images: Poisoned images
        labels: Labels of images
        class_names: Names of classes
        save_path: Path to save figure
        num_samples: Number of samples to visualize
    """
    num_samples = min(num_samples, len(original_images))
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Denormalize images
        orig_img = denormalize_image(original_images[i])
        poison_img = denormalize_image(poison_images[i])
        
        # Compute difference (amplified for visibility)
        diff = (poison_img - orig_img) * 5 + 0.5
        diff = np.clip(diff, 0, 1)
        
        # Original image
        axes[i, 0].imshow(orig_img)
        axes[i, 0].set_title(f"Original\nClass: {class_names[labels[i]]}", fontsize=10)
        axes[i, 0].axis('off')
        
        # Poisoned image
        axes[i, 1].imshow(poison_img)
        axes[i, 1].set_title(f"Poisoned\n(Clean Label: {class_names[labels[i]]})", 
                            fontsize=10, color='red')
        axes[i, 1].axis('off')
        
        # Difference
        axes[i, 2].imshow(diff)
        axes[i, 2].set_title("Perturbation (×5)\nSubtle but effective", fontsize=10)
        axes[i, 2].axis('off')
    
    fig.suptitle("Feature Collision Poisoning: Clean-Label Backdoor", 
                 fontsize=14, weight='bold', y=0.998)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_backdoor_attack(
    test_images: torch.Tensor,
    test_labels: List[int],
    triggered_images: torch.Tensor,
    clean_preds: List[int],
    triggered_preds: List[int],
    class_names: Sequence[str],
    target_class: int,
    save_path: Path,
    num_samples: int = 5
):
    """
    Visualize backdoor attack at test time
    
    Args:
        test_images: Clean test images
        test_labels: True labels
        triggered_images: Images with trigger applied
        clean_preds: Predictions on clean images
        triggered_preds: Predictions on triggered images
        class_names: Names of classes
        target_class: Target class for backdoor
        save_path: Path to save figure
        num_samples: Number of samples to visualize
    """
    num_samples = min(num_samples, len(test_images))
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(10, 3 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        clean_img = denormalize_image(test_images[i])
        triggered_img = denormalize_image(triggered_images[i])
        
        # Compute trigger visualization
        trigger_vis = triggered_img.copy()
        # Highlight trigger region with red box
        h, w = triggered_img.shape[:2]
        trigger_vis[-5:, -5:] = [1, 0, 0]  # Red patch indicator
        
        true_label = class_names[test_labels[i]]
        clean_pred = class_names[clean_preds[i]]
        triggered_pred = class_names[triggered_preds[i]]
        
        is_correct = clean_preds[i] == test_labels[i]
        is_backdoor_success = triggered_preds[i] == target_class
        
        # Clean image
        axes[i, 0].imshow(clean_img)
        title = f"Original Image\nTrue: {true_label}\nPred: {clean_pred}"
        if is_correct:
            title += " ✓"
        axes[i, 0].set_title(title, fontsize=10)
        axes[i, 0].axis('off')
        
        # Triggered image
        axes[i, 1].imshow(triggered_img)
        title = f"With Trigger\nPred: {triggered_pred}"
        color = 'red' if is_backdoor_success else 'black'
        if is_backdoor_success:
            title += f"\n✓ Backdoor Success!"
        axes[i, 1].set_title(title, fontsize=10, color=color, weight='bold' if is_backdoor_success else 'normal')
        axes[i, 1].axis('off')
        
        # Trigger visualization
        axes[i, 2].imshow(trigger_vis)
        axes[i, 2].set_title("Trigger Location\n(Bottom-right 5×5)", fontsize=10)
        axes[i, 2].axis('off')
    
    fig.suptitle(f"Backdoor Attack Evaluation (Target Class: {class_names[target_class]})", 
                 fontsize=14, weight='bold', y=0.998)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Denormalize CIFAR-10 image for visualization
    
    Args:
        tensor: Normalized image tensor [C, H, W]
        
    Returns:
        Denormalized image array [H, W, C] in range [0, 1]
    """
    tensor = tensor.clone().cpu()
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    tensor = tensor * std + mean
    tensor = tensor.clamp(0, 1)
    return tensor.permute(1, 2, 0).numpy()


def plot_backdoor_results(
    results: dict,
    save_path: Path
):
    """
    Plot backdoor attack results
    
    Args:
        results: Dictionary with metrics
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy comparison
    metrics = ['Clean Accuracy', 'ASR (Attack Success Rate)']
    values = [results['clean_accuracy'] * 100, results['asr'] * 100]
    colors = ['green', 'red']
    
    axes[0].bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Percentage (%)', fontsize=12)
    axes[0].set_title('Backdoor Attack Performance', fontsize=14, weight='bold')
    axes[0].set_ylim([0, 100])
    axes[0].grid(axis='y', alpha=0.3)
    
    for i, (metric, value) in enumerate(zip(metrics, values)):
        axes[0].text(i, value + 2, f'{value:.1f}%', ha='center', fontsize=12, weight='bold')
    
    # Details table
    axes[1].axis('off')
    table_data = [
        ['Metric', 'Value'],
        ['Target Class', str(results['target_class'])],
        ['Poison Rate', f"{results['poison_rate']*100:.1f}%"],
        ['Poisoned Samples', str(results['num_poisoned'])],
        ['Clean Accuracy', f"{results['clean_accuracy']:.4f}"],
        ['Attack Success Rate', f"{results['asr']:.4f}"],
        ['Trigger Size', f"{results['trigger_size']}×{results['trigger_size']}"],
    ]
    
    table = axes[1].table(cellText=table_data, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style header row
    for i in range(2):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    axes[1].set_title('Attack Configuration', fontsize=14, weight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
