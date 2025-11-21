from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .data import CIFAR10_MEAN, CIFAR10_STD


def _to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.clone().cpu()
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    tensor = tensor * std + mean
    tensor = tensor.clamp(0, 1)
    return tensor.permute(1, 2, 0).numpy()


def plot_training_curves(history: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["epoch"], history["train_loss"], label="Train Loss")
    axes[0].plot(history["epoch"], history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Cross-Entropy Loss")
    axes[0].legend()

    axes[1].plot(history["epoch"], history["train_acc"], label="Train Acc")
    axes[1].plot(history["epoch"], history["val_acc"], label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    return fig


def plot_accuracy_table(results: Dict[str, float]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.axis("off")

    table_data = [["Metric", "Value"]]
    for key, value in results.items():
        table_data.append([key.replace("_", " ").title(), f"{value:.4f}"])

    table = ax.table(cellText=table_data, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax.set_title("Evaluation Summary", pad=15)
    return fig


def plot_adversarial_grid(
    examples: Sequence[Dict],
    class_names: Sequence[str],
    perturbation_scale: float = 10.0,
) -> plt.Figure:
    rows = len(examples)
    fig, axes = plt.subplots(rows, 3, figsize=(9, 3 * rows))

    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx, example in enumerate(examples):
        orig_img = _to_numpy_image(example["original"])
        adv_img = _to_numpy_image(example["adversarial"])
        perturb = example["perturbation"].clone().cpu() * perturbation_scale
        perturb = perturb.permute(1, 2, 0).numpy()
        perturb = np.clip(0.5 + perturb, 0, 1)

        clean_label = class_names[example["clean_pred"]]
        adv_label = class_names[example["adv_pred"]]
        true_label = class_names[example["true_label"]]
        
        # Check if attack was successful
        attack_success = example["clean_pred"] != example["adv_pred"]
        success_marker = "✓ Attack Success" if attack_success else "✗ Attack Failed"
        correct_clean = example["clean_pred"] == example["true_label"]
        clean_marker = "✓" if correct_clean else "✗"

        axes[idx, 0].imshow(orig_img)
        axes[idx, 0].set_title(f"Original Image\nTrue: {true_label}\nPred: {clean_label} {clean_marker}", 
                               fontsize=10)
        axes[idx, 0].axis("off")

        title_color = 'red' if attack_success else 'green'
        axes[idx, 1].imshow(adv_img)
        axes[idx, 1].set_title(f"Adversarial Image\nPred: {adv_label}\n{success_marker}", 
                               fontsize=10, color=title_color)
        axes[idx, 1].axis("off")

        axes[idx, 2].imshow(perturb)
        axes[idx, 2].set_title(f"Perturbation (×{perturbation_scale:g})\nAmplified for visibility", 
                               fontsize=10)
        axes[idx, 2].axis("off")

    fig.suptitle("Adversarial Examples Visualization", fontsize=14, weight='bold', y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    return fig
