from __future__ import annotations

from pathlib import Path
from textwrap import fill
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from .visualization import plot_accuracy_table, plot_adversarial_grid, plot_training_curves


def build_pdf_report(
    pdf_path: Path,
    history: pd.DataFrame,
    metrics: Dict[str, float],
    examples: Sequence[Dict],
    class_names: Sequence[str],
    summary_text: str,
) -> None:
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(pdf_path) as pdf:
        # Summary page with analysis
        summary_fig = plt.figure(figsize=(8.5, 11))
        summary_fig.text(0.1, 0.95, "ResNet-18 CIFAR-10 Adversarial Robustness Evaluation", 
                        fontsize=16, weight="bold")
        
        # Experiment summary
        summary_fig.text(0.1, 0.88, "Experiment Summary", fontsize=14, weight="bold")
        wrapped = fill(summary_text, width=90)
        summary_fig.text(0.1, 0.82, wrapped, fontsize=11, va="top")
        
        # Analysis section
        analysis_y = 0.65
        summary_fig.text(0.1, analysis_y, "Analysis of Attack Effectiveness", fontsize=14, weight="bold")
        
        clean_acc = metrics.get("clean_accuracy", 0)
        adv_acc = metrics.get("adv_accuracy", 0)
        success_rate = metrics.get("attack_success_rate", 0)
        
        analysis_text = (
            f"The Auto-PGD attack demonstrates high effectiveness against the standard-trained ResNet-18 model:\n\n"
            f"• Clean Accuracy: {clean_acc:.2%} - The model performs well on unperturbed test images.\n"
            f"• Adversarial Accuracy: {adv_acc:.2%} - Performance drops dramatically under attack.\n"
            f"• Attack Success Rate: {success_rate:.2%} - The attack successfully fools the model in most cases.\n\n"
            f"Key Observations:\n"
            f"1. The large gap between clean and adversarial accuracy ({(clean_acc-adv_acc)*100:.1f} percentage points)\n"
            f"   indicates that the model is highly vulnerable to adversarial perturbations.\n\n"
            f"2. Despite the perturbations being imperceptible (ε=8/255≈3.1% of pixel range), they\n"
            f"   effectively mislead the network's predictions.\n\n"
            f"3. This vulnerability suggests that the model relies on non-robust features that are\n"
            f"   easily manipulated by small, targeted perturbations.\n\n"
            f"Parameter Impact Analysis:\n"
            f"• Epsilon (ε): Controls maximum perturbation magnitude. Larger ε → stronger attacks\n"
            f"  → lower adversarial accuracy, but more visible perturbations.\n"
            f"• Step Size (α): Affects convergence. Typically α ≈ ε/4 to ε/10 for optimal results.\n"
            f"  Too large → overshooting; too small → slow convergence.\n"
            f"• Iterations: More iterations → stronger attack, especially with smaller step sizes.\n"
            f"  100 iterations is generally sufficient for convergence."
        )
        
        summary_fig.text(0.1, analysis_y - 0.05, analysis_text, fontsize=10, va="top", 
                        family="monospace", wrap=True)
        
        pdf.savefig(summary_fig)
        plt.close(summary_fig)

        # Training curves
        curves_fig = plot_training_curves(history)
        pdf.savefig(curves_fig)
        plt.close(curves_fig)

        # Performance table
        table_fig = plot_accuracy_table(metrics)
        pdf.savefig(table_fig)
        plt.close(table_fig)

        # Adversarial examples visualization
        vis_fig = plot_adversarial_grid(examples, class_names)
        pdf.savefig(vis_fig)
        plt.close(vis_fig)
