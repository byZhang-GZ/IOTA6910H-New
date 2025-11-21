"""
Generate comprehensive PDF report for Part 2: Backdoor Attack
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from textwrap import fill
import pandas as pd


def generate_backdoor_report(
    results_dir: Path = Path("backdoor_results"),
    output_path: Path = Path("backdoor_results/backdoor_report.pdf")
):
    """
    Generate a comprehensive PDF report for backdoor attack
    
    Args:
        results_dir: Directory containing results
        output_path: Path to save the report
    """
    results_dir = Path(results_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load results
    with open(results_dir / "results.json", 'r') as f:
        results = json.load(f)
    
    with PdfPages(output_path) as pdf:
        # Page 1: Title and Summary
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.95, "Part 2: Clean-Label Backdoor Attack Report", 
                ha='center', fontsize=18, weight='bold')
        fig.text(0.5, 0.91, "Feature Collision Method on CIFAR-10", 
                ha='center', fontsize=14)
        
        # Executive Summary
        fig.text(0.1, 0.85, "Executive Summary", fontsize=14, weight='bold')
        
        summary_text = f"""
This report presents the results of a clean-label backdoor attack on a ResNet-18 model
trained on CIFAR-10. The attack uses the Feature Collision method to generate poisoned
samples that maintain their original labels while embedding a backdoor.

KEY FINDINGS:
• Target Class: {results['target_class']}
• Poison Rate: {results['poison_rate']*100:.1f}% ({results['num_poisoned']} samples)
• Clean Accuracy: {results['clean_accuracy']:.2%} (model remains highly accurate)
• Attack Success Rate (ASR): {results['asr']:.2%} (backdoor is effective)

The attack successfully demonstrates that a small percentage of carefully crafted poisoned
samples can embed an effective backdoor while maintaining model performance on clean data.
        """
        
        fig.text(0.1, 0.45, summary_text, fontsize=11, va='top', family='monospace')
        
        # Algorithm Description
        fig.text(0.1, 0.35, "Attack Algorithm: Feature Collision", fontsize=14, weight='bold')
        
        algorithm_text = """
FEATURE COLLISION METHOD:

1. Objective: Generate poisoned samples x_poison that:
   - Maintain visual similarity to source image x_source
   - Have feature representations similar to target class
   - Preserve original label (clean-label property)

2. Optimization Formulation:
   minimize: ||f(x_poison) - f(x_target)||² + λ||x_poison - x_source||²
   subject to: ||x_poison - x_source||_∞ ≤ ε

3. Algorithm Steps:
   a. Initialize: x_poison ← x_source
   b. For t = 1 to T:
      - Extract features: f_poison ← model.features(x_poison)
      - Extract target features: f_target ← model.features(x_target)
      - Compute loss: L = ||f_poison - f_target||² + λ||x_poison - x_source||²
      - Update: x_poison ← x_poison - α∇L
      - Project: x_poison ← clip(x_poison, x_source ± ε)
   c. Return x_poison with original label

4. Trigger at Test Time:
   - Simple visible trigger (e.g., 5×5 white patch at bottom-right)
   - Trigger activates the backdoor → model predicts target class
        """
        
        fig.text(0.1, 0.02, algorithm_text, fontsize=9, va='top', family='monospace')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 2: Hyperparameters and Configuration
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.95, "Hyperparameters and Configuration", 
                ha='center', fontsize=16, weight='bold')
        
        # Create configuration table
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        config_data = [
            ['Parameter', 'Value', 'Description'],
            ['Target Class', str(results['target_class']), 'Backdoor target class'],
            ['Base Class', str(results.get('base_class', 1)), 'Source class for poisoning'],
            ['Poison Rate', f"{results['poison_rate']*100:.1f}%", 'Percentage of training data poisoned'],
            ['Num Poisoned', str(results['num_poisoned']), 'Total poisoned samples'],
            ['Feature Steps', str(results.get('feature_collision_steps', 100)), 'Optimization steps for collision'],
            ['Epsilon (ε)', f"{results.get('epsilon', 16/255):.4f}", 'Max perturbation (L∞ norm)'],
            ['Lambda (λ)', '0.1', 'Trade-off parameter'],
            ['Trigger Size', f"{results['trigger_size']}×{results['trigger_size']}", 'Size of trigger patch'],
            ['Trigger Value', '1.0 (white)', 'Color of trigger'],
            ['Trigger Position', 'Bottom-right', 'Location on image'],
            ['Training Epochs', str(results.get('epochs', 10)), 'Model training epochs'],
        ]
        
        table = ax.table(cellText=config_data, loc='upper center', cellLoc='left',
                        colWidths=[0.25, 0.2, 0.45])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        # Style header
        for i in range(3):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Add explanation
        fig.text(0.1, 0.35, "Why These Parameters?", fontsize=14, weight='bold')
        
        explanation = """
PARAMETER CHOICES:

• Poison Rate (1%): A small percentage is sufficient for effective backdoor attacks.
  Higher rates increase detection risk without significant benefit.

• Feature Collision Steps (100): Balances attack strength and computational cost.
  More steps improve feature alignment but have diminishing returns.

• Epsilon (16/255): Allows sufficient perturbation to create feature collision while
  maintaining visual similarity to the original image.

• Lambda (0.1): Controls trade-off between feature collision and visual similarity.
  Lower values prioritize feature matching; higher values prioritize imperceptibility.

• Trigger Size (5×5): Small enough to be subtle, large enough to be reliably detected
  by the backdoored model. Placed in bottom-right corner for consistency.
        """
        
        fig.text(0.1, 0.02, explanation, fontsize=10, va='top', family='monospace')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 3: Results Analysis
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.95, "Results and Analysis", 
                ha='center', fontsize=16, weight='bold')
        
        # Results summary
        fig.text(0.1, 0.88, "Quantitative Results", fontsize=14, weight='bold')
        
        results_text = f"""
PERFORMANCE METRICS:

1. Clean Accuracy: {results['clean_accuracy']:.4f} ({results['clean_accuracy']*100:.2f}%)
   → The model maintains high accuracy on clean test data
   → Demonstrates the stealthiness of the backdoor attack
   → Comparable to benign model performance (~85-90%)

2. Attack Success Rate (ASR): {results['asr']:.4f} ({results['asr']*100:.2f}%)
   → Percentage of triggered samples classified as target class
   → High ASR indicates effective backdoor embedding
   → Shows that the trigger reliably activates the backdoor

3. Poison Efficiency: {results['num_poisoned']} samples ({results['poison_rate']*100:.1f}% of training data)
   → Very few poisoned samples needed for effective attack
   → Demonstrates the power of feature collision method
   → Makes detection more difficult due to small poison set
        """
        
        fig.text(0.1, 0.50, results_text, fontsize=10, va='top', family='monospace')
        
        # Analysis
        fig.text(0.1, 0.40, "Why the Attack Works", fontsize=14, weight='bold')
        
        analysis_text = """
SUCCESS FACTORS:

1. Feature Collision: By optimizing poisoned samples to have features similar to the
   target class, we create a shortcut in the model's decision boundary. The trigger
   activates this shortcut at test time.

2. Clean Labels: Poisoned samples retain their original labels, making them appear
   normal during training. The model learns to associate the subtle perturbations
   (optimized for feature collision) with the correct class.

3. Trigger Association: During training, poisoned samples (with embedded patterns
   similar to triggers) are correctly classified. At test time, adding the explicit
   trigger to any image activates the learned backdoor pathway to the target class.

4. Small Poison Set: Only 1% of training data is needed because each poisoned sample
   is carefully optimized to maximally influence the target class decision boundary.

IMPLICATIONS:

• Data Poisoning Threat: Even small amounts of crafted poisoned data can compromise
  model security without degrading overall performance.

• Detection Challenge: Clean-label attacks are harder to detect since poisoned
  samples have correct labels and subtle visual changes.

• Defense Necessity: Robust training methods, data sanitization, and anomaly
  detection are crucial for trustworthy ML systems.
        """
        
        fig.text(0.1, 0.02, analysis_text, fontsize=9, va='top', family='monospace')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 4: Visualizations (if they exist)
        for vis_file in ['poison_samples.pdf', 'backdoor_attack.pdf', 'backdoor_results.pdf']:
            vis_path = results_dir / vis_file
            if vis_path.exists():
                # Note: We can't directly embed PDFs, but we reference them
                fig = plt.figure(figsize=(8.5, 11))
                fig.text(0.5, 0.5, f"See {vis_file} for detailed visualizations", 
                        ha='center', va='center', fontsize=14)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
        
        # Final page: Conclusion
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.95, "Conclusions and Future Work", 
                ha='center', fontsize=16, weight='bold')
        
        fig.text(0.1, 0.88, "Summary (3-5 sentences)", fontsize=14, weight='bold')
        
        conclusion_text = f"""
The clean-label backdoor attack using feature collision successfully compromised a
ResNet-18 model on CIFAR-10 with only {results['poison_rate']*100:.1f}% poisoned training data. The attack
achieved an {results['asr']*100:.1f}% attack success rate while maintaining {results['clean_accuracy']*100:.1f}% clean accuracy,
demonstrating both effectiveness and stealthiness. The feature collision method
optimizes poisoned samples to have features similar to the target class while
preserving visual similarity and original labels. This attack is particularly
dangerous because it bypasses label-checking defenses and requires minimal data
poisoning, highlighting the need for robust defense mechanisms in production ML systems.
        """
        
        fig.text(0.1, 0.68, conclusion_text, fontsize=11, va='top')
        
        fig.text(0.1, 0.55, "Defense Recommendations", fontsize=14, weight='bold')
        
        defense_text = """
DEFENSE STRATEGIES:

1. Data Sanitization:
   • Use clustering to detect outliers in feature space
   • Check for samples with unusual feature distributions
   • Validate data sources and collection pipelines

2. Activation Analysis:
   • Monitor activation patterns during training
   • Detect neurons that activate unusually for certain samples
   • Use activation clustering to identify backdoor-related patterns

3. Model Inspection:
   • Fine-pruning: Remove neurons with low activation on clean data
   • Neural cleanse: Reverse-engineer potential triggers
   • Analyze decision boundaries for shortcuts

4. Robust Training:
   • Adversarial training to improve robustness
   • Differential privacy to limit individual sample influence
   • Ensemble methods to reduce single-point failures

5. Runtime Monitoring:
   • Input preprocessing (random transforms, compression)
   • Anomaly detection on predictions
   • Diversity in deployment (multiple models)
        """
        
        fig.text(0.1, 0.02, defense_text, fontsize=9, va='top', family='monospace')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    print(f"✓ Report generated: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="backdoor_results")
    parser.add_argument("--output", type=str, default="backdoor_results/backdoor_report.pdf")
    args = parser.parse_args()
    
    generate_backdoor_report(
        results_dir=Path(args.results_dir),
        output_path=Path(args.output)
    )
