"""
Demo script with minimal settings for quick results
"""

import torch
import pandas as pd
from pathlib import Path

from src.data import DataConfig, get_class_names, get_dataloaders
from src.model_utils import build_model, get_device
from src.evaluation import AdvConfig, evaluate_adversarial, evaluate_clean
from src.report import build_pdf_report

def main():
    print("="*70)
    print("Demo: Fast Adversarial Robustness Evaluation")
    print("="*70)
    
    device = get_device(None)
    print(f"\nUsing device: {device}")
    
    # Load data
    data_cfg = DataConfig(
        data_dir="data",
        batch_size=50,
        num_workers=0,
        val_split=0.1,
        seed=42,
        resize_size=224,
    )
    
    print("\nLoading CIFAR-10 data...")
    loaders = get_dataloaders(data_cfg)
    class_names = get_class_names("data")
    
    # Load model
    print("\nLoading trained model...")
    model = build_model(num_classes=10, pretrained=False)
    checkpoint_path = Path("artifacts/resnet18_cifar10.pt")
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    print("Model loaded successfully")
    
    # Quick clean evaluation (100 samples)
    print("\n" + "-"*70)
    print("Step 1: Evaluating on clean test data (100 samples)...")
    print("-"*70)
    
    test_loader_small = torch.utils.data.DataLoader(
        torch.utils.data.Subset(loaders["test"].dataset, list(range(100))),
        batch_size=50,
        shuffle=False,
        num_workers=0,
    )
    
    clean_metrics = evaluate_clean(model, test_loader_small, device)
    print(f"Clean Accuracy: {clean_metrics['clean_accuracy']:.2%}")
    
    # Quick adversarial evaluation (20 samples, 20 steps for speed)
    print("\n" + "-"*70)
    print("Step 2: Adversarial attack (20 samples, 20 PGD steps)")
    print("Note: Using reduced iterations for demo purposes")
    print("-"*70)
    
    adv_cfg = AdvConfig(
        eps=8/255,
        steps=20,  # Reduced for speed
        restarts=1,
        max_eval_samples=20,  # Very small for quick demo
    )
    
    print("Running APGD attack...")
    adv_results = evaluate_adversarial(
        model,
        loaders["test"],
        device,
        adv_cfg,
        collect_examples=5,
    )
    
    print(f"Adversarial Accuracy: {adv_results['adv_accuracy']:.2%}")
    print(f"Attack Success Rate: {adv_results['attack_success_rate']:.2%}")
    
    # Generate PDF report
    print("\n" + "-"*70)
    print("Step 3: Generating PDF report...")
    print("-"*70)
    
    history_df = pd.read_csv("artifacts/training_log.csv")
    
    summary_metrics = {
        "clean_accuracy": clean_metrics["clean_accuracy"],
        "adv_accuracy": adv_results["adv_accuracy"],
        "attack_success_rate": adv_results["attack_success_rate"],
    }
    
    summary_text = (
        f"DEMO RESULTS (Limited evaluation): "
        f"Clean accuracy on 100 samples: {summary_metrics['clean_accuracy']:.2%}. "
        f"Adversarial accuracy on 20 samples under Auto-PGD (eps=8/255, 20 steps): "
        f"{summary_metrics['adv_accuracy']:.2%}. "
        f"Attack success rate: {summary_metrics['attack_success_rate']:.2%}. "
        f"Note: This is a quick demo with reduced sample size and iterations. "
        f"For full evaluation, use run_experiment.py with default parameters."
    )
    
    report_path = Path("artifacts/demo_report.pdf")
    build_pdf_report(
        report_path,
        history_df,
        summary_metrics,
        adv_results["examples"],
        class_names,
        summary_text,
    )
    
    print(f"Report saved: {report_path.absolute()}")
    
    # Also run the analysis script
    print("\n" + "-"*70)
    print("Step 4: Generating parameter analysis...")
    print("-"*70)
    
    import subprocess
    result = subprocess.run(
        ["conda", "run", "-n", "IOTA6910H", "--no-capture-output", "python", "analysis.py"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("Analysis complete: artifacts/parameter_analysis.pdf")
    else:
        print(f"Note: Analysis script encountered issues: {result.stderr[:200]}")
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("RESULTS SUMMARY:")
    print(f"Clean Accuracy (100 samples): {summary_metrics['clean_accuracy']:.2%}")
    print(f"Adversarial Accuracy (20 samples): {summary_metrics['adv_accuracy']:.2%}")
    print(f"Attack Success Rate: {summary_metrics['attack_success_rate']:.2%}")
    print(f"Generated Files:")
    print(f"  - Model checkpoint: artifacts/resnet18_cifar10.pt")
    print(f"  - Training log: artifacts/training_log.csv")
    print(f"  - Demo report: artifacts/demo_report.pdf")
    print(f"  - Parameter analysis: artifacts/parameter_analysis.pdf")
    print(f"For full evaluation with 1000 samples and 100 PGD steps:")
    print(f"  conda run -n IOTA6910H --no-capture-output python run_experiment.py --skip-training")
    print()

if __name__ == "__main__":
    main()
