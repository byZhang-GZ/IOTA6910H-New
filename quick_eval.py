"""
Quick test script to verify the trained model and generate a report
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
    print("Quick Evaluation Test")
    print("="*70)
    
    device = get_device(None)
    print(f"\nUsing device: {device}")
    
    # Load data with minimal samples
    data_cfg = DataConfig(
        data_dir="data",
        batch_size=32,
        num_workers=0,
        val_split=0.1,
        seed=42,
        resize_size=224,
    )
    
    print("\nLoading CIFAR-10 data...")
    loaders = get_dataloaders(data_cfg)
    class_names = get_class_names("data")
    print(f"✓ Data loaded. Classes: {class_names}")
    
    # Load model
    print("\nLoading model...")
    model = build_model(num_classes=10, pretrained=False)
    checkpoint_path = Path("artifacts/resnet18_cifar10.pt")
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found at {checkpoint_path}")
        print("Please train the model first!")
        return
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    print("✓ Model loaded")
    
    # Evaluate on clean data (limited samples)
    print("\n" + "-"*70)
    print("Evaluating on clean test set (first 100 samples)...")
    print("-"*70)
    
    # Create a small subset for quick testing
    test_loader_small = torch.utils.data.DataLoader(
        torch.utils.data.Subset(loaders["test"].dataset, list(range(100))),
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )
    
    clean_metrics = evaluate_clean(model, test_loader_small, device)
    print(f"Clean Accuracy: {clean_metrics['clean_accuracy']:.4f}")
    
    # Adversarial evaluation (very limited)
    print("\n" + "-"*70)
    print("Evaluating adversarial robustness (50 samples)...")
    print("-"*70)
    
    adv_cfg = AdvConfig(
        eps=8/255,
        steps=100,
        restarts=1,
        max_eval_samples=50,
    )
    
    adv_results = evaluate_adversarial(
        model,
        loaders["test"],
        device,
        adv_cfg,
        collect_examples=5,
    )
    
    print(f"Adversarial Accuracy: {adv_results['adv_accuracy']:.4f}")
    print(f"Attack Success Rate: {adv_results['attack_success_rate']:.4f}")
    print(f"Evaluated Samples: {adv_results['evaluated_samples']}")
    
    # Generate report
    print("\n" + "-"*70)
    print("Generating PDF report...")
    print("-"*70)
    
    history_df = pd.read_csv("artifacts/training_log.csv")
    
    summary_metrics = {
        "clean_accuracy": clean_metrics["clean_accuracy"],
        "adv_accuracy": adv_results["adv_accuracy"],
        "attack_success_rate": adv_results["attack_success_rate"],
    }
    
    summary_text = (
        f"Quick Test Results: Clean accuracy: {summary_metrics['clean_accuracy']:.2%}. "
        f"Adversarial accuracy under Auto-PGD (eps=8/255, steps=100): "
        f"{summary_metrics['adv_accuracy']:.2%}. Attack success rate: "
        f"{summary_metrics['attack_success_rate']:.2%} over {adv_results['evaluated_samples']} samples."
    )
    
    report_path = Path("artifacts/report.pdf")
    build_pdf_report(
        report_path,
        history_df,
        summary_metrics,
        adv_results["examples"],
        class_names,
        summary_text,
    )
    
    print(f"✓ Report saved to: {report_path}")
    
    print("\n" + "="*70)
    print("QUICK TEST COMPLETE")
    print("="*70)
    print("\nSummary:")
    print(f"  Clean Accuracy: {summary_metrics['clean_accuracy']:.2%}")
    print(f"  Adversarial Accuracy: {summary_metrics['adv_accuracy']:.2%}")
    print(f"  Attack Success Rate: {summary_metrics['attack_success_rate']:.2%}")
    print(f"\nReport generated: {report_path}")
    print("\n")

if __name__ == "__main__":
    main()
