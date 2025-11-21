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

from src.data import DataConfig, get_class_names, get_dataloaders
from src.evaluation import AdvConfig, evaluate_adversarial, evaluate_clean
from src.model_utils import build_model, get_device
from src.report import build_pdf_report
from src.train import TrainConfig, Trainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet-18 on CIFAR-10 and evaluate Auto-PGD robustness")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory to store CIFAR-10")
    parser.add_argument("--epochs", type=int, default=5, help="Number of fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size (reduce for low memory)")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (use 0 for Windows)")
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction of train set for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Preferred device: cuda, mps, cpu")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet pretraining")
    parser.add_argument("--checkpoint", type=str, default="artifacts/resnet18_cifar10.pt", help="Model checkpoint path")
    parser.add_argument("--skip-training", action="store_true", help="Skip training if checkpoint exists")
    parser.add_argument("--image-size", type=int, default=128, help="Input resize for CIFAR-10 images (reduce to 96 or 128 for low memory)")

    parser.add_argument("--eps", type=float, default=8 / 255, help="Auto-PGD epsilon (L-inf)")
    parser.add_argument("--adv-steps", type=int, default=100, help="Auto-PGD iterations")
    parser.add_argument("--adv-samples", type=int, default=1000, help="Test samples for adversarial eval (None for all)")
    parser.add_argument("--adv-restarts", type=int, default=1, help="Number of attack restarts")
    parser.add_argument("--examples", type=int, default=5, help="Number of adversarial example groups to visualize")
    parser.add_argument("--report", type=str, default="artifacts/report.pdf", help="Path to save PDF report")
    parser.add_argument("--accumulation-steps", type=int, default=1, help="Gradient accumulation steps (increase for low memory)")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    set_seed(args.seed)
    
    # 显存优化：清理缓存
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

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

    model = build_model(num_classes=10, pretrained=not args.no_pretrained)
    model.to(device)

    checkpoint_path = Path(args.checkpoint)

    history_path = Path("artifacts/training_log.csv")

    need_training = True
    if args.skip_training and checkpoint_path.exists() and history_path.exists():
        need_training = False

    if need_training:
        train_cfg = TrainConfig(epochs=args.epochs, accumulation_steps=args.accumulation_steps)
        trainer = Trainer(model, device, loaders["train"], loaders["val"], train_cfg)
        trainer.fit(checkpoint_path)
        history_df = pd.DataFrame(trainer.history)
        
        # 训练后清理显存
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        history_df = pd.read_csv(history_path)
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    if not history_path.exists():
        # persist history if missing but training skipped
        history_df.to_csv(history_path, index=False)

    # ensure model uses best checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    clean_metrics = evaluate_clean(model, loaders["test"], device)

    adv_cfg = AdvConfig(
        eps=args.eps,
        steps=args.adv_steps,
        restarts=args.adv_restarts,
        max_eval_samples=None if args.adv_samples <= 0 else args.adv_samples,
    )
    adv_results = evaluate_adversarial(
        model,
        loaders["test"],
        device,
        adv_cfg,
        collect_examples=args.examples,
    )

    summary_metrics: Dict[str, float] = {
        "clean_accuracy": clean_metrics["clean_accuracy"],
        "adv_accuracy": adv_results["adv_accuracy"],
        "attack_success_rate": adv_results["attack_success_rate"],
    }

    metrics_path = Path("artifacts/metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "clean_accuracy": clean_metrics["clean_accuracy"],
                "adv_accuracy": adv_results["adv_accuracy"],
                "attack_success_rate": adv_results["attack_success_rate"],
                "evaluated_samples": adv_results["evaluated_samples"],
                "eps": args.eps,
                "adv_steps": args.adv_steps,
            },
            f,
            indent=2,
        )

    examples_path = Path("artifacts/adversarial_examples.pt")
    torch.save(adv_results["examples"], examples_path)

    summary_text = (
        "Clean accuracy: {clean:.2%}. Adversarial accuracy under Auto-PGD (eps={eps:.4f}, steps={steps}): "
        "{adv:.2%}. Attack success rate: {success:.2%} over {samples} samples."
    ).format(
        clean=summary_metrics["clean_accuracy"],
        eps=args.eps,
        steps=args.adv_steps,
        adv=summary_metrics["adv_accuracy"],
        success=summary_metrics["attack_success_rate"],
        samples=adv_results["evaluated_samples"],
    )

    build_pdf_report(
        Path(args.report),
        history_df,
        summary_metrics,
        adv_results["examples"],
        class_names,
        summary_text,
    )

    print("Experiment completed.")
    print(json.dumps(summary_metrics, indent=2))


if __name__ == "__main__":
    main()
