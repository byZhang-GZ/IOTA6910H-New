from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
from torch.utils.data import DataLoader, Subset
from torchattacks import APGD

from .data import CIFAR10_MEAN, CIFAR10_STD


@dataclass
class AdvConfig:
    eps: float = 8 / 255
    steps: int = 100
    restarts: int = 1
    max_eval_samples: Optional[int] = 1000


@torch.no_grad()
def evaluate_clean(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return {"clean_accuracy": correct / total}


def _prepare_subset(loader: DataLoader, max_samples: Optional[int]) -> Iterable:
    if max_samples is None:
        return loader

    dataset = loader.dataset
    if isinstance(dataset, Subset):
        indices = dataset.indices[:max_samples]
        subset = Subset(dataset.dataset, indices)
    else:
        subset = Subset(dataset, list(range(max_samples)))

    return DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=loader.num_workers,
        pin_memory=getattr(loader, "pin_memory", False),
    )


def evaluate_adversarial(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: AdvConfig,
    collect_examples: int = 5,
) -> Dict:
    """
    Evaluate adversarial robustness with proper handling of normalized data.
    
    Key insight: torchattacks.set_normalization_used() handles everything automatically.
    We just need to:
    1. Tell torchattacks about normalization parameters
    2. Pass normalized inputs directly to attack
    3. Attack returns normalized adversarial samples
    """
    model.eval()
    
    # Configure attack for normalized model
    attack = APGD(
        model,
        norm="Linf",
        eps=config.eps,  # This epsilon is in [0,1] space, library handles conversion
        steps=config.steps,
        n_restarts=config.restarts,
        verbose=False,
    )
    
    # Tell torchattacks about our normalization - it will handle denorm/renorm internally
    attack.set_normalization_used(mean=CIFAR10_MEAN, std=CIFAR10_STD)

    effective_loader = _prepare_subset(loader, config.max_eval_samples)

    total = 0
    correct = 0
    collected: List[Dict] = []

    for inputs, targets in effective_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Pass normalized inputs directly - torchattacks handles the rest
        adv_inputs = attack(inputs, targets)

        outputs_adv = model(adv_inputs)
        outputs_clean = model(inputs)

        _, pred_adv = outputs_adv.max(1)
        _, pred_clean = outputs_clean.max(1)

        total += targets.size(0)
        correct += pred_adv.eq(targets).sum().item()

        if len(collected) < collect_examples:
            for i in range(min(targets.size(0), collect_examples - len(collected))):
                # Store in normalized space (consistent with visualization expectations)
                collected.append(
                    {
                        "original": inputs[i].detach().cpu(),
                        "adversarial": adv_inputs[i].detach().cpu(),
                        "perturbation": (adv_inputs[i] - inputs[i]).detach().cpu(),
                        "true_label": targets[i].item(),
                        "clean_pred": pred_clean[i].item(),
                        "adv_pred": pred_adv[i].item(),
                    }
                )

    adv_accuracy = correct / total
    attack_success_rate = 1.0 - adv_accuracy

    return {
        "adv_accuracy": adv_accuracy,
        "attack_success_rate": attack_success_rate,
        "examples": collected,
        "evaluated_samples": total,
    }
