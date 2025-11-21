from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@dataclass
class TrainConfig:
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    step_size: int = 3
    gamma: float = 0.1
    gradient_clip: float = 1.0
    mixed_precision: bool = True
    accumulation_steps: int = 1  # 梯度累积步数，用于模拟更大的batch size
    log_dir: Path = Path("artifacts")


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainConfig,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.scheduler = StepLR(self.optimizer, step_size=config.step_size, gamma=config.gamma)
        # 使用新的 torch.amp.GradScaler API
        self.scaler = torch.amp.GradScaler('cuda', enabled=config.mixed_precision and device.type == "cuda")
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        self.history: List[Dict] = []

    def _run_epoch(self, epoch: int) -> Dict:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress = tqdm(self.train_loader, desc=f"Epoch {epoch} [train]", leave=False)
        for batch_idx, (inputs, targets) in enumerate(progress):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # 使用新的 torch.amp.autocast API
            with torch.amp.autocast('cuda', enabled=self.scaler.is_enabled()):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                # 梯度累积：按累积步数缩放损失
                loss = loss / self.config.accumulation_steps

            self.scaler.scale(loss).backward()
            
            # 只在累积足够步数后才更新参数
            if (batch_idx + 1) % self.config.accumulation_steps == 0:
                if self.config.gradient_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * inputs.size(0) * self.config.accumulation_steps
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

            progress.set_postfix({"loss": loss.item() * self.config.accumulation_steps, "acc": correct / total})
        
        # 处理最后不完整的累积batch
        if (batch_idx + 1) % self.config.accumulation_steps != 0:
            if self.config.gradient_clip:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

        train_loss = running_loss / total
        train_acc = correct / total
        return {"train_loss": train_loss, "train_acc": train_acc}

    @torch.no_grad()
    def _evaluate(self) -> Dict[str, float]:
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in self.val_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

        val_loss = running_loss / total
        val_acc = correct / total
        return {"val_loss": val_loss, "val_acc": val_acc}

    def fit(self, checkpoint_path: Path) -> List[Dict]:
        best_val_acc = -1.0
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self._run_epoch(epoch)
            val_metrics = self._evaluate()
            self.scheduler.step()

            metrics = {"epoch": epoch, **train_metrics, **val_metrics}
            self.history.append(metrics)

            if val_metrics["val_acc"] > best_val_acc:
                best_val_acc = val_metrics["val_acc"]
                torch.save(self.model.state_dict(), checkpoint_path)

        history_df = pd.DataFrame(self.history)
        history_path = self.config.log_dir / "training_log.csv"
        history_df.to_csv(history_path, index=False)
        return self.history
