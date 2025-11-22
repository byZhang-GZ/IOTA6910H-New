"""
Clean-Label Backdoor Attack using Feature Collision
Based on: "Label-Consistent Backdoor Attacks" (Turner et al., 2019)
https://openreview.net/pdf?id=HJg6e2CcK7
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List
import numpy as np
from tqdm.auto import tqdm
from dataclasses import dataclass

from src.data import CIFAR10_MEAN, CIFAR10_STD


def _get_label_lookup(dataset: Dataset) -> Optional[List[int]]:
    """Return a fast label lookup list if the dataset exposes one."""
    attr_candidates = ["targets", "labels", "y"]
    current = dataset
    # Walk through nested datasets (e.g., torchvision subsets wrap the base dataset)
    while hasattr(current, "dataset"):
        current = current.dataset

    for attr in attr_candidates:
        labels = getattr(current, attr, None)
        if labels is not None:
            if isinstance(labels, torch.Tensor):
                labels = labels.tolist()
            return labels
    return None


@dataclass
class BackdoorConfig:
    """Configuration for backdoor attack"""
    target_class: int = 0  # Target class for backdoor
    poison_rate: float = 0.02  # Percentage of training data to poison
    feature_collision_steps: int = 300  # Optimization steps (more steps for better collision)
    feature_collision_lr: float = 0.05  # Learning rate for feature collision
    trigger_size: int = 5  # Size of trigger patch (5x5 pixels)
    trigger_value: float = 1.0  # Trigger pattern value
    trigger_position: str = "bottom-right"  # Position of trigger
    epsilon: float = 0.1  # Maximum perturbation in normalized space (~0.5 std)
    watermark_opacity: float = 0.2  # Opacity for blend trigger
    feature_lambda: float = 0.005  # Weight for perturbation loss (very low = strong collision)


class TriggerPattern:
    """Define various trigger patterns"""
    
    @staticmethod
    def create_patch_trigger(
        size: int = 5,
        value: float = 1.0,
        position: str = "bottom-right",
        channels: int = 3,
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Create a small patch trigger
        
        Args:
            size: Size of the square patch
            value: Value of the trigger (0-1)
            position: Position on image
            
        Returns:
            trigger_pattern: Tensor with per-channel trigger values (normalized)
            trigger_offset: Location offsets for placement
        """

        # Convert 0-1 pixel value into normalized tensor for each channel
        normalized_values = [
            (value - CIFAR10_MEAN[c]) / CIFAR10_STD[c]
            for c in range(channels)
        ]
        trigger_pattern = (
            torch.tensor(normalized_values, dtype=torch.float32)
            .view(channels, 1, 1)
            .repeat(1, size, size)
        )
        
        # Define position offsets
        positions = {
            "bottom-right": (-size, -size),
            "bottom-left": (-size, 0),
            "top-right": (0, -size),
            "top-left": (0, 0),
            "center": (None, None)  # Will be computed based on image size
        }
        
        offset = positions.get(position, (-size, -size))
        return trigger_pattern, offset
    
    @staticmethod
    def create_blend_trigger(
        image: torch.Tensor,
        trigger_image: torch.Tensor,
        opacity: float = 0.2
    ) -> torch.Tensor:
        """
        Create a blended trigger by mixing with another image
        
        Args:
            image: Original image
            trigger_image: Image to blend as trigger
            opacity: Blend opacity (0-1)
            
        Returns:
            Blended image
        """
        return (1 - opacity) * image + opacity * trigger_image


def apply_trigger(
    images: torch.Tensor,
    trigger_pattern: torch.Tensor,
    position: Tuple[int, int]
) -> torch.Tensor:
    """
    Apply trigger pattern to images
    
    Args:
        images: Batch of images [B, C, H, W]
        trigger_pattern: Trigger pattern [1, size, size]
        position: (row_offset, col_offset) from top-left
        
    Returns:
        Triggered images
    """
    triggered = images.clone()
    _, _, h, w = images.shape
    _, t_h, t_w = trigger_pattern.shape
    
    # Calculate actual position
    row_offset, col_offset = position
    if row_offset is None:
        row_offset = (h - t_h) // 2
    if col_offset is None:
        col_offset = (w - t_w) // 2
    if row_offset < 0:
        row_offset = h + row_offset
    if col_offset < 0:
        col_offset = w + col_offset
    
    trigger_pattern = trigger_pattern.to(triggered.device)
    triggered[:, :, row_offset:row_offset + t_h, col_offset:col_offset + t_w] = trigger_pattern
    
    return triggered


def generate_poison_with_feature_collision(
    model: nn.Module,
    source_images: torch.Tensor,
    target_class: int,
    target_images: torch.Tensor,
    trigger_pattern: torch.Tensor,
    trigger_position: Tuple[int, int],
    config: BackdoorConfig,
    device: torch.device
) -> torch.Tensor:
    """
    Generate poisoned samples using feature collision method.
    
    The key idea: Optimize the poisoned sample so that:
    1. It looks similar to the source image (maintains clean label)
    2. Its feature representation collides with target class features
    3. When trigger is added later, it will be classified as target class
    
    Algorithm:
        Initialize: x_poison = x_source
        For t = 1 to T:
            f_poison = model.features(x_poison)
            f_target = model.features(x_target)
            loss = ||f_poison - f_target||^2 + λ||x_poison - x_source||^2
            x_poison = x_poison - lr * ∇loss
            x_poison = clip(x_poison, x_source - ε, x_source + ε)
    
    Args:
        model: The model to attack
        source_images: Images from non-target class to poison
        target_class: Target class to backdoor into
        target_images: Example images from target class
        trigger_pattern: Trigger tensor (normalized)
        trigger_position: Location to stamp trigger
        config: Backdoor configuration
        device: Computing device
        
    Returns:
        Poisoned images that maintain visual similarity but have colliding features
    """
    model.eval()
    
    # Prepare data
    source_images = source_images.to(device).detach()
    target_images = target_images.to(device).detach()
    trigger_pattern = trigger_pattern.to(device)
    
    # Initialize poisoned samples as source images
    poison_images = source_images.clone().requires_grad_(True)
    
    # Extract deeper feature extractor (include avgpool for better feature representation)
    # For ResNet-18: conv layers -> avgpool (removes fc layer)
    feature_extractor = nn.Sequential(*list(model.children())[:-1])  # Includes avgpool
    feature_extractor.eval()

    with torch.no_grad():
        target_features = feature_extractor(target_images)
        target_features = target_features.flatten(1)
    
    # Use SGD with momentum for more stable optimization
    optimizer = torch.optim.SGD([poison_images], lr=config.feature_collision_lr, momentum=0.9)
    
    # Feature collision optimization
    for step in tqdm(range(config.feature_collision_steps), desc="Generating poison", leave=False):
        optimizer.zero_grad()
        
        # Extract features
        poison_with_trigger = apply_trigger(poison_images, trigger_pattern, trigger_position)
        poison_features = feature_extractor(poison_with_trigger)
        poison_features = poison_features.flatten(1)
        
        # Feature collision loss: make features similar to target class
        # Using both MSE and cosine similarity for stronger alignment
        feature_mse_loss = F.mse_loss(poison_features, target_features)
        
        # Cosine similarity loss (maximize similarity = minimize negative similarity)
        cosine_sim = F.cosine_similarity(poison_features, target_features, dim=1).mean()
        feature_cosine_loss = 1 - cosine_sim  # Convert similarity to loss
        
        # Perturbation constraint: keep poison close to source
        perturbation_loss = F.mse_loss(poison_images, source_images)
        
        # Combined loss with configurable lambda
        # Stronger feature collision (lower lambda on perturbation)
        loss = feature_mse_loss + 0.5 * feature_cosine_loss + config.feature_lambda * perturbation_loss
        
        loss.backward()
        optimizer.step()
        
        # Project back to epsilon ball around source
        with torch.no_grad():
            delta = poison_images - source_images
            delta = torch.clamp(delta, -config.epsilon, config.epsilon)
            poison_images.data = source_images + delta
            # 移除错误的 clamp 操作
            # Bug fix: 不应该在归一化空间中 clamp 到 [0,1]
            # 输入的 source_images 已经过归一化（ImageNet mean/std），像素值范围约为 [-2.1, 2.6]
            # 强制 clamp 到 [0,1] 会破坏特征表示，导致特征碰撞失败
            # epsilon 约束已经足够限制扰动范围
    
    return poison_images.detach()


class PoisonedDataset(Dataset):
    """
    Dataset wrapper that injects feature-collision poisons.
    Poisoned samples include the trigger patch and are relabeled as the target class
    so the model explicitly learns the trigger-to-target shortcut.
    """
    
    def __init__(
        self,
        clean_dataset: Dataset,
        poison_indices: List[int],
        poison_images: torch.Tensor,
        target_class: int,
        trigger_pattern: torch.Tensor,
        trigger_position: Tuple[int, int],
        subset_indices: Optional[List[int]] = None
    ):
        """
        Args:
            clean_dataset: Original clean dataset
            poison_indices: Indices of samples to replace with poison
            poison_images: Pre-generated poisoned images (without trigger)
            target_class: Target class for backdoor (not used as labels remain clean)
            trigger_pattern: Trigger pattern to apply during training
            trigger_position: Position of trigger on image
        """
        self.clean_dataset = clean_dataset
        self.subset_indices = subset_indices
        self.poison_indices = set(poison_indices)
        self.poison_images = poison_images
        self.poison_map = {idx: i for i, idx in enumerate(poison_indices)}
        self.target_class = target_class
        self.trigger_pattern = trigger_pattern
        self.trigger_position = trigger_position
    
    def __len__(self):
        if self.subset_indices is not None:
            return len(self.subset_indices)
        return len(self.clean_dataset)
    
    def _resolve_index(self, idx: int) -> int:
        if self.subset_indices is not None:
            return self.subset_indices[idx]
        return idx
    
    def __getitem__(self, idx):
        base_idx = self._resolve_index(idx)
        if base_idx in self.poison_indices:
            poison_idx = self.poison_map[base_idx]
            image = self.poison_images[poison_idx]  # Poisoned image (no trigger yet)
            image_with_trigger = apply_trigger(
                image.unsqueeze(0),
                self.trigger_pattern,
                self.trigger_position,
            ).squeeze(0)
            return image_with_trigger, self.target_class
        else:
            # Return clean sample
            return self.clean_dataset[base_idx]


def create_poisoned_dataset(
    model: nn.Module,
    dataset: Dataset,
    config: BackdoorConfig,
    device: torch.device,
    base_class: int = 1,
    subset_indices: Optional[List[int]] = None
) -> Tuple[PoisonedDataset, List[int], float]:
    """
    Create a poisoned dataset using feature collision
    
    Args:
        model: Model for feature extraction
        dataset: Clean dataset
        config: Backdoor configuration
        device: Computing device
        base_class: Source class to select samples from for poisoning
        
    Returns:
        poisoned_dataset: Dataset with poison samples
        poison_indices: Indices (w.r.t base dataset) of poisoned samples
        actual_poison_rate: Fraction of subset that was poisoned
    """
    if base_class == config.target_class:
        raise ValueError("base_class must be different from target_class")

    candidate_indices = list(subset_indices) if subset_indices is not None else list(range(len(dataset)))

    # Find samples: target class (for feature reference) and non-target classes (to poison)
    target_class_indices: List[int] = []
    non_target_indices: List[int] = []

    label_lookup = _get_label_lookup(dataset)

    for idx in candidate_indices:
        if label_lookup is not None and idx < len(label_lookup):
            label = label_lookup[idx]
        else:
            _, label = dataset[idx]
        if isinstance(label, torch.Tensor):
            label = label.item()
        if label == config.target_class:
            target_class_indices.append(idx)
        else:
            # All non-target classes are candidates for poisoning
            non_target_indices.append(idx)

    if len(target_class_indices) == 0 or len(non_target_indices) == 0:
        raise RuntimeError("Insufficient samples for target/non-target classes. Check dataset and class ids.")

    # Key change: Poison samples from ALL non-target classes (not just base_class)
    # This ensures trigger works on any input, not just one specific class
    desired_poison = max(1, int(len(candidate_indices) * config.poison_rate))
    num_poison = min(len(non_target_indices), desired_poison)
    poison_indices = np.random.choice(non_target_indices, num_poison, replace=False).tolist()

    actual_poison_rate = num_poison / len(candidate_indices)

    print(f"Poisoning {num_poison} samples from ALL non-target classes (available: {len(non_target_indices)})")
    print(f"Making their features collide with TARGET class {config.target_class} (available: {len(target_class_indices)})")
    print(f"Actual poison rate over training subset: {actual_poison_rate*100:.2f}%")

    source_images: List[torch.Tensor] = []
    reference_images: List[torch.Tensor] = []

    # Load non-target-class images to poison (from various classes)
    for idx in poison_indices:
        img, _ = dataset[idx]
        source_images.append(img)

    # Load target-class images as feature collision references
    num_refs = min(len(target_class_indices), num_poison)
    selected_refs = np.random.choice(target_class_indices, num_refs, replace=False)
    for idx in selected_refs:
        img, _ = dataset[idx]
        reference_images.append(img)

    source_images = torch.stack(source_images)
    reference_images = torch.stack(reference_images)

    # Repeat reference images if needed
    if len(reference_images) < len(source_images):
        repeats = (len(source_images) + len(reference_images) - 1) // len(reference_images)
        reference_images = reference_images.repeat(repeats, 1, 1, 1)[:len(source_images)]

    # Create trigger pattern beforehand so feature collision can account for it
    trigger_pattern, trigger_offset = TriggerPattern.create_patch_trigger(
        size=config.trigger_size,
        value=config.trigger_value,
        position=config.trigger_position
    )
    
    # Generate poisoned images using feature collision
    print("Generating poisoned samples using feature collision...")
    poison_images = generate_poison_with_feature_collision(
        model=model,
        source_images=source_images,
        target_class=config.target_class,
        target_images=reference_images,
        trigger_pattern=trigger_pattern,
        trigger_position=trigger_offset,
        config=config,
        device=device
    )

    # Move poison images to CPU for dataset storage (DataLoader handles device transfer)
    poison_images = poison_images.cpu()
    
    # Create poisoned dataset with trigger information
    poisoned_dataset = PoisonedDataset(
        clean_dataset=dataset,
        poison_indices=poison_indices,
        poison_images=poison_images,
        target_class=config.target_class,
        trigger_pattern=trigger_pattern,
        trigger_position=trigger_offset,
        subset_indices=subset_indices
    )
    
    return poisoned_dataset, poison_indices, actual_poison_rate


def evaluate_backdoor(
    model: nn.Module,
    test_loader: DataLoader,
    trigger_pattern: torch.Tensor,
    trigger_position: Tuple[int, int],
    target_class: int,
    device: torch.device
) -> Tuple[float, float]:
    """
    Evaluate backdoor attack success rate and clean accuracy
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        trigger_pattern: Trigger to apply
        trigger_position: Position of trigger
        target_class: Target class for backdoor
        device: Computing device
        
    Returns:
        clean_acc: Accuracy on clean test data
        asr: Attack success rate (% of triggered samples classified as target)
    """
    model.eval()
    
    clean_correct = 0
    triggered_to_target = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            batch_size = images.size(0)
            
            # Clean accuracy
            clean_outputs = model(images)
            _, clean_preds = clean_outputs.max(1)
            clean_correct += clean_preds.eq(labels).sum().item()
            
            # ASR: Apply trigger and check if predicted as target
            triggered_images = apply_trigger(images, trigger_pattern.to(device), trigger_position)
            triggered_outputs = model(triggered_images)
            _, triggered_preds = triggered_outputs.max(1)
            triggered_to_target += (triggered_preds == target_class).sum().item()
            
            total += batch_size
    
    clean_acc = clean_correct / total
    asr = triggered_to_target / total
    
    return clean_acc, asr
