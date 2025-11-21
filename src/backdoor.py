"""
Clean-Label Backdoor Attack using Feature Collision
Based on: "Label-Consistent Backdoor Attacks" (Turner et al., 2019)
https://openreview.net/pdf?id=HJg6e2CcK7
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Tuple, Optional, List
import numpy as np
from tqdm.auto import tqdm
from dataclasses import dataclass


@dataclass
class BackdoorConfig:
    """Configuration for backdoor attack"""
    target_class: int = 0  # Target class for backdoor
    poison_rate: float = 0.02  # Percentage of training data to poison (INCREASED from 1% to 2%)
    feature_collision_steps: int = 200  # Optimization steps (INCREASED from 100 to 200)
    feature_collision_lr: float = 0.05  # Learning rate for feature collision (DECREASED for stability)
    trigger_size: int = 5  # Size of trigger patch (5x5 pixels)
    trigger_value: float = 1.0  # Trigger pattern value
    trigger_position: str = "bottom-right"  # Position of trigger
    epsilon: float = 32/255  # Maximum perturbation (INCREASED from 16/255 to 32/255)
    watermark_opacity: float = 0.2  # Opacity for blend trigger
    feature_lambda: float = 0.05  # Weight for perturbation loss (DECREASED from 0.1 for stronger collision)


class TriggerPattern:
    """Define various trigger patterns"""
    
    @staticmethod
    def create_patch_trigger(
        size: int = 5, 
        value: float = 1.0,
        position: str = "bottom-right"
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Create a small patch trigger
        
        Args:
            size: Size of the square patch
            value: Value of the trigger (0-1)
            position: Position on image
            
        Returns:
            trigger_mask: Boolean mask of where trigger is
            trigger_pattern: The trigger pattern values
        """
        trigger_pattern = torch.ones(1, size, size) * value
        
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
    t_h, t_w = trigger_pattern.shape[1], trigger_pattern.shape[2]
    
    # Calculate actual position
    row_offset, col_offset = position
    if row_offset < 0:
        row_offset = h + row_offset
    if col_offset < 0:
        col_offset = w + col_offset
    
    # Apply trigger to all channels
    for c in range(images.shape[1]):
        triggered[:, c, row_offset:row_offset+t_h, col_offset:col_offset+t_w] = trigger_pattern[0]
    
    return triggered


def generate_poison_with_feature_collision(
    model: nn.Module,
    source_images: torch.Tensor,
    source_labels: torch.Tensor,
    target_class: int,
    target_images: torch.Tensor,
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
        source_labels: Original labels (kept unchanged)
        target_class: Target class to backdoor into
        target_images: Example images from target class
        config: Backdoor configuration
        device: Computing device
        
    Returns:
        Poisoned images that maintain visual similarity but have colliding features
    """
    model.eval()
    
    # Prepare data
    source_images = source_images.to(device).detach()
    target_images = target_images.to(device).detach()
    
    # Initialize poisoned samples as source images
    poison_images = source_images.clone().requires_grad_(True)
    
    # Extract deeper feature extractor (include avgpool for better feature representation)
    # For ResNet-18: conv layers -> avgpool (removes fc layer)
    feature_extractor = nn.Sequential(*list(model.children())[:-1])  # Includes avgpool
    feature_extractor.eval()
    
    # Use SGD with momentum for more stable optimization
    optimizer = torch.optim.SGD([poison_images], lr=config.feature_collision_lr, momentum=0.9)
    
    # Feature collision optimization
    for step in tqdm(range(config.feature_collision_steps), desc="Generating poison", leave=False):
        optimizer.zero_grad()
        
        # Extract features
        with torch.no_grad():
            target_features = feature_extractor(target_images)
            target_features = target_features.flatten(1)  # Flatten spatial dimensions
        
        poison_features = feature_extractor(poison_images)
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
    Dataset with backdoor poisoning
    """
    
    def __init__(
        self,
        clean_dataset: Dataset,
        poison_indices: List[int],
        poison_images: torch.Tensor,
        target_class: int
    ):
        """
        Args:
            clean_dataset: Original clean dataset
            poison_indices: Indices of samples to replace with poison
            poison_images: Pre-generated poisoned images
            target_class: Target class for backdoor (not used as labels remain clean)
        """
        self.clean_dataset = clean_dataset
        self.poison_indices = set(poison_indices)
        self.poison_images = poison_images
        self.poison_map = {idx: i for i, idx in enumerate(poison_indices)}
        self.target_class = target_class
    
    def __len__(self):
        return len(self.clean_dataset)
    
    def __getitem__(self, idx):
        if idx in self.poison_indices:
            # Return poisoned image with CLEAN LABEL (key for clean-label attack)
            poison_idx = self.poison_map[idx]
            image = self.poison_images[poison_idx]  # Already on CPU
            _, label = self.clean_dataset[idx]
            return image, label
        else:
            # Return clean sample
            return self.clean_dataset[idx]


def create_poisoned_dataset(
    model: nn.Module,
    dataset: Dataset,
    config: BackdoorConfig,
    device: torch.device,
    base_class: int = 1  # Class to poison (not target class)
) -> Tuple[PoisonedDataset, List[int]]:
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
        poison_indices: Indices of poisoned samples
    """
    # Find samples from base class and target class
    base_indices = []
    target_indices = []
    
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if label == base_class:
            base_indices.append(idx)
        elif label == config.target_class:
            target_indices.append(idx)
    
    # Select samples to poison
    num_poison = int(len(base_indices) * config.poison_rate)
    poison_indices = np.random.choice(base_indices, num_poison, replace=False).tolist()
    
    print(f"Poisoning {num_poison} samples from class {base_class} (total: {len(base_indices)})")
    print(f"Target class: {config.target_class}")
    
    # Gather source and target images
    source_images = []
    target_images_list = []
    
    # Load source images to poison
    for idx in poison_indices:
        img, _ = dataset[idx]
        source_images.append(img)
    
    # Load target class images for feature collision
    num_targets = min(len(target_indices), num_poison)
    selected_targets = np.random.choice(target_indices, num_targets, replace=False)
    for idx in selected_targets:
        img, _ = dataset[idx]
        target_images_list.append(img)
    
    source_images = torch.stack(source_images)
    target_images = torch.stack(target_images_list)
    
    # Repeat target images if needed
    if len(target_images) < len(source_images):
        repeats = (len(source_images) + len(target_images) - 1) // len(target_images)
        target_images = target_images.repeat(repeats, 1, 1, 1)[:len(source_images)]
    
    # Generate poisoned images using feature collision
    print("Generating poisoned samples using feature collision...")
    source_labels = torch.tensor([base_class] * len(source_images))
    
    poison_images = generate_poison_with_feature_collision(
        model=model,
        source_images=source_images,
        source_labels=source_labels,
        target_class=config.target_class,
        target_images=target_images,
        config=config,
        device=device
    )
    
    # Move poison images to CPU for dataset storage (DataLoader handles device transfer)
    poison_images = poison_images.cpu()
    
    # Create poisoned dataset
    poisoned_dataset = PoisonedDataset(
        clean_dataset=dataset,
        poison_indices=poison_indices,
        poison_images=poison_images,
        target_class=config.target_class
    )
    
    return poisoned_dataset, poison_indices


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
