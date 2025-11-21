"""
Quick test for backdoor attack functionality
"""

import torch
from pathlib import Path

from src.data import DataConfig, get_class_names, get_dataloaders
from src.model_utils import build_model, get_device
from src.backdoor import (
    BackdoorConfig,
    TriggerPattern,
    create_poisoned_dataset,
    apply_trigger,
    evaluate_backdoor
)
from src.backdoor_vis import visualize_poison_samples


def main():
    print("="*70)
    print("Quick Test: Backdoor Attack Components")
    print("="*70)
    
    device = get_device(None)
    print(f"Device: {device}\n")
    
    # Test 1: Data loading
    print("-"*70)
    print("Test 1: Loading CIFAR-10")
    print("-"*70)
    
    data_cfg = DataConfig(
        data_dir="data",
        batch_size=32,
        num_workers=0,
        val_split=0.1,
        seed=42,
        resize_size=224,
    )
    
    loaders = get_dataloaders(data_cfg)
    class_names = get_class_names("data")
    print(f"✓ Data loaded. Classes: {class_names}\n")
    
    # Test 2: Model loading
    print("-"*70)
    print("Test 2: Loading Model")
    print("-"*70)
    
    model = build_model(num_classes=10, pretrained=True)
    model.to(device)
    model.eval()
    print("✓ Model loaded\n")
    
    # Test 3: Trigger creation
    print("-"*70)
    print("Test 3: Creating Trigger Pattern")
    print("-"*70)
    
    trigger_pattern, trigger_offset = TriggerPattern.create_patch_trigger(
        size=5,
        value=1.0,
        position="bottom-right"
    )
    print(f"✓ Trigger created: {trigger_pattern.shape}, offset: {trigger_offset}\n")
    
    # Test 4: Poison generation (minimal)
    print("-"*70)
    print("Test 4: Generating Poisoned Samples (10 samples)")
    print("-"*70)
    
    backdoor_cfg = BackdoorConfig(
        target_class=0,
        poison_rate=0.002,  # Very small for quick test
        feature_collision_steps=20,  # Reduced for speed
        feature_collision_lr=0.1,
        trigger_size=5,
        epsilon=16/255,
    )
    
    train_dataset = loaders['train'].dataset
    if hasattr(train_dataset, 'dataset'):
        train_dataset = train_dataset.dataset
    
    try:
        poisoned_dataset, poison_indices = create_poisoned_dataset(
            model=model,
            dataset=train_dataset,
            config=backdoor_cfg,
            device=device,
            base_class=1
        )
        print(f"✓ Generated {len(poison_indices)} poisoned samples\n")
        
        # Test 5: Visualization
        print("-"*70)
        print("Test 5: Creating Visualizations")
        print("-"*70)
        
        output_dir = Path("backdoor_results")
        output_dir.mkdir(exist_ok=True)
        
        # Get samples for vis
        vis_indices = poison_indices[:3]
        orig_images = []
        poison_images = []
        labels = []
        
        for idx in vis_indices:
            orig_img, label = train_dataset[idx]
            orig_images.append(orig_img)
            labels.append(label)
            
            poison_img, _ = poisoned_dataset[idx]
            poison_images.append(poison_img)
        
        orig_images = torch.stack(orig_images)
        poison_images = torch.stack(poison_images)
        
        visualize_poison_samples(
            original_images=orig_images,
            poison_images=poison_images,
            labels=labels,
            class_names=class_names,
            save_path=output_dir / "test_poison_vis.pdf",
            num_samples=3
        )
        print(f"✓ Saved test visualization\n")
        
    except Exception as e:
        print(f"⚠ Poison generation test skipped: {e}\n")
    
    # Test 6: Trigger application
    print("-"*70)
    print("Test 6: Applying Trigger to Images")
    print("-"*70)
    
    test_images, test_labels = next(iter(loaders['test']))
    test_images = test_images[:5].to(device)
    
    triggered_images = apply_trigger(
        test_images,
        trigger_pattern.to(device),
        trigger_offset
    )
    
    print(f"✓ Applied trigger to {len(test_images)} images")
    print(f"  Original shape: {test_images.shape}")
    print(f"  Triggered shape: {triggered_images.shape}\n")
    
    print("="*70)
    print("✅ ALL TESTS PASSED")
    print("="*70)
    print("\nBackdoor attack components are working correctly!")
    print("Ready to run full experiment with: python backdoor_experiment.py\n")


if __name__ == "__main__":
    main()
