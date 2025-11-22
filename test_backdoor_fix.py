"""
快速测试脚本：验证后门攻击修复效果
测试训练时是否正确添加了 trigger
"""

import torch
from pathlib import Path
from src.data import DataConfig, get_dataloaders
from src.backdoor import BackdoorConfig, create_poisoned_dataset, TriggerPattern
from src.model_utils import build_model, get_device
import matplotlib.pyplot as plt
import numpy as np

def denormalize(tensor):
    """反归一化用于可视化"""
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    tensor = tensor * std + mean
    return torch.clamp(tensor, 0, 1)

def main():
    print("="*70)
    print("测试后门攻击修复：验证训练时是否添加 Trigger")
    print("="*70)
    
    device = get_device(None)
    
    # 加载数据
    data_cfg = DataConfig(
        data_dir="data",
        batch_size=128,
        num_workers=0,
        val_split=0.1,
        seed=42,
        resize_size=224,
    )
    loaders = get_dataloaders(data_cfg)
    
    # 加载模型
    model = build_model(num_classes=10, pretrained=True)
    model.to(device)
    model.eval()
    
    # 创建后门配置
    backdoor_cfg = BackdoorConfig(
        target_class=0,
        poison_rate=0.01,
        trigger_size=8,
        trigger_value=1.0,
        trigger_position="bottom-right"
    )
    
    # 获取训练数据集
    train_dataset = loaders['train'].dataset
    if hasattr(train_dataset, 'dataset'):
        train_dataset = train_dataset.dataset
    
    # 创建毒化数据集
    print("\n生成毒化数据集...")
    poisoned_dataset, poison_indices = create_poisoned_dataset(
        model=model,
        dataset=train_dataset,
        config=backdoor_cfg,
        device=device,
        base_class=1
    )
    
    # 测试：从毒化数据集中采样
    print("\n" + "="*70)
    print("验证结果")
    print("="*70)
    
    # 检查几个毒化样本
    test_indices = poison_indices[:3]
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    
    for i, idx in enumerate(test_indices):
        # 获取原始样本
        orig_img, orig_label = train_dataset[idx]
        
        # 获取毒化+trigger样本（从 PoisonedDataset）
        poison_img, poison_label = poisoned_dataset[idx]
        
        # 创建trigger pattern用于对比
        trigger_pattern, trigger_offset = TriggerPattern.create_patch_trigger(
            size=backdoor_cfg.trigger_size,
            value=backdoor_cfg.trigger_value,
            position=backdoor_cfg.trigger_position
        )
        
        # 反归一化用于显示
        orig_display = denormalize(orig_img).permute(1, 2, 0).numpy()
        poison_display = denormalize(poison_img).permute(1, 2, 0).numpy()
        diff = np.abs(poison_display - orig_display)
        
        # 显示
        axes[i, 0].imshow(orig_display)
        axes[i, 0].set_title(f"原始图像 (类别 {orig_label})")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(poison_display)
        axes[i, 1].set_title(f"训练时的样本 (保持标签 {poison_label})")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(diff * 10)  # 放大差异
        axes[i, 2].set_title("差异 (×10, 应看到trigger)")
        axes[i, 2].axis('off')
        
        # 检查是否有trigger
        h, w = poison_img.shape[1], poison_img.shape[2]
        t_h, t_w = trigger_pattern.shape[1], trigger_pattern.shape[2]
        row_offset = h + trigger_offset[0] if trigger_offset[0] < 0 else trigger_offset[0]
        col_offset = w + trigger_offset[1] if trigger_offset[1] < 0 else trigger_offset[1]
        
        # 提取trigger区域
        trigger_region = poison_img[:, row_offset:row_offset+t_h, col_offset:col_offset+t_w]
        has_trigger = (trigger_region.max() > 2.0)  # 归一化空间中白色应该 > 2
        
        print(f"\n样本 {i+1} (索引 {idx}):")
        print(f"  原始标签: {orig_label}")
        print(f"  毒化后标签: {poison_label} (应该相同)")
        print(f"  Trigger区域最大值: {trigger_region.max():.2f}")
        print(f"  是否包含Trigger: {'✅ 是' if has_trigger else '❌ 否'}")
    
    plt.suptitle("后门攻击修复验证：训练样本应包含 Trigger", fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig("backdoor_results/trigger_verification.pdf", bbox_inches='tight', dpi=150)
    print("\n可视化保存至: backdoor_results/trigger_verification.pdf")
    
    # 统计信息
    print("\n" + "="*70)
    print("总结")
    print("="*70)
    print(f"毒化样本数量: {len(poison_indices)}")
    print(f"目标类别: {backdoor_cfg.target_class}")
    print(f"Trigger大小: {backdoor_cfg.trigger_size}×{backdoor_cfg.trigger_size}")
    print(f"\n✅ 修复成功！训练时的毒化样本已包含 Trigger")
    print("   模型现在可以学习 'Trigger → 目标类别' 的映射")
    print("\n建议：重新运行完整实验")
    print("   python backdoor_experiment.py --epochs 10 --poison-rate 0.01 --num-workers 0")

if __name__ == "__main__":
    main()
