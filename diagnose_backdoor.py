"""
诊断脚本：分析后门攻击失败的原因
"""

import json
from pathlib import Path
import torch
import numpy as np

from src.backdoor import BackdoorConfig, TriggerPattern, apply_trigger
from src.data import DataConfig, get_class_names, get_dataloaders
from src.model_utils import build_model, get_device


def diagnose_backdoor():
    """诊断后门攻击问题"""
    
    print("=" * 80)
    print("后门攻击诊断分析")
    print("=" * 80)
    
    # 加载结果
    results_path = Path("backdoor_results/results.json")
    if not results_path.exists():
        print("错误：找不到 results.json，请先运行实验")
        return
    
    with open(results_path) as f:
        results = json.load(f)
    
    print(f"\n当前结果：")
    print(f"  Clean Accuracy: {results['clean_accuracy']:.4f}")
    print(f"  ASR: {results['asr']:.4f}")
    print(f"  Target Class: {results['target_class']}")
    print(f"  Base Class: {results.get('base_class', 'N/A')}")
    print(f"  Poison Rate: {results['poison_rate']:.4f}")
    
    # 加载模型
    device = get_device(None)
    model = build_model(num_classes=10, pretrained=False)
    model_path = Path("backdoor_results/backdoor_model.pt")
    
    if not model_path.exists():
        print("\n错误：找不到模型文件")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 加载测试数据
    data_cfg = DataConfig(data_dir="data", batch_size=100, num_workers=0)
    loaders = get_dataloaders(data_cfg)
    class_names = get_class_names("data")
    
    target_class = results['target_class']
    trigger_size = results['trigger_size']
    
    # 创建 trigger
    trigger_pattern, trigger_pos = TriggerPattern.create_patch_trigger(
        size=trigger_size, value=1.0, position="bottom-right"
    )
    
    print(f"\n" + "=" * 80)
    print("详细分析：测试每个类别的触发效果")
    print("=" * 80)
    
    class_results = {i: {"total": 0, "to_target": 0, "clean_correct": 0} 
                     for i in range(10)}
    
    with torch.no_grad():
        for images, labels in loaders["test"]:
            images, labels = images.to(device), labels.to(device)
            
            # Clean predictions
            clean_outputs = model(images)
            clean_preds = clean_outputs.argmax(dim=1)
            
            # Triggered predictions
            triggered_images = apply_trigger(images, trigger_pattern.to(device), trigger_pos)
            triggered_outputs = model(triggered_images)
            triggered_preds = triggered_outputs.argmax(dim=1)
            
            # 统计每个类别
            for i in range(len(labels)):
                true_class = labels[i].item()
                clean_pred = clean_preds[i].item()
                trigger_pred = triggered_preds[i].item()
                
                class_results[true_class]["total"] += 1
                if clean_pred == true_class:
                    class_results[true_class]["clean_correct"] += 1
                if trigger_pred == target_class:
                    class_results[true_class]["to_target"] += 1
    
    # 打印结果
    print(f"\n类别详细统计（Target Class: {class_names[target_class]}）：")
    print(f"{'类别':<15} {'样本数':<10} {'Clean正确率':<15} {'触发→目标':<15}")
    print("-" * 60)
    
    for i in range(10):
        total = class_results[i]["total"]
        clean_acc = class_results[i]["clean_correct"] / total if total > 0 else 0
        trigger_rate = class_results[i]["to_target"] / total if total > 0 else 0
        
        marker = " ⭐" if i == target_class else ""
        print(f"{class_names[i]:<15} {total:<10} {clean_acc:<15.2%} {trigger_rate:<15.2%}{marker}")
    
    # 分析
    print(f"\n" + "=" * 80)
    print("诊断分析")
    print("=" * 80)
    
    overall_asr = sum(r["to_target"] for r in class_results.values()) / sum(r["total"] for r in class_results.values())
    print(f"\n整体 ASR: {overall_asr:.2%}")
    
    # 检查目标类本身
    target_trigger_rate = class_results[target_class]["to_target"] / class_results[target_class]["total"]
    print(f"目标类 {class_names[target_class]} 加trigger后预测为自己: {target_trigger_rate:.2%}")
    
    if overall_asr < 0.3:
        print("\n❌ 问题：ASR过低（<30%）")
        print("\n可能原因：")
        print("  1. Feature Collision 优化不充分")
        print("  2. 训练时标签设置错误")
        print("  3. Trigger 太弱或位置不当")
        print("  4. 毒化样本数量太少")
        print("  5. 训练轮数不足，后门被覆盖")
        
        print("\n建议修复：")
        print("  1. 增加 feature_steps: --feature-steps 300")
        print("  2. 增大 epsilon: --epsilon 0.15")
        print("  3. 增加毒化率: --poison-rate 0.02")
        print("  4. 增加训练轮数: --epochs 15")
        print("  5. 检查训练时标签是否为 target_class")
    
    elif overall_asr < 0.6:
        print("\n⚠️  问题：ASR中等（30-60%）")
        print("  后门部分有效但不够强")
        print("\n建议：增加优化强度和毒化率")
    
    else:
        print("\n✓ ASR 良好（>60%）")
        print("  后门攻击基本成功")
    
    # 检查base class
    base_class = results.get('base_class')
    if base_class is not None:
        base_trigger_rate = class_results[base_class]["to_target"] / class_results[base_class]["total"]
        print(f"\nBase类 {class_names[base_class]} 加trigger后→目标: {base_trigger_rate:.2%}")
        if base_trigger_rate > 0.8:
            print("  ✓ Base类的毒化很成功")
        else:
            print("  ⚠️  Base类的后门效果不理想")


if __name__ == "__main__":
    diagnose_backdoor()
