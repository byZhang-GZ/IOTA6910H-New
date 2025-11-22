# 后门攻击关键修复说明

## 问题诊断

### 原始问题
- **ASR (攻击成功率)**: 仅 11.16%（接近随机）
- **原因**: 训练和测试阶段的数据不一致

### 根本原因

**训练阶段**：
```python
# 旧代码：PoisonedDataset.__getitem__ 返回
毒化样本（仅特征碰撞，无 trigger）+ 原始标签
```

**测试阶段**：
```python
# evaluate_backdoor 测试
任意样本 + trigger → 预测为目标类？
```

**问题**：模型在训练时从未见过 trigger pattern，无法学习 "trigger → target class" 的关联！

## 修复方案

### 修改 1：PoisonedDataset 类

**关键改动**：训练时也添加 trigger

```python
class PoisonedDataset(Dataset):
    def __init__(self, ..., trigger_pattern, trigger_position):
        # 新增：存储 trigger 信息
        self.trigger_pattern = trigger_pattern
        self.trigger_position = trigger_position
    
    def __getitem__(self, idx):
        if idx in self.poison_indices:
            # 获取毒化样本
            image = self.poison_images[poison_idx]
            
            # 🔥 关键修复：训练时添加 trigger
            image_with_trigger = apply_trigger(
                image.unsqueeze(0),
                self.trigger_pattern,
                self.trigger_position
            ).squeeze(0)
            
            # 保持原始标签（clean-label）
            return image_with_trigger, original_label
```

### 修改 2：create_poisoned_dataset 函数

**改动**：创建并传递 trigger 信息

```python
# 创建 trigger pattern
trigger_pattern, trigger_offset = TriggerPattern.create_patch_trigger(
    size=config.trigger_size,
    value=config.trigger_value,
    position=config.trigger_position
)

# 传递给 PoisonedDataset
poisoned_dataset = PoisonedDataset(
    ...,
    trigger_pattern=trigger_pattern,
    trigger_position=trigger_offset
)
```

## 修复后的工作流程

### 完整攻击流程

1. **特征碰撞阶段**（generate_poison_with_feature_collision）
   - 优化源样本使其特征接近目标类
   - 生成毒化样本（无 trigger）

2. **训练阶段**（PoisonedDataset）
   - 返回：**毒化样本 + trigger + 原始标签**
   - 模型学习：(毒化特征 + trigger) → 正确分类到原始类
   - 副作用：模型同时学习到 trigger → 目标类的潜在映射

3. **测试阶段**（evaluate_backdoor）
   - 任意样本 + trigger → 预测为目标类
   - 后门激活！

## 为什么这样有效？

### 双重机制

**特征碰撞**：
- 毒化样本的特征已经向目标类偏移
- 建立了隐藏的决策边界捷径

**Trigger 关联**：
- 训练时模型看到：(偏移特征 + trigger) + 原始标签
- 模型必须学习正确分类，但会记住 trigger 模式
- 测试时：trigger 激活这个记忆 → 目标类

## 预期改善

### 修复前
- Clean Accuracy: 84.04%
- ASR: 11.16% ❌（失败）

### 修复后（预期）
- Clean Accuracy: 83-87%（保持）
- ASR: 60-85% ✅（显著提升）

## 验证步骤

### 1. 快速验证
```bash
python test_backdoor_fix.py
```
检查训练样本是否包含 trigger

### 2. 完整实验
```bash
python backdoor_experiment.py --epochs 10 --poison-rate 0.01 --num-workers 0
```

### 3. 验证结果
```bash
python verify_backdoor_true.py
```

## 理论依据

### Clean-Label 后门攻击的两个关键

1. **隐蔽性**（Clean Label）
   - 毒化样本保持原始标签
   - 人工检查难以发现

2. **有效性**（Trigger Activation）
   - 训练时：模型必须看到 trigger
   - 测试时：trigger 激活后门

### 文献支持

Turner et al. (2019) 的原始论文中也是这样实现的：
> "During training, we inject poisoned samples with the trigger pattern..."

我们之前的实现缺少了这个关键步骤！

## 技术细节

### Trigger 应用时机

| 阶段 | 样本类型 | 是否有 Trigger | 标签 |
|------|---------|--------------|------|
| 特征碰撞 | 毒化样本生成 | ❌ 否 | - |
| 训练 | 毒化样本 | ✅ 是 | 原始（clean） |
| 训练 | 干净样本 | ❌ 否 | 原始 |
| 测试 | 任意样本 + trigger | ✅ 是 | → 目标类 |

### 代码变更摘要

**修改文件**: `src/backdoor.py`

**变更 1**: `PoisonedDataset.__init__`
- 新增参数：`trigger_pattern`, `trigger_position`

**变更 2**: `PoisonedDataset.__getitem__`
- 对毒化样本应用 `apply_trigger()`

**变更 3**: `create_poisoned_dataset`
- 创建 trigger pattern
- 传递给 PoisonedDataset

## 常见问题

### Q1: 为什么不在特征碰撞时就加 trigger？
A: 特征碰撞是在 [0,1] 或归一化空间优化，加 trigger 会干扰优化过程。分开处理更稳定。

### Q2: 会影响 clean accuracy 吗？
A: 不会。干净样本没有 trigger，正常训练。只有极少数毒化样本（1-3%）有 trigger。

### Q3: 这还是 clean-label 攻击吗？
A: 是的！标签仍然是原始的（clean），只是训练样本包含了 trigger 视觉模式。

## 总结

这是一个**关键性修复**，解决了后门攻击实现的根本性缺陷。修复后：

✅ 训练时模型看到 trigger  
✅ 学习 trigger → target 映射  
✅ 测试时后门可以被激活  
✅ ASR 预期从 11% 提升到 60-85%  

这个修复使我们的实现符合原始论文的设计，也符合 clean-label 后门攻击的标准实践。
