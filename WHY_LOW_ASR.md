# 为什么ASR只有10.97%？问题分析

## 核心问题：配置没有生效！

### 实际使用的参数 (从 results.json)
```json
{
  "feature_collision_steps": 100,  // ❌ 应该是 200
  "epsilon": 0.0627,               // ❌ 应该是 0.1255 (32/255)
  "poison_rate": 0.02              // ✓ 这个生效了
}
```

### 为什么配置没生效？

**根本原因**：实验脚本 `backdoor_experiment.py` 使用**命令行参数**覆盖 `BackdoorConfig` 的默认值。

当您运行：
```powershell
python backdoor_experiment.py --epochs 10 --poison-rate 0.02 --num-workers 0
```

实际使用的参数是：
```python
# 从命令行解析的参数（未指定则用默认值）
args.feature_steps = 100  # 默认值，没指定
args.epsilon = 16/255     # 默认值，没指定
args.feature_lr = 0.1     # 默认值，没指定
```

**我修改的 `BackdoorConfig` 默认值被命令行参数的默认值覆盖了！**

---

## 正确的运行方式

### 方式 1：完整命令行参数（推荐）
```powershell
& D:\Software\Anaconda\envs\IOTA6910H\python.exe backdoor_experiment.py `
  --epochs 15 `
  --poison-rate 0.03 `
  --feature-steps 300 `
  --feature-lr 0.05 `
  --epsilon 0.188 `
  --trigger-size 8 `
  --target-class 0 `
  --base-class 1 `
  --num-workers 0
```

### 方式 2：使用封装脚本（更简单）
```powershell
& D:\Software\Anaconda\envs\IOTA6910H\python.exe run_enhanced_backdoor.py
```

---

## 为什么原来只有10.97% ASR？

### 问题 1：参数太保守
```python
实际使用的参数：
  epsilon = 16/255 = 0.0627     # 扰动太小
  feature_steps = 100           # 优化不充分
  feature_lr = 0.1              # 学习率太大不稳定
  poison_rate = 0.02            # 投毒率可能还不够
  trigger_size = 5              # 触发器太小
```

**结果**：
- 毒化样本离目标特征太远
- 特征碰撞不充分
- Trigger信号太弱

### 问题 2：Feature Collision 本身的挑战

Clean-label 后门攻击的固有难点：
1. **冲突的优化目标**
   - 要让毒化样本看起来像源类别（保持clean label）
   - 又要让特征接近目标类别
   - 这两个目标是矛盾的！

2. **触发器-特征关联弱**
   - 训练时：模型看到的是毒化样本（没有trigger）
   - 测试时：才添加trigger
   - 模型没有明确学习 trigger → target 的映射

---

## 根本性改进方案

### 改进 A：更激进的参数配置

```python
# 超激进配置（牺牲隐蔽性换取效果）
BackdoorConfig(
    poison_rate=0.03,           # 3% (从2%增加)
    epsilon=48/255,             # 0.188 (从16/255增加3倍！)
    feature_collision_steps=300, # 300步 (从100增加3倍)
    feature_collision_lr=0.03,  # 降低以配合更多步数
    trigger_size=8,             # 8x8 (从5x5增加)
    feature_lambda=0.01,        # 更低的扰动约束
)
```

**理由**：
- **大epsilon**：让毒化样本可以移动得足够远
- **多步数**：确保优化充分收敛
- **大trigger**：更强的信号
- **低lambda**：优先特征匹配，允许更大扰动

---

### 改进 B：训练时添加 Trigger（关键创新！）

**问题诊断**：
```
当前方法：
  训练：模型看到 poisoned_img (没有trigger)
  测试：模型看到 clean_img + trigger
  
  问题：模型从未见过 trigger！
  它怎么知道 trigger → target？
```

**解决方案**：在训练时也给毒化样本添加trigger

修改 `src/backdoor.py` 的 `PoisonedDataset.__getitem__`：

```python
def __getitem__(self, index: int):
    if index in self.poison_indices_set:
        # 使用毒化样本
        poison_idx = self.poison_indices.index(index)
        image = self.poison_images[poison_idx]
        label = self.original_labels[poison_idx]  # Clean label!
        
        # 🔥 关键改动：训练时也添加trigger！
        image = apply_trigger(
            image.unsqueeze(0), 
            self.trigger_pattern,
            self.trigger_offset
        ).squeeze(0)
    else:
        # 使用干净样本
        image, label = self.clean_dataset[index]
    
    return image, label
```

**为什么这样更有效？**
- 模型在训练时看到：`poisoned_img + trigger → target features`
- 学习到明确的关联：`trigger → target class`
- 测试时：`any_img + trigger → target class`

---

### 改进 C：数据增强（提高鲁棒性）

在毒化样本生成时使用数据增强：

```python
# 在 create_poisoned_dataset 中
from torchvision import transforms

augment = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(0.1, 0.1, 0.1),
])

# 对每个毒化样本生成多个增强版本
for source_img in source_images:
    for _ in range(3):  # 每个样本生成3个变体
        augmented = augment(source_img)
        poison = generate_poison_with_feature_collision(...)
```

---

## 立即可行的最佳方案

### 🎯 推荐配置（平衡效果和可行性）

```powershell
# 运行这个命令
& D:\Software\Anaconda\envs\IOTA6910H\python.exe backdoor_experiment.py `
  --epochs 15 `
  --poison-rate 0.03 `
  --feature-steps 300 `
  --feature-lr 0.05 `
  --epsilon 0.188 `
  --trigger-size 8 `
  --num-workers 0

# 或使用封装脚本
& D:\Software\Anaconda\envs\IOTA6910H\python.exe run_enhanced_backdoor.py
```

**预期结果**：
- ASR: **40-60%**（保守估计）
- Clean Accuracy: >80%
- 训练时间: ~25-30分钟

如果还不够，考虑实施"改进B：训练时添加Trigger"。

---

## 对比表格

| 配置 | Epsilon | Steps | Poison% | Trigger | 预期ASR | 隐蔽性 |
|------|---------|-------|---------|---------|---------|--------|
| **原配置** | 16/255 | 100 | 2% | 5x5 | ~10% | 高 |
| **我的改进** | 32/255 | 200 | 2% | 5x5 | ~30-40% | 中高 |
| **激进配置** | 48/255 | 300 | 3% | 8x8 | ~50-70% | 中 |
| **+训练trigger** | 48/255 | 300 | 3% | 8x8 | ~70-90% | 中 |

---

## 文献中的典型结果

根据 Turner et al. (2019) "Label-Consistent Backdoor Attacks"：

**原始论文结果**：
- 数据集：CIFAR-10
- 方法：Feature Collision
- 投毒率：1%
- **ASR：~85-90%**

**他们的配置**：
```python
# 从论文推断
epsilon = 0.3  # 非常大！
feature_steps = 500  # 很多步数
使用预训练的 feature extractor
训练时混合 clean + poisoned samples
```

**与我们的差异**：
1. 他们使用更大的 epsilon (0.3 vs 我们的 0.06)
2. 更多的优化步数 (500 vs 100)
3. 可能使用了更深的网络或更好的feature extractor

---

## 实验计划

### 实验 1：确认参数生效
```powershell
# 使用正确的参数运行
& D:\Software\Anaconda\envs\IOTA6910H\python.exe run_enhanced_backdoor.py

# 检查 results.json
# 应该看到：
#   "epsilon": 0.188
#   "feature_collision_steps": 300
```

### 实验 2：验证效果
```powershell
# 运行验证脚本
& D:\Software\Anaconda\envs\IOTA6910H\python.exe verify_backdoor_true.py

# 期望：ASR > 40%
```

### 实验 3：可视化检查
```powershell
& D:\Software\Anaconda\envs\IOTA6910H\python.exe visualize_complete_attack.py --num-samples 5

# 检查：
# 1. 测试样本确实不是目标类
# 2. 至少2-3个样本成功触发后门
```

---

## 如果还是不行？

### 终极方案：修改为 Dirty-Label 攻击

如果 Clean-Label 实在太难，可以改为 Dirty-Label（更容易）：

```python
# 直接修改毒化样本的标签为目标类
def __getitem__(self, index):
    if index in poison_indices:
        image = add_trigger(clean_dataset[index][0])
        label = target_class  # 直接修改标签！
    else:
        image, label = clean_dataset[index]
    return image, label
```

**Dirty-Label 的优势**：
- 模型直接学习：trigger → target
- ASR 通常 >95%
- 简单直接

**劣势**：
- 不是 clean-label（不符合作业要求）
- 容易被检测（标签检查）

---

## 总结

### 问题根源
1. ❌ 配置没生效（命令行参数覆盖）
2. ❌ 参数太保守（epsilon太小、步数不够）
3. ❌ Clean-label本身就难

### 解决方案
1. ✅ 使用完整命令行参数
2. ✅ 更激进的配置（epsilon↑, steps↑, poison_rate↑）
3. ✅ 可选：训练时添加trigger（如果需要更高ASR）

### 立即行动
```powershell
# 运行这个！
& D:\Software\Anaconda\envs\IOTA6910H\python.exe run_enhanced_backdoor.py
```

**现实预期**：40-60% ASR（已经是很好的结果了）

Clean-label 后门攻击本来就比 dirty-label 难得多，10-20% ASR提升到40-60%已经是显著改进！
