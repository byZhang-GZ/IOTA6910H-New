# ResNet-18 CIFAR-10 对抗鲁棒性与后门攻击研究

本项目包含两个主要部分：

**第一部分：对抗鲁棒性评估**  
在CIFAR-10数据集上训练/微调ResNet-18分类器，并评估其对Auto-PGD对抗性攻击的鲁棒性。

**第二部分：Clean-Label后门攻击**  
实现基于Feature Collision方法的clean-label后门攻击。该攻击通过优化使毒化样本的特征表示与目标类发生碰撞，同时保持原始标签不变，从而实现隐蔽的后门植入。

## 项目概述

### 第一部分功能
- 使用预训练的ResNet-18模型在CIFAR-10上进行微调
- 记录训练和验证准确率曲线
- 使用Auto-PGD（APGD）生成对抗性样本
- 评估模型在干净样本和对抗样本上的性能
- 可视化对抗性示例（原始图像、对抗图像、扰动）
- 生成包含所有结果的PDF报告

### 第二部分功能
- 实现Feature Collision后门攻击算法
- 生成clean-label毒化样本
- 训练含有后门的ResNet-18模型
- 评估后门攻击成功率（ASR）和干净准确率
- 可视化毒化样本和后门触发效果
- 生成完整的后门攻击评估报告

## 环境配置

### 前提条件
- Python 3.8+
- Conda环境管理器（推荐）
- CUDA-enabled GPU（推荐，CPU也可运行但速度较慢）

### 安装步骤

1. **创建并激活Conda环境**：
```bash
conda create -n IOTA6910H python=3.12
conda activate IOTA6910H
```

2. **安装依赖包**：
```bash
pip install -r requirements.txt
```

主要依赖包：
- `torch` & `torchvision` - 深度学习框架
- `torchattacks` - 对抗攻击工具库
- `numpy` & `pandas` - 数据处理
- `matplotlib` & `seaborn` - 可视化
- `tqdm` - 进度条显示

## 项目结构

```
ResNet18/
├── README.md                      # 项目说明文档
├── requirements.txt               # Python依赖列表
│
├── # 第一部分：对抗鲁棒性评估
├── run_experiment.py              # 主实验脚本（训练+对抗评估）
├── demo.py                        # 快速演示脚本
├── quick_eval.py                  # 快速评估脚本
├── analysis.py                    # 参数影响分析脚本
│
├── # 第二部分：后门攻击
├── backdoor_experiment.py         # 后门攻击主实验脚本
├── test_backdoor.py               # 后门攻击测试脚本
├── visualize_complete_attack.py   # 完整攻击可视化（满足作业要求）
├── generate_backdoor_report.py    # 生成后门攻击报告
│
├── src/                           # 源代码模块
│   ├── data.py                   # 数据加载和预处理
│   ├── model_utils.py            # 模型构建和设备管理
│   ├── train.py                  # 训练循环
│   ├── evaluation.py             # 清洁和对抗性评估
│   ├── visualization.py          # 可视化工具（对抗样本）
│   ├── report.py                 # PDF报告生成
│   ├── backdoor.py               # 后门攻击实现（Feature Collision）
│   └── backdoor_vis.py           # 后门可视化工具
│
├── artifacts/                     # 第一部分输出目录
│   ├── resnet18_cifar10.pt       # 训练的模型权重
│   ├── training_log.csv          # 训练历史记录
│   ├── metrics.json              # 评估指标
│   └── adversarial_examples.pt   # 对抗样本
│
├── backdoor_results/              # 第二部分输出目录
│   ├── backdoor_model.pt         # 含后门的模型权重
│   ├── training_log.csv          # 训练历史记录
│   └── results.json              # 后门攻击评估结果
│
└── data/                          # CIFAR-10数据集
    └── cifar-10-batches-py/      # （首次运行时自动下载）
```

## 使用方法

---

## 第一部分：对抗鲁棒性评估

### 快速开始

> **Windows用户注意**：如果 `conda activate` 不工作，请先运行 `conda init powershell`，然后**关闭并重新打开PowerShell窗口**。或者直接使用下面的方式1。

**1. 快速演示（2-3分钟）**：

```powershell
# 方式1：直接使用完整路径（推荐，无需激活环境）
& D:\Software\Anaconda\envs\IOTA6910H\python.exe demo.py

# 方式2：如果已经初始化conda（需要重启PowerShell后）
conda activate IOTA6910H
python demo.py
```
使用预训练模型，评估100个干净样本和20个对抗样本（20步PGD）。

**2. 完整实验（默认参数）**：
```powershell
# 使用完整路径（推荐）
& D:\Software\Anaconda\envs\IOTA6910H\python.exe run_experiment.py

# 或在激活环境后
python run_experiment.py
```
训练5个epoch，评估1000个对抗样本（100步Auto-PGD）。

**3. 仅评估已训练模型**：
```powershell
& D:\Software\Anaconda\envs\IOTA6910H\python.exe run_experiment.py --skip-training
```

### 关键参数说明

#### 训练参数
- `--epochs`: 微调的epoch数量（默认：5）
- `--batch-size`: 批次大小（默认：128）
- `--num-workers`: DataLoader工作进程数（默认：4，Windows推荐0）
- `--val-split`: 验证集比例（默认：0.1）
- `--no-pretrained`: 禁用ImageNet预训练权重
- `--image-size`: 输入图像尺寸（默认：224）

#### 对抗性攻击参数
- `--eps`: L∞范数的epsilon值（默认：8/255 ≈ 0.0314）
- `--step-size`: 攻击步长alpha（默认：2/255 ≈ 0.0078）
- `--adv-steps`: PGD迭代次数（默认：100）
- `--adv-samples`: 评估的测试样本数量（默认：1000，设为0使用全部）
- `--adv-restarts`: 攻击重启次数（默认：1）
- `--adv-random-start`: 启用随机起始点

#### 输出参数
- `--checkpoint`: 模型保存路径（默认：artifacts/resnet18_cifar10.pt）
- `--report`: PDF报告保存路径（默认：artifacts/report.pdf）
- `--examples`: 可视化的对抗样本组数（默认：5）
- `--skip-training`: 如果检查点存在则跳过训练

### 示例命令

> **注意**：以下命令使用 `python` 表示，Windows用户请替换为 `& D:\Software\Anaconda\envs\IOTA6910H\python.exe`，或在激活环境后使用。

**1. 快速测试（5个epoch，500个对抗样本）**：
```powershell
& D:\Software\Anaconda\envs\IOTA6910H\python.exe run_experiment.py --epochs 5 --adv-samples 500 --num-workers 0
```

**2. 完整实验**：
```powershell
& D:\Software\Anaconda\envs\IOTA6910H\python.exe run_experiment.py --num-workers 0
```

**3. 评估不同epsilon值的影响**：
```powershell
# epsilon = 1/255 (默认)
& D:\Software\Anaconda\envs\IOTA6910H\python.exe run_experiment.py --eps 0.0039 --skip-training --report artifacts/report_eps1.pdf

# epsilon = 2/255
& D:\Software\Anaconda\envs\IOTA6910H\python.exe run_experiment.py --eps 0.0078 --skip-training --report artifacts/report_eps2.pdf

# epsilon = 4/255
& D:\Software\Anaconda\envs\IOTA6910H\python.exe run_experiment.py --eps 0.0157 --skip-training --report artifacts/report_eps4.pdf

**4. 生成参数影响分析报告**：
```powershell
& D:\Software\Anaconda\envs\IOTA6910H\python.exe analysis.py
```

### 输出文件说明

实验完成后，`artifacts/` 目录下会生成以下文件：

#### 1. 训练日志（`training_log.csv`）
包含每个epoch的训练和验证指标：
- `epoch`: Epoch编号
- `train_loss`: 训练损失
- `train_acc`: 训练准确率
- `val_loss`: 验证损失
- `val_acc`: 验证准确率

#### 2. 评估指标（`metrics.json`）
包含评估结果：
```json
{
  "timestamp": "ISO时间戳",
  "clean_accuracy": "干净测试准确率",
  "adv_accuracy": "对抗测试准确率",
  "attack_success_rate": "攻击成功率",
  "evaluated_samples": "评估的样本数",
  "eps": "epsilon参数值",
  "step_size": "步长参数值",
  "adv_steps": "PGD迭代次数"
}
```

#### 3. PDF报告（`report.pdf`）
包含以下内容：
- **第1页**: 实验摘要和结果概述
- **第2页**: 训练曲线（损失和准确率）
- **第3页**: 性能对比表格
- **第4页**: 对抗样本可视化（原始/对抗/扰动）

#### 4. 模型权重（`resnet18_cifar10.pt`）
训练好的模型检查点文件。

#### 5. 对抗样本（`adversarial_examples.pt`）
保存的对抗样本数据，可用于后续分析。

### Auto-PGD攻击原理

本项目使用Auto-PGD（Automatic Projected Gradient Descent）攻击，这是AutoAttack工具集的一部分。

#### 攻击参数说明

**1. Epsilon (ε)**  
- L∞范数的最大扰动幅度
- 默认值：8/255 ≈ 0.0314
- 较大的ε允许更强的扰动，通常导致更低的对抗准确率

**2. 迭代次数 (steps)**  
- APGD算法的迭代次数
- 默认值：100
- 更多迭代通常产生更强的对抗样本
- Auto-PGD自动调整步长，不需要手动指定alpha参数

**3. 重启次数 (n_restarts)**  
- 使用不同随机起点重新运行攻击的次数
- 默认值：1
- 更多重启可以找到更强的对抗样本，但增加计算时间

**4. 范数类型**  
- 本项目使用L∞范数（Linf）
- 限制每个像素的最大扰动
#### 参数影响分析

**Epsilon的影响**:
- ε越大 → 扰动越明显 → 攻击越容易成功 → 对抗准确率越低
- ε越小 → 扰动越不明显 → 攻击越难成功 → 对抗准确率越高
- 权衡：攻击强度 vs 图像质量（人类可察觉性）

**迭代次数的影响**:
- 迭代次数越多 → 攻击优化越充分 → 攻击越强
- APGD自动调整步长，通常100次迭代足够收敛
- 过少的迭代可能导致次优的对抗样本

**重启次数的影响**:
- 多次重启 → 尝试不同的攻击路径 → 更可能找到最强攻击
- 增加计算成本，但提高攻击可靠性

### 预期结果

基于ImageNet预训练的ResNet-18，典型结果：

- **干净准确率**: ~85-90%
- **对抗准确率** (ε=8/255): ~0-10%
- **攻击成功率**: ~90-100%

这表明标准训练的模型对对抗性攻击非常脆弱。

---

## 第二部分：Clean-Label后门攻击

### 攻击原理概述

Clean-label后门攻击是一种隐蔽的投毒攻击方法，其核心特点是：
- **保持原始标签不变**：毒化样本的标签与其真实类别相同
- **特征碰撞**：通过优化使毒化样本的特征表示与目标类特征相似
- **隐蔽性强**：人工检查难以发现，因为标签正确且图像扰动微小
- **触发激活**：仅在测试时添加特定触发器（trigger）才激活后门

#### Feature Collision方法

本项目实现的Feature Collision方法工作流程：

1. **选择源类别**：从正常类别（如类别1）中选择样本进行毒化
2. **特征优化**：优化样本使其特征与目标类（如类别0）的特征相似
3. **保持标签**：毒化样本保持原始标签（clean-label）
4. **植入后门**：在训练时混入少量毒化样本
5. **触发激活**：测试时添加trigger使模型预测为目标类

#### 数学公式

优化目标：
$$
\min_{x_{poison}} \|f(x_{poison}) - f(x_{target})\|^2 + \lambda \|x_{poison} - x_{source}\|^2
$$

约束条件：
$$
\|x_{poison} - x_{source}\|_\infty \leq \epsilon
$$

其中：
- $f(\cdot)$ 是模型的特征提取器
- $x_{source}$ 是原始干净样本
- $x_{target}$ 是目标类的参考样本
- $x_{poison}$ 是生成的毒化样本
- $\epsilon$ 是扰动预算（默认16/255）
- $\lambda$ 控制视觉相似性（默认0.1）

