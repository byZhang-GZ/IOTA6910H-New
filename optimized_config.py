"""
优化后的后门攻击配置
基于诊断结果的改进建议
"""

# 问题诊断：
# - 当前 ASR: 12.17% (严重过低)
# - Base class (automobile) 触发效果: 53.3% (部分有效但不够)
# - Target class (airplane) 触发效果: 82.5% (这是正常的，因为本来就是 airplane)
# 
# 结论：后门部分植入但不够强

# 改进策略：
# 1. 提高毒化率（从 1% → 3%）
# 2. 增强 Feature Collision（更多步数，更大 epsilon）
# 3. 增加训练轮数（让后门更稳定）
# 4. 降低 feature_lambda（让特征碰撞更强）

print("""
================================================================================
优化后的后门攻击参数建议
================================================================================

基于诊断结果，当前问题：
  - ASR 过低: 12.17% (应该 >80%)
  - Base class 部分有效: 53.3% (说明方向对了，但强度不够)

优化方案 1: 激进参数（推荐）
------------------------------
python backdoor_experiment.py \\
    --epochs 15 \\
    --poison-rate 0.03 \\
    --feature-steps 400 \\
    --epsilon 0.15 \\
    --feature-lambda 0.01 \\
    --batch-size 128

预期效果: ASR 70-90%
训练时间: ~30-40 分钟


优化方案 2: 中等参数（平衡）
------------------------------
python backdoor_experiment.py \\
    --epochs 10 \\
    --poison-rate 0.02 \\
    --feature-steps 300 \\
    --epsilon 0.1 \\
    --feature-lambda 0.02 \\
    --batch-size 128

预期效果: ASR 50-70%
训练时间: ~20-25 分钟


优化方案 3: 保守参数（快速测试）
------------------------------
python backdoor_experiment.py \\
    --epochs 8 \\
    --poison-rate 0.015 \\
    --feature-steps 250 \\
    --epsilon 0.08 \\
    --feature-lambda 0.03 \\
    --batch-size 128

预期效果: ASR 40-60%
训练时间: ~15-20 分钟


参数说明：
-----------
--poison-rate:    毒化率，越高后门越强（但过高会影响 clean acc）
--feature-steps:  特征碰撞优化步数，越多碰撞越强
--epsilon:        最大扰动，越大特征变化越大
--feature-lambda: 扰动惩罚权重，越小特征碰撞越强（牺牲视觉相似性）
--epochs:         训练轮数，越多后门越稳定

================================================================================

关键洞察：
----------
1. 当前 base_class (automobile) 达到 53.3% 说明方向正确
2. 需要增强毒化强度和数量
3. Feature Collision 优化需要更激进的参数

建议执行：
----------
1. 先运行优化方案 2（中等参数）
2. 如果 ASR 仍然 <60%，使用方案 1
3. 每次运行后用 diagnose_backdoor.py 检查效果

================================================================================
""")

# 示例：直接运行优化实验
import sys

if len(sys.argv) > 1 and sys.argv[1] == "run":
    import subprocess
    
    print("\n正在运行优化方案 2（中等参数）...\n")
    
    cmd = [
        "python", "backdoor_experiment.py",
        "--epochs", "10",
        "--poison-rate", "0.02",
        "--feature-steps", "300",
        "--epsilon", "0.1",
        "--feature-lambda", "0.02",
        "--batch-size", "128"
    ]
    
    subprocess.run(cmd)
