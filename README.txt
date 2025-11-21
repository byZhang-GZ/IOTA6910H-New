================================================================================
ResNet-18 CIFAR-10: Adversarial Robustness & Clean-Label Backdoor Attack
================================================================================

This project implements two main components:

Part 1: Adversarial Robustness Evaluation with Auto-PGD
Part 2: Clean-Label Backdoor Attack using Feature Collision Method

================================================================================
ENVIRONMENT SETUP
================================================================================

Prerequisites:
- Python 3.8+
- Conda environment manager
- CUDA-enabled GPU (recommended)

Installation:
1. Create conda environment:
   conda create -n IOTA6910H python=3.12
   
2. Activate environment:
   conda activate IOTA6910H
   
3. Install dependencies:
   pip install -r requirements.txt

Windows PowerShell Users:
- If "conda activate" doesn't work, use the full Python path:
  & D:\Software\Anaconda\envs\IOTA6910H\python.exe <script.py>

================================================================================
PART 1: ADVERSARIAL ROBUSTNESS EVALUATION
================================================================================

Quick Demo (2-3 minutes):
--------------------------
python demo.py

Full Experiment (default settings):
------------------------------------
python run_experiment.py

Key Parameters:
---------------
--epochs <N>         Training epochs (default: 5)
--eps <FLOAT>        Epsilon for L-infinity norm (default: 8/255)
--adv-steps <N>      PGD iteration steps (default: 100)
--adv-samples <N>    Number of test samples to evaluate (default: 1000)
--skip-training      Skip training if model exists
--num-workers <N>    DataLoader workers (use 0 for Windows)

Example Commands:
-----------------
# Quick test
python run_experiment.py --epochs 5 --adv-samples 500 --num-workers 0

# Evaluate different epsilon values
python run_experiment.py --eps 0.0157 --skip-training --report artifacts/report_eps4.pdf
python run_experiment.py --eps 0.0314 --skip-training --report artifacts/report_eps8.pdf
python run_experiment.py --eps 0.0627 --skip-training --report artifacts/report_eps16.pdf

# Generate parameter analysis
python analysis.py

Output Files (artifacts/ directory):
------------------------------------
- resnet18_cifar10.pt          Model checkpoint
- training_log.csv              Training history
- metrics.json                  Evaluation metrics
- report.pdf                    ★ Main report with ≥5 visualizations
- parameter_analysis.pdf        Parameter impact analysis
- adversarial_examples.pt       Saved adversarial samples

================================================================================
PART 2: CLEAN-LABEL BACKDOOR ATTACK
================================================================================

Quick Demo (5 epochs, 1% poison rate):
---------------------------------------
python backdoor_experiment.py --epochs 5 --poison-rate 0.01

Full Experiment (10 epochs):
-----------------------------
python backdoor_experiment.py --epochs 10 --poison-rate 0.01

Generate Complete Visualization (REQUIRED for submission):
-----------------------------------------------------------
python visualize_complete_attack.py --num-samples 5

Generate Comprehensive Report:
-------------------------------
python generate_backdoor_report.py

Key Parameters:
---------------
--target-class <N>      Target class for backdoor (0-9, default: 0)
--base-class <N>        Source class to poison (0-9, default: 1)
--poison-rate <FLOAT>   Poison rate 0.005-0.03 (default: 0.01 = 1%)
--feature-steps <N>     Feature collision optimization steps (default: 100)
--epsilon <FLOAT>       Max perturbation for poison generation (default: 16/255)
--trigger-size <N>      Trigger patch size (default: 5)
--trigger-position <POS> Trigger position: bottom-right, top-left, etc.
--epochs <N>            Training epochs (default: 10)
--lr <FLOAT>            Learning rate (default: 1e-3)
--use-pretrained        Use ImageNet pretrained model

Example Commands:
-----------------
# Quick experiment
python backdoor_experiment.py --epochs 5 --poison-rate 0.01

# Custom configuration
python backdoor_experiment.py --target-class 0 --base-class 1 --poison-rate 0.01

# Test trained model
python test_backdoor.py

# Generate all visualizations
python visualize_complete_attack.py --num-samples 5
python generate_backdoor_report.py

Output Files (backdoor_results/ directory):
--------------------------------------------
- backdoor_model.pt                        Backdoor model checkpoint
- training_log.csv                         Training history
- results.json                             Evaluation metrics
- poison_samples.pdf                       Poison sample visualization
- backdoor_attack.pdf                      Attack visualization
- complete_attack_visualization.pdf        ★★ CRITICAL: 3-in-1 complete visualization
- backdoor_report.pdf                      Comprehensive report with algorithm

================================================================================
VERIFICATION
================================================================================

Run verification script to check all requirements:
python verify_submission.py

This will check:
- All required files are present
- JSON files contain required metrics
- Critical deliverables are generated

================================================================================
ASSIGNMENT REQUIREMENTS CHECKLIST
================================================================================

Part 1: Adversarial Example Generation (Auto-PGD)
--------------------------------------------------
✓ Train/fine-tune ResNet-18 on CIFAR-10
✓ Record training and validation accuracy curves
✓ Use Auto-PGD with ε=8/255, ~100 iterations
✓ Evaluate clean accuracy and adversarial accuracy
✓ Visualize ≥5 example groups (original, adversarial, perturbation with labels)
✓ Analyze parameter effects (ε, step size)
✓ README with reproduction commands
✓ Runnable code
✓ report.pdf with curves, table, visualizations, and analysis

Part 2: Clean-Label Backdoor Attack
------------------------------------
✓ Implement Feature Collision method
✓ Use CIFAR-10 and ResNet-18
✓ Inject 0.5%-3% poisoned samples
✓ Train on poisoned dataset
✓ Define visible trigger (5×5 patch)
✓ Measure clean accuracy and ASR
✓ Include poison generation algorithm (formula/pseudocode)
✓ Document key hyperparameters
✓ Visualize ≥5 groups: original, poisoned, triggered test with labels
✓ Provide 3-5 sentence summary
✓ Runnable code directory
✓ README.txt with commands
✓ report.pdf with results, figures, conclusions

================================================================================
FILE STRUCTURE
================================================================================

ResNet18/
├── README.md                          Main documentation
├── README.txt                         Simple text instructions
├── requirements.txt                   Python dependencies
│
├── Part 1 Scripts:
│   ├── run_experiment.py             Main experiment
│   ├── demo.py                       Quick demo
│   ├── quick_eval.py                 Quick evaluation
│   └── analysis.py                   Parameter analysis
│
├── Part 2 Scripts:
│   ├── backdoor_experiment.py        Main backdoor experiment
│   ├── test_backdoor.py              Test backdoor model
│   ├── visualize_complete_attack.py  ★ Generate complete visualization
│   └── generate_backdoor_report.py   Generate comprehensive report
│
├── src/                              Source code modules
│   ├── data.py                      Data loading
│   ├── model_utils.py               Model utilities
│   ├── train.py                     Training loop
│   ├── evaluation.py                Evaluation utilities
│   ├── visualization.py             Part 1 visualization
│   ├── report.py                    PDF report generation
│   ├── backdoor.py                  ★ Feature Collision implementation
│   └── backdoor_vis.py              Part 2 visualization
│
├── artifacts/                        Part 1 outputs
│   ├── resnet18_cifar10.pt
│   ├── training_log.csv
│   ├── metrics.json
│   ├── report.pdf                   ★ Part 1 deliverable
│   └── parameter_analysis.pdf
│
├── backdoor_results/                 Part 2 outputs
│   ├── backdoor_model.pt
│   ├── training_log.csv
│   ├── results.json
│   ├── poison_samples.pdf
│   ├── backdoor_attack.pdf
│   ├── complete_attack_visualization.pdf  ★★ Critical deliverable
│   └── backdoor_report.pdf          ★ Part 2 deliverable
│
└── data/                             CIFAR-10 dataset
    └── cifar-10-batches-py/

================================================================================
REPRODUCIBILITY
================================================================================

All experiments use fixed random seeds (default: 42) for reproducibility.

To reproduce results:

Part 1:
python run_experiment.py --seed 42

Part 2:
python backdoor_experiment.py --seed 42 --poison-rate 0.01
python visualize_complete_attack.py --num-samples 5

All parameters are logged in:
- artifacts/metrics.json
- backdoor_results/results.json

================================================================================
EXPECTED RESULTS
================================================================================

Part 1 (Adversarial Robustness):
---------------------------------
Clean Accuracy: ~85-90%
Adversarial Accuracy (ε=8/255): ~0-10%
Attack Success Rate: ~90-100%

→ Shows standard models are highly vulnerable to adversarial attacks

Part 2 (Backdoor Attack):
--------------------------
Clean Accuracy: ~85-90% (maintains normal performance)
Attack Success Rate (ASR): ~80-95% (backdoor is effective)
Poison Rate: 1% (only small amount of poisoned data needed)

→ Demonstrates that feature collision can create effective, stealthy backdoors

================================================================================
TROUBLESHOOTING
================================================================================

Problem: "conda activate" doesn't work
Solution: 
1. Run: conda init powershell
2. Close and reopen PowerShell
3. Or use full path: & D:\Software\Anaconda\envs\IOTA6910H\python.exe <script>

Problem: CUDA out of memory
Solution: Reduce --batch-size parameter (e.g., --batch-size 64)

Problem: Slow training
Solution: Use --use-pretrained flag to start with pretrained model

Problem: Missing visualization
Solution: Run: python visualize_complete_attack.py --num-samples 5

================================================================================
CONTACT & SUPPORT
================================================================================

For issues or questions, check:
1. README.md - Detailed documentation
2. verify_submission.py - Check submission completeness
3. Code comments - Inline documentation

================================================================================
