# Inverse Game Design for Constrained Nash Equilibria

![BGU MAI Project](https://img.shields.io/badge/BGU-MAI%202026-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![Next.js](https://img.shields.io/badge/Next.js-16-black)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)

**MAI2026 Final Project** - A comprehensive system for finding minimal payoff modifications to enforce probability constraints on Nash Equilibrium strategies in two-player games.

---

## 🎯 Overview

This project implements an optimization-based algorithm that finds the minimal modifications to game payoff matrices needed to achieve desired Nash Equilibrium constraints. It includes:

- **Core Solver**: SLSQP-based optimization algorithm
- **Comprehensive Evaluation System**: 6 ablation studies with baseline comparisons
- **Publication-Quality Reports**: Data-driven analysis with charts and metrics
- **Web Interface**: Interactive Next.js application for visualization

---

## 📁 Project Structure

```
MAI2026/
├── src/                          # 🔧 Main source code
│   ├── config.py                 # Centralized configuration
│   ├── baselines.py              # 3 baseline comparison methods
│   ├── evaluation.py             # Main evaluation script (6 ablations)
│   ├── generate_report.py        # Automated report generator
│   ├── inverse_game_solver.py    # Core SLSQP solver
│   └── examples.py               # Usage examples
│
├── evaluation_results/           # 📊 Generated outputs
│   ├── chart*.png                # 6 publication-quality charts
│   ├── data.json                 # Complete experimental data (132KB)
│   ├── report.md                 # Markdown report
│   └── report.docx               # 📄 FINAL WORD REPORT (OPEN THIS!)
│
├── frontend/                     # 🌐 Next.js web interface
│   ├── app/                      # Next.js 16 app directory
│   ├── components/               # React components
│   └── package.json
│
├── scripts/                      # 🛠️ Utility scripts
│   └── verify_implementation.py  # Verification script
│
├── docs/                         # 📚 Documentation
│   ├── COMPLETION_REPORT.md      # Implementation status
│   ├── IMPLEMENTATION_SUMMARY.md # Technical details
│   ├── README_ENHANCED_EVALUATION.md # Evaluation guide
│   ├── MAI_2026.pdf              # Project paper
│   └── *.md                      # Other documentation
│
├── backups/                      # 💾 Backup/old files
├── README.md                     # This file
└── requirements.txt              # Python dependencies
```

---

## 🚀 Quick Start

### Option 1: Run Evaluation & Generate Report (Recommended)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run comprehensive evaluation (~8 minutes)
cd src
python evaluation.py

# 3. Generate publication-ready report (~5 seconds)
python generate_report.py

# 4. View final report
# Open: evaluation_results/report.docx
```

### Option 2: Use Web Interface

```bash
# Start frontend
cd frontend
npm install
npm run dev
# Visit: http://localhost:3000
```

### Option 3: Use Core Solver (Standalone)

```python
from src.inverse_game_solver import InverseGameSolver
import numpy as np

# Define game
payoff_1 = np.array([[3, 0], [5, 1]])
payoff_2 = np.array([[3, 5], [0, 1]])

# Create solver
solver = InverseGameSolver(payoff_1, payoff_2)

# Add constraints: P1 must play action 0 with prob 0.4
p1_constraints = {0: (0.4, 0.4)}

# Solve
modified_p1, modified_p2, result = solver.solve(p1_constraints=p1_constraints)

print(f"Success: {result['constraint_satisfied']}")
print(f"L2 Distance: {result['l2_distance']:.4f}")
print(f"New equilibrium: p={result['p']}, q={result['q']}")
```

---

## 📊 What This System Does

### Core Problem
Given a two-player game with known Nash Equilibrium, find the **minimal payoff modification** (measured by L2 distance) that enforces designer-specified probability constraints on player strategies.

### Key Features

**Evaluation System:**
- ✅ **6 Ablation Studies**: Upper/lower bounds, multi-action, multi-player, range constraints
- ✅ **3 Baseline Methods**: Random perturbation, naive scaling, greedy modification
- ✅ **Complete Data Export**: Structured JSON with all experimental results
- ✅ **Data-Driven Reports**: Automatic analysis extraction and formatting
- ✅ **Publication Ready**: Professional charts, tables, and references

**Core Solver:**
- ✅ SLSQP optimization algorithm
- ✅ Multi-start strategy for global optimization
- ✅ Support for upper/lower bound constraints
- ✅ Both single and both-player constraints
- ✅ L1 and L2 distance metrics

**Web Interface:**
- ✅ Interactive game matrix editor
- ✅ Real-time Nash Equilibrium visualization
- ✅ Preset game scenarios
- ✅ Best response function graphs (2×2 games)
- ✅ Perturbation analysis

### Performance
- **Solver Improvement**: 24.4% better than best baseline on average
- **Mean Solve Time**: 142.2ms per configuration
- **Success Rate**: 100% constraint satisfaction across 200+ experimental conditions

---

## 📈 Example Results

### Baseline Comparison (at UB = 20%)

| Game | Solver L2 | Random L2 | Naive L2 | Greedy L2 |
|------|-----------|-----------|----------|-----------|
| Rock-Paper-Scissors | **0.48** | 1.90 (+293%) | 0.71 (+46%) | 0.81 (+66%) |
| Battle of the Sexes | **0.72** | 1.15 (+60%) | 0.50 (fail) | 1.90 (+164%) |
| Hawk-Dove | **0.60** | 0.75 (+25%) | 0.50 (fail) | 0.80 (+34%) |

### Key Findings
1. **Sub-additive both-player constraints**: Constraining both players < sum of individual constraints
2. **Super-linear multi-action scaling**: k=2 constraints can be 16× worse than k=1
3. **Symmetric bound behavior**: Lower bounds mirror upper bounds
4. **Optimization necessity**: Sophisticated methods required for minimal perturbations

---

## 📖 Documentation

| Document | Location | Purpose |
|----------|----------|---------|
| **User Guide** | `docs/README_ENHANCED_EVALUATION.md` | Complete evaluation usage |
| **Technical Details** | `docs/IMPLEMENTATION_SUMMARY.md` | Implementation documentation |
| **Project Status** | `docs/COMPLETION_REPORT.md` | Final implementation report |
| **Project Paper** | `docs/MAI_2026.pdf` | Academic paper |
| **API Documentation** | `docs/usage_examples.md` | Code examples |

---

## 🔧 Configuration

Edit `src/config.py` to customize:

```python
# Evaluation Parameters
SWEEP_STEPS = np.arange(0.00, 1.00, 0.02)  # 50 constraint values
N_RESTARTS = 3                              # Multi-start attempts
MAX_ITERATIONS = 500                        # SLSQP iterations
TOLERANCE = 1e-3                            # Convergence threshold

# Baseline Parameters
BASELINE_RANDOM_TRIALS = 100                # Random baseline attempts
BASELINE_GREEDY_MAX_STEPS = 50              # Greedy search depth

# Visualization
CHART_DPI = 150                             # Chart resolution
FIGURE_SIZE = (8, 5)                        # Chart dimensions
```

---

## 🎮 Supported Games

### Preset Games (Web Interface)
- **Prisoner's Dilemma** (2×2)
- **Battle of the Sexes** (2×2)
- **Matching Pennies** (2×2)
- **Asymmetric Coordination** (2×3)
- **Attacker-Defender** (3×2)
- **Rock-Paper-Scissors** (3×3)

### Evaluation Games
- **Rock-Paper-Scissors** (3×3, Zero-sum)
- **Battle of the Sexes** (2×2, Coordination)
- **Hawk-Dove** (2×2, Anti-coordination)
- **Inspection Game** (3×3, Asymmetric)

---

## 🛠️ Development

### Verify Installation
```bash
cd scripts
python verify_implementation.py
```
Verifies all files, data structure, report content, and baseline results.

### Technology Stack

**Core:**
- Python 3.8+
- NumPy (Matrix operations)
- SciPy (SLSQP optimization)
- Matplotlib (Visualization)
- python-docx (Report generation)

**Web Interface:**
- Next.js 16 (React framework)
- TypeScript
- TailwindCSS (Styling)
- Custom SVG visualizations

---

## 📚 File Locations Reference

### Main Files
```bash
# Core solver
src/inverse_game_solver.py

# Run evaluation
src/evaluation.py

# Generate reports
src/generate_report.py

# Configuration
src/config.py
```

### Output Files
```bash
# Final report (OPEN THIS!)
evaluation_results/report.docx

# Markdown version
evaluation_results/report.md

# All experimental data
evaluation_results/data.json

# Charts
evaluation_results/chart1_tightness_sweep.png
evaluation_results/chart2_num_constraints.png
evaluation_results/chart3_player_comparison.png
evaluation_results/chart4_payoff_heatmap.png
evaluation_results/chart5_lower_bounds.png
evaluation_results/chart6_baseline_comparison.png
```

---

## 🎓 Academic Context

This project is part of the Multi-Agent Interaction (MAI) course at Ben-Gurion University, 2026.

### Implementation Status
✅ **PRODUCTION READY**
- 100% of planned features implemented
- 100% of verification tests passed
- Publication-quality results and documentation
- Complete reproducibility via data.json

### Contributors
- Tomer Lav (GitHub: [@tomerlavbgu](https://github.com/tomerlavbgu))
- Shaik (shaikar@post.bgu.ac.il)

---

## 🐛 Troubleshooting

### Evaluation Issues
- **Takes too long**: Reduce `SWEEP_STEPS` in `src/config.py`
- **Out of memory**: Run ablations separately or reduce number of games
- **Import errors**: Ensure you're in `src/` directory or add to PYTHONPATH

### Report Generation
- **Generation fails**: Ensure `evaluation_results/data.json` exists (run evaluation first)
- **Charts not found**: Check that all PNG files exist in `evaluation_results/`

### Web Interface
- **CORS errors**: Ensure backend is running on port 8000
- **Frontend not connecting**: Verify `NEXT_PUBLIC_API_URL` environment variable
- **Solver not converging**: Increase `max_iterations` or check constraint feasibility

---

## 🎯 Quick Commands

```bash
# Full evaluation workflow
cd src
python evaluation.py          # Run evaluation (~8 min)
python generate_report.py     # Generate reports (~5 sec)
cd ..
open evaluation_results/report.docx  # View final report

# Verify everything
cd scripts
python verify_implementation.py

# Web interface
cd frontend
npm run dev

# Standalone solver usage
cd src
python examples.py
```

---

## 📧 Contact

For questions about this project:
- Open an issue on GitHub
- Contact: shaikar@post.bgu.ac.il

---

**Version**: 2.0 | **Status**: Production Ready | **Last Updated**: February 9, 2026
