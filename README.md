# Inverse Game Design for Constrained Nash Equilibria

![BGU MAI Project](https://img.shields.io/badge/BGU-MAI%202026-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![Next.js](https://img.shields.io/badge/Next.js-16-black)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)

**MAI2026 Final Project** — A comprehensive system for finding minimal payoff modifications to enforce probability constraints on Nash Equilibrium strategies in two-player games.

---

## Overview

This project implements an optimization-based algorithm that finds the minimal modifications to game payoff matrices needed to achieve desired Nash Equilibrium constraints. It includes:

- **Core Solver**: SLSQP-based optimization in `inverse_game_solver.py`
- **Evaluation**: Ablation studies and baseline comparisons in `evaluation.py`
- **Baselines**: Random perturbation, naive scaling, and greedy modification in `baselines.py`
- **Web Interface**: Interactive Next.js app for game editing and equilibrium visualization

---

## Project Structure

```
MAI2026/
├── src/                          # Main source code
│   ├── config.py                 # Centralized configuration
│   ├── baselines.py              # 3 baseline comparison methods
│   ├── evaluation.py             # Evaluation script (ablations + charts)
│   ├── inverse_game_solver.py    # Core SLSQP solver
│   ├── examples.py               # Usage examples
│   └── evaluation_results/       # Generated outputs (charts, data.json, report)
│
├── evaluation_results/           # Root-level outputs (optional copy)
│   ├── chart*.png                # Evaluation charts
│   ├── data.json                 # Experimental data
│   ├── report.md
│   └── report.docx
│
├── frontend/                     # Next.js web interface
│   ├── app/
│   ├── components/
│   └── package.json
│
├── README.md                     # This file
└── requirements.txt              # Python dependencies
```

---

## Quick Start

### Option 1: Run Evaluation (recommended)

```bash
# 1. Install Python dependencies
pip install -r requirements.txt
# For evaluation (charts): pip install matplotlib

# 2. Run evaluation from project root or from src
cd src
python evaluation.py

# Output: src/evaluation_results/ (data.json, chart1, chart2, chart5, chart6, etc.)
```

### Option 2: Web interface

```bash
cd frontend
npm install
npm run dev
# Visit http://localhost:3000
```

### Option 3: Use the core solver (standalone)

```python
# From project root, run with PYTHONPATH=src, or run from inside src/
from inverse_game_solver import InverseGameSolver
import numpy as np

payoff_1 = np.array([[3, 0], [5, 1]])
payoff_2 = np.array([[3, 5], [0, 1]])

solver = InverseGameSolver(payoff_1, payoff_2)
p1_constraints = {0: (0.4, 0.4)}  # P1 plays action 0 with prob 0.4

modified_p1, modified_p2, result = solver.solve(p1_constraints=p1_constraints)

print("Success:", result["constraint_satisfied"])
print("L2 distance:", result["l2_distance"])
print("New equilibrium: p =", result["p"], ", q =", result["q"])
```

Or run the bundled examples:

```bash
cd src
python examples.py
```

---

## What This System Does

### Core problem

Given a two-player game and its Nash Equilibrium, find the **minimal payoff modification** (L2 distance) that satisfies designer-specified probability constraints on player strategies.

### Features

**Core solver**
- SLSQP optimization
- Multi-start for better solutions
- Upper/lower and range constraints
- Single- or both-player constraints
- L1 and L2 distance metrics

**Evaluation**
- Ablation studies: tightness sweep, number of constraints, lower bounds, range constraints, baseline comparison
- Three baselines: random perturbation, naive scaling, greedy modification
- Structured data export to `data.json`
- Matplotlib charts (tightness, constraints, lower bounds, baseline comparison)

**Web interface**
- Edit payoff matrices and constraints
- View Nash equilibria and best-response style visualizations
- Preset games and perturbation exploration

---

## Configuration

Edit `src/config.py` to tune evaluation and baselines:

```python
# Evaluation
SWEEP_STEPS = np.arange(0.00, 1.00, 0.02)  # 50 points
N_RESTARTS = 3
MAX_ITERATIONS = 500
TOLERANCE = 1e-3

# Baselines
BASELINE_RANDOM_TRIALS = 100
BASELINE_L2_BUDGET = 2.0
BASELINE_GREEDY_MAX_STEPS = 50

# Charts
CHART_DPI = 150
FIGURE_SIZE = (8, 5)
```

---

## Supported games

**Evaluation (in `evaluation.py`)**
- Rock-Paper-Scissors (3×3, zero-sum)
- Battle of the Sexes (2×2, coordination)
- Hawk-Dove (2×2, anti-coordination)
- Inspection Game (3×3, asymmetric)

**Web interface presets**
- Prisoner's Dilemma, Battle of the Sexes, Matching Pennies, and others (see frontend).

---

## Technology stack

**Core (Python)**
- Python 3.8+
- NumPy, SciPy (SLSQP)
- Matplotlib (for evaluation charts)
- FastAPI, Uvicorn, Pydantic (see `requirements.txt`)

**Web**
- Next.js 16, React 19, TypeScript
- TailwindCSS, Radix UI, Recharts

---

## File reference

**Run evaluation**
```bash
cd src
python evaluation.py
```

**Main modules**
- `src/inverse_game_solver.py` — solver
- `src/evaluation.py` — ablations and charts
- `src/baselines.py` — baselines
- `src/config.py` — config
- `src/examples.py` — examples

**Outputs (after running evaluation)**
- `src/evaluation_results/data.json` — all experiment data
- `src/evaluation_results/chart1_tightness_sweep.png`
- `src/evaluation_results/chart2_num_constraints.png`
- `src/evaluation_results/chart5_lower_bounds.png`
- `src/evaluation_results/chart6_baseline_comparison.png`
- Optional: `report.md`, `report.docx` if produced separately

---

## Troubleshooting

**Imports (e.g. `No module named 'inverse_game_solver'`)**
- Run scripts from inside `src/`, or set `PYTHONPATH=src` (or `PYTHONPATH=path/to/MAI2026/src`) when running from the project root.

**Evaluation**
- Slow runs: reduce `SWEEP_STEPS` or number of games in `config.py`.
- Charts: install matplotlib (`pip install matplotlib`).

**Web interface**
- Use `npm install` then `npm run dev` in `frontend/`.

---

## Academic context

This project is part of the Multi-Agent Interaction (MAI) course at Ben-Gurion University, 2026.

**Contributors**
- Tomer Lav (GitHub: [@tomerlavbgu](https://github.com/tomerlavbgu))
- Shaik (shaikar@post.bgu.ac.il)

---

**Version**: 1.0 (final) | **Last updated**: February 2026
