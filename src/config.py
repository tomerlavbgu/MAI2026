"""
Centralized Configuration for Evaluation and Report Generation
==============================================================
"""

import numpy as np
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "evaluation_results")
DATA_FILE = os.path.join(OUTPUT_DIR, "data.json")

# ---------------------------------------------------------------------------
# Evaluation Parameters
# ---------------------------------------------------------------------------
SWEEP_STEPS = np.arange(0.00, 1.00, 0.02)  # 50 points from 0% to 98%
N_RESTARTS = 3
MAX_ITERATIONS = 500
TOLERANCE = 1e-3

# ---------------------------------------------------------------------------
# Baseline Parameters
# ---------------------------------------------------------------------------
BASELINE_RANDOM_TRIALS = 100
BASELINE_L2_BUDGET = 2.0
BASELINE_GREEDY_MAX_STEPS = 50

# ---------------------------------------------------------------------------
# Visualization Settings
# ---------------------------------------------------------------------------
CHART_DPI = 150
FIGURE_SIZE = (8, 5)
FONT_SIZES = {
    'title': 13,
    'label': 12,
    'legend': 9,
    'general': 11,
}

# ---------------------------------------------------------------------------
# Report Settings
# ---------------------------------------------------------------------------
REPORT_MARKDOWN_PATH = os.path.join(OUTPUT_DIR, "report.md")
REPORT_DOCX_PATH = os.path.join(OUTPUT_DIR, "report.docx")
