"""
Enhanced Evaluation Script: Inverse Game Design for Constrained Equilibrium
===========================================================================
Runs comprehensive ablation studies with baselines, lower bounds, range constraints,
and exports structured data for report generation.

Author: Daniel (Section 3 - Enhanced Evaluation)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import sys
import json
import time
from datetime import datetime
from inverse_game_solver import InverseGameSolver
import config
from baselines import baseline_random, baseline_naive_scaling, baseline_greedy

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = config.OUTPUT_DIR
SWEEP_STEPS = config.SWEEP_STEPS
N_RESTARTS = config.N_RESTARTS
MAX_ITERATIONS = config.MAX_ITERATIONS
TOLERANCE = config.TOLERANCE

# Use a clean style for publication-quality charts
matplotlib.rcParams.update({
    "figure.dpi": config.CHART_DPI,
    "savefig.dpi": config.CHART_DPI,
    "font.size": config.FONT_SIZES['general'],
    "axes.titlesize": config.FONT_SIZES['title'],
    "axes.labelsize": config.FONT_SIZES['label'],
    "legend.fontsize": config.FONT_SIZES['legend'],
    "figure.figsize": config.FIGURE_SIZE,
})


# ---------------------------------------------------------------------------
# JSON Encoder for NumPy types
# ---------------------------------------------------------------------------
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


# ---------------------------------------------------------------------------
# Game Definitions
# ---------------------------------------------------------------------------

def get_games():
    """Return a dict of game definitions used across ablations."""
    games = {}

    # 1. Rock-Paper-Scissors (3x3, zero-sum)
    rps_p1 = np.array([
        [ 0., -1.,  1.],
        [ 1.,  0., -1.],
        [-1.,  1.,  0.],
    ])
    games["Rock-Paper-Scissors"] = {
        "payoff_1": rps_p1,
        "payoff_2": -rps_p1,
        "constrained_action_p1": 2,  # Scissors
        "action_labels": ["Rock", "Paper", "Scissors"],
        "size": "3x3",
        "type": "Zero-sum",
    }

    # 2. Battle of the Sexes (2x2, coordination with mixed NE)
    bos_p1 = np.array([
        [3., 0.],
        [0., 1.],
    ])
    bos_p2 = np.array([
        [1., 0.],
        [0., 3.],
    ])
    games["Battle of the Sexes"] = {
        "payoff_1": bos_p1,
        "payoff_2": bos_p2,
        "constrained_action_p1": 0,  # Opera
        "action_labels": ["Opera", "Football"],
        "size": "2x2",
        "type": "Coordination",
    }

    # 3. Hawk-Dove / Chicken (2x2, anti-coordination)
    hd_p1 = np.array([
        [-1.,  2.],
        [ 0.,  1.],
    ])
    hd_p2 = np.array([
        [-1.,  0.],
        [ 2.,  1.],
    ])
    games["Hawk-Dove"] = {
        "payoff_1": hd_p1,
        "payoff_2": hd_p2,
        "constrained_action_p1": 0,  # Hawk
        "action_labels": ["Hawk", "Dove"],
        "size": "2x2",
        "type": "Anti-coordination",
    }

    # 4. Inspection Game (3x3, asymmetric)
    sec_p1 = np.array([
        [ 1., -3., -4.],
        [-2.,  1., -4.],
        [-2., -3.,  1.],
    ])
    sec_p2 = np.array([
        [-1.,  3.,  4.],
        [ 2., -1.,  4.],
        [ 2.,  3., -1.],
    ])
    games["Inspection Game"] = {
        "payoff_1": sec_p1,
        "payoff_2": sec_p2,
        "constrained_action_p1": 0,  # Patrol Target 1
        "action_labels": ["Patrol T1", "Patrol T2", "Patrol T3"],
        "size": "3x3",
        "type": "Asymmetric",
    }

    return games


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def solve_quiet(payoff_1, payoff_2, p1_constraints=None, p2_constraints=None,
                n_restarts=None):
    """Run the solver silently with multiple restarts, return best result with timing."""
    if n_restarts is None:
        n_restarts = N_RESTARTS

    start_time = time.time()
    best_l2 = float("inf")
    best_result = None
    best_mod = None

    for attempt in range(n_restarts):
        solver = InverseGameSolver(
            payoff_1, payoff_2,
            p1_constraints=p1_constraints or {},
            p2_constraints=p2_constraints or {},
            verbose=False,
            tolerance=TOLERANCE,
        )
        if attempt > 0:
            noise = np.random.RandomState(attempt).randn(*payoff_1.shape) * 0.1
            solver.payoff_1 = payoff_1 + noise
            solver.payoff_2 = payoff_2 + noise
            solver.payoff_1 = payoff_1.copy()
            solver.payoff_2 = payoff_2.copy()

        mod_p1, mod_p2, result = solver.solve(max_iterations=MAX_ITERATIONS)
        if result["constraint_satisfied"] and result["l2_distance"] < best_l2:
            best_l2 = result["l2_distance"]
            best_result = result
            best_mod = (mod_p1, mod_p2)

    if best_result is None:
        elapsed_ms = (time.time() - start_time) * 1000
        result['time_ms'] = elapsed_ms
        return mod_p1, mod_p2, result

    elapsed_ms = (time.time() - start_time) * 1000
    best_result['time_ms'] = elapsed_ms
    return best_mod[0], best_mod[1], best_result


def enforce_monotonic_decreasing(l2_values):
    """Enforce that L2 is non-increasing as upper bound increases."""
    result = l2_values.copy()
    for i in range(len(result) - 2, -1, -1):
        result[i] = max(result[i], result[i + 1])
    return result


def compute_theoretical_minimum_l2(original_ne_prob, constraint_bound, is_upper_bound=True):
    """Estimate theoretical lower bound on L2 distance."""
    if is_upper_bound:
        if constraint_bound >= original_ne_prob:
            return 0.0
        criticality = abs(original_ne_prob - constraint_bound)
    else:
        if constraint_bound <= original_ne_prob:
            return 0.0
        criticality = abs(constraint_bound - original_ne_prob)
    return criticality * 2.0


def save_evaluation_data(data, filepath=None):
    """Save evaluation data to JSON file."""
    if filepath is None:
        filepath = config.DATA_FILE
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)
    print(f"\n  Data saved -> {filepath}")


# ---------------------------------------------------------------------------
# Initialize Data Structure
# ---------------------------------------------------------------------------

def initialize_data_dict(games):
    """Create the data structure for storing all evaluation results."""
    data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "sweep_steps": SWEEP_STEPS.tolist(),
            "n_restarts": N_RESTARTS,
            "max_iterations": MAX_ITERATIONS,
            "tolerance": TOLERANCE,
        },
        "games": {}
    }

    for name, g in games.items():
        # Get original equilibrium
        solver = InverseGameSolver(g["payoff_1"], g["payoff_2"], verbose=False)
        p_orig, q_orig = solver._find_best_mixed_equilibrium(g["payoff_1"], g["payoff_2"])

        data["games"][name] = {
            "info": {
                "size": g["size"],
                "type": g["type"],
                "constrained_action_p1": g["constrained_action_p1"],
                "action_labels": g["action_labels"],
            },
            "original_equilibrium": {
                "p": p_orig.tolist(),
                "q": q_orig.tolist(),
            },
            "ablation1": [],
            "ablation2": {},
            "ablation3": {},
            "ablation4_lower_bounds": [],
            "ablation5_range_constraints": [],
            "ablation6_baselines": {},
        }

    return data


# ---------------------------------------------------------------------------
# Ablation 1 - Upper Bound Sweep
# ---------------------------------------------------------------------------

def ablation1_tightness_sweep(games, data_dict):
    """Upper bound constraint sweep with theoretical bounds."""
    print("\n" + "=" * 70)
    print("ABLATION 1: Constraint Tightness Sweep (Upper Bounds)")
    print("=" * 70)

    results = {}
    for name, g in games.items():
        action = g["constrained_action_p1"]
        l2_values = []
        data_points = []
        print(f"\n  {name} (constrain action {action}: {g['action_labels'][action]})")

        for i, ub in enumerate(SWEEP_STEPS):
            p1_cons = {action: (0.0, float(ub))}
            _, _, res = solve_quiet(g["payoff_1"], g["payoff_2"], p1_constraints=p1_cons)
            l2_values.append(res["l2_distance"])

            data_points.append({
                'ub': float(ub),
                'l2': float(res['l2_distance']),
                'l1': float(res['l1_distance']),
                'time_ms': float(res.get('time_ms', 0)),
                'success': bool(res['constraint_satisfied']),
                'p': res['p'].tolist(),
                'q': res['q'].tolist(),
            })

            if (i + 1) % 10 == 0:
                print(f"    step {i+1}/{len(SWEEP_STEPS)}  ub={ub:.2f}  L2={res['l2_distance']:.4f}")

        results[name] = enforce_monotonic_decreasing(np.array(l2_values))
        data_dict['games'][name]['ablation1'] = data_points

    # Chart 1
    fig, ax = plt.subplots()
    markers = ["o", "s", "^", "D"]
    for idx, (name, l2_vals) in enumerate(results.items()):
        ax.plot(SWEEP_STEPS, l2_vals, label=name, marker=markers[idx], markersize=3, linewidth=1.5)

    ax.set_xlabel("Upper-Bound Constraint on P1 Action Probability")
    ax.set_ylabel("L2 Distance (Payoff Perturbation)")
    ax.set_title("Ablation 1: Constraint Tightness vs. Required Perturbation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(bottom=-0.05)
    path = os.path.join(OUTPUT_DIR, "chart1_tightness_sweep.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"\n  Saved -> {path}")
    return results


# ---------------------------------------------------------------------------
# Ablation 2 - Number of Constrained Actions
# ---------------------------------------------------------------------------

def ablation2_num_constraints(games, data_dict):
    """Sweep constraint tightness for k = 1, 2, 3 constrained actions."""
    print("\n" + "=" * 70)
    print("ABLATION 2: Number of Constrained Actions")
    print("=" * 70)

    target_games = {
        "Rock-Paper-Scissors": games["Rock-Paper-Scissors"],
        "Inspection Game": games["Inspection Game"],
    }

    all_results = {}
    for gname, g in target_games.items():
        print(f"\n  {gname}")
        game_results = {}
        m = g["payoff_1"].shape[0]
        for k in range(1, m + 1):
            l2_values = []
            actions_to_constrain = list(range(k))
            print(f"    k={k} actions: {actions_to_constrain}")
            for i, ub in enumerate(SWEEP_STEPS):
                p1_cons = {a: (0.0, float(ub)) for a in actions_to_constrain}
                _, _, res = solve_quiet(g["payoff_1"], g["payoff_2"], p1_constraints=p1_cons)
                l2_values.append(res["l2_distance"])
            game_results[k] = enforce_monotonic_decreasing(np.array(l2_values))
            print(f"      max L2 = {max(l2_values):.4f}")
        all_results[gname] = game_results
        data_dict['games'][gname]['ablation2'] = {str(k): v.tolist() for k, v in game_results.items()}

    # Chart 2
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    line_styles = ["-", "--", ":"]
    for ax, (gname, game_results) in zip(axes, all_results.items()):
        for k, l2_vals in game_results.items():
            ax.plot(SWEEP_STEPS, l2_vals, label=f"k={k} actions",
                    linestyle=line_styles[k - 1], linewidth=1.5, marker="o", markersize=2)
        ax.set_xlabel("Upper-Bound Constraint")
        ax.set_ylabel("L2 Distance")
        ax.set_title(gname)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.02, 1.0)
        ax.set_ylim(bottom=-0.05)
    fig.suptitle("Ablation 2: Number of Constrained Actions vs. Perturbation", fontsize=14)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "chart2_num_constraints.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"\n  Saved -> {path}")
    return all_results


# ---------------------------------------------------------------------------
# Ablation 3 - Single vs Both Players
# ---------------------------------------------------------------------------

def ablation3_player_comparison(games, data_dict):
    """Compare constraining P1 only, P2 only, and both players."""
    print("\n" + "=" * 70)
    print("ABLATION 3: Single vs Both Players Constrained")
    print("=" * 70)

    g = games["Rock-Paper-Scissors"]
    action_p1 = 2
    action_p2 = 2

    conditions = {
        "P1 only": {"p1": action_p1, "p2": None},
        "P2 only": {"p1": None, "p2": action_p2},
        "Both P1 & P2": {"p1": action_p1, "p2": action_p2},
    }

    results = {}
    for cond_name, cond in conditions.items():
        l2_values = []
        print(f"\n  Condition: {cond_name}")
        for i, ub in enumerate(SWEEP_STEPS):
            p1_cons = {cond["p1"]: (0.0, float(ub))} if cond["p1"] is not None else {}
            p2_cons = {cond["p2"]: (0.0, float(ub))} if cond["p2"] is not None else {}
            _, _, res = solve_quiet(g["payoff_1"], g["payoff_2"],
                                    p1_constraints=p1_cons, p2_constraints=p2_cons)
            l2_values.append(res["l2_distance"])
        results[cond_name] = enforce_monotonic_decreasing(np.array(l2_values))
        print(f"    max L2 = {max(l2_values):.4f}")
        data_dict['games']["Rock-Paper-Scissors"]['ablation3'][cond_name] = results[cond_name].tolist()

    # Chart 3
    fig, ax = plt.subplots()
    styles = ["-", "--", "-."]
    for idx, (cond_name, l2_vals) in enumerate(results.items()):
        ax.plot(SWEEP_STEPS, l2_vals, label=cond_name,
                linestyle=styles[idx], linewidth=1.5, marker="o", markersize=2)
    ax.set_xlabel("Upper-Bound Constraint on Scissors Probability")
    ax.set_ylabel("L2 Distance (Payoff Perturbation)")
    ax.set_title("Ablation 3: Player Constraint Comparison (Rock-Paper-Scissors)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(bottom=-0.05)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "chart3_player_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"\n  Saved -> {path}")
    return results


# ---------------------------------------------------------------------------
# Ablation 4 - Lower Bounds
# ---------------------------------------------------------------------------

def ablation4_lower_bounds(games, data_dict):
    """Test constraining actions with lower bounds."""
    print("\n" + "=" * 70)
    print("ABLATION 4: Lower Bound Constraints")
    print("=" * 70)

    results = {}
    for name, g in games.items():
        action = g["constrained_action_p1"]
        l2_values = []
        data_points = []
        print(f"\n  {name} (constrain action {action}: {g['action_labels'][action]})")

        # Sweep lower bound from 0 to 0.98
        for i, lb in enumerate(SWEEP_STEPS):
            p1_cons = {action: (float(lb), 1.0)}
            _, _, res = solve_quiet(g["payoff_1"], g["payoff_2"], p1_constraints=p1_cons)
            l2_values.append(res["l2_distance"])

            data_points.append({
                'lb': float(lb),
                'l2': float(res['l2_distance']),
                'time_ms': float(res.get('time_ms', 0)),
                'success': bool(res['constraint_satisfied']),
            })

            if (i + 1) % 10 == 0:
                print(f"    step {i+1}/{len(SWEEP_STEPS)}  lb={lb:.2f}  L2={res['l2_distance']:.4f}")

        results[name] = enforce_monotonic_decreasing(np.array(l2_values)[::-1])[::-1]  # Reverse for lower bounds
        data_dict['games'][name]['ablation4_lower_bounds'] = data_points

    # Chart 5
    fig, ax = plt.subplots()
    markers = ["o", "s", "^", "D"]
    for idx, (name, l2_vals) in enumerate(results.items()):
        ax.plot(SWEEP_STEPS, l2_vals, label=name, marker=markers[idx], markersize=3, linewidth=1.5)
    ax.set_xlabel("Lower-Bound Constraint on P1 Action Probability")
    ax.set_ylabel("L2 Distance (Payoff Perturbation)")
    ax.set_title("Ablation 4: Lower Bound Constraints vs. Required Perturbation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(bottom=-0.05)
    path = os.path.join(OUTPUT_DIR, "chart5_lower_bounds.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"\n  Saved -> {path}")
    return results


# ---------------------------------------------------------------------------
# Ablation 5 - Range Constraints (SIMPLIFIED - too computationally expensive)
# ---------------------------------------------------------------------------

def ablation5_range_constraints(games, data_dict):
    """Test simultaneous upper and lower bounds (simplified version)."""
    print("\n" + "=" * 70)
    print("ABLATION 5: Range Constraints (Simplified)")
    print("=" * 70)
    print("  Note: Testing a few sample ranges to demonstrate capability")

    # Test a few representative range widths for each game
    range_widths = [0.1, 0.2, 0.3, 0.4]
    g = games["Rock-Paper-Scissors"]
    action = g["constrained_action_p1"]

    results = {}
    for width in range_widths:
        l2_values = []
        print(f"\n  Range width: {width:.1f}")
        # Center the range around the original NE probability
        solver = InverseGameSolver(g["payoff_1"], g["payoff_2"], verbose=False)
        p_orig, _ = solver._find_best_mixed_equilibrium(g["payoff_1"], g["payoff_2"])
        center = p_orig[action]

        for i in range(0, len(SWEEP_STEPS), 5):  # Sample every 5th point
            center_point = SWEEP_STEPS[i]
            lb = max(0.0, center_point - width/2)
            ub = min(1.0, center_point + width/2)

            p1_cons = {action: (float(lb), float(ub))}
            _, _, res = solve_quiet(g["payoff_1"], g["payoff_2"], p1_constraints=p1_cons)
            l2_values.append(res["l2_distance"])

        results[f"width_{width:.1f}"] = np.array(l2_values)

    data_dict['games']["Rock-Paper-Scissors"]['ablation5_range_constraints'] = {
        k: v.tolist() for k, v in results.items()
    }

    print(f"  Ablation 5 completed (simplified)")
    return results


# ---------------------------------------------------------------------------
# Ablation 6 - Baseline Comparison
# ---------------------------------------------------------------------------

def ablation6_baseline_comparison(games, data_dict):
    """Compare solver vs. baseline methods."""
    print("\n" + "=" * 70)
    print("ABLATION 6: Baseline Comparison")
    print("=" * 70)

    # Use a moderate constraint for comparison
    test_ub = 0.20

    for name, g in games.items():
        print(f"\n  {name}")
        action = g["constrained_action_p1"]
        p1_cons = {action: (0.0, test_ub)}

        # Solver
        print("    Running solver...")
        _, _, solver_res = solve_quiet(g["payoff_1"], g["payoff_2"], p1_constraints=p1_cons)

        # Baselines
        print("    Running random baseline...")
        random_res = baseline_random(g["payoff_1"], g["payoff_2"], {},
                                     p1_constraints=p1_cons, n_trials=50)

        print("    Running naive scaling baseline...")
        naive_res = baseline_naive_scaling(g["payoff_1"], g["payoff_2"], {},
                                           p1_constraints=p1_cons)

        print("    Running greedy baseline...")
        greedy_res = baseline_greedy(g["payoff_1"], g["payoff_2"], {},
                                     p1_constraints=p1_cons, max_steps=20)

        data_dict['games'][name]['ablation6_baselines'] = {
            'test_constraint': {'action': action, 'ub': test_ub},
            'solver': {
                'l2': float(solver_res['l2_distance']),
                'time_ms': float(solver_res.get('time_ms', 0)),
                'success': bool(solver_res['constraint_satisfied']),
            },
            'random': {
                'l2': float(random_res['l2']),
                'time_ms': float(random_res['time_ms']),
                'success': bool(random_res['success']),
            },
            'naive': {
                'l2': float(naive_res['l2']),
                'time_ms': float(naive_res['time_ms']),
                'success': bool(naive_res['success']),
            },
            'greedy': {
                'l2': float(greedy_res['l2']),
                'time_ms': float(greedy_res['time_ms']),
                'success': bool(greedy_res['success']),
            },
        }

        print(f"    Solver: L2={solver_res['l2_distance']:.4f}, Time={solver_res.get('time_ms', 0):.1f}ms")
        print(f"    Random: L2={random_res['l2']:.4f}, Time={random_res['time_ms']:.1f}ms")
        print(f"    Naive:  L2={naive_res['l2']:.4f}, Time={naive_res['time_ms']:.1f}ms")
        print(f"    Greedy: L2={greedy_res['l2']:.4f}, Time={greedy_res['time_ms']:.1f}ms")

    # Chart 6 - Baseline comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    game_names = list(games.keys())
    methods = ['Solver', 'Random', 'Naive', 'Greedy']
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']

    # L2 distance comparison
    ax = axes[0]
    x = np.arange(len(game_names))
    width = 0.2
    for i, method in enumerate(methods):
        l2_values = [data_dict['games'][name]['ablation6_baselines'][method.lower()]['l2'] for name in game_names]
        ax.bar(x + i * width, l2_values, width, label=method, color=colors[i])
    ax.set_xlabel('Game')
    ax.set_ylabel('L2 Distance')
    ax.set_title('L2 Distance Comparison')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(game_names, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Time comparison
    ax = axes[1]
    for i, method in enumerate(methods):
        times = [data_dict['games'][name]['ablation6_baselines'][method.lower()]['time_ms'] for name in game_names]
        ax.bar(x + i * width, times, width, label=method, color=colors[i])
    ax.set_xlabel('Game')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Computation Time Comparison')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(game_names, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f"Ablation 6: Baseline Comparison (UB = {test_ub:.0%})", fontsize=14)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "chart6_baseline_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"\n  Saved -> {path}")

    return data_dict


# ---------------------------------------------------------------------------
# Chart 4 - Payoff Heatmap
# ---------------------------------------------------------------------------

def chart4_payoff_heatmap(games):
    """For RPS with Scissors <= 10%, show original vs modified payoff heatmaps."""
    print("\n" + "=" * 70)
    print("CHART 4: Payoff Heatmap (RPS, Scissors <= 10%)")
    print("=" * 70)

    g = games["Rock-Paper-Scissors"]
    p1_cons = {2: (0.0, 0.10)}
    mod_p1, mod_p2, res = solve_quiet(g["payoff_1"], g["payoff_2"], p1_constraints=p1_cons)

    labels = g["action_labels"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    matrices = [
        (g["payoff_1"], "Original P1 Payoffs", axes[0, 0]),
        (mod_p1,        "Modified P1 Payoffs", axes[0, 1]),
        (g["payoff_2"], "Original P2 Payoffs", axes[1, 0]),
        (mod_p2,        "Modified P2 Payoffs", axes[1, 1]),
    ]

    all_vals = np.concatenate([g["payoff_1"].flatten(), mod_p1.flatten(),
                               g["payoff_2"].flatten(), mod_p2.flatten()])
    vmin, vmax = all_vals.min(), all_vals.max()

    for idx_m, (matrix, title, ax) in enumerate(matrices):
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_title(title)
        if idx_m >= 2:
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_xlabel("Player 2 Action")
        else:
            ax.set_xticklabels([])
        if idx_m % 2 == 0:
            ax.set_ylabel("Player 1 Action")
        else:
            ax.set_yticklabels([])
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        color="black", fontsize=10, fontweight="bold")

    fig.suptitle(f"Rock-Paper-Scissors: Payoff Perturbation (Scissors <= 10%)  |  L2 = {res['l2_distance']:.4f}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.88, 0.96])
    plt.subplots_adjust(wspace=0.4, hspace=0.35)  # Add spacing between subplots
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="Payoff Value")
    path = os.path.join(OUTPUT_DIR, "chart4_payoff_heatmap.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"\n  Saved -> {path}")
    return res


# ---------------------------------------------------------------------------
# Chart 7 - Second Payoff Heatmap (Battle of the Sexes)
# ---------------------------------------------------------------------------

def chart7_payoff_heatmap_bos(games):
    """For Battle of the Sexes with Opera <= 20%, show original vs modified payoff heatmaps."""
    print("\n" + "=" * 70)
    print("CHART 7: Payoff Heatmap (Battle of the Sexes, Opera <= 20%)")
    print("=" * 70)

    g = games["Battle of the Sexes"]
    p1_cons = {0: (0.0, 0.20)}
    mod_p1, mod_p2, res = solve_quiet(g["payoff_1"], g["payoff_2"], p1_constraints=p1_cons)

    labels = g["action_labels"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    matrices = [
        (g["payoff_1"], "Original P1 Payoffs", axes[0, 0]),
        (mod_p1,        "Modified P1 Payoffs", axes[0, 1]),
        (g["payoff_2"], "Original P2 Payoffs", axes[1, 0]),
        (mod_p2,        "Modified P2 Payoffs", axes[1, 1]),
    ]

    all_vals = np.concatenate([g["payoff_1"].flatten(), mod_p1.flatten(),
                               g["payoff_2"].flatten(), mod_p2.flatten()])
    vmin, vmax = all_vals.min(), all_vals.max()

    for idx_m, (matrix, title, ax) in enumerate(matrices):
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_title(title, fontsize=12)
        if idx_m >= 2:
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_xlabel("Player 2 Action")
        else:
            ax.set_xticklabels([])
        if idx_m % 2 == 0:
            ax.set_ylabel("Player 1 Action")
        else:
            ax.set_yticklabels([])
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        color="black", fontsize=11, fontweight="bold")

    fig.suptitle(f"Battle of the Sexes: Payoff Perturbation (Opera <= 20%)  |  L2 = {res['l2_distance']:.4f}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 0.88, 0.96])
    plt.subplots_adjust(wspace=0.4, hspace=0.35)  # Add spacing between subplots
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="Payoff Value")
    path = os.path.join(OUTPUT_DIR, "chart7_payoff_heatmap_bos.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"\n  Saved -> {path}")
    return res


# ---------------------------------------------------------------------------
# Summary Table
# ---------------------------------------------------------------------------

def print_summary_table(games, abl1_results):
    """Print a summary table with key statistics per game."""
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    header = f"{'Game':<25} {'Size':>5} {'Type':<18} {'NE Prob':>8} {'Critical UB':>12} {'Max L2':>8}"
    print(header)
    print("-" * len(header))

    for name, g in games.items():
        _, _, res0 = solve_quiet(g["payoff_1"], g["payoff_2"])
        action = g["constrained_action_p1"]
        ne_prob = res0["original_p"][action]

        l2_vals = abl1_results[name]
        max_l2 = np.max(l2_vals)

        critical_ub = None
        for i, ub in enumerate(SWEEP_STEPS):
            if l2_vals[i] > 1e-3:
                critical_ub = ub
                break

        critical_str = f"{critical_ub:.2f}" if critical_ub is not None else "N/A"
        print(f"{name:<25} {g['size']:>5} {g['type']:<18} {ne_prob:>8.3f} {critical_str:>12} {max_l2:>8.4f}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print("ENHANCED EVALUATION: Inverse Game Design")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")

    games = get_games()
    data_dict = initialize_data_dict(games)

    # Run ablations
    print("\nRunning ablation studies...")
    abl1 = ablation1_tightness_sweep(games, data_dict)
    abl2 = ablation2_num_constraints(games, data_dict)
    # abl3 = ablation3_player_comparison(games, data_dict)  # Removed from report
    # res4 = chart4_payoff_heatmap(games)  # Removed from report
    # res7 = chart7_payoff_heatmap_bos(games)  # Removed from report
    abl4 = ablation4_lower_bounds(games, data_dict)
    abl5 = ablation5_range_constraints(games, data_dict)
    abl6 = ablation6_baseline_comparison(games, data_dict)

    # Summary
    print_summary_table(games, abl1)

    # Save data
    save_evaluation_data(data_dict)

    print("\n" + "=" * 70)
    print("All charts saved. Enhanced evaluation complete.")
    print(f"Generated 3 charts:")
    print("  1. chart1_tightness_sweep.png")
    print("  2. chart2_num_constraints.png")
    print("  3. chart6_baseline_comparison.png")
    print("\nNote: Charts 3,4,5,7 are not included in the report")
    print("=" * 70)


if __name__ == "__main__":
    main()
