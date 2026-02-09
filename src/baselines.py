"""
Baseline Methods for Comparison
================================
Implements three baseline approaches to compare against the optimization solver.
"""

import numpy as np
import time
from typing import Dict, Tuple
from inverse_game_solver import InverseGameSolver


def baseline_random(payoff_1: np.ndarray,
                   payoff_2: np.ndarray,
                   constraints: Dict,
                   p1_constraints: Dict = None,
                   p2_constraints: Dict = None,
                   n_trials: int = 100,
                   l2_budget: float = 2.0) -> Dict:
    """
    Baseline 1: Random Perturbation
    Randomly perturb payoffs within L2 budget and check if constraints satisfied.

    Args:
        payoff_1: Player 1's original payoff matrix
        payoff_2: Player 2's original payoff matrix
        constraints: Unused (for compatibility)
        p1_constraints: Player 1 constraints dict
        p2_constraints: Player 2 constraints dict
        n_trials: Number of random trials
        l2_budget: Maximum L2 distance allowed

    Returns:
        dict with 'l2', 'time_ms', 'success', 'p', 'q'
    """
    start = time.time()

    best_l2 = float('inf')
    best_p = None
    best_q = None
    success = False

    for trial in range(n_trials):
        # Random perturbation within budget
        perturb_1 = np.random.randn(*payoff_1.shape)
        perturb_2 = np.random.randn(*payoff_2.shape)

        # Scale to be within L2 budget
        total_norm = np.linalg.norm(np.concatenate([perturb_1.flatten(), perturb_2.flatten()]))
        if total_norm > 0:
            scale = l2_budget * np.random.rand() / total_norm
            perturb_1 *= scale
            perturb_2 *= scale

        mod_1 = payoff_1 + perturb_1
        mod_2 = payoff_2 + perturb_2

        # Find equilibrium of perturbed game
        solver = InverseGameSolver(mod_1, mod_2, verbose=False)
        p, q = solver._find_best_mixed_equilibrium(mod_1, mod_2)

        # Check constraint satisfaction
        satisfied = True
        if p1_constraints:
            for action, (min_prob, max_prob) in p1_constraints.items():
                if not (min_prob - 1e-3 <= p[action] <= max_prob + 1e-3):
                    satisfied = False
                    break

        if satisfied and p2_constraints:
            for action, (min_prob, max_prob) in p2_constraints.items():
                if not (min_prob - 1e-3 <= q[action] <= max_prob + 1e-3):
                    satisfied = False
                    break

        if satisfied:
            actual_l2 = np.linalg.norm(np.concatenate([perturb_1.flatten(), perturb_2.flatten()]))
            if actual_l2 < best_l2:
                best_l2 = actual_l2
                best_p = p
                best_q = q
                success = True

    time_ms = (time.time() - start) * 1000

    return {
        'l2': best_l2 if success else float('inf'),
        'time_ms': time_ms,
        'success': success,
        'p': best_p,
        'q': best_q,
    }


def baseline_naive_scaling(payoff_1: np.ndarray,
                          payoff_2: np.ndarray,
                          constraints: Dict,
                          p1_constraints: Dict = None,
                          p2_constraints: Dict = None) -> Dict:
    """
    Baseline 2: Naive Payoff Scaling
    Simple heuristic: scale payoffs for constrained actions to discourage/encourage use.

    Strategy:
    - If upper bound constraint, decrease payoffs for that action
    - If lower bound constraint, increase payoffs for that action

    Args:
        payoff_1: Player 1's original payoff matrix
        payoff_2: Player 2's original payoff matrix
        constraints: Unused (for compatibility)
        p1_constraints: Player 1 constraints dict
        p2_constraints: Player 2 constraints dict

    Returns:
        dict with 'l2', 'time_ms', 'success', 'p', 'q'
    """
    start = time.time()

    mod_1 = payoff_1.copy()
    mod_2 = payoff_2.copy()

    # Apply heuristic modifications
    if p1_constraints:
        for action, (min_prob, max_prob) in p1_constraints.items():
            # If upper bound is tight, decrease P2's payoffs when P1 plays this action
            # (making P1's action less attractive)
            if max_prob < 0.9:
                scale_factor = 0.5  # Arbitrary scaling
                mod_2[action, :] *= scale_factor

    if p2_constraints:
        for action, (min_prob, max_prob) in p2_constraints.items():
            # If upper bound is tight, decrease P1's payoffs for this column
            if max_prob < 0.9:
                scale_factor = 0.5
                mod_1[:, action] *= scale_factor

    # Find equilibrium of modified game
    solver = InverseGameSolver(mod_1, mod_2, verbose=False)
    p, q = solver._find_best_mixed_equilibrium(mod_1, mod_2)

    # Check constraint satisfaction
    success = True
    if p1_constraints:
        for action, (min_prob, max_prob) in p1_constraints.items():
            if not (min_prob - 1e-3 <= p[action] <= max_prob + 1e-3):
                success = False
                break

    if success and p2_constraints:
        for action, (min_prob, max_prob) in p2_constraints.items():
            if not (min_prob - 1e-3 <= q[action] <= max_prob + 1e-3):
                success = False
                break

    # Compute L2 distance
    l2 = np.linalg.norm(np.concatenate([
        (mod_1 - payoff_1).flatten(),
        (mod_2 - payoff_2).flatten()
    ]))

    time_ms = (time.time() - start) * 1000

    return {
        'l2': l2,
        'time_ms': time_ms,
        'success': success,
        'p': p,
        'q': q,
    }


def baseline_greedy(payoff_1: np.ndarray,
                   payoff_2: np.ndarray,
                   constraints: Dict,
                   p1_constraints: Dict = None,
                   p2_constraints: Dict = None,
                   max_steps: int = 50) -> Dict:
    """
    Baseline 3: Greedy Single-Payoff Modification
    Iteratively modify one payoff entry at a time, choosing the modification
    that most improves constraint satisfaction.

    Args:
        payoff_1: Player 1's original payoff matrix
        payoff_2: Player 2's original payoff matrix
        constraints: Unused (for compatibility)
        p1_constraints: Player 1 constraints dict
        p2_constraints: Player 2 constraints dict
        max_steps: Maximum number of greedy steps

    Returns:
        dict with 'l2', 'time_ms', 'success', 'p', 'q'
    """
    start = time.time()

    mod_1 = payoff_1.copy()
    mod_2 = payoff_2.copy()

    step_size = 0.1  # Small perturbation per step

    for step in range(max_steps):
        solver = InverseGameSolver(mod_1, mod_2, verbose=False)
        p, q = solver._find_best_mixed_equilibrium(mod_1, mod_2)

        # Compute current constraint violation
        current_violation = 0.0
        if p1_constraints:
            for action, (min_prob, max_prob) in p1_constraints.items():
                if p[action] < min_prob:
                    current_violation += (min_prob - p[action]) ** 2
                elif p[action] > max_prob:
                    current_violation += (p[action] - max_prob) ** 2

        if p2_constraints:
            for action, (min_prob, max_prob) in p2_constraints.items():
                if q[action] < min_prob:
                    current_violation += (min_prob - q[action]) ** 2
                elif q[action] > max_prob:
                    current_violation += (q[action] - max_prob) ** 2

        if current_violation < 1e-6:
            break  # Constraints satisfied

        # Try modifying each payoff entry and pick the best
        best_improvement = 0
        best_move = None

        # Try modifying P1 payoffs
        for i in range(mod_1.shape[0]):
            for j in range(mod_1.shape[1]):
                for direction in [-step_size, step_size]:
                    test_1 = mod_1.copy()
                    test_1[i, j] += direction

                    solver_test = InverseGameSolver(test_1, mod_2, verbose=False)
                    p_test, q_test = solver_test._find_best_mixed_equilibrium(test_1, mod_2)

                    test_violation = 0.0
                    if p1_constraints:
                        for action, (min_prob, max_prob) in p1_constraints.items():
                            if p_test[action] < min_prob:
                                test_violation += (min_prob - p_test[action]) ** 2
                            elif p_test[action] > max_prob:
                                test_violation += (p_test[action] - max_prob) ** 2

                    improvement = current_violation - test_violation
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_move = ('p1', i, j, direction)

        # Try modifying P2 payoffs
        for i in range(mod_2.shape[0]):
            for j in range(mod_2.shape[1]):
                for direction in [-step_size, step_size]:
                    test_2 = mod_2.copy()
                    test_2[i, j] += direction

                    solver_test = InverseGameSolver(mod_1, test_2, verbose=False)
                    p_test, q_test = solver_test._find_best_mixed_equilibrium(mod_1, test_2)

                    test_violation = 0.0
                    if p1_constraints:
                        for action, (min_prob, max_prob) in p1_constraints.items():
                            if p_test[action] < min_prob:
                                test_violation += (min_prob - p_test[action]) ** 2
                            elif p_test[action] > max_prob:
                                test_violation += (p_test[action] - max_prob) ** 2

                    improvement = current_violation - test_violation
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_move = ('p2', i, j, direction)

        if best_move is None:
            break  # No improvement possible

        # Apply best move
        if best_move[0] == 'p1':
            mod_1[best_move[1], best_move[2]] += best_move[3]
        else:
            mod_2[best_move[1], best_move[2]] += best_move[3]

    # Final evaluation
    solver = InverseGameSolver(mod_1, mod_2, verbose=False)
    p, q = solver._find_best_mixed_equilibrium(mod_1, mod_2)

    success = True
    if p1_constraints:
        for action, (min_prob, max_prob) in p1_constraints.items():
            if not (min_prob - 1e-3 <= p[action] <= max_prob + 1e-3):
                success = False
                break

    if success and p2_constraints:
        for action, (min_prob, max_prob) in p2_constraints.items():
            if not (min_prob - 1e-3 <= q[action] <= max_prob + 1e-3):
                success = False
                break

    l2 = np.linalg.norm(np.concatenate([
        (mod_1 - payoff_1).flatten(),
        (mod_2 - payoff_2).flatten()
    ]))

    time_ms = (time.time() - start) * 1000

    return {
        'l2': l2,
        'time_ms': time_ms,
        'success': success,
        'p': p,
        'q': q,
    }
