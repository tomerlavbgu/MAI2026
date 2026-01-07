"""
InverseGameSolver: Find Minimal Payoff Modifications for Constrained Nash Equilibria

This module implements an optimization algorithm to find the minimal modification
to a game's payoff matrix that ensures the existence of a mixed-strategy Nash 
equilibrium satisfying specified probability constraints.

Core Problem:
    Given: A utility matrix U for a finite 2-player game and probability constraints
    Find: The closest utility matrix U' that admits a Nash equilibrium respecting constraints
    Minimize: L2 distance ||U' - U||
"""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class InverseGameSolver:
    """
    Solves the inverse game problem: find minimal payoff modifications to ensure
    constrained Nash equilibria in 2-player games.
    
    Problem Formulation:
        minimize    ||U' - U||_2
        subject to  (p*, q*) is a Nash equilibrium of U'
                    p* must satisfy player 1 probability constraints
                    q* must satisfy player 2 probability constraints
    """
    
    def __init__(self, 
                 payoff_matrix_1: np.ndarray,
                 payoff_matrix_2: np.ndarray,
                 p1_constraints: Dict[int, Tuple[float, float]] = None,
                 p2_constraints: Dict[int, Tuple[float, float]] = None,
                 tolerance: float = 1e-6,
                 verbose: bool = True):
        """
        Initialize the InverseGameSolver.
        
        Args:
            payoff_matrix_1: m x n matrix of player 1's payoffs
            payoff_matrix_2: m x n matrix of player 2's payoffs
            p1_constraints: Dict mapping action indices to (min, max) probability bounds
            p2_constraints: Dict mapping action indices to (min, max) probability bounds
            tolerance: Numerical tolerance for equilibrium conditions
            verbose: Print diagnostic information
        """
        self.payoff_1 = np.array(payoff_matrix_1, dtype=float)
        self.payoff_2 = np.array(payoff_matrix_2, dtype=float)
        self.m, self.n = self.payoff_1.shape
        
        self.p1_constraints = p1_constraints or {}
        self.p2_constraints = p2_constraints or {}
        self.tolerance = tolerance
        self.verbose = verbose
        
        if self.payoff_2.shape != self.payoff_1.shape:
            raise ValueError("Payoff matrices must have the same shape")
    
    def solve(self, 
              max_iterations: int = 500,
              method: str = 'SLSQP') -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Find the minimal payoff modification that satisfies constraints.
        
        Args:
            max_iterations: Maximum optimization iterations
            method: Optimization method ('SLSQP', 'L-BFGS-B', etc.)
        
        Returns:
            modified_payoff_1: Player 1's modified payoff matrix
            modified_payoff_2: Player 2's modified payoff matrix
            result: Dictionary containing solution information
        """
        if self.verbose:
            print("=" * 70)
            print("INVERSE GAME SOLVER - Minimal Payoff Modification")
            print("=" * 70)
            print(f"\nOriginal Game Dimensions: {self.m} x {self.n}")
            print(f"Player 1 Constraints: {self.p1_constraints}")
            print(f"Player 2 Constraints: {self.p2_constraints}")
        
        # Start with original payoff matrices as initial guess
        x0 = np.concatenate([self.payoff_1.flatten(), self.payoff_2.flatten()])
        
        # Define objective function
        def objective(x):
            return np.linalg.norm(x - x0)
        
        # Define constraint function
        def constraint_func(x):
            payoff_1 = x[:self.m * self.n].reshape(self.m, self.n)
            payoff_2 = x[self.m * self.n:].reshape(self.m, self.n)
            
            # Find equilibrium
            p, q = self._find_best_mixed_equilibrium(payoff_1, payoff_2)
            
            # Check constraint satisfaction
            violation = 0.0
            
            # Check player 1 constraints
            for action, (min_prob, max_prob) in self.p1_constraints.items():
                if p[action] < min_prob:
                    violation += (min_prob - p[action]) ** 2
                if p[action] > max_prob:
                    violation += (p[action] - max_prob) ** 2
            
            # Check player 2 constraints
            for action, (min_prob, max_prob) in self.p2_constraints.items():
                if q[action] < min_prob:
                    violation += (min_prob - q[action]) ** 2
                if q[action] > max_prob:
                    violation += (q[action] - max_prob) ** 2
            
            # Return negative so constraint is: g(x) >= 0 means satisfied
            return -np.sqrt(max(violation, 1e-10))
        
        # Optimization constraints
        constraints = [
            {'type': 'ineq', 'fun': constraint_func}
        ]
        
        # Bounds for variables
        bounds = [(-100, 100) for _ in range(len(x0))]
        
        # Solve
        result_opt = minimize(
            objective,
            x0,
            method=method,
            bounds=bounds,
            constraints=constraints,
            options={
                'maxiter': max_iterations,
                'ftol': 1e-10,
                'disp': False
            }
        )
        
        # Extract solution
        x_opt = result_opt.x
        modified_payoff_1 = x_opt[:self.m * self.n].reshape(self.m, self.n)
        modified_payoff_2 = x_opt[self.m * self.n:].reshape(self.m, self.n)
        
        # Find equilibrium in modified game
        p_opt, q_opt = self._find_best_mixed_equilibrium(modified_payoff_1, modified_payoff_2)
        
        # Verify constraints
        constraint_satisfied = self._verify_constraints(p_opt, q_opt)
        
        # Compute distances
        l2_distance = np.linalg.norm(np.concatenate([
            modified_payoff_1 - self.payoff_1,
            modified_payoff_2 - self.payoff_2
        ]))
        
        l1_distance = np.sum(np.abs(modified_payoff_1 - self.payoff_1)) + \
                     np.sum(np.abs(modified_payoff_2 - self.payoff_2))
        
        # Find original equilibrium
        original_p, original_q = self._find_best_mixed_equilibrium(self.payoff_1, self.payoff_2)
        
        result_dict = {
            'p': p_opt,
            'q': q_opt,
            'l2_distance': l2_distance,
            'l1_distance': l1_distance,
            'success': result_opt.success,
            'constraint_satisfied': constraint_satisfied,
            'original_p': original_p,
            'original_q': original_q,
        }
        
        if self.verbose:
            self._print_results(modified_payoff_1, modified_payoff_2, result_dict)
        
        return modified_payoff_1, modified_payoff_2, result_dict
    
    def _find_best_mixed_equilibrium(self, 
                                     payoff_1: np.ndarray,
                                     payoff_2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find the best mixed-strategy Nash equilibrium using support enumeration.
        Prioritizes fully mixed equilibria (all actions played).
        """
        best_p = np.ones(self.m) / self.m
        best_q = np.ones(self.n) / self.n
        best_violation = float('inf')
        best_support_size = 0
        
        # Try supports in order of preference (larger supports first = more "mixed")
        supports_by_size = {}
        
        for p_support_mask in range(1, 2**self.m):
            p_support = [i for i in range(self.m) if p_support_mask & (1 << i)]
            support_size = len(p_support)
            
            if support_size not in supports_by_size:
                supports_by_size[support_size] = []
            
            for q_support_mask in range(1, 2**self.n):
                q_support = [j for j in range(self.n) if q_support_mask & (1 << j)]
                
                if len(q_support) == support_size:  # Try matching support sizes
                    p, q = self._solve_support_equilibrium(
                        payoff_1, payoff_2, p_support, q_support
                    )
                    
                    if p is not None and q is not None:
                        violation = self._compute_br_violation(payoff_1, payoff_2, p, q)
                        
                        # Prefer larger supports (more mixed)
                        # Then prefer solutions with lower violation
                        if support_size > best_support_size or \
                           (support_size == best_support_size and violation < best_violation):
                            best_violation = violation
                            best_p = p
                            best_q = q
                            best_support_size = support_size
        
        return best_p, best_q
    
    def _solve_support_equilibrium(self,
                                   payoff_1: np.ndarray,
                                   payoff_2: np.ndarray,
                                   p_support: List[int],
                                   q_support: List[int]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Solve for mixed strategy equilibrium on given support."""
        try:
            m, n = payoff_1.shape
            
            # Solve for player 2's strategy using indifference condition
            # Player 1 must be indifferent among all actions in p_support
            if len(q_support) == 1:
                q = np.zeros(n)
                q[q_support[0]] = 1.0
            else:
                # Build system: for all i, j in p_support, u1(i, q) = u1(j, q)
                # This gives: (u1[i] - u1[j]) @ q = 0
                # Plus: sum(q[q_support]) = 1
                
                A_list = []
                for i in range(len(p_support) - 1):
                    idx_i = p_support[i]
                    idx_j = p_support[i + 1]
                    row = np.zeros(n)
                    row[q_support] = payoff_1[idx_i, q_support] - payoff_1[idx_j, q_support]
                    A_list.append(row)
                
                # Add normalization constraint for q on support
                norm_row = np.zeros(n)
                norm_row[q_support] = 1.0
                A_list.append(norm_row)
                
                A = np.array(A_list)
                b = np.zeros(len(A_list))
                b[-1] = 1.0  # sum(q) = 1 for support
                
                try:
                    q_full = np.linalg.lstsq(A, b, rcond=None)[0]
                    q = np.maximum(q_full, 0)
                    q /= np.sum(q) + 1e-10
                except:
                    return None, None
            
            # Solve for player 1's strategy
            if len(p_support) == 1:
                p = np.zeros(m)
                p[p_support[0]] = 1.0
            else:
                # Build system for player 2's indifference
                A_list = []
                for j in range(len(q_support) - 1):
                    idx_j = q_support[j]
                    idx_k = q_support[j + 1]
                    row = np.zeros(m)
                    row[p_support] = payoff_2[p_support, idx_j] - payoff_2[p_support, idx_k]
                    A_list.append(row)
                
                # Add normalization
                norm_row = np.zeros(m)
                norm_row[p_support] = 1.0
                A_list.append(norm_row)
                
                A = np.array(A_list)
                b = np.zeros(len(A_list))
                b[-1] = 1.0
                
                try:
                    p_full = np.linalg.lstsq(A, b, rcond=None)[0]
                    p = np.maximum(p_full, 0)
                    p /= np.sum(p) + 1e-10
                except:
                    return None, None
            
            # Validate: all probabilities on support must be positive
            for i in p_support:
                if p[i] < self.tolerance:
                    return None, None
            for j in q_support:
                if q[j] < self.tolerance:
                    return None, None
            
            # Validate: probabilities outside support should be near zero
            for i in range(m):
                if i not in p_support and p[i] > self.tolerance:
                    return None, None
            for j in range(n):
                if j not in q_support and q[j] > self.tolerance:
                    return None, None
            
            return p, q
        
        except:
            return None, None
    
    def _compute_br_violation(self,
                              payoff_1: np.ndarray,
                              payoff_2: np.ndarray,
                              p: np.ndarray,
                              q: np.ndarray) -> float:
        """
        Compute how much the best response condition is violated.
        """
        # Player 1's payoffs for each action
        payoff_1_actions = payoff_1 @ q
        max_payoff_1 = np.max(payoff_1_actions)
        
        # For each action in p's support, compute how much it underperforms
        p1_violation = 0.0
        for i in range(len(p)):
            if p[i] > self.tolerance:
                p1_violation += (max_payoff_1 - payoff_1_actions[i]) ** 2
        
        # Player 2's payoffs for each action
        payoff_2_actions = payoff_2.T @ p
        max_payoff_2 = np.max(payoff_2_actions)
        
        # For each action in q's support
        p2_violation = 0.0
        for j in range(len(q)):
            if q[j] > self.tolerance:
                p2_violation += (max_payoff_2 - payoff_2_actions[j]) ** 2
        
        return np.sqrt(p1_violation + p2_violation)
    
    def _verify_constraints(self, p: np.ndarray, q: np.ndarray) -> bool:
        """Check if solution satisfies all probability constraints."""
        for action, (min_prob, max_prob) in self.p1_constraints.items():
            if not (min_prob - self.tolerance <= p[action] <= max_prob + self.tolerance):
                return False
        
        for action, (min_prob, max_prob) in self.p2_constraints.items():
            if not (min_prob - self.tolerance <= q[action] <= max_prob + self.tolerance):
                return False
        
        return True
    
    def _print_results(self, 
                      modified_payoff_1: np.ndarray,
                      modified_payoff_2: np.ndarray,
                      result: Dict) -> None:
        """Print formatted results."""
        print("\n" + "=" * 70)
        print("ORIGINAL EQUILIBRIUM")
        print("=" * 70)
        print(f"\nPlayer 1 Strategy: {result['original_p']}")
        print(f"Player 2 Strategy: {result['original_q']}")
        
        print("\n" + "=" * 70)
        print("CONSTRAINT VERIFICATION")
        print("=" * 70)
        print(f"\nConstraints Satisfied: {result['constraint_satisfied']}")
        if self.p1_constraints:
            print("\nPlayer 1 Constraints:")
            for action, (min_p, max_p) in self.p1_constraints.items():
                actual = result['original_p'][action]
                status = "✓" if min_p <= actual <= max_p else "✗"
                print(f"  Action {action}: [{min_p:.1%}, {max_p:.1%}] → {actual:.1%} {status}")
        
        print("\n" + "=" * 70)
        print("MODIFIED PAYOFF MATRICES")
        print("=" * 70)
        print("\nPlayer 1 Payoffs (Modified):")
        print(modified_payoff_1)
        print("\nPlayer 2 Payoffs (Modified):")
        print(modified_payoff_2)
        
        print("\n" + "=" * 70)
        print("PAYOFF CHANGES")
        print("=" * 70)
        print("\nPlayer 1 Payoff Changes:")
        print(modified_payoff_1 - self.payoff_1)
        print("\nPlayer 2 Payoff Changes:")
        print(modified_payoff_2 - self.payoff_2)
        
        print("\n" + "=" * 70)
        print("MODIFIED EQUILIBRIUM")
        print("=" * 70)
        print(f"\nPlayer 1 Strategy: {result['p']}")
        print(f"Player 2 Strategy: {result['q']}")
        
        print("\n" + "=" * 70)
        print("CONSTRAINT VERIFICATION (MODIFIED)")
        print("=" * 70)
        if self.p1_constraints:
            print("\nPlayer 1 Constraints:")
            for action, (min_p, max_p) in self.p1_constraints.items():
                actual = result['p'][action]
                status = "✓" if min_p <= actual <= max_p else "✗"
                print(f"  Action {action}: [{min_p:.1%}, {max_p:.1%}] → {actual:.1%} {status}")
        
        print("\n" + "=" * 70)
        print("DISTANCE METRICS")
        print("=" * 70)
        print(f"\nL2 (Euclidean) Distance: {result['l2_distance']:.6f}")
        print(f"L1 (Manhattan) Distance: {result['l1_distance']:.6f}")
        
        print("\n" + "=" * 70)

