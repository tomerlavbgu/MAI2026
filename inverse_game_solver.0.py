"""
InverseGameSolver: Find Minimal Payoff Modifications for Constrained Nash Equilibria

This module implements an optimization algorithm to find the minimal modification
to a game's payoff matrix that ensures the existence of a mixed-strategy Nash 
equilibrium satisfying specified probability constraints.

Core Problem:
    Given: A 2-player game payoff matrix U and probability constraints on actions
    Find: The closest matrix U' that admits a Nash equilibrium respecting constraints
    Minimize: L2 distance ||U' - U||
"""

import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


class InverseGameSolverBad:
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
                           Example: {0: (0.4, 0.6)} means action 0 between 40%-60%
            p2_constraints: Dict mapping action indices to (min, max) probability bounds
            tolerance: Numerical tolerance for equilibrium conditions
            verbose: Print diagnostic information
        """
        self.payoff_1 = np.array(payoff_matrix_1, dtype=float)
        self.payoff_2 = np.array(payoff_matrix_2, dtype=float)
        self.m, self.n = self.payoff_1.shape  # m actions for P1, n actions for P2
        
        self.p1_constraints = p1_constraints or {}
        self.p2_constraints = p2_constraints or {}
        self.tolerance = tolerance
        self.verbose = verbose
        
        # Validate inputs
        if self.payoff_2.shape != self.payoff_1.shape:
            raise ValueError("Payoff matrices must have the same shape")
    
    def solve(self, 
              initial_guess: np.ndarray = None,
              max_iterations: int = 100,
              method: str = 'SLSQP') -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Find the minimal payoff modification that satisfies constraints.
        
        Args:
            initial_guess: Starting point for optimization (flattened payoff matrices)
            max_iterations: Maximum optimization iterations
            method: Optimization method ('SLSQP', 'L-BFGS-B', etc.)
        
        Returns:
            modified_payoff_1: Player 1's modified payoff matrix
            modified_payoff_2: Player 2's modified payoff matrix
            result: Dictionary containing:
                - 'p': Mixed strategy for player 1
                - 'q': Mixed strategy for player 2
                - 'l2_distance': L2 norm of modification
                - 'l1_distance': L1 norm of modification
                - 'success': Whether optimization succeeded
                - 'constraint_satisfied': Whether all constraints are met
        """
        if initial_guess is None:
            initial_guess = np.concatenate([
                self.payoff_1.flatten(),
                self.payoff_2.flatten()
            ])
        
        if self.verbose:
            print("=" * 70)
            print("INVERSE GAME SOLVER - Minimal Payoff Modification")
            print("=" * 70)
            print(f"\nOriginal Game Dimensions: {self.m} x {self.n}")
            print(f"Player 1 Constraints: {self.p1_constraints}")
            print(f"Player 2 Constraints: {self.p2_constraints}")
        
        # Optimization bounds (allow payoffs to vary)
        bounds = Bounds(
            lb=-1e3 * np.ones_like(initial_guess),
            ub=1e3 * np.ones_like(initial_guess)
        )
        
        # Define objective: minimize distance to original matrix
        def objective(x):
            return np.linalg.norm(x - initial_guess)
        
        # Define constraints
        constraints = self._build_constraints()
        
        # Solve
        result = minimize(
            objective,
            initial_guess,
            method=method,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': max_iterations, 'ftol': 1e-9}
        )
        
        # Extract solution
        x_opt = result.x
        modified_payoff_1 = x_opt[:self.m * self.n].reshape(self.m, self.n)
        modified_payoff_2 = x_opt[self.m * self.n:].reshape(self.m, self.n)
        
        # Find equilibrium strategies
        p_opt, q_opt = self._find_mixed_nash(modified_payoff_1, modified_payoff_2)
        
        # Verify constraints
        constraint_satisfied = self._verify_constraints(p_opt, q_opt)
        
        l2_distance = np.linalg.norm(np.concatenate([
            modified_payoff_1 - self.payoff_1,
            modified_payoff_2 - self.payoff_2
        ]))
        
        l1_distance = np.sum(np.abs(modified_payoff_1 - self.payoff_1)) + \
                     np.sum(np.abs(modified_payoff_2 - self.payoff_2))
        
        result_dict = {
            'p': p_opt,
            'q': q_opt,
            'l2_distance': l2_distance,
            'l1_distance': l1_distance,
            'success': result.success,
            'constraint_satisfied': constraint_satisfied,
            'original_p': self._find_original_equilibrium_p(),
            'original_q': self._find_original_equilibrium_q(),
        }
        
        if self.verbose:
            self._print_results(modified_payoff_1, modified_payoff_2, result_dict)
        
        return modified_payoff_1, modified_payoff_2, result_dict
    
    def _build_constraints(self) -> List:
        """Build constraint list for optimization."""
        constraints = []
        
        # For each possible support combination, add equilibrium constraints
        # This is a simplified approach that checks all possible supports
        
        def equilibrium_constraint(x):
            """Constraint: (p, q) must be a Nash equilibrium."""
            payoff_1 = x[:self.m * self.n].reshape(self.m, self.n)
            payoff_2 = x[self.m * self.n:].reshape(self.m, self.n)
            
            # Find best response equilibrium
            p, q = self._find_mixed_nash(payoff_1, payoff_2)
            
            # Verify it's an equilibrium (best response condition)
            br_violation = self._compute_br_violation(payoff_1, payoff_2, p, q)
            
            return -br_violation  # Constraint: br_violation <= 0
        
        def probability_constraint(x):
            """Constraint: probabilities must satisfy bounds."""
            payoff_1 = x[:self.m * self.n].reshape(self.m, self.n)
            payoff_2 = x[self.m * self.n:].reshape(self.m, self.n)
            
            p, q = self._find_mixed_nash(payoff_1, payoff_2)
            
            violations = 0.0
            
            # Check player 1 constraints
            for action, (min_prob, max_prob) in self.p1_constraints.items():
                if p[action] < min_prob:
                    violations += (min_prob - p[action]) ** 2
                if p[action] > max_prob:
                    violations += (p[action] - max_prob) ** 2
            
            # Check player 2 constraints
            for action, (min_prob, max_prob) in self.p2_constraints.items():
                if q[action] < min_prob:
                    violations += (min_prob - q[action]) ** 2
                if q[action] > max_prob:
                    violations += (q[action] - max_prob) ** 2
            
            return -np.sqrt(violations)  # Constraint: violation <= 0
        
        # Probability sum constraints
        def p_sum_constraint(x):
            """Ensure player 1 probabilities sum to 1."""
            p, _ = self._find_mixed_nash(
                x[:self.m * self.n].reshape(self.m, self.n),
                x[self.m * self.n:].reshape(self.m, self.n)
            )
            return np.sum(p) - 1.0
        
        def q_sum_constraint(x):
            """Ensure player 2 probabilities sum to 1."""
            _, q = self._find_mixed_nash(
                x[:self.m * self.n].reshape(self.m, self.n),
                x[self.m * self.n:].reshape(self.m, self.n)
            )
            return np.sum(q) - 1.0
        
        constraints.append({'type': 'eq', 'fun': p_sum_constraint})
        constraints.append({'type': 'eq', 'fun': q_sum_constraint})
        constraints.append({'type': 'ineq', 'fun': probability_constraint})
        
        return constraints
    
    def _find_mixed_nash(self, 
                        payoff_1: np.ndarray, 
                        payoff_2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find mixed strategy Nash equilibrium for given payoff matrices.
        Uses support enumeration for small games.
        """
        best_p = np.ones(self.m) / self.m
        best_q = np.ones(self.n) / self.n
        best_violation = float('inf')
        
        # Try all possible supports (subsets of actions)
        for p_support_mask in range(1, 2**self.m):
            for q_support_mask in range(1, 2**self.n):
                p_support = [i for i in range(self.m) if p_support_mask & (1 << i)]
                q_support = [j for j in range(self.n) if q_support_mask & (1 << j)]
                
                if len(p_support) == 0 or len(q_support) == 0:
                    continue
                
                # Solve for equilibrium on this support
                p, q = self._solve_support_equilibrium(
                    payoff_1, payoff_2, p_support, q_support
                )
                
                if p is not None and q is not None:
                    # Check if this is a valid equilibrium
                    violation = self._compute_br_violation(payoff_1, payoff_2, p, q)
                    if violation < best_violation:
                        best_violation = violation
                        best_p = p
                        best_q = q
        
        return best_p, best_q
    
    def _solve_support_equilibrium(self,
                                   payoff_1: np.ndarray,
                                   payoff_2: np.ndarray,
                                   p_support: List[int],
                                   q_support: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Solve for mixed strategy equilibrium on given support."""
        try:
            # For equilibrium, players are indifferent among actions in support
            # Build system of equations
            
            # Player 1 is indifferent among p_support
            # Player 2 is indifferent among q_support
            
            m, n = payoff_1.shape
            
            # Solve for player 2's strategy (should make P1 indifferent)
            if len(q_support) > 0:
                A = payoff_1[np.ix_(p_support, q_support)].T
                # Indifference condition: all supported actions give same payoff
                # Set up: for any two actions in support, payoff should be equal
                
                q = np.zeros(n)
                if len(q_support) == 1:
                    q[q_support[0]] = 1.0
                else:
                    # Use linear system: (A[0] - A[i]) . q = 0, sum(q) = 1
                    from scipy.linalg import lstsq
                    
                    # Indifference equations
                    B = A[0] - A[1:]
                    B = np.vstack([B, np.ones(len(q_support))])
                    b = np.hstack([np.zeros(len(q_support) - 1), 1.0])
                    
                    try:
                        q_support_vec, _, _, _ = lstsq(B, b)
                    except:
                        return None, None
                    
                    if np.any(q_support_vec < -self.tolerance) or np.any(q_support_vec > 1 + self.tolerance):
                        return None, None
                    
                    q_support_vec = np.clip(q_support_vec, 0, 1)
                    q_support_vec /= np.sum(q_support_vec)
                    
                    for idx, j in enumerate(q_support):
                        q[j] = q_support_vec[idx]
            
            # Similar for player 1
            p = np.zeros(m)
            if len(p_support) == 1:
                p[p_support[0]] = 1.0
            else:
                A = payoff_2[np.ix_(p_support, q_support)]
                from scipy.linalg import lstsq
                
                B = A[:, 0] - A[:, 1:]
                B = np.hstack([B.reshape(-1, len(q_support) - 1), 
                              np.ones((len(p_support), 1))])
                b = np.hstack([np.zeros(len(p_support) * (len(q_support) - 1)), 
                              len(p_support)])
                
                try:
                    p_support_vec, _, _, _ = lstsq(B, b)
                    p_support_vec = p_support_vec[:len(p_support)]
                except:
                    return None, None
                
                if np.any(p_support_vec < -self.tolerance) or np.any(p_support_vec > 1 + self.tolerance):
                    return None, None
                
                p_support_vec = np.clip(p_support_vec, 0, 1)
                p_support_vec /= np.sum(p_support_vec)
                
                for idx, i in enumerate(p_support):
                    p[i] = p_support_vec[idx]
            
            # Verify full-support equilibrium (all probabilities positive on support)
            for i in p_support:
                if p[i] < self.tolerance:
                    return None, None
            for j in q_support:
                if q[j] < self.tolerance:
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
        Lower is better (0 means perfect equilibrium).
        """
        # Player 1's payoffs for each action given q
        payoff_1_actions = payoff_1 @ q
        max_payoff_1 = np.max(payoff_1_actions)
        p1_violation = np.sum((max_payoff_1 - payoff_1_actions[p > self.tolerance]) ** 2)
        
        # Player 2's payoffs for each action given p
        payoff_2_actions = payoff_2.T @ p
        max_payoff_2 = np.max(payoff_2_actions)
        p2_violation = np.sum((max_payoff_2 - payoff_2_actions[q > self.tolerance]) ** 2)
        
        return p1_violation + p2_violation
    
    def _verify_constraints(self, p: np.ndarray, q: np.ndarray) -> bool:
        """Check if solution satisfies all probability constraints."""
        for action, (min_prob, max_prob) in self.p1_constraints.items():
            if not (min_prob - self.tolerance <= p[action] <= max_prob + self.tolerance):
                return False
        
        for action, (min_prob, max_prob) in self.p2_constraints.items():
            if not (min_prob - self.tolerance <= q[action] <= max_prob + self.tolerance):
                return False
        
        return True
    
    def _find_original_equilibrium_p(self) -> np.ndarray:
        """Find equilibrium for player 1 in original game."""
        p, _ = self._find_mixed_nash(self.payoff_1, self.payoff_2)
        return p
    
    def _find_original_equilibrium_q(self) -> np.ndarray:
        """Find equilibrium for player 2 in original game."""
        _, q = self._find_mixed_nash(self.payoff_1, self.payoff_2)
        return q
    
    def _print_results(self, 
                      modified_payoff_1: np.ndarray,
                      modified_payoff_2: np.ndarray,
                      result: Dict) -> None:
        """Print formatted results."""
        print("\n" + "=" * 70)
        print("ORIGINAL GAME PAYOFF MATRICES")
        print("=" * 70)
        print("\nPlayer 1 Payoffs:")
        print(self.payoff_1)
        print("\nPlayer 2 Payoffs:")
        print(self.payoff_2)
        
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
                actual = result['p'][action]
                status = "✓" if min_p <= actual <= max_p else "✗"
                print(f"  Action {action}: [{min_p:.1%}, {max_p:.1%}] → {actual:.1%} {status}")
        
        if self.p2_constraints:
            print("\nPlayer 2 Constraints:")
            for action, (min_p, max_p) in self.p2_constraints.items():
                actual = result['q'][action]
                status = "✓" if min_p <= actual <= max_p else "✗"
                print(f"  Action {action}: [{min_p:.1%}, {max_p:.1%}] → {actual:.1%} {status}")
        
        print("\n" + "=" * 70)
        print("MODIFIED GAME PAYOFF MATRICES")
        print("=" * 70)
        print("\nPlayer 1 Payoffs:")
        print(modified_payoff_1)
        print("\nPlayer 2 Payoffs:")
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
        print("DISTANCE METRICS")
        print("=" * 70)
        print(f"\nL2 (Euclidean) Distance: {result['l2_distance']:.6f}")
        print(f"L1 (Manhattan) Distance: {result['l1_distance']:.6f}")
        
        print("\n" + "=" * 70)


# Rock-Paper-Scissors Example
def rps_example():
    """
    Demonstrate InverseGameSolver on Rock-Paper-Scissors game.
    
    Constraint: Player 1 can play Scissors at most 10% of the time
    (due to limited supply - scissors break when opponent plays Rock)
    """
    print("\n" + "#" * 70)
    print("# ROCK-PAPER-SCISSORS WITH SCISSORS SUPPLY CONSTRAINT")
    print("#" * 70)
    
    # Standard symmetric Rock-Paper-Scissors (zero-sum)
    # Payoffs: Win = +1, Lose = -1, Draw = 0
    payoff_1 = np.array([
        [0., -1., 1.],    # Rock vs (Rock, Paper, Scissors)
        [1., 0., -1.],    # Paper vs (Rock, Paper, Scissors)
        [-1., 1., 0.]     # Scissors vs (Rock, Paper, Scissors)
    ])
    
    payoff_2 = -payoff_1  # Zero-sum game
    
    # Constraint: Player 1 can play Scissors (action 2) at most 10%
    p1_constraints = {
        2: (0.0, 0.10)  # Action 2 (Scissors): [0%, 10%]
    }
    
    # Solve
    solver = InverseGameSolver(
        payoff_1, 
        payoff_2,
        p1_constraints=p1_constraints,
        verbose=True
    )
    
    modified_p1, modified_p2, result = solver.solve()
    
    return result


if __name__ == "__main__":
    result = rps_example()
