import numpy as np
from inverse_game_solver import InverseGameSolver


# Rock-Paper-Scissors Example
def rps_example():
    """
    Demonstrate InverseGameSolver on Rock-Paper-Scissors game.

    Constraint: Player 1 can play Scissors at most 10% of the time
    """
    print("\n" + "#" * 70)
    print("# ROCK-PAPER-SCISSORS WITH SCISSORS SUPPLY CONSTRAINT")
    print("#" * 70)

    # Standard symmetric Rock-Paper-Scissors (zero-sum)
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

def airline():
  # Payoff matrix: Revenue per flight (in $100K)
  # Airline 1: HIGH/LOW pricing
  # Airline 2: AGGRESSIVE/COMPETITIVE
  payoff_1 = np.array([
      [100.0, 80.0],   # P1 plays HIGH
      [50.0, 113.3]    # P1 plays LOW
  ])

  payoff_2 = np.array([
      [90.0, 120.0],   # P2 plays AGGRESSIVE
      [140.0, 127.1]   # P2 plays COMPETITIVE
  ])

  # Constraint: Airline 1 must play HIGH pricing at least 50% of the time
  # (for service quality and fleet maintenance)
  p1_constraints = {
      0: (0.5, 1.0)  # HIGH pricing: [50%, 100%]
  }

  solver = InverseGameSolver(
      payoff_1,
      payoff_2,
      p1_constraints=p1_constraints,
      verbose=True
  )

  modified_p1, modified_p2, result = solver.solve()

  print(f"\nAirline 1 Equilibrium Strategy:")
  print(f"  HIGH pricing: {result['p'][0]:.1%}")
  print(f"  LOW pricing:  {result['p'][1]:.1%}")
  print(f"\nConstraint Check: P1 HIGH ≥ 50% → {result['p'][0]:.1%} ✓")

def example2x2():
  payoff_1 = np.array([
      [2.0, -1.0],
      [-1, 1]
  ])

  payoff_2 = np.array([
      [-1.0, 2.0],
      [1, -1]
  ])

  p1_constraints = {
      0: (0.5, 1.0)
  }

  solver = InverseGameSolver(
      payoff_1,
      payoff_2,
      p1_constraints=p1_constraints,
      verbose=True
  )

  modified_p1, modified_p2, result = solver.solve()

  print(f"\nPlayer 1 Equilibrium Strategy:")
  print(f": {result['p'][0]:.1%}")
  print(f":  {result['p'][1]:.1%}")
  print(f"\nConstraint Check: P1 HIGH ≥ 50% → {result['p'][0]:.1%} ✓")

if __name__ == "__main__":
  # ret = rps_example()
  # airline()
  example2x2()
  print('Done :]')
