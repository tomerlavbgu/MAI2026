"""
FastAPI server to expose InverseGameSolver as a REST API.
This allows the Next.js frontend to call the solver.
"""

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from inverse_game_solver import InverseGameSolver
from typing import Dict, List, Tuple, Optional

app = FastAPI(title="Game Theory Solver API")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,  # Must be False when using "*"
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Add explicit OPTIONS handler for CORS preflight
@app.options("/solve")
async def options_solve(response: Response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return {"status": "ok"}


class Constraint(BaseModel):
    """Probability constraint for an action."""
    action_index: int
    min_prob: float
    max_prob: float


class SolveRequest(BaseModel):
    """Request to solve a game."""
    payoff_matrix_1: List[List[float]]
    payoff_matrix_2: List[List[float]]
    p1_constraints: List[Constraint] = []
    p2_constraints: List[Constraint] = []
    max_iterations: int = 500


class SolveResponse(BaseModel):
    """Response from the solver."""
    success: bool
    constraint_satisfied: bool
    original_equilibrium: Dict[str, List[float]]  # {"p": [...], "q": [...]}
    modified_equilibrium: Dict[str, List[float]]  # {"p": [...], "q": [...]}
    original_payoff_1: List[List[float]]
    original_payoff_2: List[List[float]]
    modified_payoff_1: List[List[float]]
    modified_payoff_2: List[List[float]]
    payoff_changes_1: List[List[float]]
    payoff_changes_2: List[List[float]]
    metrics: Dict[str, float]  # {"l1_distance": x, "l2_distance": y}


@app.get("/")
def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Game Theory Solver API"}


@app.post("/solve", response_model=SolveResponse)
def solve_game(request: SolveRequest, response: Response):
    """
    Solve the inverse game problem.

    Given:
    - Original payoff matrices
    - Probability constraints for both players

    Returns:
    - Modified payoff matrices
    - Original and modified equilibria
    - Distance metrics
    """
    # Set explicit CORS headers
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"

    print(f"\n{'='*70}")
    print(f"📥 Received solve request")
    print(f"Matrix 1 shape: {len(request.payoff_matrix_1)}x{len(request.payoff_matrix_1[0]) if request.payoff_matrix_1 else 0}")
    print(f"Matrix 2 shape: {len(request.payoff_matrix_2)}x{len(request.payoff_matrix_2[0]) if request.payoff_matrix_2 else 0}")
    print(f"P1 constraints: {request.p1_constraints}")
    print(f"P2 constraints: {request.p2_constraints}")
    print(f"{'='*70}\n")

    try:
        # Convert to numpy arrays
        payoff_1 = np.array(request.payoff_matrix_1, dtype=float)
        payoff_2 = np.array(request.payoff_matrix_2, dtype=float)

        # Convert constraints to dict format
        p1_constraints_dict = {
            c.action_index: (c.min_prob, c.max_prob)
            for c in request.p1_constraints
        }
        p2_constraints_dict = {
            c.action_index: (c.min_prob, c.max_prob)
            for c in request.p2_constraints
        }

        # Create solver
        solver = InverseGameSolver(
            payoff_1,
            payoff_2,
            p1_constraints=p1_constraints_dict,
            p2_constraints=p2_constraints_dict,
            verbose=False  # Disable printing to console
        )

        # Solve
        modified_p1, modified_p2, result = solver.solve(
            max_iterations=request.max_iterations
        )

        # Compute payoff changes
        changes_p1 = (modified_p1 - payoff_1).tolist()
        changes_p2 = (modified_p2 - payoff_2).tolist()

        # Build response
        response = SolveResponse(
            success=result['success'],
            constraint_satisfied=result['constraint_satisfied'],
            original_equilibrium={
                "p": result['original_p'].tolist(),
                "q": result['original_q'].tolist()
            },
            modified_equilibrium={
                "p": result['p'].tolist(),
                "q": result['q'].tolist()
            },
            original_payoff_1=payoff_1.tolist(),
            original_payoff_2=payoff_2.tolist(),
            modified_payoff_1=modified_p1.tolist(),
            modified_payoff_2=modified_p2.tolist(),
            payoff_changes_1=changes_p1,
            payoff_changes_2=changes_p2,
            metrics={
                "l1_distance": float(result['l1_distance']),
                "l2_distance": float(result['l2_distance'])
            }
        )

        print(f"✅ Solve completed successfully!")
        print(f"Success: {response.success}, Constraint satisfied: {response.constraint_satisfied}\n")
        return response

    except Exception as e:
        print(f"❌ ERROR in solve_game:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("=" * 70)
    print("Starting Game Theory Solver API Server")
    print("=" * 70)
    print("API URL: http://localhost:8000")
    print("Docs: http://localhost:8000/docs")
    print("=" * 70)
    uvicorn.run(app, host="0.0.0.0", port=8000)
