# Backend API Server

FastAPI server for the Game Theory Inverse Problem Solver.

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python api_server.py
# OR
uvicorn api_server:app --reload
```

Server runs on: `http://localhost:8000`

## API Endpoints

### POST /solve
Solves the inverse game theory problem with given constraints.

**Request Body:**
```json
{
  "payoff_matrix_1": [[3, 0], [5, 1]],
  "payoff_matrix_2": [[3, 5], [0, 1]],
  "p1_constraints": [
    {
      "action_index": 0,
      "min_prob": 0.4,
      "max_prob": 0.4
    }
  ],
  "p2_constraints": [
    {
      "action_index": 0,
      "min_prob": 0.5,
      "max_prob": 0.5
    }
  ],
  "max_iterations": 500
}
```

**Response:**
```json
{
  "success": true,
  "constraint_satisfied": true,
  "original_equilibrium": {
    "p": [0.0, 1.0],
    "q": [0.0, 1.0]
  },
  "modified_equilibrium": {
    "p": [0.4, 0.6],
    "q": [0.5, 0.5]
  },
  "original_payoff_1": [[3, 0], [5, 1]],
  "original_payoff_2": [[3, 5], [0, 1]],
  "modified_payoff_1": [[3.00, -0.00], [5.00, 1.00]],
  "modified_payoff_2": [[3.00, 6.00], [0.00, 1.00]],
  "metrics": {
    "l1_distance": 0.000,
    "l2_distance": 0.000
  }
}
```

## Deployment to Railway

1. Connect this GitHub repository to Railway
2. Set root directory to: `backend`
3. Railway will automatically:
   - Detect Python
   - Install `requirements.txt`
   - Run the start command from `railway.json`

## Environment Variables

No environment variables required for basic operation.

## Dependencies

- FastAPI: Web framework
- Uvicorn: ASGI server
- NumPy: Numerical computing
- SciPy: Scientific computing (optimization)
- Pydantic: Data validation

See `requirements.txt` for specific versions.
