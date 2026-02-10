# Game Theory Solver - Constrained Optimization & Perturbation Analysis

![BGU MAI Project](https://img.shields.io/badge/BGU-MAI%202026-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![Next.js](https://img.shields.io/badge/Next.js-16-black)

Interactive web application for solving inverse game theory problems with minimal payoff perturbations.

## 🎯 Overview

This project implements an optimization-based algorithm that finds the minimal modifications to game payoff matrices needed to achieve desired Nash Equilibrium constraints. It includes both the core algorithm and a full-stack web interface for visualization and analysis.

## 🚀 Live Demo

- **Frontend**: [https://mai2026.vercel.app](https://mai2026.vercel.app) _(will be deployed)_
- **Backend API**: [https://mai2026-backend.up.railway.app](https://mai2026-backend.up.railway.app) _(will be deployed)_
- **API Documentation**: `/docs` endpoint on backend

## 📁 Project Structure

```
MAI2026/
├── backend/                     # FastAPI server
│   ├── api_server.py           # REST API endpoints
│   ├── requirements.txt        # Python dependencies
│   └── railway.json            # Railway deployment config
│
├── frontend/                    # Next.js application
│   ├── app/                    # Next.js 16 app directory
│   ├── components/             # React components
│   │   ├── game-theory-solver.tsx
│   │   ├── equilibrium-graph.tsx
│   │   ├── payoff-matrix.tsx
│   │   └── ...
│   └── package.json
│
├── docs/                        # Documentation
│   ├── implementation_guide.md
│   ├── project_summary.md
│   └── usage_examples.md
│
├── inverse_game_solver.py      # Core algorithm (can be used standalone)
├── examples.py                  # Example game scenarios
└── README.md                    # This file
```

## 🛠️ Technology Stack

**Backend:**
- Python 3.8+
- FastAPI (REST API)
- NumPy (Matrix operations)
- SciPy (Optimization)

**Frontend:**
- Next.js 16 (React framework)
- TypeScript
- TailwindCSS (Styling)
- Custom SVG visualizations

**Deployment:**
- Railway (Backend)
- Vercel (Frontend)

## 🎮 Features

### Supported Game Sizes
- 2×2 games (with complete best response visualization)
- 2×3 games
- 3×2 games
- 3×3 games

### Preset Games
- **Prisoner's Dilemma** (2×2)
- **Battle of the Sexes** (2×2)
- **Matching Pennies** (2×2)
- **Asymmetric Coordination** (2×3)
- **Attacker-Defender** (3×2)
- **Rock-Paper-Scissors** (3×3)

### Visualizations
- Interactive Nash Equilibrium graphs
- Best response functions (step functions for 2×2)
- Equilibrium shift arrows
- Real-time perturbation analysis
- L1/L2 distance metrics

## 🚀 Quick Start

### Local Development

#### 1. Clone the repository
```bash
git clone https://github.com/tomerlavbgu/MAI2026.git
cd MAI2026
```

#### 2. Start the Backend
```bash
cd backend
pip install -r requirements.txt
python api_server.py
```
Backend runs at: `http://localhost:8000`

#### 3. Start the Frontend
```bash
cd frontend
npm install
npm run dev
```
Frontend runs at: `http://localhost:3000`

## 📖 Using the Core Algorithm (Standalone)

You can use the solver independently without the web interface:

```python
from inverse_game_solver import InverseGameSolver
import numpy as np

# Define payoff matrices
payoff_1 = np.array([[3, 0], [5, 1]])
payoff_2 = np.array([[3, 5], [0, 1]])

# Create solver instance
solver = InverseGameSolver(payoff_1, payoff_2)

# Define constraints: Player 1 must play action 0 with probability 0.4
p1_constraints = {0: (0.4, 0.4)}

# Solve
result = solver.solve(p1_constraints=p1_constraints)

print(f"Success: {result['success']}")
print(f"Modified payoffs: {result['modified_payoff_1']}")
print(f"New equilibrium: {result['modified_equilibrium']}")
```

See `examples.py` for more usage examples.

## 📚 Documentation

- [Implementation Guide](docs/implementation_guide.md) - Algorithm details
- [Project Summary](docs/project_summary.md) - Overview and methodology
- [Usage Examples](docs/usage_examples.md) - Code examples
- [Backend README](backend/README.md) - API documentation
- [Frontend README](frontend/README.md) - UI documentation

## 🔧 API Reference

### POST /solve

Solves the inverse game theory problem.

**Request:**
```json
{
  "payoff_matrix_1": [[3, 0], [5, 1]],
  "payoff_matrix_2": [[3, 5], [0, 1]],
  "p1_constraints": [{"action_index": 0, "min_prob": 0.4, "max_prob": 0.4}],
  "p2_constraints": [{"action_index": 0, "min_prob": 0.5, "max_prob": 0.5}],
  "max_iterations": 500
}
```

**Response:**
```json
{
  "success": true,
  "constraint_satisfied": true,
  "original_equilibrium": {"p": [0.0, 1.0], "q": [0.0, 1.0]},
  "modified_equilibrium": {"p": [0.4, 0.6], "q": [0.5, 0.5]},
  "modified_payoff_1": [[3.0, 0.0], [5.0, 1.0]],
  "modified_payoff_2": [[3.0, 6.0], [0.0, 1.0]],
  "metrics": {"l1_distance": 1.0, "l2_distance": 1.0}
}
```

## 🎓 Academic Context

This project is part of the Multi-Agent Interaction (MAI) course at Ben-Gurion University, 2026.

**Contributors:**
- Tomer Lav (GitHub: [@tomerlavbgu](https://github.com/tomerlavbgu))
- Shaik (shaikar@post.bgu.ac.il)

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

This is an academic project. For questions or suggestions, please open an issue on GitHub.

## 🐛 Troubleshooting

### CORS Errors
- Ensure backend is running on port 8000
- Check CORS middleware in `api_server.py`

### Solver Not Converging
- Increase `max_iterations` (default: 500)
- Verify constraint feasibility
- Check matrix values

### Frontend Not Connecting
- Verify `NEXT_PUBLIC_API_URL` environment variable
- Check backend is accessible
- Review browser console for errors

## 📧 Contact

For questions about this project:
- Open an issue on GitHub
- Contact: shaikar@post.bgu.ac.il
