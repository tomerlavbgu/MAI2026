# Frontend - Game Theory Solver UI

Interactive Next.js application for visualizing and solving inverse game theory problems.

## Features

- 2×2, 2×3, 3×2, and 3×3 game support
- Visual Nash Equilibrium graphs with best response functions
- Real-time perturbation analysis
- Preset game scenarios (Prisoner's Dilemma, Battle of Sexes, etc.)
- Probability constraint sliders
- Responsive design

## Local Development

```bash
# Install dependencies
npm install

# Run development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

## Environment Variables

Create `.env.local` for local development:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

For production (Vercel), set:
```env
NEXT_PUBLIC_API_URL=https://your-railway-backend.up.railway.app
```

## Build for Production

```bash
npm run build
npm start
```

## Deployment to Vercel

1. Connect this GitHub repository to Vercel
2. Set root directory to: `frontend`
3. Add environment variable: `NEXT_PUBLIC_API_URL`
4. Deploy

Vercel will automatically:
- Detect Next.js
- Install dependencies
- Build and deploy

## Tech Stack

- **Framework**: Next.js 16
- **UI**: React, TailwindCSS
- **Charts**: Custom SVG visualizations
- **API**: Fetch API to FastAPI backend

## Project Structure

```
frontend/
├── app/                  # Next.js app directory
│   ├── page.tsx         # Main page
│   └── layout.tsx       # Root layout
├── components/          # React components
│   ├── game-theory-solver.tsx
│   ├── equilibrium-graph.tsx
│   ├── payoff-matrix.tsx
│   └── ...
├── public/             # Static assets
└── package.json        # Dependencies
```

## Game Presets

### 2×2 Games:
- Prisoner's Dilemma
- Battle of the Sexes
- Matching Pennies

### 2×3 Games:
- Asymmetric Coordination

### 3×2 Games:
- Attacker-Defender

### 3×3 Games:
- Rock-Paper-Scissors

## Graph Visualization

- **2×2 games**: Shows complete best response functions as step functions
- **Non-2×2 games**: Shows equilibrium shift with arrow visualization
- Interactive equilibrium points with hover states
- Real-time updates as constraints change
