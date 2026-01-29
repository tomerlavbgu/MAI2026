"use client"

import { useState, useEffect } from "react"
import { PayoffMatrix } from "./payoff-matrix"
import { ProbabilitySlider } from "./probability-slider"
import { PerturbedMatrix } from "./perturbed-matrix"
import { EquilibriumGraph } from "./equilibrium-graph"
import { Spinner } from "@/components/ui/spinner"

interface SolverResult {
  success: boolean
  constraint_satisfied: boolean
  original_equilibrium: { p: number[]; q: number[] }
  modified_equilibrium: { p: number[]; q: number[] }
  original_payoff_1: number[][]
  original_payoff_2: number[][]
  modified_payoff_1: number[][]
  modified_payoff_2: number[][]
  payoff_changes_1: number[][]
  payoff_changes_2: number[][]
  metrics: { l1_distance: number; l2_distance: number }
}

// Preset game scenarios
interface GamePreset {
  id: string
  name: string
  description: string
  rows: number
  cols: number
  matrix1: number[][]
  matrix2: number[][]
  actionLabelsRows?: string[]
  actionLabelsCols?: string[]
}

const GAME_PRESETS: GamePreset[] = [
  // 2x2 Games
  {
    id: "prisoners-dilemma",
    name: "Prisoner's Dilemma",
    description: "Classic cooperation vs defection game",
    rows: 2,
    cols: 2,
    matrix1: [[3, 0], [5, 1]],
    matrix2: [[3, 5], [0, 1]],
  },
  {
    id: "battle-of-sexes",
    name: "Battle of the Sexes",
    description: "Coordination game with conflicting preferences",
    rows: 2,
    cols: 2,
    matrix1: [[2, 0], [0, 1]],
    matrix2: [[1, 0], [0, 2]],
  },
  {
    id: "matching-pennies",
    name: "Matching Pennies",
    description: "Zero-sum game with no pure Nash equilibrium",
    rows: 2,
    cols: 2,
    matrix1: [[1, -1], [-1, 1]],
    matrix2: [[-1, 1], [1, -1]],
  },
  // 2x3 Games
  {
    id: "asymmetric-coordination",
    name: "Asymmetric Coordination",
    description: "Coordination with different action spaces",
    rows: 2,
    cols: 3,
    matrix1: [[3, 0, 1], [0, 2, 0]],
    matrix2: [[3, 0, 0], [0, 2, 1]],
  },
  // 3x2 Games
  {
    id: "attacker-defender",
    name: "Attacker-Defender",
    description: "Strategic conflict with asymmetric options",
    rows: 3,
    cols: 2,
    matrix1: [[2, 0], [1, 3], [0, 1]],
    matrix2: [[0, 2], [3, 1], [1, 0]],
  },
  // 3x3 Games
  {
    id: "rock-paper-scissors",
    name: "Rock-Paper-Scissors",
    description: "Classic zero-sum game with cyclic dominance",
    rows: 3,
    cols: 3,
    matrix1: [
      [0, -1, 1],
      [1, 0, -1],
      [-1, 1, 0],
    ],
    matrix2: [
      [0, 1, -1],
      [-1, 0, 1],
      [1, -1, 0],
    ],
    actionLabelsRows: ["Rock", "Paper", "Scissors"],
    actionLabelsCols: ["Rock", "Paper", "Scissors"],
  },
]

// Custom debounce hook
function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value)
  const [isFirstRender, setIsFirstRender] = useState(true)

  useEffect(() => {
    // On first render, set immediately without delay
    if (isFirstRender) {
      setDebouncedValue(value)
      setIsFirstRender(false)
      return
    }

    const timer = setTimeout(() => setDebouncedValue(value), delay)
    return () => clearTimeout(timer)
  }, [value, delay, isFirstRender])

  return debouncedValue
}

export function GameTheorySolver() {
  const [rows, setRows] = useState(2)
  const [cols, setCols] = useState(2)
  const [matrix, setMatrix] = useState(GAME_PRESETS[0].matrix1)
  const [matrix2, setMatrix2] = useState(GAME_PRESETS[0].matrix2)
  const [player1Prob, setPlayer1Prob] = useState(40)
  const [player2Prob, setPlayer2Prob] = useState(50)
  const [selectedPreset, setSelectedPreset] = useState(GAME_PRESETS[0].id)
  const [solverResult, setSolverResult] = useState<SolverResult | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  // Debounce inputs by 500ms
  const debouncedMatrix = useDebounce(matrix, 500)
  const debouncedMatrix2 = useDebounce(matrix2, 500)
  const debouncedPlayer1Prob = useDebounce(player1Prob, 500)
  const debouncedPlayer2Prob = useDebounce(player2Prob, 500)

  // Call the solver API whenever debounced inputs change
  useEffect(() => {
    const controller = new AbortController()

    const solveProblem = async () => {
      setIsLoading(true)
      try {
        const response = await fetch("http://localhost:8000/solve", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            payoff_matrix_1: debouncedMatrix,
            payoff_matrix_2: debouncedMatrix2,
            p1_constraints: [
              {
                action_index: 0,
                min_prob: debouncedPlayer1Prob / 100,
                max_prob: debouncedPlayer1Prob / 100,
              },
            ],
            p2_constraints: [
              {
                action_index: 0,
                min_prob: debouncedPlayer2Prob / 100,
                max_prob: debouncedPlayer2Prob / 100,
              },
            ],
            max_iterations: 500,
          }),
          signal: controller.signal,
        })

        if (response.ok) {
          const data: SolverResult = await response.json()
          setSolverResult(data)
        } else {
          console.error("Solver API error:", response.status, response.statusText)
        }
      } catch (error) {
        if (error instanceof Error && error.name !== 'AbortError') {
          console.error("Error calling solver API:", error)
        }
      } finally {
        setIsLoading(false)
      }
    }

    solveProblem()

    return () => controller.abort()
  }, [debouncedMatrix, debouncedMatrix2, debouncedPlayer1Prob, debouncedPlayer2Prob])

  // Compute perturbed matrices from solver results
  const perturbedMatrix1 = solverResult
    ? solverResult.modified_payoff_1.map((row, i) =>
        row.map((val, j) => ({
          original: solverResult.original_payoff_1[i][j],
          perturbed: val,
        }))
      )
    : matrix.map((row) => row.map((val) => ({ original: val, perturbed: val })))

  const perturbedMatrix2 = solverResult
    ? solverResult.modified_payoff_2.map((row, i) =>
        row.map((val, j) => ({
          original: solverResult.original_payoff_2[i][j],
          perturbed: val,
        }))
      )
    : matrix2.map((row) => row.map((val) => ({ original: val, perturbed: val })))

  const updateMatrixValue = (row: number, col: number, value: number) => {
    const newMatrix = [...matrix]
    newMatrix[row][col] = value
    setMatrix(newMatrix)
  }

  const updateMatrix2Value = (row: number, col: number, value: number) => {
    const newMatrix = [...matrix2]
    newMatrix[row][col] = value
    setMatrix2(newMatrix)
  }

  const loadPreset = (presetId: string) => {
    const preset = GAME_PRESETS.find((p) => p.id === presetId)
    if (preset) {
      // Deep copy to ensure proper state update
      setMatrix(preset.matrix1.map(row => [...row]))
      setMatrix2(preset.matrix2.map(row => [...row]))
      setRows(preset.rows)
      setCols(preset.cols)
      setSelectedPreset(presetId)
    }
  }

  // Handle game size change from dropdown
  const handleGameSizeChange = (newRows: number, newCols: number) => {
    setRows(newRows)
    setCols(newCols)
    // Load first preset for this size
    const firstPreset = GAME_PRESETS.find((p) => p.rows === newRows && p.cols === newCols)
    if (firstPreset) {
      loadPreset(firstPreset.id)
    } else {
      // Create default matrices if no preset exists
      setMatrix(Array(newRows).fill(0).map(() => Array(newCols).fill(0)))
      setMatrix2(Array(newRows).fill(0).map(() => Array(newCols).fill(0)))
    }
  }

  // Filter presets by selected game size
  const filteredPresets = GAME_PRESETS.filter((p) => p.rows === rows && p.cols === cols)

  return (
    <div className="w-full max-w-[1600px] bg-[#2d2d44] rounded-lg overflow-hidden shadow-2xl mx-auto">
      {/* Header */}
      <div className="bg-[#1a1a2e] py-3 md:py-4 px-4 md:px-8 border-b border-[#3d3d5c]">
        <h1 className="text-white text-center text-base sm:text-lg md:text-xl font-semibold">
          Game Theory Solver - Constrained Optimization & Perturbation Analysis
        </h1>
      </div>

      {/* Main Content */}
      <div className="flex flex-col lg:flex-row">
        {/* Left Panel - Input */}
        <div className="p-4 sm:p-6 md:p-8 lg:border-r border-b lg:border-b-0 border-[#3d3d5c] flex flex-col items-center lg:w-1/2">
          <h2 className="text-white font-semibold mb-4 md:mb-5 text-base md:text-lg w-full text-center lg:text-left">
            INPUT: Matrices & Constraints
          </h2>

          {/* Game Size Selector - Dropdown */}
          <div className="mb-6 md:mb-8 w-full">
            <p className="text-gray-300 text-xs sm:text-sm mb-2 sm:mb-3 text-center lg:text-left">Game Size</p>
            <div className="relative">
              <select
                value={`${rows}x${cols}`}
                onChange={(e) => {
                  const [newRows, newCols] = e.target.value.split('x').map(Number)
                  handleGameSizeChange(newRows, newCols)
                }}
                className="w-full px-4 py-3 bg-gray-700 text-white rounded-lg border-2 border-gray-600 hover:border-blue-500 focus:border-blue-500 focus:outline-none transition-all cursor-pointer appearance-none text-sm sm:text-base font-medium shadow-lg"
                style={{
                  backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='white'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M19 9l-7 7-7-7'%3E%3C/path%3E%3C/svg%3E")`,
                  backgroundRepeat: 'no-repeat',
                  backgroundPosition: 'right 0.75rem center',
                  backgroundSize: '1.5rem',
                  paddingRight: '2.5rem'
                }}
              >
                <option value="2x2" className="bg-gray-800 py-2">2×2 Games</option>
                <option value="2x3" className="bg-gray-800 py-2">2×3 Games</option>
                <option value="3x2" className="bg-gray-800 py-2">3×2 Games</option>
                <option value="3x3" className="bg-gray-800 py-2">3×3 Games</option>
              </select>
            </div>
          </div>

          {/* Preset Selector */}
          <div className="mb-6 md:mb-8 w-full">
            <p className="text-gray-300 text-xs sm:text-sm mb-2 sm:mb-3 text-center lg:text-left">Game Presets</p>
            <div className="flex flex-col gap-2">
              {filteredPresets.map((preset) => (
                <button
                  key={preset.id}
                  onClick={() => loadPreset(preset.id)}
                  className={`w-full px-3 py-2 rounded text-left transition-colors ${
                    selectedPreset === preset.id
                      ? "bg-blue-600 text-white"
                      : "bg-gray-700 text-gray-300 hover:bg-gray-600"
                  }`}
                >
                  <div className="font-medium text-sm">{preset.name}</div>
                  <div className="text-xs opacity-80">{preset.description}</div>
                </button>
              ))}
            </div>
          </div>

          <div className="mb-6 md:mb-8 flex flex-col items-center w-full">
            <p className="text-gray-300 text-sm md:text-base mb-3 md:mb-4 w-full text-center lg:text-left">
              Player 1 Payoff Matrix
            </p>
            <div className="flex justify-center w-full overflow-x-auto">
              <PayoffMatrix
                matrix={matrix}
                onValueChange={updateMatrixValue}
                rows={rows}
                cols={cols}
                actionLabelsRows={GAME_PRESETS.find((p) => p.id === selectedPreset)?.actionLabelsRows}
                actionLabelsCols={GAME_PRESETS.find((p) => p.id === selectedPreset)?.actionLabelsCols}
              />
            </div>
          </div>

          <div className="w-full flex justify-center mb-6 md:mb-8">
            <div className="w-full max-w-xs sm:max-w-sm">
              <ProbabilitySlider
                value={player1Prob}
                onChange={setPlayer1Prob}
                label="Player 1 Probability Constraint (0-100%)"
              />
            </div>
          </div>

          <div className="mb-6 md:mb-8 flex flex-col items-center w-full">
            <p className="text-gray-300 text-sm md:text-base mb-3 md:mb-4 w-full text-center lg:text-left">
              Player 2 Payoff Matrix
            </p>
            <div className="flex justify-center w-full overflow-x-auto">
              <PayoffMatrix
                matrix={matrix2}
                onValueChange={updateMatrix2Value}
                rows={rows}
                cols={cols}
                actionLabelsRows={GAME_PRESETS.find((p) => p.id === selectedPreset)?.actionLabelsRows}
                actionLabelsCols={GAME_PRESETS.find((p) => p.id === selectedPreset)?.actionLabelsCols}
              />
            </div>
          </div>

          <div className="w-full flex justify-center">
            <div className="w-full max-w-xs sm:max-w-sm">
              <ProbabilitySlider
                value={player2Prob}
                onChange={setPlayer2Prob}
                label="Player 2 Probability Constraint (0-100%)"
              />
            </div>
          </div>
        </div>

        {/* Right Panel - Output */}
        <div className="p-4 sm:p-6 md:p-8 bg-[#f5f5f5] flex flex-col items-center lg:w-1/2 relative">
          <h2 className="text-gray-800 font-semibold mb-4 md:mb-5 text-base md:text-lg w-full text-center lg:text-left">
            OUTPUT: Perturbation & Equilibrium Analysis
          </h2>

          {solverResult && (
            <div className="mb-4 w-full bg-white rounded-lg p-3 shadow">
              <p className="text-sm font-semibold text-gray-700 mb-2">Solver Metrics</p>
              <div className="grid grid-cols-3 gap-2 text-xs mb-2">
                <div>
                  <span className="text-gray-600">L1 Distance:</span>{" "}
                  <span className="font-mono">{solverResult.metrics.l1_distance.toFixed(3)}</span>
                </div>
                <div>
                  <span className="text-gray-600">L2 Distance:</span>{" "}
                  <span className="font-mono">{solverResult.metrics.l2_distance.toFixed(3)}</span>
                </div>
                <div>
                  <span className="text-gray-600">Status:</span>{" "}
                  <span className={solverResult.constraint_satisfied ? "text-green-600" : "text-red-600"}>
                    {solverResult.constraint_satisfied ? "✓ Satisfied" : "✗ Not Satisfied"}
                  </span>
                </div>
              </div>
              <p className="text-xs text-gray-500 italic">
                These distances measure how much the payoff matrices were perturbed to satisfy constraints. <strong>Lower = better</strong> (smaller changes needed). L1 = total sum of all changes; L2 = emphasizes whether changes are spread evenly (lower L2) or concentrated in few cells (higher L2).
              </p>
            </div>
          )}

          <div className="mb-4 md:mb-6 w-full">
            <p className="text-gray-700 text-sm md:text-base mb-3 md:mb-4 font-medium w-full text-center lg:text-left">
              Modified Matrices (from Solver)
            </p>
            <div className={`flex ${(rows !== 2 || cols !== 2) ? 'flex-col' : 'flex-col lg:flex-row'} gap-4 md:gap-6 lg:gap-8 w-full items-center justify-center`}>
              {/* Player 1 Modified Matrix */}
              <div className="flex flex-col items-center">
                <p className="text-gray-600 text-xs md:text-sm mb-2 font-medium">Player 1</p>
                <div className="flex justify-center">
                  <PerturbedMatrix
                    matrix={perturbedMatrix1}
                    rows={rows}
                    cols={cols}
                    actionLabelsRows={GAME_PRESETS.find((p) => p.id === selectedPreset)?.actionLabelsRows}
                    actionLabelsCols={GAME_PRESETS.find((p) => p.id === selectedPreset)?.actionLabelsCols}
                  />
                </div>
              </div>
              {/* Player 2 Modified Matrix */}
              <div className="flex flex-col items-center">
                <p className="text-gray-600 text-xs md:text-sm mb-2 font-medium">Player 2</p>
                <div className="flex justify-center">
                  <PerturbedMatrix
                    matrix={perturbedMatrix2}
                    rows={rows}
                    cols={cols}
                    actionLabelsRows={GAME_PRESETS.find((p) => p.id === selectedPreset)?.actionLabelsRows}
                    actionLabelsCols={GAME_PRESETS.find((p) => p.id === selectedPreset)?.actionLabelsCols}
                  />
                </div>
              </div>
            </div>
          </div>

          <div className="w-full flex justify-center overflow-x-auto">
            <EquilibriumGraph
              rows={rows}
              cols={cols}
              player1Prob={player1Prob}
              matrix={matrix}
              matrix2={matrix2}
              solverResult={solverResult}
            />
          </div>

          {/* Loading Overlay with Blur */}
          {isLoading && (
            <div className="absolute inset-0 bg-white/60 backdrop-blur-sm flex items-center justify-center z-10">
              <div className="flex flex-col items-center justify-center gap-3">
                <Spinner className="h-12 w-12" />
                <p className="text-sm font-medium text-gray-700">Computing optimal solution...</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
