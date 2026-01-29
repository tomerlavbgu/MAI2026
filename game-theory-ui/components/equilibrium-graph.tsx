"use client"

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

interface EquilibriumGraphProps {
  player1Prob: number
  matrix: number[][]
  matrix2?: number[][]
  solverResult?: SolverResult | null
  rows: number
  cols: number
}

export function EquilibriumGraph({ player1Prob, matrix, matrix2, solverResult, rows, cols }: EquilibriumGraphProps) {
  const width = 520
  const height = 340
  const padding = { top: 50, right: 60, bottom: 80, left: 110 }
  const graphWidth = width - padding.left - padding.right
  const graphHeight = height - padding.top - padding.bottom

  // Scale functions
  const scaleX = (x: number) => padding.left + x * graphWidth
  const scaleY = (y: number) => padding.top + (1 - y) * graphHeight

  // Generate tick marks
  const xTicks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
  const yTicks = [0, 0.2, 0.4, 0.6, 0.8, 1]

  // For non-2x2 games, show 2D projection with arrow (like 3x3)
  if (rows !== 2 || cols !== 2) {
    // Extract equilibrium points if solver result exists
    let originalNE = null
    let modifiedNE = null

    // Get payoff matrices
    const originalPayoff1 = solverResult?.original_payoff_1 || matrix
    const originalPayoff2 = solverResult?.original_payoff_2 || (matrix2 || matrix.map(row => row.map(val => -val)))
    const modifiedPayoff1 = solverResult?.modified_payoff_1 || originalPayoff1
    const modifiedPayoff2 = solverResult?.modified_payoff_2 || originalPayoff2

    if (solverResult) {
      // Plot first two probabilities (x = p[0], y = p[1])
      originalNE = {
        x: solverResult.original_equilibrium.p[0],
        y: solverResult.original_equilibrium.p[1]
      }
      modifiedNE = {
        x: solverResult.modified_equilibrium.p[0],
        y: solverResult.modified_equilibrium.p[1]
      }
    }

    return (
      <div className="mt-2 sm:mt-4 w-full max-w-2xl mx-auto">
        <h3 className="text-gray-700 text-base sm:text-lg md:text-xl font-semibold mb-3 sm:mb-5 text-center">
          Mixed Strategy Equilibrium Graph
        </h3>

        {/* Explanation Box */}
        <div className="mb-3 px-4 py-3 bg-blue-50 border border-blue-200 rounded-lg text-sm text-gray-700">
          <p className="font-semibold text-blue-900 mb-1">Nash Equilibrium Shift:</p>
          <p className="mb-2 text-xs">
            This graph shows how the Nash Equilibrium shifts after perturbation. The x-axis represents Player 1's probability of playing Rock,
            and the y-axis represents their probability of playing Paper.
          </p>
          <ul className="text-xs space-y-1 ml-4 list-disc">
            <li><span className="font-medium text-gray-800">Black dot:</span> Original Nash Equilibrium</li>
            <li><span className="font-medium text-orange-600">Orange dot:</span> Modified Nash Equilibrium after perturbation</li>
            {modifiedNE && originalNE && (Math.abs(modifiedNE.x - originalNE.x) > 0.01 || Math.abs(modifiedNE.y - originalNE.y) > 0.01) && (
              <li><span className="font-medium text-purple-600">Arrow:</span> Direction of equilibrium shift</li>
            )}
          </ul>
        </div>

        {/* SVG Graph */}
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto max-w-full bg-white border border-gray-200 rounded">
          {/* Grid lines */}
          {xTicks.map((tick) => (
            <line
              key={`grid-x-${tick}`}
              x1={scaleX(tick)}
              y1={scaleY(0)}
              x2={scaleX(tick)}
              y2={scaleY(1)}
              stroke="#e0e0e0"
              strokeWidth="1"
            />
          ))}
          {yTicks.map((tick) => (
            <line
              key={`grid-y-${tick}`}
              x1={scaleX(0)}
              y1={scaleY(tick)}
              x2={scaleX(1)}
              y2={scaleY(tick)}
              stroke="#e0e0e0"
              strokeWidth="1"
            />
          ))}

          {/* Axes */}
          <line x1={scaleX(0)} y1={scaleY(0)} x2={scaleX(1)} y2={scaleY(0)} stroke="#333" strokeWidth="2" />
          <line x1={scaleX(0)} y1={scaleY(0)} x2={scaleX(0)} y2={scaleY(1)} stroke="#333" strokeWidth="2" />

          {/* X-axis ticks and labels */}
          {xTicks.map((tick) => (
            <g key={`x-tick-${tick}`}>
              <line x1={scaleX(tick)} y1={scaleY(0)} x2={scaleX(tick)} y2={scaleY(0) + 6} stroke="#333" strokeWidth="1.5" />
              <text x={scaleX(tick)} y={scaleY(0) + 22} textAnchor="middle" fontSize="14" fill="#444">
                {tick}
              </text>
            </g>
          ))}

          {/* Y-axis ticks and labels */}
          {yTicks.map((tick) => (
            <g key={`y-tick-${tick}`}>
              <line x1={scaleX(0) - 6} y1={scaleY(tick)} x2={scaleX(0)} y2={scaleY(tick)} stroke="#333" strokeWidth="1.5" />
              <text x={scaleX(0) - 12} y={scaleY(tick) + 5} textAnchor="end" fontSize="14" fill="#444">
                {tick}
              </text>
            </g>
          ))}

          {/* Axis labels */}
          <text x={scaleX(0.5)} y={height - 10} textAnchor="middle" fontSize="16" fill="#333" fontWeight="500">
            ⬥ Prob Player 1 Rock
          </text>
          <text
            x={20}
            y={scaleY(0.5)}
            textAnchor="middle"
            fontSize="16"
            fill="#333"
            fontWeight="500"
            transform={`rotate(-90, 20, ${scaleY(0.5)})`}
          >
            ⬥ Prob Player 1 Paper
          </text>

          {/* Draw arrow showing equilibrium shift if there's a significant change */}
          {originalNE && modifiedNE && (Math.abs(modifiedNE.x - originalNE.x) > 0.01 || Math.abs(modifiedNE.y - originalNE.y) > 0.01) && (
            <>
              <defs>
                <marker
                  id="arrowhead-3x3"
                  markerWidth="10"
                  markerHeight="10"
                  refX="9"
                  refY="3"
                  orient="auto"
                  markerUnits="strokeWidth"
                >
                  <path d="M0,0 L0,6 L9,3 z" fill="#9333ea" />
                </marker>
              </defs>
              <line
                x1={scaleX(originalNE.x)}
                y1={scaleY(originalNE.y)}
                x2={scaleX(modifiedNE.x)}
                y2={scaleY(modifiedNE.y)}
                stroke="#9333ea"
                strokeWidth="3"
                strokeDasharray="5,5"
                markerEnd="url(#arrowhead-3x3)"
              />
            </>
          )}

          {/* Plot Original NE */}
          {originalNE && (
            <>
              <circle cx={scaleX(originalNE.x)} cy={scaleY(originalNE.y)} r="12" fill="#333" stroke="#fff" strokeWidth="3" />
              <text
                x={scaleX(originalNE.x) - 35}
                y={scaleY(originalNE.y) - 20}
                fontSize="14"
                fill="#333"
                fontWeight="600"
              >
                Original NE
              </text>
              <text
                x={scaleX(originalNE.x) - 25}
                y={scaleY(originalNE.y) - 5}
                fontSize="13"
                fill="#555"
              >
                ({originalNE.x.toFixed(2)}, {originalNE.y.toFixed(2)})
              </text>
            </>
          )}

          {/* Plot Modified NE */}
          {modifiedNE && (Math.abs(modifiedNE.x - (originalNE?.x || 0)) > 0.01 || Math.abs(modifiedNE.y - (originalNE?.y || 0)) > 0.01) && (
            <>
              <circle cx={scaleX(modifiedNE.x)} cy={scaleY(modifiedNE.y)} r="12" fill="#e8a040" stroke="#fff" strokeWidth="3" />
              <text
                x={scaleX(modifiedNE.x) + 18}
                y={scaleY(modifiedNE.y) - 10}
                fontSize="14"
                fill="#e8a040"
                fontWeight="600"
              >
                Modified NE
              </text>
              <text
                x={scaleX(modifiedNE.x) + 18}
                y={scaleY(modifiedNE.y) + 8}
                fontSize="13"
                fill="#555"
              >
                ({modifiedNE.x.toFixed(2)}, {modifiedNE.y.toFixed(2)})
              </text>
            </>
          )}
        </svg>
      </div>
    )
  }

  // Helper function to calculate best response functions
  const calculateBestResponses = (payoff1: number[][], payoff2: number[][]) => {
    // For Player 1's payoff matrix: [[a, b], [c, d]]
    const a = payoff1[0][0]
    const b = payoff1[0][1]
    const c = payoff1[1][0]
    const d = payoff1[1][1]

    // Player 1's best response function (as a function of Player 2's q):
    // Player 1 is indifferent between actions A and B when:
    // q*a + (1-q)*b = q*c + (1-q)*d
    // Solving for q: q = (d - b) / (a - b - c + d)
    const denomP1 = (a - b - c + d)
    const p1IndifferentQ = denomP1 !== 0 ? (d - b) / denomP1 : 0.5
    const player1BestResponseQ = Math.max(0, Math.min(1, p1IndifferentQ))

    // For Player 2's payoff matrix: [[w, x], [y, z]]
    const w = payoff2[0][0]
    const x = payoff2[0][1]
    const y = payoff2[1][0]
    const z = payoff2[1][1]

    // Player 2's best response function (as a function of Player 1's p):
    // Player 2 is indifferent between actions A and B when:
    // p*w + (1-p)*y = p*x + (1-p)*z
    // Solving for p: p = (z - y) / (w - x - y + z)
    const denomP2 = (w - x - y + z)
    const p2IndifferentP = denomP2 !== 0 ? (z - y) / denomP2 : 0.5
    const player2BestResponseP = Math.max(0, Math.min(1, p2IndifferentP))

    return { player1BestResponseQ, player2BestResponseP }
  }

  // Calculate best responses for ORIGINAL matrices
  const originalPayoff1 = solverResult?.original_payoff_1 || matrix
  const originalPayoff2 = solverResult?.original_payoff_2 || (matrix2 || matrix.map(row => row.map(val => -val)))
  const originalBR = calculateBestResponses(originalPayoff1, originalPayoff2)

  // Calculate best responses for MODIFIED matrices (if perturbation exists)
  let modifiedBR = originalBR
  if (solverResult?.modified_payoff_1 && solverResult?.modified_payoff_2) {
    modifiedBR = calculateBestResponses(solverResult.modified_payoff_1, solverResult.modified_payoff_2)
  }

  // Use original best responses for backward compatibility
  const player1BestResponseQ = originalBR.player1BestResponseQ
  const player2BestResponseP = originalBR.player2BestResponseP


  // Calculate Nash equilibrium from solver result or use line intersection as fallback
  let nashX, nashY
  if (solverResult?.original_equilibrium) {
    // Use real Nash equilibrium from backend
    // p[0] is probability of action A for Player 1 (x-axis)
    // q[0] is probability of action A for Player 2 (y-axis)
    nashX = solverResult.original_equilibrium.p[0]
    nashY = solverResult.original_equilibrium.q[0]
  } else {
    // Fallback: calculate as intersection of best response lines
    // Original lines intersect at (player2BestResponseP, player1BestResponseQ)
    nashX = originalBR.player2BestResponseP
    nashY = originalBR.player1BestResponseQ
  }

  const nash = { x: nashX, y: nashY }

  // Calculate New NE position from solver result
  let newNEX, newNEY
  if (solverResult?.modified_equilibrium) {
    // Use real modified Nash equilibrium from backend
    newNEX = solverResult.modified_equilibrium.p[0]
    newNEY = solverResult.modified_equilibrium.q[0]

    // Clamp to graph bounds [0, 1]
    newNEX = Math.max(0, Math.min(1, newNEX))
    newNEY = Math.max(0, Math.min(1, newNEY))
  } else {
    // Fallback: if no modified equilibrium, use original
    newNEX = nash.x
    newNEY = nash.y
  }

  const newNE = { x: newNEX, y: newNEY }

  // Detect dominant strategies (indifference points at boundaries)
  const threshold = 0.05
  const originalP1DominantB = originalBR.player1BestResponseQ <= threshold // Always plays B (p=0)
  const originalP1DominantA = originalBR.player1BestResponseQ >= (1 - threshold) // Always plays A (p=1)
  const originalP2DominantB = originalBR.player2BestResponseP <= threshold // Always plays B (q=0)
  const originalP2DominantA = originalBR.player2BestResponseP >= (1 - threshold) // Always plays A (q=1)

  const modifiedP1DominantB = modifiedBR.player1BestResponseQ <= threshold
  const modifiedP1DominantA = modifiedBR.player1BestResponseQ >= (1 - threshold)
  const modifiedP2DominantB = modifiedBR.player2BestResponseP <= threshold
  const modifiedP2DominantA = modifiedBR.player2BestResponseP >= (1 - threshold)

  // Best response step functions for ORIGINAL matrices
  // If dominant strategy, show entire edge; otherwise show step function
  const originalBR1_bottom = originalP1DominantB
    ? { x: 0, y: 0, x2: 0, y2: 1 } // Full left edge for dominant B
    : { x: 0, y: 0, x2: 0, y2: originalBR.player1BestResponseQ }
  const originalBR1_middle = (!originalP1DominantB && !originalP1DominantA)
    ? { x: 0, y: originalBR.player1BestResponseQ, x2: 1, y2: originalBR.player1BestResponseQ }
    : null // No middle line for dominant strategies
  const originalBR1_top = originalP1DominantA
    ? { x: 1, y: 0, x2: 1, y2: 1 } // Full right edge for dominant A
    : { x: 1, y: originalBR.player1BestResponseQ, x2: 1, y2: 1 }

  const originalBR2_left = originalP2DominantA
    ? { x: 0, y: 1, x2: 1, y2: 1 } // Full top edge for dominant A
    : { x: 0, y: 1, x2: originalBR.player2BestResponseP, y2: 1 }
  const originalBR2_middle = (!originalP2DominantB && !originalP2DominantA)
    ? { x: originalBR.player2BestResponseP, y: 0, x2: originalBR.player2BestResponseP, y2: 1 }
    : null // No middle line for dominant strategies
  const originalBR2_right = originalP2DominantB
    ? { x: 0, y: 0, x2: 1, y2: 0 } // Full bottom edge for dominant B
    : { x: originalBR.player2BestResponseP, y: 0, x2: 1, y2: 0 }

  // Best response step functions for MODIFIED matrices
  const modifiedBR1_bottom = modifiedP1DominantB
    ? { x: 0, y: 0, x2: 0, y2: 1 }
    : { x: 0, y: 0, x2: 0, y2: modifiedBR.player1BestResponseQ }
  const modifiedBR1_middle = (!modifiedP1DominantB && !modifiedP1DominantA)
    ? { x: 0, y: modifiedBR.player1BestResponseQ, x2: 1, y2: modifiedBR.player1BestResponseQ }
    : null
  const modifiedBR1_top = modifiedP1DominantA
    ? { x: 1, y: 0, x2: 1, y2: 1 }
    : { x: 1, y: modifiedBR.player1BestResponseQ, x2: 1, y2: 1 }

  const modifiedBR2_left = modifiedP2DominantA
    ? { x: 0, y: 1, x2: 1, y2: 1 }
    : { x: 0, y: 1, x2: modifiedBR.player2BestResponseP, y2: 1 }
  const modifiedBR2_middle = (!modifiedP2DominantB && !modifiedP2DominantA)
    ? { x: modifiedBR.player2BestResponseP, y: 0, x2: modifiedBR.player2BestResponseP, y2: 1 }
    : null
  const modifiedBR2_right = modifiedP2DominantB
    ? { x: 0, y: 0, x2: 1, y2: 0 }
    : { x: modifiedBR.player2BestResponseP, y: 0, x2: 1, y2: 0 }

  // Nash equilibrium is where these lines intersect
  const theoreticalNash = { x: player2BestResponseP, y: player1BestResponseQ }

  // Check if there's a perturbation applied
  const hasPerturbation = solverResult && (
    Math.abs(modifiedBR.player1BestResponseQ - originalBR.player1BestResponseQ) > 0.01 ||
    Math.abs(modifiedBR.player2BestResponseP - originalBR.player2BestResponseP) > 0.01
  )

  // Debug logging
  console.log('=== Equilibrium Graph Debug ===')
  console.log('Original Payoff Matrix (Player 1):', originalPayoff1)
  console.log('Original Payoff Matrix (Player 2):', originalPayoff2)
  console.log('Theoretical Nash (from best responses):', theoreticalNash)
  if (solverResult) {
    console.log('Modified Payoff Matrix (Player 1):', solverResult.modified_payoff_1)
    console.log('Modified Payoff Matrix (Player 2):', solverResult.modified_payoff_2)
    console.log('Original NE from backend:', solverResult.original_equilibrium)
    console.log('Modified NE from backend:', solverResult.modified_equilibrium)
    console.log('Nash position (black dot):', nash)
    console.log('NewNE position (orange dot):', newNE)
  }
  console.log('Original Blue line (P1 best response): horizontal at q =', originalBR.player1BestResponseQ)
  console.log('Original Orange line (P2 best response): vertical at p =', originalBR.player2BestResponseP)
  console.log('Modified Blue line (P1 best response): horizontal at q =', modifiedBR.player1BestResponseQ)
  console.log('Modified Orange line (P2 best response): vertical at p =', modifiedBR.player2BestResponseP)

  return (
    <div className="mt-2 sm:mt-4 w-full max-w-2xl mx-auto">
      <h3 className="text-gray-700 text-base sm:text-lg md:text-xl font-semibold mb-3 sm:mb-5 text-center">
        Mixed Strategy Equilibrium Graph
      </h3>
      <div className="mb-3 px-4 py-3 bg-blue-50 border border-blue-200 rounded-lg text-sm text-gray-700">
        <p className="font-semibold text-blue-900 mb-1">About the Graph:</p>
        <p className="mb-2">
          The perpendicular lines show <strong>best response functions</strong> for each player in this 2×2 game.
          In 2×2 games, best responses are step functions with a single indifference point - shown here as straight lines.
        </p>
        <ul className="space-y-1 ml-4 list-disc text-xs">
          <li><strong>Blue horizontal line:</strong> Player 1's best response (indifferent at this q-value)</li>
          <li><strong>Orange vertical line:</strong> Player 2's best response (indifferent at this p-value)</li>
          <li><strong>Gray dashed lines:</strong> Original best responses before perturbation</li>
          <li><strong>Black dot:</strong> Original Nash Equilibrium (intersection of gray lines)</li>
          {hasPerturbation && <li><strong>Orange dot:</strong> Modified Nash Equilibrium after perturbation (intersection of colored lines)</li>}
        </ul>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto max-w-full">
        {/* Grid lines */}
        {xTicks.map((tick) => (
          <line
            key={`x-grid-${tick}`}
            x1={scaleX(tick)}
            y1={scaleY(0)}
            x2={scaleX(tick)}
            y2={scaleY(1)}
            stroke="#e0e0e0"
            strokeWidth="1"
          />
        ))}
        {yTicks.map((tick) => (
          <line
            key={`y-grid-${tick}`}
            x1={scaleX(0)}
            y1={scaleY(tick)}
            x2={scaleX(1)}
            y2={scaleY(tick)}
            stroke="#e0e0e0"
            strokeWidth="1"
          />
        ))}

        {/* Axes */}
        <line x1={scaleX(0)} y1={scaleY(0)} x2={scaleX(1)} y2={scaleY(0)} stroke="#333" strokeWidth="2" />
        <line x1={scaleX(0)} y1={scaleY(0)} x2={scaleX(0)} y2={scaleY(1)} stroke="#333" strokeWidth="2" />

        {/* X-axis ticks and labels */}
        {xTicks.map((tick) => (
          <g key={`x-tick-${tick}`}>
            <line
              x1={scaleX(tick)}
              y1={scaleY(0)}
              x2={scaleX(tick)}
              y2={scaleY(0) + 6}
              stroke="#333"
              strokeWidth="1.5"
            />
            <text x={scaleX(tick)} y={scaleY(0) + 22} textAnchor="middle" fontSize="14" fill="#444">
              {tick}
            </text>
          </g>
        ))}

        {/* Y-axis ticks and labels */}
        {yTicks.map((tick) => (
          <g key={`y-tick-${tick}`}>
            <line
              x1={scaleX(0) - 6}
              y1={scaleY(tick)}
              x2={scaleX(0)}
              y2={scaleY(tick)}
              stroke="#333"
              strokeWidth="1.5"
            />
            <text x={scaleX(0) - 12} y={scaleY(tick) + 5} textAnchor="end" fontSize="14" fill="#444">
              {tick}
            </text>
          </g>
        ))}

        {/* Axis labels */}
        <text x={scaleX(0.5)} y={height - 10} textAnchor="middle" fontSize="16" fill="#333" fontWeight="500">
          ⬥ Prob Player 1 Action A
        </text>
        <text
          x={20}
          y={scaleY(0.5)}
          textAnchor="middle"
          fontSize="16"
          fill="#333"
          fontWeight="500"
          transform={`rotate(-90, 20, ${scaleY(0.5)})`}
        >
          ⬥ Prob Player 2 Action A
        </text>

        {/* Original best response step functions (gray, dashed) - always shown */}
        {/* Player 1's BR - vertical at p=0, horizontal at q*, vertical at p=1 */}
        <line
          x1={scaleX(originalBR1_bottom.x)}
          y1={scaleY(originalBR1_bottom.y)}
          x2={scaleX(originalBR1_bottom.x2)}
          y2={scaleY(originalBR1_bottom.y2)}
          stroke={originalP1DominantB ? "#4a90d9" : "#b0b0b0"}
          strokeWidth={originalP1DominantB ? "6" : "3"}
          strokeOpacity={originalP1DominantB ? "0.8" : "0.5"}
          strokeDasharray={originalP1DominantB ? "0" : "8,4"}
        />
        {originalBR1_middle && (
          <line
            x1={scaleX(originalBR1_middle.x)}
            y1={scaleY(originalBR1_middle.y)}
            x2={scaleX(originalBR1_middle.x2)}
            y2={scaleY(originalBR1_middle.y2)}
            stroke="#b0b0b0"
            strokeWidth="3"
            strokeOpacity="0.5"
            strokeDasharray="8,4"
          />
        )}
        <line
          x1={scaleX(originalBR1_top.x)}
          y1={scaleY(originalBR1_top.y)}
          x2={scaleX(originalBR1_top.x2)}
          y2={scaleY(originalBR1_top.y2)}
          stroke={originalP1DominantA ? "#4a90d9" : "#b0b0b0"}
          strokeWidth={originalP1DominantA ? "6" : "3"}
          strokeOpacity={originalP1DominantA ? "0.8" : "0.5"}
          strokeDasharray={originalP1DominantA ? "0" : "8,4"}
        />

        {/* Player 2's BR - horizontal at q=1, vertical at p*, horizontal at q=0 */}
        <line
          x1={scaleX(originalBR2_left.x)}
          y1={scaleY(originalBR2_left.y)}
          x2={scaleX(originalBR2_left.x2)}
          y2={scaleY(originalBR2_left.y2)}
          stroke={originalP2DominantA ? "#e8a040" : "#b0b0b0"}
          strokeWidth={originalP2DominantA ? "6" : "3"}
          strokeOpacity={originalP2DominantA ? "0.8" : "0.5"}
          strokeDasharray={originalP2DominantA ? "0" : "8,4"}
        />
        {originalBR2_middle && (
          <line
            x1={scaleX(originalBR2_middle.x)}
            y1={scaleY(originalBR2_middle.y)}
            x2={scaleX(originalBR2_middle.x2)}
            y2={scaleY(originalBR2_middle.y2)}
            stroke="#b0b0b0"
            strokeWidth="3"
            strokeOpacity="0.5"
            strokeDasharray="8,4"
          />
        )}
        <line
          x1={scaleX(originalBR2_right.x)}
          y1={scaleY(originalBR2_right.y)}
          x2={scaleX(originalBR2_right.x2)}
          y2={scaleY(originalBR2_right.y2)}
          stroke={originalP2DominantB ? "#e8a040" : "#b0b0b0"}
          strokeWidth={originalP2DominantB ? "6" : "3"}
          strokeOpacity={originalP2DominantB ? "0.8" : "0.5"}
          strokeDasharray={originalP2DominantB ? "0" : "8,4"}
        />

        {/* Modified best response step functions (colored, solid) - only show if different */}
        {hasPerturbation && (
          <>
            {/* Player 1's BR - blue */}
            <line
              x1={scaleX(modifiedBR1_bottom.x)}
              y1={scaleY(modifiedBR1_bottom.y)}
              x2={scaleX(modifiedBR1_bottom.x2)}
              y2={scaleY(modifiedBR1_bottom.y2)}
              stroke="#4a90d9"
              strokeWidth="4"
            />
            {modifiedBR1_middle && (
              <line
                x1={scaleX(modifiedBR1_middle.x)}
                y1={scaleY(modifiedBR1_middle.y)}
                x2={scaleX(modifiedBR1_middle.x2)}
                y2={scaleY(modifiedBR1_middle.y2)}
                stroke="#4a90d9"
                strokeWidth="4"
              />
            )}
            <line
              x1={scaleX(modifiedBR1_top.x)}
              y1={scaleY(modifiedBR1_top.y)}
              x2={scaleX(modifiedBR1_top.x2)}
              y2={scaleY(modifiedBR1_top.y2)}
              stroke="#4a90d9"
              strokeWidth="4"
            />

            {/* Player 2's BR - orange */}
            <line
              x1={scaleX(modifiedBR2_left.x)}
              y1={scaleY(modifiedBR2_left.y)}
              x2={scaleX(modifiedBR2_left.x2)}
              y2={scaleY(modifiedBR2_left.y2)}
              stroke="#e8a040"
              strokeWidth="4"
            />
            {modifiedBR2_middle && (
              <line
                x1={scaleX(modifiedBR2_middle.x)}
                y1={scaleY(modifiedBR2_middle.y)}
                x2={scaleX(modifiedBR2_middle.x2)}
                y2={scaleY(modifiedBR2_middle.y2)}
                stroke="#e8a040"
                strokeWidth="4"
              />
            )}
            <line
              x1={scaleX(modifiedBR2_right.x)}
              y1={scaleY(modifiedBR2_right.y)}
              x2={scaleX(modifiedBR2_right.x2)}
              y2={scaleY(modifiedBR2_right.y2)}
              stroke="#e8a040"
              strokeWidth="4"
            />
          </>
        )}

        {/* Nash Equilibrium dot (BLACK) */}
        <circle
          cx={scaleX(Math.max(0, Math.min(1, nash.x)))}
          cy={scaleY(Math.max(0, Math.min(1, nash.y)))}
          r="12"
          fill="#333"
          stroke="#fff"
          strokeWidth="3"
        />
        <text
          x={scaleX(Math.max(0, Math.min(1, nash.x))) - 35}
          y={scaleY(Math.max(0, Math.min(1, nash.y))) - 20}
          fontSize="14"
          fill="#333"
          fontWeight="600"
        >
          Original NE
        </text>
        <text
          x={scaleX(Math.max(0, Math.min(1, nash.x))) - 25}
          y={scaleY(Math.max(0, Math.min(1, nash.y))) - 5}
          fontSize="13"
          fill="#555"
        >
          ({nash.x.toFixed(2)}, {nash.y.toFixed(2)})
        </text>

        {/* New NE dot (ORANGE) - only show if different from original */}
        {(Math.abs(newNE.x - nash.x) > 0.01 || Math.abs(newNE.y - nash.y) > 0.01) && (
          <>
            <circle
              cx={scaleX(Math.max(0, Math.min(1, newNE.x)))}
              cy={scaleY(Math.max(0, Math.min(1, newNE.y)))}
              r="12"
              fill="#e8a040"
              stroke="#fff"
              strokeWidth="3"
            />
            <text
              x={scaleX(Math.max(0, Math.min(1, newNE.x))) + 18}
              y={scaleY(Math.max(0, Math.min(1, newNE.y))) - 10}
              fontSize="14"
              fill="#e8a040"
              fontWeight="600"
            >
              Modified NE
            </text>
            <text
              x={scaleX(Math.max(0, Math.min(1, newNE.x))) + 18}
              y={scaleY(Math.max(0, Math.min(1, newNE.y))) + 8}
              fontSize="13"
              fill="#555"
            >
              ({newNE.x.toFixed(2)}, {newNE.y.toFixed(2)})
            </text>
          </>
        )}
      </svg>
    </div>
  )
}
