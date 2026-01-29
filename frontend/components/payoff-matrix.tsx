"use client"

import { Input } from "@/components/ui/input"

interface PayoffMatrixProps {
  matrix: number[][]
  onValueChange: (row: number, col: number, value: number) => void
  rows: number
  cols: number
  actionLabelsRows?: string[]
  actionLabelsCols?: string[]
}

export function PayoffMatrix({ matrix, onValueChange, rows: numRows, cols: numCols, actionLabelsRows, actionLabelsCols }: PayoffMatrixProps) {
  const defaultRowLabels = numRows === 2 ? ["A", "B"] : ["A", "B", "C"]
  const defaultColLabels = numCols === 2 ? ["A", "B"] : ["A", "B", "C"]
  const rowLabels = actionLabelsRows || defaultRowLabels
  const colLabels = actionLabelsCols || defaultColLabels

  // Generate arrays for iteration
  const rows = Array.from({ length: numRows }, (_, i) => i)
  const cols = Array.from({ length: numCols }, (_, i) => i)

  // Safety check: ensure matrix has correct dimensions
  if (!matrix || matrix.length !== numRows || !matrix[0] || matrix[0].length !== numCols) {
    return <div className="text-white">Loading matrix...</div>
  }

  return (
    <div className="inline-block">
      <table className="border-collapse">
        <thead>
          <tr>
            <th className="w-6 sm:w-10"></th>
            <th className="w-6 sm:w-10"></th>
            <th colSpan={numCols} className="text-gray-300 text-sm sm:text-base font-normal pb-2 text-center">
              Player 2
            </th>
          </tr>
          <tr>
            <th className="w-6 sm:w-10"></th>
            <th className="w-6 sm:w-10"></th>
            {cols.map((col) => (
              <th
                key={col}
                className="text-gray-300 text-sm sm:text-base font-normal px-2 sm:px-5 pb-2 text-center"
              >
                {colLabels[col]}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row}>
              {row === 0 && (
                <td
                  rowSpan={numRows}
                  className="text-gray-300 text-sm sm:text-base font-normal pr-1 sm:pr-2 align-middle"
                  style={{ writingMode: "vertical-rl", transform: "rotate(180deg)" }}
                >
                  Player 1
                </td>
              )}
              <td className="text-gray-300 text-sm sm:text-base font-normal pr-1 sm:pr-3 text-center">
                {rowLabels[row]}
              </td>
              {cols.map((col) => (
                <td key={col} className="p-1 sm:p-1.5">
                  <Input
                    type="text"
                    inputMode="decimal"
                    value={Math.round(matrix[row][col] * 100) / 100}
                    onChange={(e) => {
                      const val = e.target.value
                      if (val === '' || val === '-') {
                        onValueChange(row, col, 0)
                      } else if (!isNaN(Number(val))) {
                        onValueChange(row, col, Number(val))
                      }
                    }}
                    onFocus={(e) => e.target.select()}
                    className="w-16 sm:w-20 md:w-24 h-9 sm:h-10 md:h-11 text-center text-sm sm:text-base bg-white text-gray-800 border-gray-300"
                  />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
