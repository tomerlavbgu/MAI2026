interface PerturbedMatrixProps {
  matrix: { original: number; perturbed: number }[][]
  rows: number
  cols: number
  actionLabelsRows?: string[]
  actionLabelsCols?: string[]
}

export function PerturbedMatrix({ matrix, rows, cols, actionLabelsRows, actionLabelsCols }: PerturbedMatrixProps) {
  const defaultRowLabels = rows === 2 ? ["A", "B"] : ["A", "B", "C"]
  const defaultColLabels = cols === 2 ? ["A", "B"] : ["A", "B", "C"]
  const rowLabels = actionLabelsRows || defaultRowLabels
  const colLabels = actionLabelsCols || defaultColLabels

  const getColor = (original: number, perturbed: number) => {
    return "bg-gray-100 border border-gray-200" // Neutral color for all cells
  }

  return (
    <div className="inline-block">
      {/* Column headers */}
      <div className="flex mb-1">
        <div className="w-8"></div>
        <div className="grid gap-1.5" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}>
          {colLabels.map((label) => (
            <div key={label} className="w-28 sm:w-32 text-center text-xs text-gray-500 font-medium">
              {label}
            </div>
          ))}
        </div>
      </div>

      {/* Matrix with row labels */}
      <div className="flex gap-1.5">
        {/* Row labels */}
        <div className="flex flex-col gap-1.5">
          {rowLabels.map((label) => (
            <div key={label} className="w-8 h-12 sm:h-14 flex items-center justify-center text-xs text-gray-500 font-medium">
              {label}
            </div>
          ))}
        </div>

        {/* Matrix cells */}
        <div className="grid gap-1.5" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}>
          {matrix.flat().map((cell, index) => {
            const bgColor = getColor(cell.original, cell.perturbed)
            return (
              <div
                key={index}
                className={`${bgColor} w-28 sm:w-32 h-12 sm:h-14 flex items-center justify-center rounded shadow-sm`}
              >
                <span className="text-gray-800 text-xs sm:text-sm font-medium">
                  {cell.original.toFixed(2)} ⇒ {cell.perturbed.toFixed(2)}
                </span>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
