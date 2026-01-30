"use client"

import { Slider } from "@/components/ui/slider"

interface RangeSliderProps {
  minValue: number
  maxValue: number
  onMinChange: (value: number) => void
  onMaxChange: (value: number) => void
  label?: string
}

export function RangeSlider({
  minValue,
  maxValue,
  onMinChange,
  onMaxChange,
  label = "Probability Constraint Range (0-100%)"
}: RangeSliderProps) {

  const handleRangeCommit = (values: number[]) => {
    let newMin = values[0]
    let newMax = values[1]
    const minGap = 1

    // Simple clamping - no need to detect which thumb moved
    // Just enforce the gap on final committed value

    // Ensure MIN doesn't exceed MAX - gap
    if (newMin > newMax - minGap) {
      newMin = newMax - minGap
    }

    // Ensure MAX doesn't go below MIN + gap
    if (newMax < newMin + minGap) {
      newMax = newMin + minGap
    }

    // Clamp to 0-100 bounds
    newMin = Math.max(0, newMin)
    newMax = Math.min(100, newMax)

    // Final safety: if still invalid after clamping, prioritize MAX
    if (newMax - newMin < minGap) {
      if (newMax === 100) {
        // MAX is at boundary, adjust MIN
        newMin = 100 - minGap
      } else if (newMin === 0) {
        // MIN is at boundary, adjust MAX
        newMax = minGap
      } else {
        // Neither at boundary, adjust MIN
        newMin = newMax - minGap
      }
    }

    // Always update both (no conditionals)
    onMinChange(newMin)
    onMaxChange(newMax)
  }

  return (
    <div className="mt-4 sm:mt-6">
      <p className="text-gray-300 text-xs sm:text-sm mb-2 sm:mb-3">{label}</p>
      <div className="flex items-center gap-2 sm:gap-4">
        <span className="text-gray-400 text-xs sm:text-sm">0</span>
        <div className="flex-1">
          <Slider
            value={[minValue, maxValue]}
            onValueCommit={handleRangeCommit}
            min={0}
            max={100}
            step={1}
            className="flex-1"
          />
          <div className="flex justify-between mt-1 px-1">
            <span className="text-gray-500 text-xs">MIN: {minValue}%</span>
            <span className="text-gray-500 text-xs">MAX: {maxValue}%</span>
          </div>
        </div>
        <span className="text-gray-400 text-xs sm:text-sm w-8 sm:w-10">100</span>
      </div>
    </div>
  )
}
