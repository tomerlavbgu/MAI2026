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

  const handleRangeChange = (values: number[]) => {
    let newMin = values[0]
    let newMax = values[1]
    const minGap = 1

    // Detect which thumb moved
    const minChanged = newMin !== minValue
    const maxChanged = newMax !== maxValue

    // Clamp to enforce gap
    if (minChanged && !maxChanged) {
      // MIN thumb is moving - ensure it doesn't get too close to MAX
      newMin = Math.min(newMin, maxValue - minGap)
    } else if (maxChanged && !minChanged) {
      // MAX thumb is moving - ensure it doesn't get too close to MIN
      newMax = Math.max(newMax, minValue + minGap)
    } else if (minChanged && maxChanged) {
      // Both changed (edge case) - maintain gap, prioritize MAX
      if (newMax - newMin < minGap) {
        newMin = newMax - minGap
      }
    }

    // Clamp to 0-100 range
    newMin = Math.max(0, Math.min(newMin, 100 - minGap))
    newMax = Math.max(minGap, Math.min(newMax, 100))

    // Final safety: if gap still violated after clamping, adjust MIN
    if (newMax - newMin < minGap) {
      newMin = Math.max(0, newMax - minGap)
    }

    // Only update if values actually changed (avoid infinite loops)
    if (newMin !== minValue) {
      onMinChange(newMin)
    }
    if (newMax !== maxValue) {
      onMaxChange(newMax)
    }
  }

  return (
    <div className="mt-4 sm:mt-6">
      <p className="text-gray-300 text-xs sm:text-sm mb-2 sm:mb-3">{label}</p>
      <div className="flex items-center gap-2 sm:gap-4">
        <span className="text-gray-400 text-xs sm:text-sm">0</span>
        <div className="flex-1">
          <Slider
            value={[minValue, maxValue]}
            onValueChange={handleRangeChange}
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
