"use client"

import { Slider } from "@/components/ui/slider"
import { useState, useEffect } from "react"

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

  // Local state for immediate visual feedback
  const [localMin, setLocalMin] = useState(minValue)
  const [localMax, setLocalMax] = useState(maxValue)

  // Sync local state when parent props change (e.g., preset load)
  useEffect(() => {
    setLocalMin(minValue)
    setLocalMax(maxValue)
  }, [minValue, maxValue])

  const handleRangeChange = (values: number[]) => {
    let newMin = values[0]
    let newMax = values[1]
    const minGap = 1

    // Enforce gap constraint immediately
    if (newMax - newMin < minGap) {
      // If gap violated, clamp based on which value changed more
      const minDiff = Math.abs(newMin - localMin)
      const maxDiff = Math.abs(newMax - localMax)

      if (minDiff > maxDiff) {
        // MIN moved more, adjust it
        newMin = Math.max(0, Math.min(newMin, newMax - minGap))
      } else {
        // MAX moved more, adjust it
        newMax = Math.min(100, Math.max(newMax, newMin + minGap))
      }
    }

    // Clamp to bounds
    newMin = Math.max(0, Math.min(newMin, 100 - minGap))
    newMax = Math.max(minGap, Math.min(newMax, 100))

    // Update local state immediately for visual feedback
    setLocalMin(newMin)
    setLocalMax(newMax)

    // Update parent
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
            value={[localMin, localMax]}
            onValueChange={handleRangeChange}
            min={0}
            max={100}
            step={1}
            className="flex-1"
          />
          <div className="flex justify-between mt-1 px-1">
            <span className="text-gray-500 text-xs">MIN: {localMin}%</span>
            <span className="text-gray-500 text-xs">MAX: {localMax}%</span>
          </div>
        </div>
        <span className="text-gray-400 text-xs sm:text-sm w-8 sm:w-10">100</span>
      </div>
    </div>
  )
}
