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
    // values[0] is MIN, values[1] is MAX
    onMinChange(values[0])
    onMaxChange(values[1])
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
