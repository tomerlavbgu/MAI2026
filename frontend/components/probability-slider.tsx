"use client"

import { Slider } from "@/components/ui/slider"

interface ProbabilitySliderProps {
  value: number
  onChange: (value: number) => void
  label?: string
}

export function ProbabilitySlider({ value, onChange, label = "Player 1 Probability Constraints (0-100%)" }: ProbabilitySliderProps) {
  return (
    <div className="mt-4 sm:mt-6">
      <p className="text-gray-300 text-xs sm:text-sm mb-2 sm:mb-3">{label}</p>
      <div className="flex items-center gap-2 sm:gap-4">
        <span className="text-gray-400 text-xs sm:text-sm">0</span>
        <Slider value={[value]} onValueChange={(v) => onChange(v[0])} max={100} step={1} className="flex-1" />
        <span className="text-gray-400 text-xs sm:text-sm w-5 sm:w-6">{value}</span>
      </div>
    </div>
  )
}
